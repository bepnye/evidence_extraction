import copy
import random
import torch.nn as nn
import argparse
import sys
from load_data import *
from load_CDR_data import load_CDR
from model import *
from utils import *
import torch.optim as optim
from sklearn.metrics import f1_score, classification_report
import numpy as np
import transformers
from sklearn.cluster import AgglomerativeClustering as agg_cluster
import itertools
sys.path.append('../')
from classes import Entity, Span

CUT_OFF = 512
ev_inf_label_config = {'E1': 1, 'E2': 2, 'NULL': 3, 'NO_RELATION': 3}
cdr_label_config = {'E1': 0, 'E2': 1, 'NULL': 2, 'NO_RELATION': 0}

### Get set of indicies for a mention ### 
get_mention_set = lambda e: set(flatten([list(range(m.i, m.f)) for m in e.mentions]))

# FLATTEN A LIST OF LISTS #
flatten = lambda l: [item for sublist in l for item in sublist]

def overlaps(e1, e2):  
    """ Do these two entity objects overlap at all?"""
    return bool(get_mention_set(e1) & get_mention_set(e2))         

def get_dataset(dataset):
    """ Generate the dataset based on what is requested. """
    generate_data = {'evidence-inference': load_data, 'CDR': load_CDR}.get(dataset)
    return generate_data()

def create_model(dataset, bert_backprop, ner_path):
    """
    Create a model with the given specifications and return it.

    @param dataset specifies what dataset to use so we know how big our output dim should be.
    @param bert_backprop determines if we are to backprop through BERT.
    @return a model set up with the given specification.
    """
    assert(dataset in set(['evidence-inference', 'CDR']))
    output_dimensions = {'evidence-inference': 4, 'CDR': 2}.get(dataset)
    ner_dimensions = {'evidence-inference': 4, 'CDR': 3}.get(dataset)
    print("Loading relation model")
    #relation_model = BERTVergaPytorch(output_dimensions, bert_backprop=bert_backprop, initialize_bert=False).cuda()
    relation_model = VergaClone(output_dimensions, bert_backprop=bert_backprop, initialize_bert=False).cuda()
    print("Loading NER model")
    #ner_model = transformers.BertForTokenClassification.from_pretrained(ner_path, num_labels=ner_dimensions, output_hidden_states=True).cuda()
    ner_model = NotQuiteVergaNER(num_classes=ner_dimensions, bert_dim=relation_model.bert_dim).cuda()
    return ner_model, relation_model

def soft_scoring(batch_inputs, batch_orig_inputs, predictions, no_relation_label):
    """ Generate true positives, false negatives, and false positives. """
    """
    true_tuples = { (i, o, r) for (i, o), r in true_relations }
    pred_tuples = set()
    for (pred_i, pred_o) pred_r in model_outputs:
        match_found = False
        for (true_i, true_o), _ in true_relations:
            if overlaps(pred_i, true_i) and overlaps(pred_o, true_o):
                pred_tuples.add((true_i, true_o, pred_r))
                match_found = True

        if not match_found:
            pred_tuples.add((pred_i, true_i, pred_r))
     tp = len(true_tuples & pred_tuples)
     fp = len(pred_tuples - true_tuples)
     fn = len(true_tuples - pred_tuples)
    """
    pred_offset = 0
    scores = []
    assert len(batch_inputs) != 0
    assert len(batch_orig_inputs) != 0
    for inputs, orig_inputs in zip(batch_inputs, batch_orig_inputs):
        true_tuples = [(i, o, r) for (i, o), r in zip(orig_inputs['relations'], orig_inputs['labels'])]
        true_tuples = list(filter(lambda x: x[-1] != no_relation_label, true_tuples))
        pred_tuples = set()
        for (pred_i, pred_o), pred_r in zip(inputs['relations'], predictions[pred_offset:pred_offset+len(inputs['relations'])]):
            match_found = False
            if pred_r.cpu().item() == no_relation_label: 
                pred_offset += 1
                continue

            for (true_i, true_o, _) in true_tuples:
                if overlaps(pred_i, true_i) and overlaps(pred_o, true_o):
                    pred_tuples.add((true_i, true_o, pred_r.cpu().item()))
                    match_found = True

                if not(match_found):
                    pred_tuples.add((pred_i, pred_o, pred_r.cpu().item()))
                    # pseudo code is wrong 

            pred_offset += 1

        true_tuples = set(true_tuples)
        tp = len(true_tuples & pred_tuples)
        fp = len(pred_tuples - true_tuples)
        fn = len(true_tuples - pred_tuples)
        prec = tp / (tp + fp) if (tp + fp) != 0 else 0
        rec = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) != 0 else 0 
        scores.append((prec, rec, f1))

    return scores

def collapse_mentions(bert_mentions):
    """ 
    Take a list of bert tokens + Nones and collapse into mentions... 
    Example:
        [e1, e2, e3, None, None, e4, e5, None, None] -> [meanpool(e1, e2, e3), meanpool(e4, e5)]
    """
    to_collapse = []
    collapsed_mentions = []
    token_offsets = []
    start_span = 0
    for idx, mention in enumerate(bert_mentions):
        if mention is None and len(to_collapse) != 0:  
            collapsed_mentions.append(torch.mean(torch.stack(to_collapse), dim = 0))
            to_collapse = []
            token_offsets.append((start_span, idx))
        elif not(mention is None) and len(to_collapse) == 0:
            start_span = idx
            to_collapse.append(mention)
        elif not(mention is None): 
            to_collapse.append(mention)

    return collapsed_mentions, token_offsets

def create_entities_from_offsets(text, groups, offsets):
    """ 
    Create N entities where N is the number of unique groups in @param groups. Each 
    of these entities has spans described by offset.

    @param text is a word piece array of max size CUT_OFF (i.e. the abstract).
    @param groups  is an array corresponding to which offsets belong to what group.
    @param offsets is an array of tuples corresponding to start/end spans.
    @return a list of entities in length equal to len(set(groups))
    """
    entity_array = [None] * len(groups) # init to the max size possible
    for grp, span in zip(groups, offsets):
        selected_text = text[span[0]:span[1]] # this is tokenized... sorry :(
        if entity_array[grp] is None: entity_array[grp] = Entity(Span(-1, -1, 'N/A'), 'N/A')
        entity_array[grp].mentions.append(Span(span[0], span[1], selected_text))
       
    return list(filter(lambda x: not(x is None), entity_array))

def create_simulated_relations(inputs, intv_groups, intv_offsets, out_groups, out_offsets, label_config, test_set = False):
    """
    Create intervention and outcome groups with corresponding relations
    """ 
    relation_map = [None] * len(inputs['text'])
    ### Alright... we have a bunch of interventions + outcome pairs that we need to get labels for.
    ### What we need to know is if this intervention/outcome group has any offsets that align with an
    ### entity in the gold standard set. If we see something that has overlaping intervals for BOTH
    ### the outcome and intervention, then awesome (we take that label to be ours), otherwise we 
    ### have a NULL label.
    # ENTITY: def __init__(self, span, entity_type, label = None)
    # SPAN:   def __init__(self, i, f, text, label = None) 
    outcome_entities = create_entities_from_offsets(text=inputs['text'], groups=out_groups, offsets=out_offsets)
    intv_entities = create_entities_from_offsets(text=inputs['text'], groups=intv_groups, offsets=intv_offsets)
    pairs = itertools.product(intv_entities, outcome_entities) # every combo of intervention + outcome 

    ### Start assembling new inputs (so as to not modify data) for this batch ### 
    batch_labels, batch_input = [], {'text': inputs['text'], 'relations': [], 'labels': []}
    shuffled = list(zip(inputs['relations'].copy(), inputs['labels'].copy()))
    random.shuffle(shuffled)
    shuffled_relations = [s[0] for s in shuffled]
    shuffled_labels    = [s[1] for s in shuffled]
    for intv, out in pairs:
        idx, success = 0, False
        for e1, e2 in shuffled_relations:
            intv_overlap = bool(get_mention_set(e1) & get_mention_set(intv)) # do my sets overlap
            out_overlap  = bool(get_mention_set(e2) & get_mention_set(out)) 
            if intv_overlap and out_overlap:
                batch_labels.append(shuffled_labels[idx])
                batch_input['relations'].append((intv, out))
                batch_input['labels'].append(shuffled_labels[idx])
                success = True
                break
            
            idx += 1

        ### couldn't find anything ###
        if not(success) and (random.uniform(0, 1) < 0.01 or test_set):
            batch_labels.append(label_config.get('NO_RELATION')) # add null
            batch_input['labels'].append(label_config.get('NO_RELATION'))
            batch_input['relations'].append((intv, out))
    
    return batch_input, batch_labels

def get_arg_scores(batch_inputs, batch_ner_scores):
    batch_labels = []
    for inputs, ner_scores in zip(batch_inputs, batch_ner_scores):
        word_pieces = TOKENIZER.tokenize(TOKENIZER.decode(inputs['text'], clean_up_tokenization_spaces=False))
        ner_scores = torch.argmax(ner_scores, dim = -1)
        assert(min(512, len(word_pieces)) <= len(ner_scores))
        last_valid = 0
        labels = []
        for idx, w in enumerate(word_pieces):
            if idx >= CUT_OFF: 
                break
            elif "##" in w:
                labels.append(ner_scores[last_valid])
            else:
                last_valid = idx
                labels.append(ner_scores[idx])
        
        labels.extend(ner_scores[idx + 1:])
        batch_labels.append(torch.stack(labels))

    return batch_labels

def agglomerative_coref(inputs, ner_scores, bert_embedds, true_labels, label_config, test_set = False): 
    """ 
    Perform COREF grouping and modify the labels to correspond to the ones given by the argmax ner scores.
    @param inputs       is what we will eventually pass to the relation model
    @param ner_scores   is the scoring of each token as population/intervention/outcome/NULL.
    @param bert_embedds is the token embeddings for each word piece in the abstract.
    @param true_labels  is what the ner_scores SHOULD be once argmaxed.
    @param label_config what is the mapping of E1 and E2 to values (given as a dictionary -> {'E1': 0, 'E2': 1,...})? 
    @param test_set     are we running on the test set data? 
    """
    arg_scores = get_arg_scores(inputs, ner_scores)#torch.stack([torch.argmax(x, dim = 1) for x in ner_scores])
    new_inputs = []
    batch_labels = []
    for i in range(len(arg_scores)):
        intervention_embeds, intv_offsets = collapse_mentions([bert_embedds[i][idx] if sc == label_config['E1'] else None for idx, sc in enumerate(arg_scores[i])])
        outcome_embeds, outcome_offsets = collapse_mentions([bert_embedds[i][idx] if sc == label_config['E2'] else None for idx, sc in enumerate(arg_scores[i])])      
        if len(intervention_embeds) == 0 or len(outcome_embeds) == 0: continue
        intv_model = agg_cluster(n_clusters=None, affinity='euclidean', linkage='ward', distance_threshold=5)
        out_model  = agg_cluster(n_clusters=None, affinity='euclidean', linkage='ward', distance_threshold=5)  
        intv_groups = intv_model.fit_predict(torch.stack(intervention_embeds).cpu().detach().numpy()) if len(intervention_embeds) != 1 else [0]
        out_groups  = out_model.fit_predict(torch.stack(outcome_embeds).cpu().detach().numpy()) if len(outcome_embeds) != 1 else [0]
        new_input, new_labels = create_simulated_relations(inputs[i], intv_groups, intv_offsets, out_groups, outcome_offsets, label_config, test_set)
        
        ### Add to our list of new inputs + labels ### 
        new_inputs.append(new_input)
        batch_labels.extend(new_labels)

    return new_inputs, batch_labels

def evaluate_model(relation_model, ner_model, label_config, criterion, test, epoch, batch_size, teacher_forcing=False):
    # evaluate on validation set
    relation_model.eval()
    ner_model.eval()
    test_outputs = []
    test_labels  = []
    test_ner_labels = []
    test_ner_scores = []
    test_loss = 0
    all_scores = []
    for batch_range in range(0, len(test), batch_size):
        data = test[batch_range: batch_range + batch_size]
        inputs, labels, ner_labels = extract_data(data, label_config)
        if len(inputs) == 0: continue
        ner_batch_labels = PaddedSequence.autopad([lab[:CUT_OFF] for lab in ner_labels], batch_first=True, padding_value=label_config.get('NULL'), device='cuda')
        text_inputs = [torch.tensor(input_['text'][:CUT_OFF]).cuda() for input_ in inputs]
        padded_text = PaddedSequence.autopad(text_inputs, batch_first = True, padding_value=0, device='cuda')

        # run model through validation data
        ner_mask=padded_text.mask(on=1.0, off=0.0, dtype=torch.float, device=padded_text.data.device)
        with torch.no_grad():
            orig_inputs = inputs
            assert padded_text.data.size()[:2] == ner_batch_labels.data.size()[:2]
            ner_loss, ner_scores, hidden_states = ner_model(padded_text.data, attention_mask=ner_mask, labels = ner_batch_labels.data)
            if not(teacher_forcing):
                inputs, labels = agglomerative_coref(inputs, ner_scores, hidden_states[-2], ner_labels[batch_range: batch_range + batch_size], label_config, test_set = True)
                if len(labels) == 0: continue
                relation_outputs = relation_model(inputs, hidden_states[-2])
                scores = soft_scoring(inputs, orig_inputs, relation_outputs.argmax(dim=-1), label_config['NO_RELATION'])
            else:
                relation_outputs = relation_model(inputs, hidden_states[-2])
                scores = [(0, 0, 0)]

            all_scores.extend(scores)
    
        loss = criterion(relation_outputs, torch.tensor(labels).cuda())
        test_loss += loss.item()
        test_outputs.extend(relation_outputs.cpu().numpy()) # or something like this
        test_labels.extend(labels)

        # GET NER SCORES
        test_ner_scores.extend(ner_scores.cpu().numpy())
        test_ner_labels.extend(ner_labels)
   
    if len(test_ner_labels) == 0: 
        print("Error! Not enough labels. Please fix.")
        return 0

    soft_prec = np.mean([x[0] for x in all_scores])
    soft_rec = np.mean([x[1] for x in all_scores])
    soft_f1 = np.mean([x[2] for x in all_scores]) 

    ### REMOVE PADDED PREDICTIONS, GET NER F1 ###
    for idx, n in enumerate(test_ner_labels):
        new_size = min(CUT_OFF, len(n))
        test_ner_labels[idx] = n[:new_size]
        test_ner_scores[idx] = [np.argmax(x) for x in test_ner_scores[idx][:new_size]]

    ### FLATTEN LIST OF LISTS TO LIST ###
    test_ner_labels = flatten(test_ner_labels)
    test_ner_scores = flatten(test_ner_scores)

    ### GET AND PRINT SCORE ### 
    ner_f1 = f1_score(test_ner_labels, test_ner_scores, average='macro')
    print("NER CLASSIFICATION REPORT:\n")
    print(classification_report(test_ner_labels, test_ner_scores))

    ### DO CALCULATIONS FOR FINAL F1 ###
    outputs = [np.argmax(x) for x in test_outputs]
    labels  = test_labels
    f1 = f1_score(labels, outputs, average = 'macro')
    if max(labels) > 1 or max(outputs) > 1:
        print("BINARY CLASSIFICATION REPORT:\n{}".format(classification_report(labels, outputs)))
        outputs = [1 if x != 3 else 0 for x in outputs]
        labels  = [1 if x != 3 else 0 for x in labels]
        bin_f1  = f1_score(labels, outputs, average = 'macro')
    else:
        bin_f1 = 0

    print("FULL TASK REPORT\n{}".format(classification_report(labels, outputs)))
    print("Epoch {}\nSoft F1: {}\nSoft Precision: {}\nSoft Recall: {}\nF1 score: {}\nBinary F1: {}\nLoss: {}\nNER_F1: {}\n".format(epoch, soft_f1, soft_prec, soft_rec, f1, bin_f1, test_loss, ner_f1))
    return f1

def train_model(ner_model, relation_model, df, parameters):
    """ Take a model and train it with the given data. """
    # get parameters of how to train model
    epochs     = parameters.epochs
    batch_size = parameters.batch_size
    balance_classes = parameters.balance_classes
    learning_rate   = parameters.lr
    ner_loss_weighting = parameters.ner_loss_weight
    teacher_force_ratio = parameters.teacher_forcing_ratio
    teacher_force_decay = parameters.teacher_forcing_decay
    label_config = {'evidence-inference': ev_inf_label_config, 'CDR': cdr_label_config}.get(parameters.dataset)
    path_ner = "ner_model_lr_{}_epochs_{}_ner_loss_w_{}_start_tfr_{}_decay_{}.pth".format(learning_rate, epochs, ner_loss_weighting, teacher_force_ratio, teacher_force_decay)
    path_relation = "relation_model_lr_{}_epochs_{}_ner_loss_w_{}_start_tfr_{}_decay_{}.pth".format(learning_rate, epochs, ner_loss_weighting, teacher_force_ratio, teacher_force_decay)           
    assert(ner_loss_weighting <= 1.0 and ner_loss_weighting >= 0.0)

    # split data, set up our optimizers
    best_model = None
    max_f1_score = 0 # best f1 score seen thus far
    train, dev, test = split_data(df, parameters.percent_train)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(ner_model.parameters()) + list(relation_model.parameters()), lr = learning_rate)
    for epoch in range(epochs):
        ner_model.train()
        relation_model.train()
        # define losses to use later
        label_offset  = 0 
        training_loss = 0
        train_data, train_labels, ner_labels = extract_data(train, label_config, balance_classes == 'True')        
        teacher_force = True if random.uniform(0, 1) < teacher_force_ratio else False

        # single epoch train
        for batch_range in range(0, len(train_data), batch_size):
            inputs = train_data[batch_range: batch_range + batch_size] 
            ner_batch_labels = PaddedSequence.autopad([lab[:CUT_OFF] for lab in ner_labels[batch_range: batch_range + batch_size]], batch_first=True, padding_value=label_config['NULL'], device='cuda')
            text_inputs = [torch.tensor(input_['text'][:CUT_OFF]).cuda() for input_ in inputs]    
            padded_text = PaddedSequence.autopad(text_inputs, batch_first = True, padding_value=0, device='cuda')

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backwards + optimize
            ner_mask=padded_text.mask(on=1.0, off=0.0, dtype=torch.float, device=padded_text.data.device)
            ner_loss, ner_scores, hidden_states = ner_model(padded_text.data,
                                                            attention_mask=ner_mask,
                                                            labels = ner_batch_labels.data)
            
            if not(teacher_force): 
                inputs, labels = agglomerative_coref(inputs, ner_scores, hidden_states[-2], ner_labels[batch_range: batch_range + batch_size], label_config)
                if len(labels) == 0: continue
                relation_outputs = relation_model(inputs, hidden_states[-2])
            else:
                relation_outputs = relation_model(inputs, hidden_states[-2])
                labels = train_labels[label_offset: label_offset + len(relation_outputs)]

            # loss
            relation_loss = criterion(relation_outputs, torch.tensor(labels).cuda())
            loss = ner_loss_weighting * ner_loss + (1 - ner_loss_weighting) * relation_loss
            loss.backward()
            optimizer.step()

            # add loss
            training_loss += loss.item()

            # next labels
            label_offset += len(relation_outputs)

        ### Update teacher forcing ### 
        teacher_force_ratio = min(0, teacher_force_ratio - teacher_force_decay)

        ### Print the losses and evaluate on the dev set ###
        print("Epoch {} Training Loss: {}\n".format(epoch, training_loss))
        print("train scores with teacher forcing")
        f1_score = evaluate_model(relation_model, ner_model, label_config, criterion, train, epoch, batch_size)
        evaluate_model(relation_model, ner_model, label_config, criterion, train, epoch, batch_size, teacher_forcing = True)
        print("dev scores with teacher forcing")
        f1_score = evaluate_model(relation_model, ner_model, label_config, criterion, dev, epoch, batch_size)
        evaluate_model(relation_model, ner_model, label_config, criterion, dev, epoch, batch_size, teacher_forcing = True)

        # update our scores to find the best possible model
        best_model   = (copy.deepcopy(ner_model), copy.deepcopy(relation_model))  if max_f1_score < f1_score else best_model
        max_f1_score = max(max_f1_score, f1_score)

    print("Final test run:\n")
    evaluate_model(best_model[1], best_model[0], label_config, criterion, test, epoch, batch_size)
    torch.save(best_model[0], path_ner)
    torch.save(best_model[1], path_relation)

def main(args=sys.argv[1:]):
    ### Use arg parser to read in data. ###
    parser = argparse.ArgumentParser(description="Parses input specifications of how to run model.")
    parser.add_argument("--dataset", dest='dataset', required=True, help="Which dataset to use.")
    parser.add_argument("--epochs", dest='epochs', type=int, required=True, help="How many epochs to run model for.")
    parser.add_argument("--learning_rate", dest='lr', type=float, required=True, help="What should the learning rate be?")
    parser.add_argument("--batch_size", dest='batch_size', type=int, required=True, help="What should the batch_size be?")
    parser.add_argument("--balance_classes", dest="balance_classes", default=False, help="Should we balance the classes for you?")
    parser.add_argument("--bert_backprop", dest="bert_backprop", default=False, help="Should we backprop through BERT?")
    parser.add_argument("--percent_train", dest="percent_train", type=float, default=1, help="What percent if the training data should we use?")
    parser.add_argument("--ner_loss_weight", dest="ner_loss_weight", type=float, required=True, help="Relative loss weight for NER task (b/w 0 and 1)")
    parser.add_argument("--teacher_forcing_ratio", dest="teacher_forcing_ratio", type=float, required=True, help="What teacher forcing ratio do you want during training?")
    parser.add_argument("--teacher_forcing_decay", dest="teacher_forcing_decay", type=float, required=True, help="What decay after each epoch?")
    parser.add_argument("--teacher_forcing_evaluation", dest="teacher_forcing_evaluation", action='store_true', help="Should we evaluate with real NER labels?")
    args = parser.parse_args()
    print("Running with the given arguments:\n\n{}".format(args))

    get_ner_path = lambda x: {'evidence-inference': NER_BERT_LOCATION, 'CDR': CDR_NER_BERT_LOCATION}.get(x)
    ## GET THE DATA ###
    df = get_dataset(args.dataset)

    ### LOAD THE MODEL ###
    ner_model, relation_model = create_model(dataset=args.dataset,
                                             bert_backprop=args.bert_backprop == 'True',
                                             ner_path=get_ner_path(args.dataset))

    ### TRAIN ###
    train_model(ner_model, relation_model, df, args)

if __name__ == '__main__':
    main()
