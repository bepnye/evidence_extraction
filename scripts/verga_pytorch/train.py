import copy
import random
import torch.nn as nn
import argparse
import sys
from load_data import *
from load_CDR_data import load_CDR  
from model import * 
import torch.optim as optim 
from sklearn.metrics import f1_score, classification_report
import numpy as np
CUT_OFF = 500 

def find_entity_match(entity_list, str_):
    """ 
    Find which entity matches the str_ description (i.e. compare entity.text to str_).
    If there are no matches, return None, otherwise return the matched entity.
    @param entity_list a list of entities to search in.
    @param str_        a string to match the entities with.
    """
    for e in entity_list:
        if e.text == str_:
            return e

    return None

def label_to_val(label):
    """ Convert the @param label (is type string) to a natural number. """
    return {'INCR': 0, 'DECR': 1, 'SAME': 2, 'NULL': 3, 'Null': 0, 'CID': 1}.get(label)

def split_data(df):
    """ Split the data into train, dev, and test. """
    random.shuffle(df)
    train_split = int(len(df) * 0.6)
    dev_split   = int(len(df) * 0.8)
    return df[:train_split], df[train_split:dev_split], df[dev_split:]

def to_segmentation_ids(tokens):
    """ 
    Take @param tokens and iterate through it to get segmentation ids. 
    Convert text at end to correct format (of token values).
    """
    start = 0
    segments = []
    for t in tokens:
        if t == '[SEP]':
            start += 1

        segments.append(start)

    return TOKENIZER.convert_tokens_to_ids(tokens), segments

def extract_data(df, balance_classes = False):
    """ 
    Extract the data from the classes and reformat it. 
    
    Goal: Create tuples of 3 -> (entity1, entity2, relation).
    
    The current entity map is (string1, string2): string3, 
    where string1 and string2 are string versions of the entities,
    and string 3 is the label. 
    """
    random.shuffle(df)
    all_data   = []
    labels = []
    invalid_entry = 0
    for d in df:
        mapping   = d.entity_map[0]
        relations = d.entity_map[1] # this is a dictionary
        doc_data  = []
        for key in relations.keys():
            # find what entity matches us
            entity1, entity2 = find_entity_match(mapping, key[0]), find_entity_match(mapping, key[1])
            entity1.mentions = list(filter(lambda m: m.i < CUT_OFF, entity1.mentions))
            entity2.mentions = list(filter(lambda m: m.i < CUT_OFF, entity2.mentions))
            if len(entity1.mentions) == 0 or len(entity2.mentions) == 0:
                invalid_entry += 1
                continue

            doc_data.append((entity1, entity2))
            labels.append(label_to_val(relations[key]))
            assert(not(entity1 is None) and not(entity2 is None)) 
            
        text, segment_ids = to_segmentation_ids(d.tokenized_text)
        all_data.append({'text': text, 'segment_ids': segment_ids, 'relations': doc_data})

    #print("Number of invalid ({}) / Total Samples ({}) = {}".format(invalid_entry, len(labels), round(invalid_entry/ len(labels), 2) if len(labels) != 0 else 0))
    if balance_classes:
        label_types = set(labels)
        smallest_class = min([labels.count(s) for s in label_types])
        num_taken = [0 for _ in label_types]
        
        n_labels, n_docs = 0, 0
        new_data, new_labels = [], []
        while (sum(num_taken) < (smallest_class * len(label_types))):
            copied_data = copy.deepcopy(all_data[n_docs])
            new_relation_list = []
            for x in all_data[n_docs]['relations']: 
                if num_taken[labels[n_labels]] < smallest_class:
                    new_relation_list.append(x)
                    new_labels.append(labels[n_labels])
                    num_taken[labels[n_labels]] += 1

                n_labels += 1
                copied_data['relations'] = new_relation_list    

            if len(new_relation_list) != 0:
                new_data.append(copied_data)

            n_docs += 1

        return new_data, new_labels
    else:
        return all_data, labels         

def create_model(dataset, bert_backprop, ner_loss):
    """ 
    Create a model with the given specifications and return it.

    @param dataset specifies what dataset to use so we know how big our output dim should be.
    @param bert_backprop determines if we are to backprop through BERT.
    @param ner_loss determines if we are to add NER loss to our outputs.
    @return a model set up with the given specification.
    """
    output_dimensions = {'evidence-inference': 4, 'CDR': 2}.get(dataset)
    
    return BERTVergaPytorch(output_dimensions, bert_backprop = bert_backprop, ner_loss = ner_loss)

def train_model(model, df, parameters): 
    """ Take a model and train it with the given data. """
    # get parameters of how to train model 
    epochs     = parameters.epochs
    batch_size = parameters.batch_size
    balance_classes = parameters.balance_classes
    learning_rate   = parameters.lr

    # split data, set up our optimizers
    train, dev, test = split_data(df)
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr = learning_rate) 
    for epoch in range(epochs):
        # define losses to use later
        training_loss = 0
        dev_loss      = 0 
       
        train_data, train_labels = extract_data(train, balance_classes)
        label_offset = 0
        # single epoch train
        for batch_range in range(0, len(train), batch_size): 
            inputs = train_data[batch_range: batch_range + batch_size]
           
            if len(inputs) == 0: continue
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backwards + optimize
            outputs = model(inputs)
            labels  = train_labels[label_offset: label_offset + len(outputs)]
            loss = criterion(outputs, torch.tensor(labels).cuda())
            loss.backward()
            optimizer.step()

            # add loss 
            training_loss += loss.item()
        
            # next labels
            label_offset += len(outputs)
         
        # evaluate on validation set
        dev_outputs = []
        dev_labels  = []
        for batch_range in range(0, len(dev), batch_size):
            data = dev[batch_range: batch_range + batch_size]
            inputs, labels = extract_data(data) 
            if len(labels) == 0: continue 

            # run model through validation data
            with torch.no_grad():
                outputs = model(inputs)
            
            loss = criterion(outputs, torch.tensor(labels).cuda())
            dev_loss += loss.item()
            dev_outputs.extend(outputs.cpu().numpy()) # or something like this
            dev_labels.extend(labels)
        
        outputs = [np.argmax(x) for x in dev_outputs]
        labels  = dev_labels
        f1 = f1_score(labels, outputs, average = 'macro')
        if max(labels) > 1 or max(outputs) > 1:
            print(classification_report(labels, outputs))
            outputs = [1 if x != 3 else 0 for x in outputs]
            labels  = [1 if x != 3 else 0 for x in labels]
            bin_f1  = f1_score(labels, outputs, average = 'macro')
        else:
            bin_f1 = 0

        print(classification_report(labels, outputs))
        print("Epoch {}\nDev F1 score: {}\nDev bin F1: {}\nDev Loss: {}\nTraining Loss: {}\n\n".format(epoch, f1, bin_f1, dev_loss, training_loss))
        
    test_outputs = []
    test_labels  = []
    for batch_range in range(0, len(test), batch_size):
        data = test[batch_range: batch_range + batch_size]
        inputs, labels = extract_data(data)
        if len(labels) == 0: continue

        with torch.no_grad():
            outputs = model(inputs)

        test_outputs.extend(outputs.cpu().numpy())
        test_labels.extend(labels)

    outputs = [np.argmax(x) for x in test_outputs]
    labels  = test_labels
    f1 = f1_score(labels, outputs, average = 'macro')
    
    if max(labels) > 1 or max(outputs) > 1:
        print(classification_report(labels, outputs))
        outputs = [1 if x != 3 else 0 for x in outputs]
        labels  = [1 if x != 3 else 0 for x in labels]
        bin_f1  = f1_score(labels, outputs, average = 'macro')
    else:
        bin_f1 = 0

    print(classification_report(labels, outputs))
    print("Test F1 score: {}\nBinary Test F1 score: {}".format(f1, bin_f1))

def main(args=sys.argv[1:]): 
    ### Use arg parser to read in data. ###
    parser = argparse.ArgumentParser(description="Parses input specifications of how to run model.")
    parser.add_argument("--dataset", dest='dataset', required=True, help="Which dataset to use.")
    parser.add_argument("--epochs", dest='epochs', type=int, required=True, help="How many epochs to run model for.")
    parser.add_argument("--learning_rate", dest='lr', type=float, required=True, help="What should the learning rate be?")
    parser.add_argument("--batch_size", dest='batch_size', type=int, required=True, help="What should the batch_size be?")
    parser.add_argument("--balance_classes", dest="balance_classes", type=bool, required=True, help="Should we balance the classes for you?")
    parser.add_argument("--bert_backprop", dest="bert_backprop", type=bool, required=True, help="Should we backprop through BERT?")
    parser.add_argument("--ner_loss", dest="ner_loss", type=str, default="NULL", help="Should we add NER loss to model (select 'joint' or 'alternate'") 
    print("Running with the given arguments:\n\n{}".format(parser))

    ### GET THE DATA ### 
    df = get_dataset(parser.dataset) 

    ### LOAD THE MODEL ###
    model = create_model(dataset=parser.dataset, bert_backprop=parser.bert_backprop, ner_loss=parser.ner_loss) 

    ### TRAIN ###
    train_model(model, df, parser)

if __name__ == '__main__':
    main()
