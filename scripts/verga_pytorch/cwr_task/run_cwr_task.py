import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

from utils import extract_data
from load_data import * 
from transformers import * 
from model import *
from load_CW4R_data import load_cwr4c_data

CUT_OFF = 500
RELATION_PATH = '../relation_model.pth'
NER_PATH = '../ner_model.pth'

def create_model():
    """
    Load the NER model and the relation model. 
    """
    relation_model = torch.load('../entire_relation.pth').cuda()
    ner_model = torch.load('../entire_ner.pth').cuda()
    
    #transformers.BertForTokenClassification.load_state_dict(torch.load(NER_PATH)).cuda()
    return ner_model, relation_model

def evaluate_model(relation_model, ner_model, test, batch_size):
    # evaluate on validation set
    test_outputs = []
    test_labels  = []
    test_ner_labels = []
    test_ner_scores = []
    test_loss = 0
    for batch_range in range(0, len(test), batch_size):
        data = test[batch_range: batch_range + batch_size]
        inputs, labels, _ = extract_data(data)
        text_inputs = [torch.tensor(input_['text'][:CUT_OFF]).cuda() for input_ in inputs]
        padded_text = PaddedSequence.autopad(text_inputs, batch_first = True, padding_value=0, device='cuda')

        # run model through validation data
        ner_mask=padded_text.mask(on=1.0, off=0.0, dtype=torch.float, device=padded_text.data.device)
        with torch.no_grad():
            ner_loss, ner_scores, hidden_states = ner_model(padded_text.data, attention_mask=ner_mask, labels = ner_batch_labels.data)
            relation_outputs, _ = relation_model(inputs, hidden_states[-2])
            
        loss = criterion(relation_outputs, torch.tensor(labels).cuda())
        test_loss += loss.item()
        test_outputs.extend(relation_outputs.cpu().numpy()) # or something like this
        test_labels.extend(labels)

        # GET NER SCORES
        test_ner_scores.extend(ner_scores.cpu().numpy())
        test_ner_labels.extend(ner_labels)

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
    print("Epoch {}\nF1 score: {}\nBinary F1: {}\nLoss: {}\nNER_F1: {}\n".format(epoch, f1, bin_f1, test_loss, ner_f1))
    return f1


def main(): 
    docs, diff = load_cwr4c_data()
    n, r = create_model()
    evaluate_model(n, r, docs, 8)
    import pdb; pdb.set_trace()

main()

