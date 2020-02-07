import copy
import random
from transformers import *

TOKENIZER = BertTokenizer.from_pretrained('/home/eric/evidence-inference/evidence_inference/models/structural_attn/scibert_scivocab_uncased')
#from load_data import TOKENIZER
CUT_OFF = 500

def label_to_val(label):
    """ Convert the @param label (is type string) to a natural number. """
    return {'INCR': 0, 'DECR': 1, 'SAME': 2, 'NULL': 3, 'Null': 0, 'CID': 1}.get(label)

def split_data(df):
    """ Split the data into train, dev, and test. """
    random.shuffle(df)
    train_split = int(len(df) * 0.6)
    dev_split   = int(len(df) * 0.8)
    return df[:train_split], df[train_split:dev_split], df[dev_split:]

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

def to_segmentation_ids(tokens):
    """
    Generate segment ids for each token, denoting which sentence each token is from.

    @param tokens are the tokens to iterate through and generate ids for.
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

        ### TO DO REMOVE THIS (we modify text here so keep it for now) ### 
        text, _  = to_segmentation_ids(d.tokenized_text)
        if len(doc_data) == 0: continue
        all_data.append({'text': text, 'relations': doc_data})
    
    if balance_classes:
        return balance_label_classes(all_data, labels)
    else:
        return all_data, labels

def balance_label_classes(all_data, labels):
    """
    Balance the classes in a way such that there is an equal distribution of all
    label types.

    @param all_data is a list of dictionaries, where each dictionary contains
    information about relationships (which is where labels are generated from).
    @param labels is an array of labels for each relation in each dictionary of
    all_data.
    @return a new balanced array of dictionaries, and the corresponding labels.
    """
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
