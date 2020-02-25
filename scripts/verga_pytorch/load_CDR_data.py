import sys
import glob
sys.path.append('..')
from classes import *
from load_data import Tokenized_Doc, TOKENIZER
import pickle

dir_ = './CDR_data/'
files = glob.glob(dir_ + '*.txt')

def word_piece_to_char_offset(word_piece_text, start, end):
    """ 
    Take an offset into word piece text, and convert it into character offsets.
    @param word_piece_text is a list of word pieces corresponding with the given start + end.
    @param start is the word piece token to start on.
    @param end is the word piece token to end on.
    @return a tuple corresponding to the new start/end
    """ 
    text = ''
    ch_start, ch_end = -1, -1
    word_pieces = word_piece_text.split(' ')
    for idx, wp in enumerate(word_pieces): 
        filtered_wp = wp.replace('@@', '')
        if idx == start:
            ch_start = len(text)

        if idx == end:
            ch_end   = len(text) # this may need to be -1 

        # only add a space if '@@' does not appear
        text += filtered_wp + ('' if wp != filtered_wp else ' ') 

    return (ch_start, ch_end)

def parse_CDR_document(doc_array):
    """ 
    Parse the document data (as list format) into Documents, Entities, and Spans. 
    @param doc_array is a 11 dimensional array with the first 5 entries corresponding 
    to entity1, the next 5 to entity 2, and the last to the text.
    @return a document class, and a relation entity map
    """
    text  = doc_array[-1]
    label = doc_array[-2]
    norm_text = text.replace('@@ ', '')

    # create entity 1
    entity1 = Entity(Span(-1, -1, doc_array[2]), doc_array[2])
    e1_idx_spans = zip(doc_array[3].split(':'), doc_array[4].split(':'))
    for s, e in e1_idx_spans:
        ch_start, ch_end = word_piece_to_char_offset(text, int(s), int(e))
        entity1.mentions.append(Span(ch_start, ch_end, doc_array[2]))

    # create entity 2
    entity2 = Entity(Span(-1, -1, doc_array[7]), doc_array[7])
    e2_idx_spans = zip(doc_array[8].split(':'), doc_array[9].split(':'))
    for s, e in e2_idx_spans:
        ch_start, ch_end = word_piece_to_char_offset(text, int(s), int(e))
        entity2.mentions.append(Span(ch_start, ch_end, doc_array[7]))

    document   = Doc(doc_array[10], norm_text)
    entity_map = ([entity1, entity2], {(entity1.text, entity2.text): label})
    return document, entity_map

def load_CDR_file(f):
    """ 
    Load a single CDR file, and parse into documents, relations, and entities. 
    @param f is a file location (string).
    """ 
    with open(f) as tmp:
        documents = tmp.read().split('\n')[:-1]
    
    docs, relation_maps = [], []
    for d in documents:
        doc_obj, rm_obj = parse_CDR_document(d.split('\t'))
        docs.append(doc_obj)
        relation_maps.append(rm_obj)

    id_to_entity_map = {} # id_ to entity_map
    id_to_document = {} # id_ to documents
    for d in docs: id_to_document[d.id] = d
    for entity_map, doc in zip(relation_maps, docs):
        if not(doc.id in id_to_entity_map):
            id_to_entity_map[doc.id] = entity_map
        else:
            id_to_entity_map[doc.id][0].extend(entity_map[0])
            id_to_entity_map[doc.id][1].update(entity_map[1])

    filtered_docs = []
    filtered_relations = []
    for id_ in id_to_document.keys():
        filtered_docs.append(id_to_document[id_])
        filtered_relations.append(id_to_entity_map[id_])

    return filtered_docs, filtered_relations

def main(files, tokenizer):
    """ Main function that takes in a list of CDR files to parse. """
    documents, relation_maps = [], []
    for f in files:
        d, rm = load_CDR_file(f) # list of documents
        documents.extend(d) # we don't care about the previous train/dev/val split
        relation_maps.extend(rm)
   
    tokenized_docs = []
    for d, entity_map in zip(documents, relation_maps):
        tokenized_docs.append(Tokenized_Doc(d.text, entity_map, tokenizer)) 

    return tokenized_docs

def load_CDR():
    """ TODO """
    with open('CDR_data.p', 'rb') as tmp:
        data = pickle.load(tmp)

    return data

#main(files, TOKENIZER)
