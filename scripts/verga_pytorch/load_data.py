import os
import sys
sys.path.append('..')
sys.path.append('../..')
import config
import pickle
import processing
import itertools
import process_evidence_inference
sys.path.append('/home/eric/bran/src/processing/utils/')
#from word_piece_tokenizer import WordPieceTokenizer as WPT
from transformers import *
global TOKENIZER, SEP_TOKEN, CLS_TOKEN, SEP_SIZE, CLS_SIZE
TOKENIZER = BertTokenizer.from_pretrained('/home/eric/evidence-inference/evidence_inference/models/structural_attn/scibert_scivocab_uncased') #WPT('/home/eric/bran/data/cdr/word_piece_vocabs/just_train_2500/word_pieces.txt', entity_str = 'ENTITY_')  
SEP_TOKEN = " [SEP] "
CLS_TOKEN = " [CLS] "
CLS_SIZE = len(CLS_TOKEN)
SEP_SIZE = len(SEP_TOKEN)

import spacy
import string
nlp = spacy.load("en_core_web_sm")

def text_to_sentence_offsets(text):
    """
    Takes a string text variable and calculates the index of sentences.

    @param text is a string text to be used for calculations
    @return an array of tuples corresponding to the start/end of each sentence.
    """
    doc = nlp(text)
    sentences = list(doc.sents)
    sentence_st_end = []
    previous_offset = 0
    for s in sentences:
        sentence_st_end.append((previous_offset, previous_offset + len(s.text)))
        previous_offset += len(s.text)

    return sentence_st_end

def character_index_to_token(text, ch_st, ch_end, tokenizer):
    """ 
    Takes a character index and converts it to a token index. 
    
    @param text is the NON-TOKENIZED plain text 
    @param ch_st is the character start of span.
    @param ch_end is the character end of the span.
    @param tokenizer is an object with function tokenize that turns strings -> list of tokens.
    @return a tuple denoting the tokenized start + tokenized end.
    """
    before = tokenizer.tokenize(text[:ch_st])
    span   = tokenizer.tokenize(text[ch_st:ch_end])
    return (len(before), len(before) + len(span))   

def add_CLS_token(text, entity_map):
    """
    Add the CLS token to the beginning of the text and recalculate offsets.
    @param entity_map is the offset and string for all entities..
    @return new text, new entity map.
    """
    final_text = CLS_TOKEN + text
    for entity in entity_map:
        for span in entity.mentions:
            span.i += CLS_SIZE
            span.f += CLS_SIZE

    return final_text, entity_map

def insert_sep_tokens(text, tokenizer, entity_map):
    """ 
    Insert [SEP] tokens at the end of each sentence. Return a tokenized array of the text,
    the new text, and the new CHARACTER offsets, since we are modifying the text as well.
    The length of the output should be equal to tokenizer.tokenize(text)

    @param text is a string text representation.
    @param tokenizer is an object with function tokenize that turns strings -> list of tokens.
    @param entity_map is the list of entities that need to be modified.
    @return the new text, new tokenized version of the text, and the updated entity list.
    """
    assert(not(SEP_TOKEN in text))
    offset   = 0
    counts   = [[0, 0] for _ in entity_map]
    final_text = ""
    sentences  = [str(x.text) for x in list(nlp(text).sents)]
    for idx, s in enumerate(sentences):
        offset      += len(s) 
        offset_text = text[offset:]
        whitespace  = offset_text[:offset_text.index(sentences[idx + 1])] if idx < len(sentences) - 1 else ""
        offset      += len(whitespace)
        n_occur     = final_text.count(SEP_TOKEN)
    
        for list_idx, seen_idx in enumerate(counts):
            start_seen_idx, end_seen_idx = seen_idx
            mention_len = len(entity_map[list_idx].mentions)
            
            while (start_seen_idx < mention_len and entity_map[list_idx].mentions[start_seen_idx].i <= offset):
                span = entity_map[list_idx].mentions[start_seen_idx]
                span.i += SEP_SIZE * n_occur
                counts[list_idx][0] += 1
                start_seen_idx += 1

            while (end_seen_idx < mention_len and entity_map[list_idx].mentions[end_seen_idx].f <= offset):
                span = entity_map[list_idx].mentions[end_seen_idx]
                span.f += SEP_SIZE * n_occur
                counts[list_idx][1] += 1
                end_seen_idx += 1

        final_text += s + whitespace + SEP_TOKEN  
    
    final_text, entity_map = add_CLS_token(final_text, entity_map)
    return final_text, tokenizer.tokenize(final_text), entity_map

class Tokenized_Doc:
    """ A document that has all data represented as tokens. """

    def __init__(self, text, entity_map, tokenizer):
        """ 
        @param text is the original text.
        @param entity_map is a tuple of a list of entities and relations.
        @param tokenizer is an object with function tokenize that turns strings -> list of tokens.
        @param token_frames should be a list of tokenized frames consisting of Tokenized_Frame class.
        """
        # modify mentions to be dictionaries... 
        # we need to insert [SEP] tokens at the correct spots.                                                
        text, tokenized_text, updated_entity_map = insert_sep_tokens(text, tokenizer, entity_map[0])
        for entity in updated_entity_map:
            for mention in entity.mentions:
                start, end = character_index_to_token(text, mention.i, mention.f, tokenizer)
                mention.i = start
                mention.f = end
        
        self.entity_map     = entity_map
        self.text           = text
        self.tokenized_text = tokenized_text

class Tokenized_Frame:
    """ 
    An evidence frame where everything is tokenized and all offsets are in 
    based on token indices. 
    """
    
    def __init__(self, frame, tokenizer):
        """
        This function takes the original frame, and converts it into a tokenized 
        representation by tokenizing strings and converting character offsets into token
        offsets. 

        @param frame is an untokenized Frame class. 
        @param bert_tokenizer is the method of converting text into tokens.
        """
        self.ev = tokenizer.tokenize(frame.ev.text)
        self.i = tokenizer.tokenize(frame.i.text)
        self.c = tokenizer.tokenize(frame.c.text)
        self.o = tokenizer.tokenize(frame.o.text)
        self.label = frame.label

def load_data(tokenizer = None):
    if os.path.exists('./data.p'):
        # because we wrote in wb, we most read in rb
        with open('data.p', 'rb') as tmp: 
            data = pickle.load(tmp)

        return data

    if tokenizer is None:
        global TOKENIZER
        tokenizer = TOKENIZER

    if os.path.exists('./doc_entities.p'):
        with open('./doc_entities.p', 'rb') as tmp:
            doc_entities = pickle.load(tmp)
            
        with open('./docs_normal.p', 'rb') as tmp:
            docs = pickle.load(tmp)
        
        #import pdb; pdb.set_trace()
    else:
        docs = process_evidence_inference.read_docs(abst_only = True)
        print("DOCS LOADED")
        for d in docs: d.replace_acronyms()
        print("ACRONYM REPLACEMENT")
        processing.add_ner_output(docs, '../../data/ner/ev_inf.json')
        print("NER OUTPUTS ADDED")
        doc_entities = [processing.extract_distant_info(d) for d in docs]
#        import pdb; pdb.set_trace()

    # for every doc, create a tokenized document which contains tokenized text,
    # and tokenized Frames.
    tokenized_docs = []
    counter = 0
    for d, entity_map in zip(docs, doc_entities):
        if counter % 25 == 0: print("On document {}\n".format(counter))
        tokenized_docs.append(Tokenized_Doc(d.text, entity_map, tokenizer))
        counter += 1
       
    return tokenized_docs
"""
data = load_data()
import pdb; pdb.set_trace()
import pickle 
pickle.dump(data, open('data.p', 'wb'))
"""
