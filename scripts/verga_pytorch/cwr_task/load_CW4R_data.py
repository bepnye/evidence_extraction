# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 01:00:35 2020

@author: Eric Lehman
"""
import sys
sys.path.append('../')
sys.path.append('../../') # I'm disgusting -- sorry.
sys.path.append('../../../') # we unfortunately need this for config.py 
import glob
import pandas as pd
from classes import Span, Entity
from load_data import Tokenized_Doc, TOKENIZER
EXT = './cwr_data/'
DELIMITER = ';;;' # from CWR data
DATAFRAME_F  = EXT + 'wb_rankings.csv'
MAIN_LABELS  = EXT + 'top10k_only_wb_annotated_111_updated_relevancy.csv'
ANNOT_LABELS = list(filter(lambda x: not(MAIN_LABELS in x) and not(DATAFRAME_F) in x, glob.glob(EXT + '*.csv')))   
import re

def find_all(text, s):
    """ find all occurances of s in the text -> return list of tuples corresponding to start + end. """
    res = []
    for i in range(len(text)):
        l = text[i: i + len(s)]
        if l == s: res.append((i, i + len(s)))

    return res

def annotation_file_to_doc(annot_file, df):
    """ 
    Given an annotation file, and a dataframe, return a list of tokenized docs based off 
    of the given annotation file.
    @param annot_file is a string location of the file
    @param df is a pandas dataframe containing abstract + title info.
    """
    seen = set()
    docs = []
    ann = pd.read_csv(annot_file)
    orig_idx = list(df['Original Index (Not important)'].values)
    difficulty_level = []

    for idx, row in ann.iterrows():
        if row.ID in seen: continue
        matched_row_idx = int(str(row.ID).split('.')[0])
        offset = int(str(row.ID).split('.')[1]) if '.' in str(row.ID) else 0 ### ONLY USED FOR INTERVENTION ### 
        matched_row = df.iloc[orig_idx.index(matched_row_idx)]
        if type(matched_row.Abstract) != str: continue

        # Here are the items we are interested in # 
        matched_intv = matched_row['Matched Intervention (Word Embeddings)'].split(DELIMITER)[offset]
        matched_out  = matched_row['Matched Outcome (Word Embeddings)']

        # Get the mentions # 
        e1_mentions = [Span(start, end, matched_intv) for start, end in find_all(matched_row['Abstract'], matched_row['Article Intervention (Word Embeddings)'].split(DELIMITER)[offset])]
        e2_mentions = [Span(start, end, matched_out) for start, end in find_all(matched_row['Abstract'], matched_row['Article Outcome (Word Embeddings)'])]
        e1 = Entity(Span(-1, -1, matched_intv), 'intervention')
        e2 = Entity(Span(-1, -1, matched_out), 'outcome')
        e1.mentions = e1_mentions
        e2.mentions = e2_mentions
   
        if len(e1.mentions) == 0 or len(e2.mentions) == 0: continue

        ### ASSEMBLE ###
        entity_map = ([e1, e2], {(e1.text, e2.text): 1 if row.Label else 0})
        doc = Tokenized_Doc(matched_row['Abstract'], entity_map, TOKENIZER) # text, entity_map, tokenizer
        docs.append(doc)
        difficulty_level.append(row['Selected Reasoning'])
        seen.add(row.ID)

    return docs, difficulty_level

def main_file_to_doc(df):
    """ 
    Take in a dataframe, and produce a list of tokenized documents.
    """
    annotated = df[df['NewA'] == 1.0]
    labels    = annotated['Relevant'].values
    abstracts = annotated['Abstract'].values 
    article_intv = annotated['Article.Intervention..Word.Embeddings.'].values
    article_out  = annotated['Article.Outcome..Word.Embeddings.'].values
    matched_intv = annotated['Matched.Intervention..Word.Embeddings.'].values
    matched_out  = annotated['Matched.Outcome..Word.Embeddings.'].values

    ### Create Docs and Entity map ### 
    docs = []
    for i in range(len(annotated)):
        e1_mentions = [Span(start, end, matched_intv[i]) for start, end in find_all(abstracts[i], article_intv[i])]
        e2_mentions = [Span(start, end, matched_out[i]) for start, end in find_all(abstracts[i], article_out[i])]
        e1 = Entity(Span(-1, -1, matched_intv[i]), 'intervention')
        e2 = Entity(Span(-1, -1, matched_out[i]), 'outcome')
        e1.mentions = e1_mentions
        e2.mentions = e2_mentions
        
        if len(e1.mentions) == 0 or len(e2.mentions) == 0: continue

        ### ASSEMBLE ###
        entity_map = ([e1, e2], {(e1.text, e2.text): labels[i]})
        doc = Tokenized_Doc(abstracts[i], entity_map, TOKENIZER) # text, entity_map, tokenizer
        docs.append(doc)

    return docs

def load_cwr4c_data():
    ### GET THE LABEL FILES ### 
    df = pd.read_csv(DATAFRAME_F)
    tokenized_docs = main_file_to_doc(pd.read_csv(MAIN_LABELS))
    difficulty_levels = ['NA'] * len(tokenized_docs)

    # get annotator files # 
    for f in ANNOT_LABELS: 
        new_docs, new_level_analysis = annotation_file_to_doc(f, df)
        tokenized_docs.extend(new_docs)
        difficulty_levels.extend(new_level_analysis)

    return tokenized_docs, difficulty_levels


#load_cwr4c_data()
