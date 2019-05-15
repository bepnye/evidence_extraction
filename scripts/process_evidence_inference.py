import sys, random
from collections import namedtuple, defaultdict
import numpy as np
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
from imp import reload
from Levenshtein import distance as string_distance

sys.path.append('/home/ben/Desktop/evidence-inference/evidence_inference/preprocess/')
import ico_reader
reload(ico_reader)
import utils

DATA_DIR = '../data/'

def sample_evs(data):
  ss = []
  ds = []
  for d in data.values():
    for f in d['frames']:
      s = d['sents'][f['ev_idx']][2]
      e = f['ev']
      ss.append(e in s)
  return ss

def find_overlaps(ev, spans):
  overlaps = [utils.overlap(ev[0], ev[1], s[0], s[1]) for s in spans]
  return overlaps

def fix_offsets(ev, i, f, text):
  span = text[i:f]
  if ev == span:
    pass
  elif ev in text[i-5:f+5]:
    i = text.index(ev, i-5)
    f = i + len(ev)
  elif string_distance(ev.strip(' '), span.strip(' ')) <= 3:
    ev = span.strip(' ')
    i = text.index(ev, i-5)
    f = i + len(ev)
  else:
    i = -1
    f = -1

  return ev, i, f


def read_data(docs = None):

  docs = docs or ico_reader.read_data(abst_only = False)

  group_pmcids = { \
          'train': ico_reader.pre.train_document_ids(),
          'test':  ico_reader.pre.test_document_ids(),
          'dev':   ico_reader.pre.validation_document_ids(), }
  for group, pmcids in group_pmcids.items():
      valid_pmcids = [p for p in pmcids if p in docs]
      print('{:03} pmcids, {:03} with data for {}'.format(len(pmcids), len(valid_pmcids), group))
      group_pmcids[group] = valid_pmcids

  data = {}

  matches = defaultdict(int)
  for group, pmcids in group_pmcids.items():
    with open('%s.tsv' %group, 'w') as fout:
      for pmcid in pmcids:

        doc = docs[int(pmcid)]
        new_doc = {}
        text = ico_reader.pre.extract_raw_text(doc['article'])
        sents = utils.span_sent_tokenize(text)

        new_doc['text'] = text
        new_doc['sents'] = sents
        new_doc['frames'] = []

        for pid, p in doc['prompts'].items():

          frame = {}
          frame['i'] = p['Intervention']
          frame['c'] = p['Comparator']
          frame['o'] = p['Outcome']
          frame['anns'] = []

          for a in p['Annotations']:
            ann = {}
            i = a['Evidence Start']
            f = a['Evidence End']
            ev = a['Annotations']
            ev, i, f = fix_offsets(ev, i, f, text)
            sent_mask = find_overlaps((i, f, ev), sents)
            ann['label'] = utils.LABEL_TO_ID[a['Label']]
            ann['evidence'] = ev
            ann['text_index_i'] = i
            ann['text_index_f'] = f
            ann['sent_indices'] = [i for i,m in enumerate(sent_mask) if m]
            frame['anns'].append(ann)
          new_doc['frames'].append(frame)

        data[pmcid] = new_doc

  return data

def write_data(data):
  DATA_DIR = 'data/ev-inf/'
  for pmid, doc in data.items():
    with open('{}/{}.txt'.format(DATA_DIR, pmid), 'w') as fout:
      fout.write(doc['text'])
    with open('{}/{}.sents'.format(DATA_DIR, pmid), 'w') as fout:
      lines = []
      for i,s in enumerate(doc['sents']):
        lines.append('{}\t{}'.format(i, s))
      fout.write('\n'.join(lines))
    f_cols = 'i c o label evidence text_index_i text_index_f sent_indices'.split()
    with open('{}/{}.frames'.format(DATA_DIR, pmid), 'w') as fout:
      lines = []
      for f in doc['frames']:
        lines.append('\t'.join([str(f[col]) for col in f_cols]))
      fout.write('\n'.join(lines))

def write_pmcid_splits(data):
  group_pmids = { \
          'test':  ico_reader.pre.test_document_ids(),
          'train': ico_reader.pre.train_document_ids(),
          'dev':   ico_reader.pre.validation_document_ids(), }
  
  print('Cleaning PMCID splits')
  for group, pmids in group_pmids.items():
    valid_pmids = [p for p in pmids if int(p) in data]
    print('{:03} pmids, {:03} with data for {}'.format(len(pmids), len(valid_pmids), group))
    with open('{}/{}_ids.txt'.format(DATA_DIR, group), 'w') as fout:
      fout.write('\n'.join([str(x) for x in valid_pmids]))
