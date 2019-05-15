import sys, random
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/home/ben/Desktop/evidence-inference/evidence_inference/preprocess/')
import ico_reader
from utils import *

def format_str(s):
  return s.replace('\t', ' <TAB> ').replace('\n', ' <NEWLINE>').replace('\r', '<NEWLINE>')

def sample_negs(doc_sents, ev_sents, group):
  valid_negs = [s for s in doc_sents if not any([sample_overlap(s, s_ev) for s_ev in ev_sents])]

  if group == 'test':
    return valid_negs

  samples = []
  for ev_s in ev_sents:
    if not valid_negs:
      continue

    matches = [s for s in valid_negs if abs(1 - (s.ev_f-s.ev_i)/(ev_s.ev_f-ev_s.ev_i)) < 0.2]
    if not matches:
      matches = valid_negs

    random.shuffle(matches)
    samples.append(matches[0])
    valid_negs.remove(matches[0])
  return samples

def sample_ev_sents(docs = None):
  if not docs:
    docs = ico_reader.read_data(abst_only = False)

  group_pmids = { \
          'test':  ico_reader.pre.test_document_ids(),
          'train': ico_reader.pre.train_document_ids(),
          'dev':   ico_reader.pre.validation_document_ids(), }
  
  print('Sampling evidence examples for pmids')
  for group, pmids in group_pmids.items():
      valid_pmids = [p for p in pmids if p in docs]
      print('{:03} pmids, {:03} with data for {}'.format(len(pmids), len(valid_pmids), group))
      group_pmids[group] = valid_pmids

  for group, pmids in group_pmids.items():
    with open('%s.tsv' %group, 'w') as fout:
      for pmid in pmids:
        doc = docs[int(pmid)]

        ev_sents = set()
        for pid, p in doc['prompts'].items():
          for a in p['Annotations']:
            ev_sents.add(Sample(p['Intervention'], p['Comparator'], p['Outcome'], \
                    a['Annotations'], a['Evidence Start'], a['Evidence End'], a['Label'], \
                    pid, pmid))

        doc_sents = span_sent_tokenize(doc['article'].to_raw_str())
        for s in ev_sents:
          text_a = format_str('{} <SEP> {} <SEP> {}'.format(s.i, s.c, s.o))
          text_b = format_str(s.ev)
          fout.write('\t'.join([str(pmid), '1', text_a, text_b]) + '\n')
          text_a = format_str('{} <SEP> {} <SEP> {}'.format(s.c, s.i, s.o))
          text_b = format_str(s.ev)
          fout.write('\t'.join([str(pmid), '0', text_a, text_b]) + '\n')
        #neg_sents = sample_negs(doc_sents, ev_sents, group)
        #for s in neg_sents:
        #  neg_lens.append(len(s[2]))
        #  fout.write('\t'.join([str(pmid), '0', format_str(s[2])]) + '\n')

    print('Finished sampling')

def sample_neg_pairs(pos_samples, sample_fields = 'ico'):
  all_fields = { f: [getattr(s, f) for s in pos_samples] for f in sample_fields }
  neg_samples = []
  for s in pos_samples:
    # pick random field to replace
    s_fields = s._asdict()

    neg_field = random.choice(sample_fields)
    cur_x = s_fields[neg_field]

    # pick random value from corpus to replace with
    valid_samples = [x for x in all_fields[neg_field] if x != cur_x]
    new_x = random.choice(valid_samples)

    # replace sampled field, set target label to false
    s_fields[neg_field] = new_x
    s_fields['label'] = '0'
    neg_samples.append(Sample(**s_fields))

  return neg_samples

def sample_ico_ev_pairs(docs = None):
  if not docs:
    docs = ico_reader.read_data(abst_only = False)

  group_pmids = { \
          'test':  ico_reader.pre.test_document_ids(),
          'train': ico_reader.pre.train_document_ids(),
          'dev':   ico_reader.pre.validation_document_ids(), }
  
  print('Sampling evidence examples for pmids')
  for group, pmids in group_pmids.items():
      valid_pmids = [p for p in pmids if p in docs]
      print('{:03} pmids, {:03} with data for {}'.format(len(pmids), len(valid_pmids), group))
      group_pmids[group] = valid_pmids

  for group, pmids in group_pmids.items():
    with open('%s.tsv' %group, 'w') as fout:
      pos_samples = []

      for pmid in pmids:
        doc = docs[int(pmid)]
        for pid, p in doc['prompts'].items():
          for a in p['Annotations']:
            pos_samples.append(Sample(p['Intervention'], p['Comparator'], p['Outcome'], \
                a['Annotations'], a['Evidence Start'], a['Evidence End'], '1', \
                pid, pmid))
            
      neg_samples = []#sample_neg_pairs(pos_samples, 'ico')
      all_samples = pos_samples + neg_samples
      all_samples = sorted(all_samples, key = lambda s: s.pmid)
      for s in all_samples:
        text_a = format_str('{}\t{}\t{}'.format(s.i, s.c, s.o))
        text_b = format_str('{}'.format(s.ev))
        fout.write('\t'.join([str(s.pmid), str(pid), s.label, text_a, text_b]) + '\n')

  #counts = [[v[f] for v in all_negs] for f in 'icon']
  #for c in counts:
  #  print(np.mean(c), np.std(c))
  #plt.hist(counts, bins = range(1, 10), density = True, rwidth = 0.9, align = 'left', \
  #    label = ['I', 'C', 'O', 'n'], color = ['red', 'green', 'blue', 'purple'])
  #plt.title('Per-document diversity of ICO frames')
  #plt.legend()
  #plt.ylabel('Frequency')
  #plt.xlabel('Number of unique strings')
  #plt.show()


if __name__ == '__main__':
    sample_ev_sents()
