from collections import namedtuple
from itertools import groupby
from nltk.tokenize import sent_tokenize

Sample = namedtuple('Sample', 'i c o ev ev_i ev_f label pid pmid')

DATA_DIR = '../data/'
NER_INPUT_FIELDS = 'token pmid token_idx sent_idx dummy'
SENT_INPUT_FIELDS = 'dummy pmid sent_idx sent'

LABEL_TO_ID = { \
        'significantly decreased':   '0',
        'no significant difference': '1',
        'significantly increased':   '2', }

def joinstr(seq, sep = '\t'):
  return sep.join([str(s) for s in seq])

def readlines(fname):
  return open(fname).read().strip('\n').split('\n')

def group_ids(group):
  return open('{}/{}_ids.txt'.format(DATA_DIR, group)).read().split('\n')

def norm_d(d):
  N = sum(d.values())
  return { k: v/N for k,v in d.items() }

def overlap(x1, x2, y1, y2):
  return x1 <= y2 and y1 <= x2

def contained(x1, x2, y1, y2):
  return x1 >= y1 and x2 <= y2

def sample_overlap(s1, s2):
  x1, x2 = s1.ev_i, s1.ev_f
  y1, y2 = s2.ev_i, s2.ev_f
  return overlap(x1, x2, y1, y2)

def condense_labels(labels, neg_class = '0'):
  groups = [(k, sum(1 for _ in g)) for k,g in groupby(labels)]
  spans = []
  i = 0
  for label, length in groups:
    if label != neg_class:
      spans.append((i, i+length, label))
    i += length
  return spans

def span_sent_tokenize(t):
  sents = []
  for s in sent_tokenize(t):
    sents += [p for p in s.split('\n') if p]
  i = 0
  spans = []
  for s in sents:
    i = t.find(s, i)
    f = i + len(s)
    assert t[i:f] == s
    spans.append((i,f,s))
    i = f
  return spans
