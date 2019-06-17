from collections import namedtuple
from itertools import groupby
from nltk import tokenize as tokenizer

frame_cols = 'i c o label evidence ev_i ev_f sent_indices'.split()
Frame = namedtuple('Frame', frame_cols)
Sent = namedtuple('Sent', 'i f s')

DATA_DIR = '../data/'
NER_INPUT_FIELDS = 'token pmid token_idx sent_idx dummy'
SENT_INPUT_FIELDS = 'dummy pmid sent_idx sent'

LABEL_TO_ID = { \
        'significantly decreased':   '0',
        'no significant difference': '1',
        'significantly increased':   '2', }

def clean_str(s):
  return str(s).replace('\t', ' <TAB> ').replace('\n', ' <NEWLINE>').replace('\r', '<NEWLINE>')

def joinstr(seq, sep = '\t'):
  return sep.join([str(s) for s in seq])

def readlines(fname):
  return open(fname).read().strip('\n').split('\n')

def read_tsv(fname):
  return [l.split('\t') for l in open(fname).read().strip('\n').split('\n')]

def read_frames(fname):
  frames = []
  for l in readlines(fname):
    try:
      f = Frame(*l.split('\t'))
      frames.append(f)
    except TypeError:
      print(l)
      raise
  return frames

def read_sents(fname):
  sents = []
  for l in readlines(fname):
    try:
      s = Sent(*l.split('\t'))
      sents.append(s)
    except TypeError:
      print(l)
      raise
  return sents

def group_ids(group):
  return open('{}/{}_ids.txt'.format(DATA_DIR, group)).read().split('\n')

def norm_d(d):
  N = sum(d.values())
  return { k: v/N for k,v in d.items() }

def overlap(x1, x2, y1, y2):
  return x1 <= y2 and y1 <= x2

def contained(x1, x2, y1, y2):
  return x1 >= y1 and x2 <= y2

def frame_overlap(s1, s2):
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

def span_tokenize(txt, tokenize):
  chunks = tokenize(txt)
  spans = []
  i = 0
  for chunk in chunks:
    # undo the PTB conversion of double quotes
    if chunk in ["``", "''"]:
      chunk = '"'
    i = txt.find(chunk, i)
    f = i + len(chunk)
    if (txt[i:f] == chunk):
      spans.append((i,f,chunk))
      i = f
    else:
      print('Unable to find token:', i, f, chunk, txt[i:f])
  return spans

def sent_tokenize(txt):
  return span_tokenize(txt, tokenizer.sent_tokenize)

def word_tokenize(txt):
  return span_tokenize(txt, tokenizer.word_tokenize)
