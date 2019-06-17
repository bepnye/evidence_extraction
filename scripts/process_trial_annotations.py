import utils, re, os
import json
import pandas as pd
from minimap import minimap

def process_data(df):
  print('Formatting data...')
  data = {}

  n = df.size

  data = {}

  for idx, row in df.iterrows():
    if idx > 0 and idx % 10000 == 0:
      write_data(data)
      data = {}
      print('{} / {}'.format(idx, n))

    doc_id = '{}'.format(idx)
    a = {}
    text = row['ab']

    if type(text) != str or len(text) == 0:
      continue

    a['title'] = row['ti']
    a['text'] = text
    a['sample_size'] = row['num_randomized']
    a['sents'] = utils.sent_tokenize(text)

    data[doc_id] = a

  return data

def write_data(data):
  fdir = '../data/sent_classifier/trialstreamer/{}_{}/'.format(min(data.keys()), max(data.keys()))
  os.system('mkdir -p {}'.format(fdir))
  sent_fout = open('{}/dev.tsv'.format(fdir), 'w')
  for doc_id in data:
    for (i, f, s) in data[doc_id]['sents']:
      sent_fout.write('{}\t{}\t{}\t{}\n'.format(doc_id, i, f, s.replace('\n', '<NEWLINE>')))
