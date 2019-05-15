import sys, os, json
from itertools import product, combinations

import numpy as np

import utils

def generate_sents(output_name, group = 'dev'):

  outdir = '{}/ico_acceptor/{}/'.format(utils.DATA_DIR, output_name)
  try:
    os.mkdir(outdir)
  except OSError:
    print('Target dir: {} already exists'.format(outdir))

  sent_dir = '{}/documents/sents/'.format(utils.DATA_DIR)
  pmids = utils.group_ids(group)

  sample_c = open('{}/{}_sample_c.txt'.format(outdir, group), 'w')
  sample_o = open('{}/{}_sample_o.txt'.format(outdir, group), 'w')
  sample_ic = open('{}/{}_sample_ic.txt'.format(outdir, group), 'w')
  sample_co = open('{}/{}_sample_co.txt'.format(outdir, group), 'w')
  sample_none = open('{}/{}_sample_none.txt'.format(outdir, group), 'w')
  for pmid in pmids:

    all_ner_f = '{}/documents/txts/{}.ner_test'.format(utils.DATA_DIR, pmid)
    all_ner = json.loads(open(all_ner_f).read())
    sents = utils.readlines('{}/{}.sents'.format(sent_dir, pmid))
    ev_labels = [int(l) for l in \
        utils.readlines('{}/{}.bert_ev_binary'.format(sent_dir, pmid))]
    ners = [json.loads(l) for l in \
        open('{}/{}.ner_test'.format(sent_dir, pmid)).readlines()]

    frame_idx = 0

    for sent_idx, (s, ev, ner) in enumerate(zip(sents, ev_labels, ners)):
      if ev:
        if s == 'ABSTRACT ': # data artifact due to text generation
          continue

        n_i = len(ner['i'])
        n_o = len(ner['o'])

        if n_i >= 2:
          i_pairs = combinations(ner['i'], 2)
          if n_o >= 1:
            for (i, c), o in product(i_pairs, ner['o']):
              sample_none.write(utils.joinstr([pmid, sent_idx, frame_idx, i, c, o, s]))
              sample_none.write('\n')
              frame_idx += 1
          else: # n_o == 0
            o_spans = all_ner['o']
            for i, c in i_pairs:
              for o in o_spans:
                sample_o.write(utils.joinstr([pmid, sent_idx, frame_idx, i, c, o, s]))
                sample_o.write('\n')
              frame_idx += 1

        elif n_i == 1:
          i = ner['i'][0]
          if n_o >= 1:
            c_spans = all_ner['i']
            for o in ner['o']:
              for c in c_spans:
                sample_c.write(utils.joinstr([pmid, sent_idx, frame_idx, i, c, o, s]))
                sample_c.write('\n')
              frame_idx += 1
          else: # n_o == 0
            c_spans = [c for c in all_ner['i'] if c != i]
            o_spans = all_ner['o']
            for c, o in product(c_spans, o_spans):
              sample_co.write(utils.joinstr([pmid, sent_idx, frame_idx, i, c, o, s]))
              sample_co.write('\n')
            frame_idx += 1

        else: # n_i == 0
          if n_o >= 1:
            ic_pairs = combinations(all_ner['i'], 2)
            for o in ner['o']:
              for i, c in ic_pairs:
                sample_ic.write(utils.joinstr([pmid, sent_idx, frame_idx, i, c, o, s]))
                sample_ic.write('\n')
              frame_idx += 1

          else: # n_o == 0
            pass # too hard! punt!

if __name__ == '__main__':
  generate_sents(*sys.argv[1:])
