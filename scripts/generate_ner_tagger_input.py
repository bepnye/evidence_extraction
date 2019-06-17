import sys, os
import utils

def generate_conll(output_name, group = 'dev'):

  outdir = '{}/ner_tagger_input/{}/'.format(utils.DATA_DIR, output_name)
  os.mkdir(outdir)

  with open('{}/{}.txt'.format(outdir, group), 'w') as fout:

    pmids = utils.group_ids(group)
    for pmid in pmids:
      tokens = utils.readlines('{}/documents/tokens/{}.tokens'.format(utils.DATA_DIR, pmid))
      sent_idxs = utils.readlines('{}/documents/tokens/{}.sent_index'.format(utils.DATA_DIR, pmid))
      text_idxs = utils.readlines('{}/documents/tokens/{}.text_index'.format(utils.DATA_DIR, pmid))
      fout.write('-DOCSTART- -X- O\n')
      cur_sent_idx = -1
      for token, sent_idx, (text_i, text_f) in zip(tokens, sent_idxs, text_idxs):
        if sent_idx != cur_sent_idx:
          fout.write('\n')
          cur_sent_idx = sent_idx
        fout.write('0 {} {} {} {}\n'.format(token, token, pmid, sent_idx, text_i, text_f))
      fout.write('\n')

if __name__ == '__main__':
  generate_conll(sys.argv[1])
