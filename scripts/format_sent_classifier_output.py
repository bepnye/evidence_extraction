import sys
from itertools import groupby
from operator import itemgetter
import utils

def format_results(fdir, group = 'dev'):
  input_lines = utils.readlines('{}/sent_classifier/{}/{}.tsv'.format(utils.DATA_DIR, fdir, group))
  output_lines = utils.readlines('{}/sent_classifier/{}/{}_results.tsv'.format(utils.DATA_DIR, fdir, group))

  assert len(input_lines) == len(output_lines)
  assert utils.SENT_INPUT_FIELDS == 'dummy pmid sent_idx sent'

  input_data = [l.split('\t') for l in input_lines]
  output_probs = [[float(x) for x in l.split('\t')] for l in output_lines]
  output_preds = [l.index(max(l)) for l in output_probs]

  all_data = [inputs + [p] for inputs, p in zip(input_data, output_preds)]
  doc_data = groupby(all_data, itemgetter(1))
  for pmid, lines in doc_data:
    with open('{}/documents/sents/{}.bert_{}'.format(utils.DATA_DIR, pmid, fdir), 'w') as fout:
      for _, pmid, _, sent, label in lines:
        fout.write('{}\n'.format(label))

if __name__ == '__main__':
  format_results(*sys.argv[1:])
