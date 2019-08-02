import sys
import os
from glob import glob
from shutil import copyfile


from utils import readlines
sys.path.append('..')
import config

def process_group(group, ann_dirs, ann_names):
  docs = {}
  pmids = readlines(os.path.join(config.EBM_NLP_DIR, 'pmids_{}.txt'.format(group)))

  with open('../data/id_splits/ebm_nlp/{}.txt'.format(group), 'w') as fout:
    fout.write('\n'.join(pmids))

  for pmid in pmids:
    f_src = os.path.join(config.EBM_NLP_DIR, 'documents', '{}.tokens'.format(pmid))
    f_dest = os.path.join('../data/documents/tokens/', '{}.tokens'.format(pmid))
    copyfile(f_src, f_dest)

    for d, name in zip(ann_dirs, ann_names):
      f_src = os.path.join(config.EBM_NLP_DIR, 'annotations', 'aggregated', d, '{}.AGGREGATED.ann'.format(pmid))
      f_dest = '../data/documents/tokens/{}.{}'.format(pmid, name)
      copyfile(f_src, f_dest)

if __name__ == '__main__':
  process_group('test',  ['starting_spans/outcomes/test/gold', 'starting_spans/interventions/test/gold'], ['o_labels', 'i_labels'])
  process_group('train', ['starting_spans/outcomes/train/', 'starting_spans/interventions/train/'], ['o_labels', 'i_labels'])
  process_group('dev',   ['starting_spans/outcomes/train/', 'starting_spans/interventions/train/'], ['o_labels', 'i_labels'])
