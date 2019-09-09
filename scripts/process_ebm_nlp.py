import sys
import os
from glob import glob
from shutil import copyfile


from utils import readlines
sys.path.append('..')
import config

def process_group(group, e, phase = 'starting_spans', label_suffix = ''):
  docs = {}
  pmids = readlines(os.path.join(config.EBM_NLP_DIR, 'pmids_{}.txt'.format(group)))

  group_fname = '../data/id_splits/ebm_nlp/{}.txt'.format(group)
  if not os.path.isfile(group_fname):
    with open(group_fname, 'w') as fout: 
      fout.write('\n'.join(pmids))

  for pmid in pmids:
    f_src = os.path.join(config.EBM_NLP_DIR, 'documents', '{}.tokens'.format(pmid))
    f_dest = os.path.join('../data/documents/tokens/', '{}.tokens'.format(pmid))
    copyfile(f_src, f_dest)
    
    f_src = os.path.join(config.EBM_NLP_DIR, 'documents', '{}.txt'.format(pmid))
    f_dest = os.path.join('../data/documents/txts/', '{}.txt'.format(pmid))
    copyfile(f_src, f_dest)

    f_src = os.path.join(config.EBM_NLP_DIR, 'annotations', 'aggregated', phase, e, group, '{}.AGGREGATED.ann'.format(pmid))
    if os.path.isfile(f_src):
      f_dest = '../data/documents/tokens/{}.{}{}'.format(pmid, e, label_suffix)
      copyfile(f_src, f_dest)

if __name__ == '__main__':
  for group in ['test', 'train', 'dev']:
    for e in ['participants', 'interventions', 'outcomes']:
      process_group(group, e, 'starting_spans')
      process_group(group, e, 'hierarchical_labels', '_detailed')
