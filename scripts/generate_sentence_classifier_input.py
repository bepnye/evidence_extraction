import sys, os, json
import utils
from random import shuffle

def generate_ev_trinary(group = 'dev', output_name = 'ev_trinary'):

  outdir = '{}/sent_classifier/{}/'.format(utils.DATA_DIR, output_name)
  try:
    os.system('mkdir -p {}'.format(outdir))
  except OSError:
    print('Target dir: {} already exists'.format(outdir))
    input('Proceeding with generation...')

  # parse the files out in to their expected fields
  pmids = utils.group_ids(group)
  with open('{}/{}.tsv'.format(outdir, group), 'w') as fout:
    for pmid in pmids:
      try:
        frames = utils.read_frames('{}/documents/frames/{}.frames'.format(utils.DATA_DIR, pmid))
      except TypeError:
        print(pmid)
        print('{}/documents/frames/{}.frames'.format(utils.DATA_DIR, pmid))
        input()
      for f in frames:
        fout.write('\t'.join(map(str, [f.label, f.i, f.c, f.o, f.evidence])) + '\n')


def generate_ev_binary(group = 'dev', output_name = 'ev_binary'):
  outdir = '{}/sent_classifier/{}/'.format(utils.DATA_DIR, output_name)
  try:
    os.system('mkdir -p {}'.format(outdir))
  except OSError:
    print('Target dir: {} already exists'.format(outdir))
    input('Proceeding with generation...')

  pmids = utils.group_ids('ev_inf', group)
  with open('{}/{}.tsv'.format(outdir, group), 'w') as fout:
    for pmid in pmids:
      sents = utils.readlines('{}/documents/sents/{}.sents'.format(utils.DATA_DIR, pmid))
      frames = utils.readlines('{}/documents/sents/{}.frame_idx'.format(utils.DATA_DIR, pmid))

      pos_sents = [s for s, fs in zip(sents, frames) if len(fs) >= 1]
      neg_sents = [s for s, fs in zip(sents, frames) if len(fs) == 0]

      print(pmid)
      print(pos_sents)
      input()
      shuffle(neg_sents)
      neg_sents = neg_sents[:len(pos_sents)]
        
      for s in pos_sents:
        fout.write('{}\t{}\n'.format(1, s))
      for s in neg_sents:
        fout.write('{}\t{}\n'.format(0, s))


if __name__ == '__main__':
  for group in ['train', 'dev']:
    generate_ev_binary(group)
