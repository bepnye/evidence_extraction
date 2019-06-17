import sys, os, json
import utils

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

  pmids = utils.group_ids(group)
  with open('{}/{}.tsv'.format(outdir, group), 'w') as fout:

    for pmid in pmids:
      frames = utils.read_frames('{}/documents/frames/{}.frames'.format(utils.DATA_DIR, pmid))
      sents = utils.readlines('{}/documents/sents/{}.sents'.format(utils.DATA_DIR, pmid))


      # collect all sentences without an evidence span in them
      sents = dict(enumerate(sents))
      for f in frames:
        for idx in json.loads(f.sent_indices):
          try:
            del sents[idx]
          except KeyError:
            pass

      # Negative sample non-evidence sentences from within the document
      for f in frames:
        is_different = lambda idx_s: idx_s[1] not in f.evidence and f.evidence not in idx_s[1]
        valid_negs = list(filter(is_different, sents.items()))

        if not valid_negs:
          continue

        # try to length-match each positive with a negative
        sorted_negs = sorted(valid_negs, key = lambda idx_s: abs(len(f.evidence) - len(idx_s[1])))
        neg_idx, neg_s = sorted_negs[0]
        
        fout.write('\t'.join(map(str, [1, f.evidence])) + '\n')
        fout.write('\t'.join(map(str, [0, neg_s])) + '\n')

        del sents[neg_idx]


if __name__ == '__main__':
  for group in ['train', 'dev']:
    generate_ev_trinary(group)
    generate_ev_binary(group)
