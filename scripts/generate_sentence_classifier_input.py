import sys, os
import utils

def generate_sents(output_name, group = 'dev'):

  outdir = '{}/sent_classifier/{}/'.format(utils.DATA_DIR, output_name)
  try:
    os.mkdir(outdir)
  except OSError:
    print('Target dir: {} already exists'.format(outdir))

  with open('{}/{}.tsv'.format(outdir, group), 'w') as fout:

    pmids = utils.group_ids(group)
    for pmid in pmids:
      sents = utils.readlines('{}/documents/sents/{}.sents'.format(utils.DATA_DIR, pmid))
      for s_idx, s in enumerate(sents):
        assert utils.SENT_INPUT_FIELDS = 'dummy pmid sent_idx sent'
        fout.write('0\t{}\t{}\t{}\n'.format(pmid, s_idx, s))

if __name__ == '__main__':
  generate_sents(*sys.argv[1:])
