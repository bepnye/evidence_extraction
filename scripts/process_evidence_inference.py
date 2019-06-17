import sys, random, os
from collections import namedtuple, defaultdict
import numpy as np
import matplotlib.pyplot as plt
from imp import reload
from Levenshtein import distance as string_distance

sys.path.append('/home/ben/Desktop/evidence-inference/evidence_inference/preprocess/')
import preprocessor
import utils

def sample_evs(data):
  ss = []
  ds = []
  for d in data.values():
    for f in d['frames']:
      s = d['sents'][f['ev_idx']][2]
      e = f['ev']
      ss.append(e in s)
  return ss

def find_overlaps(ev, spans):
  overlaps = [utils.overlap(ev[0], ev[1], s[0], s[1]) for s in spans]
  return overlaps

def fix_offsets(ev, i, f, text):
  span = text[i:f]
  if ev == span:
    pass
  elif ev in text[i-5:f+5]:
    i = text.index(ev, i-5)
    f = i + len(ev)
  elif string_distance(ev.strip(' '), span.strip(' ')) <= 3:
    ev = span.strip(' ')
    i = text.index(ev, i-5)
    f = i + len(ev)
  else:
    i = -1
    f = -1

  return ev, i, f

def read_and_preprocess_data(abst_only = False):
    print('Reading prompts and annotations')
    prompts = [r for _,r in preprocessor.read_prompts().iterrows()]
    annos = [a for _,a in preprocessor.read_annotations().iterrows()]

    docs = {}
    print('Reading articles')
    for prompt in prompts:
        pmcid = prompt['PMCID']
        if pmcid not in docs:
            docs[pmcid] = { 'article': preprocessor.get_article(pmcid),
                            'pmcid': pmcid,
                            'prompts': {}, }
        prompt['Annotations'] = []
        docs[pmcid]['prompts'][prompt['PromptID']] = prompt

    print('Processing annotations')
    for anno in annos:
        ev = anno['Annotations']
        if ev and type(ev) == str:
            doc = docs[anno['PMCID']]
            prompt = doc['prompts'][anno['PromptID']]
            prompt['Annotations'].append(anno)

    pmcids_docs = list(docs.items())
    for pmcid, doc in pmcids_docs:
        for pid, prompt in list(doc['prompts'].items()):
            if not prompt['Annotations']:
                del doc['prompts'][pid]
        if not doc['prompts']:
            del docs[pmcid]
    
    n_prompts = sum([len(d['prompts']) for d in docs.values()])
    print('Retained {}/{} prompts with nonzero annotations'.format(n_prompts, len(prompts)))
    print('Retained {}/{} docs with nonzero prompts'.format(len(docs), len(pmcids_docs)))
    

    if abst_only:
        pmcids_docs = list(docs.items())
        for pmcid, doc in pmcids_docs:
            for pid, prompt in list(doc['prompts'].items()):
                if not all([anno['In Abstract'] for anno in prompt['Annotations']]):
                    del doc['prompts'][pid]
            if not doc['prompts']:
                del docs[pmcid]

        n_prompts = sum([len(d['prompts']) for d in docs.values()])
        print('Retained {}/{} prompts with all annotations in abstract'.format(n_prompts, len(prompts)))
        print('Retained {}/{} docs with nonzero prompts'.format(len(docs), len(pmcids_docs)))

    return docs


def read_data(docs = None):

  docs = docs or read_and_preprocess_data(abst_only = False)

  group_pmcids = { \
          'train': preprocessor.train_document_ids(),
          'test':  preprocessor.test_document_ids(),
          'dev':   preprocessor.validation_document_ids(), }
  for group, pmcids in group_pmcids.items():
      valid_pmcids = [p for p in pmcids if p in docs]
      print('{:03} pmcids, {:03} with data for {}'.format(len(pmcids), len(valid_pmcids), group))
      group_pmcids[group] = valid_pmcids

  data = {}

  matches = defaultdict(int)
  for group, pmcids in group_pmcids.items():
    for pmcid in pmcids:

      doc = docs[int(pmcid)]
      new_doc = {}
      text = preprossor.extract_raw_text(doc['article'])
      title = doc['article'].get_title()
      sents = utils.sent_tokenize(text)

      new_doc['text'] = text
      new_doc['title'] = title
      new_doc['sents'] = sents
      new_doc['frames'] = []

      for pid, p in doc['prompts'].items():

        evs = set()
        label = p['Annotations'][0]['Label']

        for a in p['Annotations']:
          assert a['Label'] == label
          i = a['Evidence Start']
          f = a['Evidence End']
          ev = a['Annotations']
          ev, i, f = fix_offsets(ev, i, f, text)
          ev_tuple = (ev, i, f)
          evs.add(ev_tuple)

        for ev_tuple in evs:
          frame = {}
          frame['i'] = p['Intervention']
          frame['c'] = p['Comparator']
          frame['o'] = p['Outcome']
          frame['label'] = utils.LABEL_TO_ID[label]

          ev, i, f = ev_tuple
          sent_mask = find_overlaps((i, f, ev), sents)
          frame['evidence'] = ev
          frame['ev_i'] = i
          frame['ev_f'] = f
          frame['sent_indices'] = [i for i,m in enumerate(sent_mask) if m]
          new_doc['frames'].append(frame)

      data[pmcid] = new_doc

  return data

def write_data(data):

  for outtype in ['txt', 'sents', 'frames', 'tokens', 'titles']:
    outdir = '{}/documents/{}/'.format(utils.DATA_DIR, outtype)
    if not os.path.isdir(outdir):
      os.system('mkdir -p {}'.format(outdir))

  for pmid, doc in data.items():

    with open('{}/documents/txt/{}.txt'.format(utils.DATA_DIR, pmid), 'w') as fout:
      fout.write(doc['text'])

    with open('{}/documents/titles/{}.title'.format(utils.DATA_DIR, pmid), 'w') as fout:
      fout.write(doc['title'])

    with open('{}/documents/sents/{}.sents'.format(utils.DATA_DIR, pmid), 'w') as fout:
      lines = [utils.clean_str(s) for i,f,s in doc['sents']]
      fout.write('\n'.join(lines))

    with open('{}/documents/sents/{}.txt_idxs'.format(utils.DATA_DIR, pmid), 'w') as fout:
      lines = [str(i)+'\t'+str(f) for i,f,s in doc['sents']]
      fout.write('\n'.join(lines))

    token_lines = { ext: [] for ext in ['tokens', 'sent_idxs', 'txt_idxs'] }
    for idx, (sent_i, sent_f, s) in enumerate(doc['sents']):
      tokens = utils.word_tokenize(s)
      for t_i, t_f, t in tokens:
        token_lines['tokens'].append(utils.clean_str(t))
        token_lines['sent_idxs'].append(str(idx))
        token_lines['txt_idxs'].append('{}\t{}'.format(sent_i + t_i, sent_i + t_f))
    for ext, lines in token_lines.items():
      with open('{}/documents/tokens/{}.{}'.format(utils.DATA_DIR, pmid, ext), 'w') as fout:
        fout.write('\n'.join(lines))


    with open('{}/documents/frames/{}.frames'.format(utils.DATA_DIR, pmid), 'w') as fout:
      lines = ['\t'.join(map(lambda col: utils.clean_str(f[col]), utils.frame_cols)) for f in doc['frames']]
      fout.write('\n'.join(lines))

def write_pmcid_splits(data):
  group_pmids = { \
          'test':  preprocessor.test_document_ids(),
          'train': preprocessor.train_document_ids(),
          'dev':   preprocessor.validation_document_ids(), }
  
  print('Cleaning PMCID splits')
  for group, pmids in group_pmids.items():
    valid_pmids = [p for p in pmids if int(p) in data]
    print('{:03} pmids, {:03} with data for {}'.format(len(pmids), len(valid_pmids), group))
    with open('{}/{}_ids.txt'.format(utils.DATA_DIR, group), 'w') as fout:
      fout.write('\n'.join([str(x) for x in valid_pmids]))

if __name__ == '__main__':
  data = read_data()
  write_data(data)
  write_pmcid_splits(data)
