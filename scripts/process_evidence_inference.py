import sys, random, os, csv
from collections import namedtuple, defaultdict, Counter
import numpy as np
from imp import reload
from Levenshtein import distance as string_distance

sys.path.append('..')
import config
sys.path.append(os.path.join(config.EV_INF_DIR, 'evidence_inference', 'preprocess'))
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

FIXES = { 'unchanged': 0, 'wiggled': 0, 'uncorrupted': 0, '???': 0 }
def fix_offsets(ev, i, f, text):
  if ev == text[i:f]:
    FIXES['unchanged'] += 1
    return ev, i, f

  search_range = 10
  if ev in text[i-search_range:f+search_range]:
    i = text.index(ev, i-search_range)
    f = i + len(ev)
    FIXES['wiggled'] += 1
    return ev, i, f

  min_dist = max(3, len(ev)*0.05)
  min_span = ''
  min_i = 0
  min_f = 0
  for i_offset in range(-search_range, search_range):
    for f_offset in range(-search_range, search_range):
      span = text[i + i_offset:f + f_offset]
      dist = string_distance(ev.strip(' '), span.strip(' '))
      if dist <= min_dist:
        min_dist = dist
        min_span = span
        min_i = i+i_offset
        min_f = f+f_offset
  if min_span:
    FIXES['uncorrupted'] += 1
    return min_span, min_i, min_f

  i = -1
  f = -1
  FIXES['???'] += 1
  return ev, i, f

def check_offsets(docs = None):
  anns = preprocessor.read_annotations().to_dict('records')
  if not docs:
    pmcids = list(preprocessor.train_document_ids()) + \
             list(preprocessor.test_document_ids()) + \
             list(preprocessor.validation_document_ids())
    docs = { pmcid: {'txt': preprocessor.extract_raw_text(preprocessor.get_article(pmcid))}  for pmcid in pmcids }

  for idx, a in enumerate(anns):
    if idx % int(len(anns)/100) == 0:
      print('{}/{}'.format(idx, len(anns)))
    if a['PMCID'] in docs:
      ev = a['Annotations']
      if ev and type(ev) == str:
        ev, i, f = fix_offsets(ev, a['Evidence Start'], a['Evidence End'], docs[a['PMCID']]['txt'])
        a['ev'] = ev
        a['i'] = i
        a['f'] = f
  print(FIXES.items())
  return anns

def read_docs(abst_only = False):
  print('Reading prompts and annotations')
  prompts = preprocessor.read_prompts().to_dict('records')
  annotations = preprocessor.read_annotations().to_dict('records')

  def init_doc(pmcid):
    doc = { 'pmcid': pmcid, 'prompts': {} }
    article = preprocessor.get_article(pmcid)
    doc['title'] = article.get_title()
    if (abst_only):
      doc['txt'] = preprocessor.extract_raw_abstract(article)
      doc['offset'] = len("TITLE:\n" + doc['title'] + "\n\n")
    else:
      doc['txt'] = preprocessor.extract_raw_text(article)
      doc['offset'] = 0
    return doc

  docs = {}
  print('Reading articles')
  for prompt in prompts:
    pmcid = prompt['PMCID']
    if pmcid not in docs:
      docs[pmcid] = init_doc(pmcid)

    prompt['anns'] = []
    docs[pmcid]['prompts'][prompt['PromptID']] = prompt

  n_anns = 0
  bad_offsets = []
  print('Processing annotations')
  for ann in annotations:
    if not abst_only or ann['In Abstract']:
      ev = ann['Annotations']
      if ev and type(ev) == str:
        doc = docs[ann['PMCID']]
        prompt = doc['prompts'][ann['PromptID']]
        ev = ann['Annotations']
        i = ann['Evidence Start'] - doc['offset']
        f = ann['Evidence End'] - doc['offset']

        ev, i, f = fix_offsets(ev, i, f, doc['txt'])
        if i >= 0:
          ann['i'] = i
          ann['f'] = f
          ann['ev'] = ev
          prompt['anns'].append(ann)
          n_anns += 1
        else:
          bad_offsets.append((ev, doc['txt'][ann['Evidence Start']:ann['Evidence End']]))

  pmcids_docs = list(docs.items())
  for pmcid, doc in pmcids_docs:
    for pid, prompt in list(doc['prompts'].items()):
      if not prompt['anns']:
        del doc['prompts'][pid]
    if not doc['prompts']:
      del docs[pmcid]
  
  n_prompts = sum([len(d['prompts']) for d in docs.values()])
  print('Retained {}/{} valid annotations ({} w/ bad offsets)'.format(\
      n_anns, len(annotations), len(bad_offsets)))
  print('Retained {}/{} prompts with nonzero annotations'.format(n_prompts, len(prompts)))
  print('Retained {}/{} docs with nonzero prompts'.format(len(docs), len(pmcids_docs)))
  
  return docs

def view_abst_anns(docs):
  with open('out_ben.csv', 'w') as fp:
    fields = ['PMCID', 'Intervention', 'Comparator', 'Outcome']
    writer = csv.DictWriter(fp, fieldnames=fields)
    writer.writeheader()
    for pmid, doc in docs.items():
      for p in doc['prompts'].values():
        for a in p['anns']:
          assert doc['txt'][a['i']:a['f']] == a['ev']
      writer.writerow({f:p[f] for f in fields})


def process_docs(docs = None):

  docs = docs or read_and_preprocess_docs(abst_only = False)

  group_pmcids = { \
      'train': preprocessor.train_document_ids(),
      'test':  preprocessor.test_document_ids(),
      'dev':   preprocessor.validation_document_ids(), }

  for group, pmcids in group_pmcids.items():
    valid_pmcids = [p for p in pmcids if p in docs]
    print('{:03} pmcids, {:03} with data for {}'.format(len(pmcids), len(valid_pmcids), group))
    group_pmcids[group] = valid_pmcids

  data = {}
  for group, pmcids in group_pmcids.items():
    for pmcid in pmcids:

      doc = docs[int(pmcid)]
      text = doc['txt']
      sents = utils.sent_tokenize(text)

      new_doc = {}
      new_doc['text'] = text
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
  data = process_docs()
  write_data(data)
  write_pmcid_splits(data)
