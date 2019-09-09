import sys
import os
import glob
from random import shuffle
from itertools import combinations

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, recall_score, precision_score
from scipy.cluster.hierarchy import dendrogram, linkage

import matplotlib.pyplot as plt

sys.path.append('..')
import config
import utils
sys.path.append(os.path.join(config.EV_INF_DIR, 'evidence_inference', 'preprocess'))
import preprocessor

import minimap as mmap

#from pycorenlp import StanfordCoreNLP
#nlp = StanfordCoreNLP('http://localhost:9000')

from bert_serving.client import BertClient

def get_ev_inf():
  print('Reading prompts and annotations')
  prompts = preprocessor.read_prompts().to_dict('records')
  annotations = preprocessor.read_annotations().to_dict('records')

  def init_doc(pmcid):
    doc = { 'pmcid': pmcid, 'prompts': {} }
    article = preprocessor.get_article(pmcid)
    doc['title'] = article.get_title()
    doc['txt'] = preprocessor.extract_raw_abstract(article)
    doc['offset'] = len("TITLE:\n" + doc['title'] + "\n\n")
    doc['i'] = { 'spans': set() } 
    doc['o'] = { 'spans': set() }
    return doc

  docs = {}
  print('Reading articles')
  for prompt in prompts:
    pmcid = prompt['PMCID']
    if pmcid not in docs:
      docs[pmcid] = init_doc(pmcid)

    prompt['anns'] = []
    docs[pmcid]['prompts'][prompt['PromptID']] = prompt
    docs[pmcid]['i']['spans'].add(prompt['Intervention'])
    docs[pmcid]['i']['spans'].add(prompt['Comparator'])
    docs[pmcid]['o']['spans'].add(prompt['Outcome'])

  print('Processing annotations')
  n_anns = 0
  bad_offsets = []
  for ann in annotations:
    if ann['In Abstract']:
      ev = ann['Annotations']
      if ev and type(ev) == str:
        doc = docs[ann['PMCID']]
        prompt = doc['prompts'][ann['PromptID']]
        ann['ev'] = ann['Annotations']
        ann['i'] = ann['Evidence Start'] - doc['offset']
        ann['f'] = ann['Evidence End'] - doc['offset']
        prompt['anns'].append(ann)
        n_anns += 1

  pmcids_docs = list(docs.items())
  for pmcid, doc in pmcids_docs:
    for pid, prompt in list(doc['prompts'].items()):
      if not prompt['anns']:
        del doc['prompts'][pid]
    if not doc['prompts']:
      del docs[pmcid]

  for pmcid, doc in docs.items():
    doc['i']['spans'] = list(doc['i']['spans'])
    doc['o']['spans'] = list(doc['o']['spans'])
  
  n_prompts = sum([len(d['prompts']) for d in docs.values()])
  print('Retained {}/{} valid annotations ({} w/ bad offsets)'.format(\
      n_anns, len(annotations), len(bad_offsets)))
  print('Retained {}/{} prompts with nonzero annotations'.format(n_prompts, len(prompts)))
  print('Retained {}/{} docs with nonzero prompts'.format(len(docs), len(pmcids_docs)))
  
  return docs

agg_names = ['hmm', 'mv', 'union', 'intersection']

def get_docs(default_span = 'mv'):
  ebm_nlp = '/Users/elizabethwagoner/Desktop/ben/EBM-NLP/ebm_nlp_2_00'
  ann_dir = '{}/annotations'.format(ebm_nlp)

  pmids = utils.readlines('../data/id_splits/ebm_nlp/test.txt')
  docs = { p: { 'i': {}, 'o': {} } for p in pmids }

  for p in pmids:

    token_fname = os.path.join('../data/documents/tokens/', '{}.tokens'.format(p))
    tokens = utils.readlines(token_fname)
    docs[p]['tokens'] = tokens

    for el in ['interventions', 'outcomes']:

      agg_fname = '{}/aggregated/starting_spans/{}/test/{}.AGGREGATED.ann'.format(ann_dir, el, p)
      indv_fnames = glob.glob('{}/individual/phase_1/{}/test/gold/{}.*.ann'.format(ann_dir, el, p))
      e = el[0]
      
      docs[p][e]['hmm'] = list(map(int, utils.readlines(agg_fname)))
      docs[p][e]['indv'] = []
      for f in indv_fnames:
        docs[p][e]['indv'].append(list(map(int, utils.readlines(f))))
      docs[p][e]['avg'] = list(map(np.mean, zip(*docs[p][e]['indv'])))
      
      agg_strats = [\
          ('mv',           lambda x: int(x + 0.5)),
          ('union',        lambda x: int(x > 0)),
          ('intersection', lambda x: int(x))]

      for name, func in agg_strats:
        docs[p][e][name] = list(map(func, docs[p][e]['avg']))

      spans = utils.condense_labels(docs[p][e][default_span])
      docs[p][e]['spans'] = [' '.join(tokens[i:f]) for i, f, l in spans]

  return docs

def get_agreement(labels):
  agreement = { e : { a: {} for a in agg_names } for e in 'io' }

  for e in 'io':
    for p in labels:
      for agg_name in agg_names:
        agg = labels[p][e][agg_name]
        corrs = [pearsonr(ls, agg)[0] for ls in labels[p][e]['indv']]
        agreement[e][agg_name][p] = list(filter(lambda m: not np.isnan(m), corrs))
        
  return agreement

def plot_agreement(agreement = None, e = 'o'):
  agreement = agreement or get_agreement(get_labels())

  agg_names = list(agreement[e].keys())
  agg_corrs = { n: [np.mean(corrs) for corrs in agreement[e][n].values()] for n in agg_names }
  fig, axes = plt.subplots(4, 1, sharey = True, sharex = True)
  for ax, (n, corrs) in zip(axes, agg_corrs.items()):
    counts, edges, patches = ax.hist(corrs, bins = np.arange(0, 1, 0.02), label = n)
    cdf = np.cumsum(counts)
    ax2 = ax.twinx()
    ax2.plot(edges[1:], cdf, color = 'red')
    ax.title.set_text('{}: {}'.format(n, np.mean(list(filter(lambda x: not np.isnan(x), corrs)))))
  plt.tight_layout()
  plt.show()

def embed_umls():
  pass

def compute_embs(docs):
  EMBS = {}
  bc = BertClient()
  for pmid, d in docs.items():
    for e in ['i', 'o']:
      spans = [s for s in d[e]['spans'] if s not in EMBS]
      if spans:
        vecs = bc.encode(spans)
        for s, v in zip(spans, vecs):
          EMBS[s] = v
  return EMBS

def compute_mm(docs):
  MM = {}
  for pmid, d in docs.items():
    for e in ['i', 'o']:
      for s in d[e]['spans']:
        if s not in MM:
          MM[s] = mmap.minimap(s)
  return MM

class Graph:
  
  class Node:
    def __init__(self, nid, name):
      self.id = nid
      self.name = name
      self.neighbor_ids = set()
      self.visited = False

  def __init__(self, nodes, edges):
    self.nodes = { i: self.Node(i, n) for i, n in enumerate(nodes) }
    for u, v in edges:
      self.nodes[u].neighbor_ids.add(v)
      self.nodes[v].neighbor_ids.add(u)

  def get_clusters(self):

    for n in self.nodes.values():
      n.visited = False

    clusters = []
    for nid in self.nodes:
      if not self.nodes[nid].visited:
        clusters.append(self.bfs(nid))
    return clusters
  
  def bfs(self, nid):
    found = []
    queue = [nid]
    while queue:
      nid = queue.pop(0)
      if not self.nodes[nid].visited:
        found.append(nid)
        self.nodes[nid].visited = True
        for nid in self.nodes[nid].neighbor_ids:
          queue.append(nid)
    return found


def print_cui_clusters(docs, mm):

  def get_cuis(s):
    return [m['cui'] for m in mm[s]]
  def share_cuis(s1, s2):
    return len(set(get_cuis(s1)).intersection(get_cuis(s2))) > 0

  for e in 'io':
    for p, d in docs.items():
      print('PMID {}'.format(p))
      txts = d[e]['spans']
      edges = []
      for i, s1 in enumerate(txts):
        for k, s2 in enumerate(txts):
          if i == k:
            continue
          if s1 == s2 or share_cuis(s1, s2):
            edges.append([i, k])
      clusters = Graph(txts, edges).get_clusters()
      for cid, ids in enumerate(clusters):
        print('\tCLUSTER {}'.format(cid))
        for i in ids:
          print('\t\t\t\t\t\t', txts[i])
      print()

def print_bert_clusters(docs, embs, e = 'i'):
  #keys = { s: i for i, s in enumerate(embs.keys()) }
  #X = np.stack(embs.values())
  ds = list(docs.values())
  shuffle(ds)
  for d in ds:
    spans = d[e]['spans']
    if not spans:
      continue
    X = np.stack([embs[s] for s in spans])
    Z = linkage(X, 'ward')
    fig, axes = plt.subplots(1, 1)
    dendrogram(Z, ax = axes, orientation = 'left', labels = spans)
    axes.set_xlim([20, -1])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
  docs = get_docs()
  mm = compute_mm(docs)
  print_cui_clusters(docs, mm)
