import os
from collections import namedtuple

import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

import utils
import eval

Chunk = namedtuple('Chunk', 'tokens labels')


def score_groups(docs):
  coverage = { 'i': [], 'o': [] }
  for d in docs.values():
    for gid, chunks in d['g'].items():
      target_e = gid[0]
      tagged = [c for c in chunks if target_e in c.labels]
      coverage[target_e].append((len(chunks), len(tagged)))

  plot_coverage(coverage['i'])
  plot_coverage(coverage['o'])


def plot_coverage(pairs):
  X, Y = zip(*pairs)
  data = np.zeros((max(Y)+1, max(X)+1))
  for x, y in pairs:
    data[y][x] += 1
  #ax = sns.heatmap(data, mask = (data == 0), linewidth=0.5)
  #ax.invert_yaxis()
  pred_totals = data.sum(axis = 1)
  sns.barplot(x = list(range(len(pred_totals))), y = pred_totals)
  plt.show()

if __name__ == '__main__':
  parse_file('/home/ben/Desktop/evidence_extraction/models/ner_tagger/data/io/results/pred.txt')
