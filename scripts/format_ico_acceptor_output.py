import sys, json
from itertools import groupby
from operator import itemgetter

import nltk

import utils

def generate_frames(output_name, group = 'dev'):
  icodir = '{}/ico_acceptor/{}/'.format(utils.DATA_DIR, output_name)

  data = {}

  for sample in ['x', 'c', 'o']:
    input_lines = utils.readlines('{}/{}_sample_{}.txt'.format(icodir, group, sample))
    input_lines = [l.split('\t') for l in input_lines]
    output_lines = utils.readlines('{}/{}_sample_{}_results.txt'.format(icodir, group, sample))
    output_lines = [l.split('\t') for l in output_lines]

    #sample_c.write(utils.joinstr([pmid, sent_idx, frame_idx, i, c, o, s])) don't show Jay
    assert len(input_lines) == len(output_lines)

    all_lines = [i_l + o_l for i_l, o_l in zip(input_lines, output_lines)]

    for pmid, pmid_lines in groupby(all_lines, itemgetter(0)):
      if pmid not in data:
        data[pmid] = {}
      for frame_idx, frame_lines in groupby(pmid_lines, itemgetter(2)):
        pmids, sent_idxs, frame_idxs, i_s, c_s, o_s, s_s, p0s, p1s = zip(*frame_lines)
        assert len(set(pmids)) == 1
        assert len(set(sent_idxs)) == 1
        assert len(set(frame_idxs)) == 1
        assert len(set(s_s)) == 1

        if len(nltk.tokenize.word_tokenize(s_s[0])) < 10:
          continue
    
        sent_idx = sent_idxs[0]
        frame_idx = frame_idxs[0]
        ev_span = s_s[0]

        top_frames = sorted(zip(p1s, i_s, c_s, o_s), key = itemgetter(0), reverse = True)

        assert frame_idx not in data[pmid]
        frame = {
            'sent_idx': sent_idx,
            'frame_idx': frame_idx,
            'ev': ev_span,
            'icos': top_frames[:5],
            'sample': sample,
            }
        data[pmid][frame_idx] = frame

  for pmid, frames in data.items():
    with open('{}/documents/frames/{}.bert_frames'.format(utils.DATA_DIR, pmid), 'w') as fout:
      for frame in frames.values():
        fout.write(json.dumps(frame) + '\n')

if __name__ == '__main__':
  generate_frames(*sys.argv[1:])
