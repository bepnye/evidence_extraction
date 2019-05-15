import json

import utils

def compare(group = 'dev'):
  pmids = utils.group_ids(group)

  with open('frames.csv', 'w') as fout:
    fout.write('\t'.join('label i c o span'.split()) + '\n')
    for pmid in pmids:
      try:
        bert_frames = utils.readlines('{}/documents/frames/{}.bert_frames'.format(utils.DATA_DIR, pmid))
      except FileNotFoundError:
        # womp womp no frames for this doc
        continue
      gold_frames = utils.readlines('{}/documents/frames/{}.frames'.format(utils.DATA_DIR, pmid))
      gold_sent_idxs = utils.readlines('{}/documents/frames/{}.sent_idxs'.format(utils.DATA_DIR, pmid))

      gold_lookup = { i: l.split('\t') for i,l in enumerate(gold_sent_idxs) if len(l.split('\t')) == 1 }

      for frame_str in bert_frames:
        frame = json.loads(frame_str)
        matching_frames = [i for i, idxs in gold_lookup.items() if frame['sent_idx'] in idxs]
        if matching_frames:
          matching_frame = gold_frames[matching_frames[0]]
          i, c, o, _, _, ev = matching_frame.split('\t')
          if ev not in frame['ev']:
            continue
          fout.write('\t'.join(['gold', i, c, o, ev]) + '\n')

          ico = frame['icos'][0]
          i_score = ''
          c_score = ''
          o_score = ''
          if o.lower() == ico[3].lower():
            o_score = '5'
          if i.lower() == ico[1].lower():
            i_score = '5'
          elif i.lower() == ico[2].lower():
            ico[1], ico[2] = ico[2], ico[1]
            i_score = '5'
            if frame['sample'] == 'c': frame['sample'] = 'i'
          if c.lower() == ico[2].lower():
            c_score = '5'
          fout.write('\t'.join([frame['sample'], ico[1], ico[2], ico[3], frame['ev']]) + '\n')
          #fout.write('\t'.join(['',i_score,c_score,o_score,'']) + '\n')

if __name__ == '__main__':
  compare()
