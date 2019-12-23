def add_minimap_output(docs):
  for pmid, d in docs.items():
    d['mm_spans'] = tools.mm_text(d['text'])
  add_minimap_frames(docs)

def add_minimap_frames(docs):
  for pmid, d in docs.items():
    d['mm_frames'] = {}
    for f in d['frames']:
      fd = f._asdict()
      for e in 'ico':
        if fd[e] not in d['mm_frames']:
          uids = set()
          for span, uid in tools.mm_text(fd[e]):
            uids.update(uid)
          d['mm_frames'][fd[e]] = uids

def get_seq_labels(target_spans, labeled_spans):
  labels = [0 for s in target_spans]
  for l_s in labeled_spans:
    for i, t_s in enumerate(target_spans):
      if overlap(l_s, t_s):
        labels[i] = 1
  return labels

def compare_minimap_frames(docs):
  precs = []
  recls = []
  for pmid, d in docs.items():
    print('DOCUMENTS:', pmid)
    for f in d['frames']:
      fd = f._asdict()
      for e in 'ico':
        print()
        print('\t', 'FRAME ELEMENTS:', e)
        print('\t', fd[e])
        f_uids = d['mm_frames'][fd[e]]
        for uid in f_uids:
          print('\t\t', '[{}]'.format(uid))
        true_spans = [s for s in fd['{}_spans'.format(e)] if s.i >= 0]
        print()
        print('\t', 'ANNOTATOR')
        for s in true_spans:
          print('\t', s.text)
        print()
        print('\t', 'MINIMAP')
        for s, s_uids in d['mm_spans']:
          if len(f_uids & s_uids) >= 1:
            print('\t', s.s)
            for uid in s_uids:
              print('\t\t', '{} [{}]'.format('+' if uid in f_uids else '-', uid))
      print()
      input()
      print('\n'*10)
        
def write_brat_output(docs):
  pmid_docs = defaultdict(dict)
  for d in docs:
    pmid_docs[d['id']][d['worker']] = d

  for pmid, worker_docs in pmid_docs.items():
    lines = []
    t = 0
    for worker, doc in worker_docs.items():
      for g_id, g_spans in doc['groups'].items():
        for s in g_spans:
          if s.i >= 0:
            lines.append('T{}\t{}_{} {} {}\t{}'.format(t, g_id[0], worker, s.i, s.f, s.text))
            lines.append('A{}\tGroup T{} {}'.format(t, t, g_id[2:]))
            t += 1
    with open('../data/brat/{}.text'.format(pmid), 'w') as fout:
      fout.write(doc['text'])
    with open('../data/brat/{}.ann'.format(pmid), 'w') as fout:
      fout.write('\n'.join(lines))
    
