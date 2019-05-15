import sys, json
from itertools import groupby
from operator import itemgetter

import utils

def format_ner(ner_dir, group = 'dev'):
  input_text = open('{}/ner_tagger/{}/{}.txt'.format(utils.DATA_DIR, ner_dir, group)).read().strip()
  output_text = open('{}/ner_tagger/{}/pred.txt'.format(utils.DATA_DIR, ner_dir)).read().strip()

  input_seqs = input_text.split('\n\n')
  output_seqs = output_text.split('\n\n')
  assert len(input_seqs) == len(output_seqs)

  seq_matches = [0,0]

  all_spans = {'i': set(), 'o': set()}
  doc_spans = {'i': set(), 'o': set()}
  doc_sent_spans = []
  expected_sent_idx = 0
  cur_pmid = None

  for input_seq, output_seq in zip(input_seqs, output_seqs):

    input_lines = [l.split() for l in input_seq.split('\n')]
    output_lines = [l.split() for l in output_seq.split('\n')]

    tokens, pmids, token_idxs, sent_idxs, dummy = zip(*input_lines)
    true_labels, pred_labels, out_tokens = zip(*output_lines)

    assert len(set(pmids)) == 1
    pmid = pmids[0]
    if pmid != cur_pmid:
      if cur_pmid != None:
        write_doc(ner_dir, cur_pmid, doc_spans, doc_sent_spans)
      doc_spans = {'i': set(), 'o': set()}
      doc_sent_spans = []
      cur_pmid = pmid
    
    assert len(set(sent_idxs)) == 1
    sent_idx = int(sent_idxs[0])
    if sent_idx == 0:
      expected_sent_idx = 0

    if sent_idx > expected_sent_idx:
      print('Skipping empty sentence output for', expected_sent_idx)
      assert expected_sent_idx < sent_idx
      while sent_idx > expected_sent_idx:
        doc_sent_spans.append({})
        expected_sent_idx += 1
    elif sent_idx < expected_sent_idx:
      print('it is effed')
      input()

    if len(input_lines) != len(output_lines):
      pad_len = len(input_lines) - len(output_lines)
      pred_labels = pred_labels + tuple(['0']*pad_len)

    sent_spans = {'i': set(), 'o': set()}
    label_spans = utils.condense_labels(pred_labels)
    for i, f, l in label_spans:
      if l == '[SEP]':
        continue
      string = ' '.join(tokens[i:f])
      doc_spans[l].add(string)
      all_spans[l].add(string)
      sent_spans[l].add(string)

    doc_sent_spans.append(sent_spans)
    expected_sent_idx += 1

  # flush final doc
  write_doc(ner_dir, cur_pmid, doc_spans, doc_sent_spans)

def write_doc(ner_dir, pmid, doc_spans, doc_sent_spans):
  with open('{}/documents/sents/{}.ner_{}'.format(utils.DATA_DIR, pmid, ner_dir), 'w') as fout:
    for s in doc_sent_spans:
      json_doc_sent_spans = { k: list(v) for k, v in s.items() }
      fout.write('{}\n'.format(json.dumps(json_doc_sent_spans)))
  with open('{}/documents/txts/{}.ner_{}'.format(utils.DATA_DIR, pmid, ner_dir), 'w') as fout:
    json_doc_spans = { k: list(v) for k, v in doc_spans.items() }
    fout.write('{}'.format(json.dumps(json_doc_spans)))


if __name__ == '__main__':
  format_ner(sys.argv[1])
