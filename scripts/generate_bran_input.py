import sys
import itertools

import classes
import processing
import process_coref_annes
import process_evidence_inference

sys.path.append('/home/eric/bran/src/processing/utils/')
from word_piece_tokenizer import WordPieceTokenizer as WPT
tokenizer = WPT('/home/eric/bran/data/cdr/word_piece_vocabs/just_train_2500/word_pieces.txt', entity_str = 'ENTITY_')

def find_wp_indices(span, wp_spans):
	s_i = -1; s_f = -1
	for idx, wp in enumerate(wp_spans):
		if wp.i >= span.i:
			s_i = idx
			break
	# technically only need to search from s_i onwards
	for idx, wp in enumerate(wp_spans):
		if wp.f >= span.f:
			s_f = idx + 1 # add one so [i:f] includes the final wp token
			break
	return s_i, s_f

def get_wp_spans(text):
	wp_tokens = tokenizer.tokenize(text)
	cur_i = 0
	wp_spans = []
	for t in wp_tokens:
		t_text = t.strip('@@')
		i = text.find(t_text, cur_i)
		f = i + len(t_text)
		wp_spans.append(classes.Span(i, f, t))
		cur_i = f
	return wp_spans

def format_doc_bran_data(entities, entity_pair_rels, doc):
	entity_map = { e.text: e for e in entities }
	wp_spans = get_wp_spans(doc.text)
	entity_mention_locations = {}
	for e_str, e in entity_map.items():
		if e.mentions:
			wp_offsets = [find_wp_indices(m, wp_spans) for m in e.mentions]
			wp_starts, wp_ends = zip(*wp_offsets)
		else:
			wp_starts = [0]
			wp_ends = [0]
		entity_mention_locations[e_str] = (wp_starts, wp_ends)

	def get_e_cols(e):
		return [\
		  entity_map[e].name, entity_map[e].type, e, \
		  ':'.join(map(str, entity_mention_locations[e][0])), \
		  ':'.join(map(str, entity_mention_locations[e][1]))]
	pos_rows = []
	neg_rows = []
	for (e1, e2), r in entity_pair_rels.items():
		row = get_e_cols(e1) + get_e_cols(e2) + [doc.id, r, ' '.join([wp.text for wp in wp_spans])]
		if r == 'NULL':
			neg_rows.append(row)
		else:
			pos_rows.append(row)

	ner_tokens = [wp.text for wp in wp_spans]
	ner_docids = [doc.id for _ in wp_spans]
	ner_labels = ['O' for _ in wp_spans]
	ner_enames = [-1  for _ in wp_spans]
	# only assign NER labels for the NER spans that actually got assigned as mentions
	# this is necessary in order to use the "name" from the assigned entity!
	for e in entities:
		for span in e.mentions:
			wp_i, wp_f = find_wp_indices(span, wp_spans)
			ner_enames[wp_i] = e.name
			ner_labels[wp_i] = 'B-'+e.type
			for idx in range(wp_i+1, wp_f):
				ner_labels[idx] = 'I-'+e.type
	ner_rows = list(zip(ner_tokens, ner_labels, ner_enames, ner_docids))
	return pos_rows, neg_rows, ner_rows

def write_bran_data_group(docs, bran_data, group, top_dir = '../../bran/data/cre/evinf'):
	pos_fp = open('{}/positive_CRE_{}.txt'.format(top_dir, group), 'w')
	neg_fp = open('{}/negative_CRE_{}.txt'.format(top_dir, group), 'w')
	ner_fp = open('{}/ner_CRE_{}.txt'.format(top_dir, group), 'w')
	print('Writing data for {} docs to {}'.format(len(docs), pos_fp))
	for (es, eps), doc in zip(bran_data, docs):
		pos_rows, neg_rows, ner_rows = format_doc_bran_data(es, eps, doc)
		for r in pos_rows: pos_fp.write('\t'.join(map(str, r)) + '\n')
		for r in neg_rows: neg_fp.write('\t'.join(map(str, r)) + '\n')
		for r in ner_rows: ner_fp.write('\t'.join(map(str, r)) + '\n')
		ner_fp.write('\n')

def process_bran_data():
	ev_inf_docs = process_evidence_inference.read_docs(True)
	coref_docs = process_coref_annes.read_docs(True)
	# TODO: save the sf_lf_map info somewhere so I don't have to recompute
	for d in ev_inf_docs: d.replace_acronyms()
	for d in coref_docs: d.replace_acronyms()
	ner_top = '../data/ner/'
	processing.add_ner_output(ev_inf_docs, '{}/ev_inf.json'.format(ner_top))
	processing.add_ner_output(coref_docs, '{}/coref_dev.json'.format(ner_top))
	train_data = [processing.extract_distant_info(d) for d in ev_inf_docs]
	test_data = [processing.extract_unsupervised_info(d) for d in coref_docs]
	write_bran_data_group(ev_inf_docs[0:300], train_data[0:300], 'dev')
	write_bran_data_group(ev_inf_docs[300:], train_data[300:], 'train')
	write_bran_data_group(coref_docs, test_data, 'test')

process_bran_data()
