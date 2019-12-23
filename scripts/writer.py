from collections import defaultdict
import os
import json
import classes
import processing
import random
import operator

def dummy_label(source_labels):
	return ['0']*len(source_labels)


def first_char(source_labels, label_pref = operator.itemgetter(0), neg_label = '0'):
	ner_labels = []
	for token_label_list in source_labels:
		if len(token_label_list) == 0: label = neg_label
		if len(token_label_list) == 1: label = token_label_list[0][0]
		if len(token_label_list) >= 2: label = label_pref(token_label_list)[0]
		ner_labels.append(label)
	return ner_labels

def write_ner_data(docs, label_fn, fdir, allow_acronyms = False):
	group_docs = defaultdict(list)
	for doc in docs:
		group_docs[doc.group].append(doc)

	def init_seq(pmid):
		return { 'pmid': pmid, 'tokens': [], 'labels': [], 'offsets': [] }

	for group, doc_list in group_docs.items():
		rows = []
		for doc in doc_list:
			if doc.has_acronyms and not allow_acronyms:
				print('Skipping doc without acronym subs: {}'.format(doc.id))
				continue
			if not doc.parsed:
				doc.parse_text()
			seq = init_seq(doc.id)
			source_labels = doc.get_token_labels()
			doc_labels = label_fn(source_labels)
			for t, l in zip(doc.tokens, doc_labels):
				seq['tokens'].append(t.text)
				seq['labels'].append(l)
				seq['offsets'].append((t.i, t.f))
				if t.text == '.':
					rows.append(seq)
					seq = init_seq(doc.id)
			# flush any extra tokens after last "."
			if len(seq['tokens']) > 0:
				rows.append(seq)

		fname = os.path.join(fdir, '{}.json'.format(group))
		with open(fname, 'w') as fout:
			fout.write(json.dumps(rows, indent = 2))
