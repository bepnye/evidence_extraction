from collections import defaultdict
import os, json, random, operator, glob

import classes
import processing
import utils

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
			if not (allow_acronyms or doc.has_sf_lf_map()):
				print('Skipping doc without acronym subs: {}'.format(doc.id))
				continue
			if not doc.parsed:
				doc.parse_text()
			seq = init_seq(doc.id)
			#source_labels = doc.get_token_labels()
			#doc_labels = label_fn(source_labels)
			doc_labels = ['0']*len(doc.tokens)
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

def write_sent_data(docs, fdir, balance_classes = False):
	group_docs = defaultdict(list)
	for doc in docs:
		group_docs[doc.group].append(doc)
	for group, doc_list in group_docs.items():
		fout = open('{}/{}.tsv'.format(fdir, group), 'w')
		for doc in doc_list:
			sent_labels = [any([utils.s_overlap(s, f.ev) for f in doc.frames]) for s in doc.sents]
			pos_sents = [s.text for s, l in zip(doc.sents, sent_labels) if l]
			neg_sents = [s.text for s, l in zip(doc.sents, sent_labels) if not l]
			if balance_classes:
				neg_samples = []
				for pos_s in pos_sents:
					neg_sents = sorted(neg_sents, key = lambda s: abs(len(s) - len(pos_s)))
					try:
						neg_samples.append(neg_sents.pop(0))
						neg_samples.append(neg_sents.pop(0))
					except IndexError:
						print('Warning: unable to sample enough negatives from doc {}'.format(doc.id))
				neg_sents = neg_samples
			for s in pos_sents:
				fout.write('1\t{}\t{}\n'.format(doc.id, utils.clean_str(s)))
			for s in neg_sents:
				fout.write('0\t{}\t{}\n'.format(doc.id, utils.clean_str(s)))

def write_sent_data_pipeline(docs, fdir):
	group_docs = defaultdict(list)
	for doc in docs:
		group_docs[doc.group].append(doc)
	for group, doc_list in group_docs.items():
		fout = open('{}/{}.tsv'.format(fdir, group), 'w')
		for doc in doc_list:
			for sent in doc.sents:
				fout.write('0\t{}\t{}\n'.format(doc.id, utils.clean_str(sent.text)))

def write_o_ev_data(docs, fdir, add_i = False):
	group_docs = defaultdict(list)
	for doc in docs:
		group_docs[doc.group].append(doc)
	for group, doc_list in group_docs.items():
		fout = open('{}/{}.tsv'.format(fdir, group), 'w')
		for doc in doc_list:
			for frame in doc.frames:
				sents = utils.s_overlaps(frame.ev, doc.sents)
				ev_text = utils.clean_str(doc.text[sents[0].i:sents[-1].f])
				o_text = utils.clean_str(frame.o.text)
				if add_i:
					o_text = '{} effect on {}'.format(utils.clean_str(frame.i.text), o_text)
				fout.write('{}\t{}\t{}\t{}\n'.format(frame.label+1, doc.id, o_text, ev_text))

def write_o_ev_data_pipeline(docs, fdir):
	fout = open('{}/{}.tsv'.format(fdir, docs[0].group), 'w')
	for doc in docs:
		assert doc.group == 'test' or doc.group == 'testtest'
		for ev_span in doc.labels['BERT_ev']:
			for o_span in utils.s_overlaps(ev_span, doc.labels['NER_o']):
				fout.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format('0', doc.id, o_span.i, o_span.f, \
						utils.clean_str(o_span.text), utils.clean_str(ev_span.text)))

def intro_group_ev(doc, i, ev):
	ev_sents = [s for idx, s in enumerate(doc.sents) if utils.s_overlap(s, ev)]
	group_idx = doc.text.lower().find('group')
	context_sents = [s for idx, s in enumerate(doc.sents) if s.i <= group_idx <= s.f or 2 <= idx <= 4]
	return ' '.join([s.text for s in ev_sents + context_sents])

def ev_abst(doc, i, ev):
	ev_s = [s.text for idx, s in enumerate(doc.sents) if utils.s_overlap(s, ev)]
	other_s = [s.text for idx, s in enumerate(doc.sents) if not utils.s_overlap(s, ev)]
	return ' '.join(ev_s + other_s)

def first_and_ev(doc, i, ev):
	i_idx = doc.text.lower().find(i.text.lower().strip('. ,)('))
	g_idx = doc.text.lower().find('group')
	sents = [s for idx, s in enumerate(doc.sents) if \
				utils.s_overlap(s, ev) or \
				s.i <= i_idx <= s.f or \
				s.i <= g_idx <= s.f]
	return ' '.join([s.text for s in sents])

def write_i_c_data(docs, context_fn, fdir, neg_prob = 0.5):
	group_docs = defaultdict(list)
	for doc in docs:
		group_docs[doc.group].append(doc)
	for group, doc_list in group_docs.items():
		fout = open('{}/{}.tsv'.format(fdir, group), 'w')
		for doc in doc_list:
			visited_frames = {}
			for f in doc.frames:
				k = (f.i.text, f.c.text, f.ev.text)
				if k not in visited_frames:
					context_text = utils.clean_str(context_fn(doc, f.i, f.ev))
					fout.write('2\t{}\t{}\t{}\n'.format(doc.id, utils.clean_str(f.i.text), context_text))
					fout.write('1\t{}\t{}\t{}\n'.format(doc.id, utils.clean_str(f.c.text), context_text))
					visited_frames[k] = True
					if random.random() <= neg_prob:
						neg_i = get_neg_i(doc, f)
						fout.write('0\t{}\t{}\t{}\n'.format(doc.id, utils.clean_str(neg_i.text), context_text))

def get_neg_i(d, f):
	if random.random() <= 0.5:
		neg_i = f.o
	elif random.random() <= 0.75:
		neg_i = classes.Span(-1, -1, '{} vs. {}'.format(f.i.text, f.c.text))
	else:
		t_idx_i = random.randint(0, len(d.tokens)-4)
		tokens = d.tokens[t_idx_i:t_idx_i+random.randint(2,4)]
		neg_i = classes.Span(-1, -1, d.text[tokens[0].i:tokens[-1].f])
	return neg_i

def write_i_c_data_pipeline(docs, context_fn, fdir):
	group_docs = defaultdict(list)
	for doc in docs:
		group_docs[doc.group].append(doc)
	for group, doc_list in group_docs.items():
		fout = open('{}/{}.tsv'.format(fdir, group), 'w')
		for doc in doc_list:
			for ev in doc.labels['BERT_ev']:
				visited_frames = {}
				for i in doc.labels['NER_i']:
					k = i.text
					if k not in visited_frames:
						context_text = utils.clean_str(context_fn(doc, i, ev))
						fout.write('0\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format( \
								doc.id, ev.i, ev.f, i.i, i.f, utils.clean_str(i.text), context_text))
						visited_frames[k] = True
