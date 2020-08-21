import glob, os, json, re

import pandas as pd
import numpy as np

import classes
import writer
import minimap

"""
input schema:

data = [
	{
		"abstract": string,
		"pmid": string,
		"p": list of strings,
		"i": list of strings,
		"o": list of strings
	}
]
"""
def process_json_data(data):
	docs = []
	for d in data:
		doc = classes.Doc(d['pmid'], d['abstract'])
		for e in 'pio':
			for span in d[e]:
				for m in re.finditer(re.escape(span), doc.text):
					doc.labels['NER_'+e].append(classes.Span(m.start(), m.end(), span))
		for span in d.get('ev', []):
			for m in re.finditer(re.escape(span), doc.text):
				doc.labels['BERT_ev'].append(classes.Span(m.start(), m.end(), span))
		doc.group = 'test'
		doc.parse_text()
		docs.append(doc)
	return docs

def read_shard_docs(data_dir):
	print('\t\tcreating Docs for {}'.format(data_dir))
	fnames = glob.glob('{}/*.text'.format(data_dir))
	docs = [classes.Doc(os.path.basename(f).strip('.text'), open(f).read()) for f in fnames]
	docs = [d for d in docs if d.text]
	for d in docs:
		d.parse_text()
		d.group = 'test'
		d.sf_lf_map = {} # already acronym'd
	return docs

def get_icos(d):
	icos = []
	for ev in d.labels['BERT_ev']:
		if not hasattr(ev, 'pred_os'):
			continue
		for pred_o in ev.pred_os:
			icos.append({ \
					'i': ev.pred_i.text,
					'c': ev.pred_c.text,
					'o': pred_o.text,
					'l': pred_o.label,
					'i_mesh': list(get_mesh(ev.pred_i.text)),
					'c_mesh': list(get_mesh(ev.pred_c.text)),
					'o_mesh': list(get_mesh(pred_o.text)),
					'ev': ev.text })
	return icos

def generate_trialstreamer_inputs(docs):
	rows = []
	for d in docs:
		r = {}
		r['text'] = d.text
		r['pmid'] = d.id
		r['p_spans'] = [(s.i, s.f) for s in d.labels['NER_p']]
		r['i_spans'] = [(s.i, s.f) for s in d.labels['NER_i']]
		r['o_spans'] = [(s.i, s.f) for s in d.labels['NER_o']]
		r['p_mesh'] = [m['mesh_term'] for m in minimap.get_unique_terms([s.text for s in d.labels['NER_p']])]
		r['i_mesh'] = [m['mesh_term'] for m in minimap.get_unique_terms([s.text for s in d.labels['NER_i']])]
		r['o_mesh'] = [m['mesh_term'] for m in minimap.get_unique_terms([s.text for s in d.labels['NER_o']])]
		r['frames'] = get_icos(d)
		rows.append(r)
	return rows

def write_trialstreamer_inputs(docs, top):
	fname = '{}/pipeline_data.json'.format(top)
	if not os.path.isfile(fname):
		print('\t\tconstructing JSON')
		rows = generate_trialstreamer_inputs(docs)
		json.dump(rows, open(fname, 'w'))

def add_json_cuis(top):
	print('Loading pipeline data from', top)
	docs = json.load(open(os.path.join(top, 'pipeline_data.json'), 'r'))
	for doc in docs:
		for e in 'pio':
			spans = [doc['text'][i:f] for i,f in doc[e+'_spans']]
			mesh = minimap.get_unique_terms(spans)
			doc[e+'_cuis'] = [m['cui'] for m in mesh]
			doc[e+'_duis'] = [m['mesh_ui'] for m in mesh]
	json.dump(docs, open('{}/pipeline_data.json'.format(top), 'w'))

def add_json_titles(top):
	docs = json.load(open(os.path.join(top, 'pipeline_data.json'), 'r'))
	for doc in docs:
		doc['title'] = open('{}/{}.title'.format(top, doc['pmid'])).read()
	json.dump(docs, open('{}/pipeline_data.json'.format(top), 'w'))

def print_ico(ico):
	print(ico['ev'].replace('\n', '\\n'))
	for e in 'ico':
		print('\t{}: [{}] {}'.format(e, ico[e], '|'.join(ico[e+'_mesh'])))

def generate_shard_files():
	print('Reading trial_annotations.csv')
	df = pd.read_csv('/home/ben/Desktop/forked_trialstreamer/trialstreamer/data/trial_annotations.csv')
	start_idx = 550000
	shard_size = 10000
	for i,f in list(zip(range(start_idx,len(df),shard_size), range(start_idx+shard_size,len(df),shard_size))):
		print('parsing shard {}_{}'.format(i,f))
		os.system('mkdir -p ../data/trialstreamer/{}_{}'.format(i,f))
		for idx,r in df.ix[i:f,:].iterrows():
			if type(r.ab) != str: continue
			d = classes.Doc(idx, r.ab)
			d.replace_acronyms()
			open('../data/trialstreamer/{}_{}/{}.text'.format(i,f,idx), 'w').write(d.text)
			open('../data/trialstreamer/{}_{}/{}.title'.format(i,f,idx), 'w').write(r.ti)
