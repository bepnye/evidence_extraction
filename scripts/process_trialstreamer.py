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

def process_covid_data():
	top = '../data/covid/'
	fnames = glob.glob('{}/json/*/*.json'.format(top))
	print('Processing {} files...'.format(len(fnames)))
	docs = []
	for f in fnames:
		j = json.load(open(f))
		pmid = j['paper_id']
		title = j['metadata']['title']
		abst = '\n\n'.join([p['text'] for p in j['abstract']])
		body = '\n\n'.join([p['text'] for p in j['body_text']])
		text = '\n\n\n'.join([abst, body])
		doc = classes.Doc(pmid, text)
		doc.group = 'test'
		docs.append(doc)
		with open('{}/docs/{}.abst'.format(top, pmid),  'w') as fp: fp.write(abst)
		with open('{}/docs/{}.body'.format(top, pmid),  'w') as fp: fp.write(body)
		with open('{}/docs/{}.title'.format(top, pmid), 'w') as fp: fp.write(title)
	return docs

def process_cwr_data():
	df = pd.read_csv('../data/cures_within_reach/cwr.csv')
	df = df[~df.Relevant.apply(np.isnan)]
	df = df[df['Matched.Outcome..Word.Embeddings.'].apply(lambda o: type(o) == str)]
	docs = {}
	for idx, r in df.iterrows():
		if r.PMID in docs:
			print('Ignoring dupe id: {}'.format(r.PMID))
			continue
		if type(r.Abstract) is not str:
			continue
		text = r.Abstract.replace('\r', '')
		text = re.sub('\n+', '\n', text)
		doc = classes.Doc(r.PMID, text)
		doc.group = 'test'
		with open('../data/cures_within_reach/{}.text'.format(r.PMID), 'w') as fp:
			fp.write(doc.text)
		with open('../data/cures_within_reach/{}.title'.format(r.PMID), 'w') as fp:
			fp.write(r.Title)

		p_match = r['Article.Population..Word.Embeddings.']
		i_match = r['Article.Intervention..Word.Embeddings.']
		o_match = r['Article.Outcome..Word.Embeddings.']

		p_query = r['Matched.Population..Word.Embeddings.']
		i_query = r['Matched.Intervention..Word.Embeddings.']
		o_query = r['Matched.Outcome..Word.Embeddings.']

		doc.query = (p_query, i_query, o_query)
		doc.match = (p_match, i_match, o_match)
		doc.relevant = r.Relevant

		docs[r.PMID] = doc
	return list(docs.values())

def read_cwr_docs(data_dir = '../data/cures_within_reach/'):
	fnames = glob.glob('{}/*.text'.format(data_dir))
	docs = [classes.Doc(os.path.basename(f).strip('.text'), open(f).read()) for f in fnames]
	docs = [d for d in docs if d.text]
	for d in docs:
		d.parse_text()
		d.group = 'test'
		d.sf_lf_map = {} # already acronym'd
	return docs

def read_shard_docs(data_dir):
	print('\t\tcreating Docs for {}'.format(shard))
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
		rows = generate_trialstreamer_inputs(docs, top)
		json.dump(rows, open(fname, 'w'))

def print_ico(ico):
	print(ico['ev'].replace('\n', '\\n'))
	for e in 'ico':
		print('\t{}: [{}] {}'.format(e, ico[e], '|'.join(ico[e+'_mesh'])))

def FLATTEN_MESH(m):
	if m in ['Progression-Free Survival', 'Survival Rate', 'Event-Free Survival', 'Cumulative Survival Rate', 'Death', 'Mortality']:
		return 'Survival'
	return m

def get_mesh(s):
	return set([FLATTEN_MESH(m['mesh_term']) for m in minimap.get_unique_terms([s])])

def get_cwr_metadata(docs, verbose = False):
	pmid_to_query = {}
	for d in docs:
		if d.labels['NER_p']:
			doc_p_mesh = set.union(*[get_mesh(s.text) for s in d.labels['NER_p']])
			query_p_mesh = get_mesh(d.query[0])
			p_match = query_p_mesh <= doc_p_mesh
		else:
			p_match = False
		pmid_to_query[d.id] = {
			'p': d.query[0],
			'i': d.query[1],
			'o': d.query[2],
			'p_mesh': list(get_mesh(d.query[0])),
			'i_mesh': list(get_mesh(d.query[1])),
			'o_mesh': list(get_mesh(d.query[2])),
			'l': d.relevant,
			'p_match': p_match }
		
	return pmid_to_query

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
