import traceback, glob, json
import pandas as pd
from collections import defaultdict
from functools import partial

from Bio import Entrez
Entrez.email = 'bepnye@gmail.com'

import utils
import classes
import minimap
import run_pipeline as run

def process_eric_data():
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

def read_eric_docs(data_dir = '../data/cures_within_reach/eric_data'):
	fnames = glob.glob('{}/*.text'.format(data_dir))
	docs = [classes.Doc(os.path.basename(f).strip('.text'), open(f).read()) for f in fnames]
	docs = [d for d in docs if d.text]
	for d in docs:
		d.parse_text()
		d.group = 'test'
		d.sf_lf_map = {} # already acronym'd
	return docs

def read_55(data_dir = '../data/cures_within_reach/55_sample'):
	df = pd.read_csv('{}/55_sample.csv'.format(data_dir))
	df.rename(columns = {c: c.lstrip() for c in df.columns}, inplace = True)
	docs = []
	for idx, r in df.iterrows():
		doc = classes.Doc(r.pmid, r.abstract)
		doc.qp = r.disease
		doc.qi = r.drugs
		doc.parse_text()
		doc.group = 'test'
		docs.append(doc)
	return docs

def read_covid(data_dir = '../data/cures_within_reach/covid'):
	df = pd.read_csv('{}/covid_docs.csv'.format(data_dir))
	docs = []
	for idx, r in df.iterrows():
		abst = r.Abstract if type(r.Abstract) == str else ''
		doc = classes.Doc(r.EntrezUID, '{}\n\n{}'.format(r.Title, abst))
		doc.parse_text()
		doc.group = 'test'
		docs.append(doc)
	return docs

def annotate_docs(docs, top = \
		'/home/ben/Desktop/evidence_extraction/data/cures_within_reach/55_sample'):
	run.dump_phase1_ner(top, docs)
	run.exec_phase1_ner(top, 'ebm_p')
	run.load_phase1_ner(top, docs)
	run.exec_phase1_ner(top, 'ebm_i')
	run.load_phase1_ner(top, docs)
	run.exec_phase1_ner(top, 'ebm_o')
	run.load_phase1_ner(top, docs)

	run.dump_phase1_ev(top, docs)
	run.exec_phase1_ev(top)
	run.load_phase1_ev(top, docs, label_fn = partial(utils.prob_thresh, thresh = 0.5))
	
	run.dump_phase2_o_ev(top, docs)
	run.exec_phase2_o_ev(top)
	run.load_phase2_o_ev(top, docs)
	
	run.dump_phase2_ic_ev(top, docs)
	run.exec_phase2_ic_ev(top)
	run.load_phase2_ic_ev(top, docs)

def get_mesh_terms(strs):
	if not strs:
		return []
	if type(strs) != list:
		strs = [strs]
	if type(strs[0]) == classes.Span:
		strs = [s.text for s in strs]
	return list(set([m['mesh_term'] for m in minimap.get_unique_terms(strs)]))

def get_master_list(e):
	strs = [l.strip() for l in open('../data/cures_within_reach/master_{}.csv'.format(e)).readlines()]
	mesh = get_mesh_terms(strs)
	return mesh

def get_p_list():
	return get_master_list('p')
def get_i_list():
	return get_master_list('i')
def get_o_list():
	return get_master_list('o')

def load_master_lists():
	global master_p
	global master_i
	global master_o
	master_p = get_p_list()
	master_i = get_i_list()
	master_o = get_o_list()

def markup_docs(docs, has_query = True):
	for d in docs:
		d.p = d.labels['NER_p']
		d.i = d.labels['NER_i']
		d.o = d.labels['NER_o']
		d.p_mesh = get_mesh_terms(d.labels['NER_p'])
		d.i_mesh = get_mesh_terms(d.labels['NER_i'])
		d.o_mesh = get_mesh_terms(d.labels['NER_o'])
		if has_query:
			d.qp_mesh = get_mesh_terms([d.qp])
			d.qi_mesh = get_mesh_terms([d.qi])
		d.frames = []
		if not d.i:
			continue
		for s in d.labels['BERT_ev']:
			i = s.pred_i
			c = s.pred_c
			for o in s.pred_os:
				d.frames.append({
					'i': i.text,
					'o': o.text,
					'i_mesh': get_mesh_terms([i]),
					'o_mesh': get_mesh_terms([o]),
					'ev': s.text,
					'label': o.label
				})

def intersects(l1, l2):
	return len(set(l1).intersection(l2)) > 0

def p_subset(d):
	return intersects(d.qp_mesh, d.p_mesh)

def i_subset(d):
	return intersects(d.qi_mesh, d.i_mesh)

def o_subset(d):
	return intersects(d.qo_mesh, d.o_mesh)

def str_match(d):
	for s in d.p:
		if d.qp in s.text:
			return True
	return False

def raw_str(d):
	return d.qp in d.text

def match_p(docs):
	results = defaultdict(int)
	for d in docs:
		matched = False
		if p_subset(d):
			matched = True; results['mesh_hit'] += 1
		if str_match(d):
			matched = True; results['str_hit'] += 1
		if set(all_terms).intersection(d.p_mesh):
			matched = True; results['global'] += 1
		if raw_str(d):
			matched = True; results['raw'] += 1
		if not matched:
			results['miss'] += 1
	return results

def i_frame(d):
	for f in d.frames:
		if intersects(d.qi_mesh, f['i_mesh']):
			return True
	return False

def match_i(docs):
	results = defaultdict(int)
	for d in docs:
		matched = False
		if i_subset(d):
			matched = True; results['mesh_hit'] += 1
		if i_frame(d):
			matched = True; results['frame_hit'] += 1
		if not matched:
			print(doc.i)
			print(doc.qi)
			results['miss'] += 1
	return results

def format_strs(l, sep = '\n'):
	return sep.join(['[{}]'.format(s) for s in l])

def enc_label(l):
	return { -1: 'decreased', 0: 'unchanged', 1: 'increased' }[l]

def format_frames(fs, label_key = 'label'):
	frame_strs = []
	for f in fs:
		f_dict = { k: f[k] for k in ['i', 'c', 'o', 'ev'] }
		f_dict[label_key] = enc_label(f[label_key])
		frame_strs.append(json.dumps(f_dict, indent = 2))
	return '\n'.join(frame_strs)

def get_55_df(docs):
	rows = []
	for d in docs:
		r = {}
		r['text'] = d.text
		r['query_p'] = d.qp
		r['query_i'] = d.qi
		r['p_model'] = format_strs([s.text for s in d.p])
		r['i_model'] = format_strs([s.text for s in d.i])
		r['o_model'] = format_strs([s.text for s in d.o])
		r['query_p_terms'] = d.qp_mesh
		r['model_p_terms'] = format_strs(d.p_mesh)
		r['p_terms_match'] = p_subset(d)
		r['query_i_terms'] = d.qi_mesh
		r['model_i_terms'] = format_strs(d.i_mesh)
		r['i_terms_match'] = i_subset(d)
		r['frames'] = format_frames(d.frames)
		r['i_frame_match'] = i_frame(d)
		r['model_o_terms'] = format_strs(d.o_mesh)
		rows.append(r)
	return pd.DataFrame(rows)

def global_frame_match_io(frames):
	for f in frames:
		if intersects(d.i_mesh, master_i) and intersects(d.o_mesh, master_o):
			return True
	return False
def global_frame_match_i(frames, i_terms):
	for f in frames:
		if intersects(f['i_mesh'], i_terms):
			return True
	return False

def get_covid_df(docs):
	rows = []
	for d in docs:
		r = {}
		r['text'] = d.text
		r['p_model'] = format_strs([s.text for s in d.p])
		r['i_model'] = format_strs([s.text for s in d.i])
		r['o_model'] = format_strs([s.text for s in d.o])
		r['model_p_terms'] = format_strs(d.p_mesh)
		r['model_i_terms'] = format_strs(d.i_mesh)
		r['model_o_terms'] = format_strs(d.o_mesh)
		r['p_terms_match'] = intersects(master_p, d.p_mesh)
		r['i_terms_match'] = intersects(master_i, d.i_mesh)
		r['o_terms_match'] = intersects(master_o, d.o_mesh)
		r['frames'] = format_frames(d.frames)
		r['frames_match_i'] = global_frame_match_i(d)
		r['frames_match_i_and_o'] = global_frame_match_io(d)
		rows.append(r)
	return pd.DataFrame(rows)

def get_test_set_pmids():
	return ['31717', '27308', '14423', '342629', '231363', '358342', '281252']

def get_new_pmids():
	return [26275735, 22124104, 18809614, 25765952, 29129443, 14736927, 31337877, 23161898, 14736927, 15972865]

def get_entrez_docs(fname = '../data/cures_within_reach/entrez_downloads/docs.csv'):
	df = pd.read_csv(open(fname))
	docs = []
	for idx, r in df.iterrows():
		doc = classes.Doc(r.pmid, r.abst)
		doc.title = r.title
		doc.group = 'test'
		doc.parse_text()
		docs.append(doc)
	return docs

def download_abstracts(pmids, output_fname = 'entrez_docs.csv'):
	handle = Entrez.efetch(db='pubmed', id=pmids, retmode = 'xml')
	records = Entrez.read(handle)

	docs = []
	for doc in records['PubmedArticle']:
		row = {}

		pmid = str(doc['MedlineCitation']['PMID'])
		row['pmid'] = pmid

		ab_strs = doc['MedlineCitation']['Article']['Abstract']['AbstractText']
		ab_text = '\n\n'.join(ab_strs)
		row['abst'] = ab_text

		ab_xml = '<article><front><article-meta><title-group><article-title></article-title></title-group><article-id>'+pmid+'</article-id>'
		ab_xml += '<abstract>'
		for s in ab_strs:
			ab_xml += '<sec>'
			if 'Label' in s.attributes:
				ab_xml += '<title>'+s.attributes['Label']+'</title>'
			ab_xml += '<p>'+str(s)+'</p>'
			ab_xml += '</sec>'
		ab_xml += '</abstract></article-meta></front><body></body></article>'
		row['xml'] = ab_xml

		try:
			mesh = doc['MedlineCitation']['MeshHeadingList']
			mesh_terms = [str(m['DescriptorName']) for m in mesh]
			row['mesh'] = mesh_terms
		except KeyError:
			print('Unable to parse MeSH')
			pass

		title = doc['MedlineCitation']['Article']['ArticleTitle']
		row['title'] = title

		docs.append(row)

	df = pd.DataFrame(docs)
	df.to_csv(open(output_fname, 'w'), index = False)


def extract_ts_docs():
	i_v1 = ['Colchicine', 'Chloroquine', 'Hydroxychloroquine', 'Azithromycin', 'Lopinavir', 'Ritonavir', 'Prazosin', 'Remdesivir', 'Tocilizumab']
	i_v2 = ['propranolol', 'clarithromycin', 'diclofenac', 'estramustine', 'axitinib', 'methylprednisolone', 'oseltamivir', 'sildenafil citrate', 'n-acetylcysteine', 'acetylcysteine', 'metformin', 'simvastatin', 'pravastatin']
	i_v3 = ['ketorolac', 'dexamethasone', 'flurbiprofen', 'artesunate', 'nitroglycerin']
	i_strs = i_v1 + i_v2 + i_v3
	i_terms = set(get_mesh_terms(i_strs))
	o_terms = set(master_o)
	p_terms = set(master_p)
	shard_jsons = glob.glob('../data/trialstreamer/*/*.json')
	all_matches = []
	for f in shard_jsons:
		docs = json.load(open(f))
		shard_matches = []
		for d in docs:
			if i_terms.intersection(d['i_mesh']) and \
				 p_terms.intersection(d['p_mesh']):
				shard_matches.append(d)
		print('{}/{} matches in {}'.format(len(shard_matches), len(docs), f))
		all_matches += shard_matches
	return get_ts_df(all_matches, i_terms)

ti_to_pmid = None
def get_pmid(ti):
	global ti_to_pmid
	if not ti_to_pmid:
		ti_pmid_df = pd.read_csv('../data/trialstreamer/pmid_ti_idx.csv')
		ti_to_pmid = dict(zip(ti_pmid_df.ti.values, ti_pmid_df.pmid.values))
	return ti_to_pmid.get(ti, 'UNKNOWN')

def get_ts_df(docs, i_terms):
	rows = []
	for d in docs:	
		r = {}
		r['gid'] = d['pmid']
		r['pmid'] = get_pmid(d['title'])
		r['text'] = d['text']
		r['p_model'] = format_strs([d['text'][i:f] for i,f in d['p_spans']])
		r['i_model'] = format_strs([d['text'][i:f] for i,f in d['i_spans']])
		r['o_model'] = format_strs([d['text'][i:f] for i,f in d['o_spans']])
		r['model_p_terms'] = format_strs(d['p_mesh'])
		r['model_i_terms'] = format_strs(d['i_mesh'])
		r['model_o_terms'] = format_strs(d['o_mesh'])
		r['o_terms_match'] = intersects(master_o, d['o_mesh'])
		r['frames'] = format_frames(d['frames'], 'l')
		r['frames_match_i'] = global_frame_match_i(d['frames'], i_terms)
		rows.append(r)
	return pd.DataFrame(rows)

def format_eli_frames(doc):
	frames = []
	for ev in doc.labels['BERT_ev']:
		for pred_o in ev.pred_os:
			frames.append({ \
					'i': ev.pred_i.text,
					'c': ev.pred_c.text,
					'o': pred_o.text,
					'ev': ev.text,
					'label': pred_o.label })
	return format_frames(frames)

def get_eli_df(docs):
	rows = []
	for d in docs:
		r = {}
		r['pmid'] = d.id
		r['title'] = d.title
		r['text'] = d.text
		r['p_model'] = '\n'.join([s.text for s in d.labels['NER_p']])
		r['i_model'] = '\n'.join([s.text for s in d.labels['NER_i']])
		r['o_model'] = '\n'.join([s.text for s in d.labels['NER_o']])
		r['p_model_terms'] = '\n'.join(get_mesh_terms(r['p_model']))
		r['i_model_terms'] = '\n'.join(get_mesh_terms(r['i_model']))
		r['o_model_terms'] = '\n'.join(get_mesh_terms(r['o_model']))
		r['frames'] = format_eli_frames(d)
		rows.append(r)
	return pd.DataFrame(rows)
