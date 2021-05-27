import traceback, glob, json, itertools
import pandas as pd
from collections import defaultdict, Counter
from functools import partial

from Bio import Entrez
Entrez.email = 'bepnye@gmail.com'

import utils
import classes
import minimap
import run_pipeline as run
from utils import safe_div

def load_mesh_terms(fname = 'minimap/full_mesh.json'):
	terms = json.load(open(fname))
	uid_terms = { t['uid']: t for t in terms }
	return uid_terms

def get_uid_terms(uids, include_children = False, uid_terms = None):
	uid_terms = uid_terms or load_mesh_terms()
	matches = [uid_terms[uid] for uid in uids]
	if include_children:
		root_tns = sum([m['tree_numbers'] for m in matches], [])
		for term in uid_terms.values():
			for child_tn, root_tn in itertools.product(term['tree_numbers'], root_tns):
				if child_tn.startswith(root_tn + '.'):
					matches.append(term)
					break
	return matches

def get_mesh_warnings(doc_df, query_terms, min_freq_thresh = 0.05, min_ratio_thresh = 10):
	n = len(doc_df)
	print('Counting true MeSH...')
	true_mesh = doc_df.mesh.values
	true_counts = Counter(sum(true_mesh, []))
	true_freqs = defaultdict(float, [(k, v/n) for k, v in true_counts.items()])
	
	print('Mapping abstracts...')
	pred_mesh = doc_df.pred_mesh.values
	pred_counts = Counter(sum(pred_mesh, []))
	pred_freqs = defaultdict(float, [(k, v/n) for k, v in pred_counts.items()])

	print('Generating comparison...')
	pred_mesh = sorted(pred_freqs.keys(), key = lambda m: pred_freqs[m], reverse = True)
	print('NEU\tPubMed\tMeSH\t\tTerm')
	for q in query_terms:
		q_mesh = get_mesh_terms(q)
		for m in q_mesh:
			if pred_freqs[m] > min_freq_thresh:
				if safe_div(pred_freqs[m], true_freqs[m]) > min_ratio_thresh or true_freqs[m] == 0:
					print('{:.3f}\t{:.3f}\t[{}]\t"{}"'.format(pred_freqs[m], true_freqs[m], m, q))

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

def get_mesh_terms(strs, key = 'mesh_term'):
	if not strs:
		return []
	if type(strs) != list:
		strs = [strs]
	if type(strs[0]) == classes.Span:
		strs = [s.text for s in strs]
	return list(set([m[key] for m in minimap.get_unique_terms(strs)]))

def get_mesh_uids(s):
	return [m['mesh_ui'] for m in minimap.get_unique_terms([s])]

def get_master_list(e, map_terms = False):
	terms = [l.strip() for l in open('../data/cures_within_reach/master_{}.csv'.format(e)).readlines()]
	if map_terms:
		terms = get_mesh_terms(terms)
	return terms

def get_p_list():
	return get_master_list('p')
def get_i_list():
	return get_master_list('i')
def get_o_list():
	return get_master_list('o_pruned')

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
				d.frames.append(classes.Frame(
						i, c, o, s, o.label))
				'''
				d.frames.append({
					'i': i.text,
					'o': o.text,
					'i_mesh': get_mesh_terms([i]),
					'o_mesh': get_mesh_terms([o]),
					'ev': s.text,
					'label': o.label
				})
				'''

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
		f_dict = { k: f[k] for k in ['i', 'c', 'o', 'i_mesh', 'c_mesh', 'o_mesh', 'ev'] }
		f_dict[label_key] = enc_label(f[label_key])
		frame_strs.append(json.dumps(f_dict, indent = 2))
	return '\n'.join(frame_strs)

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

def sanitize(s):
	s_clean = s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;');
	return s_clean

def download_abstracts(pmids, output_fname = 'entrez_docs.csv'):
	handle = Entrez.efetch(db='pubmed', id=pmids, retmode = 'xml')
	records = Entrez.read(handle)
	return parse_abstracts(records)

def parse_abstracts(xml_file):
	docs = []
	for doc in xml_file['PubmedArticle']:
		row = {}

		pmid = str(doc['MedlineCitation']['PMID'])
		row['pmid'] = pmid

		try:
			ab_strs = [s for s in doc['MedlineCitation']['Article']['Abstract']['AbstractText']]
			ab_text = '\n\n'.join(ab_strs)
			row['abst'] = ab_text

			ab_xml = '<article><front><article-meta><title-group><article-title></article-title></title-group><article-id>'+pmid+'</article-id>'
			ab_xml += '<abstract>'
			for s in ab_strs:
				ab_xml += '<sec>'
				if 'Label' in s.attributes:
					ab_xml += '<title>'+s.attributes['Label']+'</title>'
				ab_xml += '<p>'+sanitize(str(s))+'</p>'
				ab_xml += '</sec>'
			ab_xml += '</abstract></article-meta></front><body></body></article>'
			row['xml'] = ab_xml
		except KeyError:
			print('Unable to parse abst for {}'.format(pmid))
			row['abst'] = '<ERR>'
			row['xml'] = '<ERR>'

		try:
			mesh = doc['MedlineCitation']['MeshHeadingList']
			mesh_terms = [str(m['DescriptorName']) for m in mesh]
			row['mesh'] = mesh_terms
		except KeyError:
			print('Unable to parse MeSH for {}'.format(pmid))
			row['mesh'] = []

		title = doc['MedlineCitation']['Article']['ArticleTitle']
		row['title'] = title

		docs.append(row)

	df = pd.DataFrame(docs)
	df.to_csv(open(output_fname, 'w'), index = False)
	return df

def load_druglist(list_name):
	fname = '../data/cures_within_reach/druglist_{}.txt'.format(list_name)
	return [l.strip() for l in open(fname).readlines()]

def doc_field_matches(docs, field, match_list):
	matches = []
	for d in docs:
		if match_list.intersection(d[field]):
			matches.append(d)
	return matches

def extract_ts_docs(docs_match_fn):
	shard_jsons = sorted(glob.glob('/media/data/ben/trialstreamer/*/pipeline_data.json'))
	all_matches = []
	for f in shard_jsons:
		docs = json.load(open(f))
		shard_matches = docs_match_fn(docs)
		print('{}/{} matches in {}'.format(len(shard_matches), len(docs), f))
		all_matches += shard_matches
	return all_matches

ti_to_pmid = None
def get_pmid(ti):
	global ti_to_pmid
	if not ti_to_pmid:
		ti_pmid_df = pd.read_csv('../data/trialstreamer/pmid_ti_idx.csv')
		ti_to_pmid = dict(zip(ti_pmid_df.ti.values, ti_pmid_df.pmid.values))
	return ti_to_pmid.get(ti, 'UNKNOWN')

def get_ts_df(docs):
	rows = []
	for d in docs:
		r = {}
		r['gid'] = d['pmid']
		r['pmid'] = get_pmid(d['title'])
		r['text'] = d['text']
		r['p'] = format_strs([d['text'][i:f] for i,f in d['p_spans']])
		r['i'] = format_strs([d['text'][i:f] for i,f in d['i_spans']])
		r['o'] = format_strs([d['text'][i:f] for i,f in d['o_spans']])
		r['p_terms'] = format_strs(d['p_mesh'])
		r['i_terms'] = format_strs(d['i_mesh'])
		r['o_terms'] = format_strs(d['o_mesh'])
		r['frames'] = format_frames(d['frames'], 'l')
		rows.append(r)
	return pd.DataFrame(rows)

def run_ts_query():
	def read_lines(fname):
		return set(open(fname).read().strip().split('\n'))
	
	p_uid_roots = ['D009370', 'D009371', 'D009376', 'D009381', 'D016609']
	p_terms = get_uid_terms(p_uid_roots, include_children = True)
	p_terms.append({'name': 'Cancer', 'uid': 'D009369'})
	p_uids = set([m['uid'] for m in p_terms])

	#i_strs = read_lines('../data/cwr/drug_list_oct_extended.txt')
	i_strs = read_lines('../data/cwr/drug_list_nov_unmapped.txt')
	i_terms = minimap.get_unique_terms(list(i_strs))
	i_uids = set([m['mesh_ui'] for m in i_terms])

	o_strs = read_lines('../data/cwr/master_o_pruned.csv')
	o_terms = minimap.get_unique_terms(list(o_strs))
	o_uids = set([m['mesh_ui'] for m in o_terms])
	
	print('Final query UID counts:')
	print('\tP: {}'.format(len(p_uids)))
	print('\tI: {}'.format(len(i_uids)))
	print('\tO: {}'.format(len(o_uids)))

	# Potential source of errors:
	#   get_uid_terms(...) returns the official MeSH Heading string as the name
	#   minimap.get_unique_terms(...) uses a mapping from UMLS, and not always the preffered MeSH term
	# for example, minimap.get_unique_terms(['Breast Cancer']) returns
	#   ('Breast Cancer', 'D001943'), but the MeSH heading is
	#   ('Breast Neoplasms', 'D001943')
	uid_to_name = {}
	for m in p_terms: uid_to_name[m['uid']] = m['name']
	for m in i_terms: uid_to_name[m['mesh_ui']] = m['mesh_term']
	for m in o_terms: uid_to_name[m['mesh_ui']] = m['mesh_term']
	
	print('Matching P terms')
	p_match_fn = partial(doc_field_matches, field= 'p_duis', match_list = p_uids)
	p_matches = extract_ts_docs(p_match_fn)
	print('\t matched {} P terms'.format(len(p_matches)))
	print('Matching I terms')
	ip_matches = [d for d in p_matches if i_uids.intersection(d['i_duis'])]
	print('\t matched {} I terms'.format(len(ip_matches)))

	# Skipping the string matching - no new articles found w/ the list of 50 unmapped strings
	"""
	def str_in_doc(s, doc, field = 'text'):
		if field == 'text':
			texts = [doc[field].lower()]
		elif 'spans' in field:
			texts = [doc['text'][i:f].lower() for i,f in doc[field]]
		else:
			raise NotImplementedError
		return any([s.lower() in t for t in texts])
	i_strs_extra = read_lines('../data/cwr/drug_list_oct_extended_missing_uids.txt')
	ip_matches_pmids = set([d['pmid'] for d in ip_matches])
	i_strs_extra_matches = [d for d in p_matches if any([str_in_doc(s, d, 'i_spans') for s in i_strs_extra])]
	i_strs_extra_matches_new = [d for d in i_strs_extra_matches if d['pmid'] not in ip_matches_pmids]
	print('Found {} docs matches {} terms ({} new)'.format( \
			len(i_strs_extra_matches), len(i_strs_extra), len(i_strs_extra_matches_new)))
	ip_matches += i_strs_extra_matches_new
	"""

	return ip_matches, p_uids, i_uids, o_uids, uid_to_name

def generate_ts_query_df(docs, p_uids, i_uids, o_uids, uid_to_name):
	rows = []
	for doc_idx, d in enumerate(docs):
		print(doc_idx)
		r = {}
		r['text'] = d['text']
		r['pmid'] = get_pmid(d['title'])
		r['p'] = format_strs([d['text'][i:f] for i,f in d['p_spans']])
		r['i'] = format_strs([d['text'][i:f] for i,f in d['i_spans']])
		r['o'] = format_strs([d['text'][i:f] for i,f in d['o_spans']])
		r['p_terms'] = format_strs(d['p_mesh'])
		r['i_terms'] = format_strs(d['i_mesh'])
		r['o_terms'] = format_strs(d['o_mesh'])
		r['frames'] = format_frames(d['frames'], 'l')
		r['p_terms_whichquerymatched'] = format_strs([uid_to_name[ui] for ui in p_uids.intersection(d['p_duis'])])
		r['i_terms_whichquerymatched'] = format_strs([uid_to_name[ui] for ui in i_uids.intersection(d['i_duis'])])
		r['o_terms_whichquerymatched'] = format_strs([uid_to_name[ui] for ui in o_uids.intersection(d['o_duis'])])
		i_frame_idx = set()
		c_frame_idx = set()
		o_frame_idx = set()
		for idx, f in enumerate(d['frames']):
			if i_uids.intersection(get_mesh_terms(f['i'], 'mesh_ui')): i_frame_idx.add(idx)
			if i_uids.intersection(get_mesh_terms(f['c'], 'mesh_ui')): c_frame_idx.add(idx)
			if o_uids.intersection(get_mesh_terms(f['o'], 'mesh_ui')): o_frame_idx.add(idx)
		r['i_frame_whichindexmatches'] = '\n'.join(map(str, sorted(list(i_frame_idx))))
		r['c_frame_whichindexmatches'] = '\n'.join(map(str, sorted(list(c_frame_idx))))
		r['o_frame_whichindexmatches'] = '\n'.join(map(str, sorted(list(o_frame_idx))))

		rows.append(r)
	return pd.DataFrame(rows)
