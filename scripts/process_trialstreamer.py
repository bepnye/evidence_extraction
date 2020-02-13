import glob, os, json, re

import pandas as pd
import numpy as np

import processing
import writer
import minimap

DATA_DIR='/home/ben/Desktop/evidence_extraction/data/trialstreamer/'

def process_cwr_data():
	df = pd.read_csv('../data/cures_within_reach/cwr.csv')
	#df = df[~df.Relevant.apply(np.isnan)]
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
		doc = processing.classes.Doc(r.PMID, text)
		doc.replace_acronyms()
		doc.group = 'test'
		with open('../data/cures_within_reach/{}.text'.format(r.PMID), 'w') as fp:
			fp.write(doc.text)
		with open('../data/cures_within_reach/{}.title'.format(r.PMID), 'w') as fp:
			fp.write(r.Title)
		#i_target = minimap.minimap(r['Matched.Intervention..Word.Embeddings.'])
		#o_target = minimap.minimap(r['Matched.Outcome..Word.Embeddings.'])
		#doc.target = (i_target, o_target, r.Relevant)
		docs[r.PMID] = doc
	return list(docs.values())

def read_shard_docs(shard, data_dir = DATA_DIR):
	print('\t\tcreating Docs for {}'.format(shard))
	fnames = glob.glob('{}/{}/*.text'.format(data_dir, shard))
	docs = [processing.classes.Doc(os.path.basename(f).strip('.text'), open(f).read()) for f in fnames]
	docs = [d for d in docs if d.text]
	for d in docs:
		d.parse_text()
		d.group = 'test'
		d.sf_lf_map = {} # already acronym'd
	return docs

def write_phase1_inputs(s, docs, data_dir = DATA_DIR):
	top = '{}/{}'.format(data_dir, s)
	if not os.path.isfile('{}/ner/test.json'.format(top)):
		print('\t\twriting ner inputs...')
		docs = docs or read_shard_docs(s)
		os.system('mkdir -p {}/ner'.format(top))
		os.system('mkdir -p {}/ner/results'.format(top))
		writer.write_ner_data(docs, writer.dummy_label, '{}/ner/'.format(top), allow_acronyms = True)
	if not os.path.isfile('{}/ev/test.tsv'.format(top)):
		print('\t\twriting ev inputs...')
		docs = docs or read_shard_docs(s)
		os.system('mkdir -p {}/ev'.format(top))
		os.system('mkdir -p {}/ev/results'.format(top))
		writer.write_sent_data_pipeline(docs, '{}/ev/'.format(top))

def load_phase1_outputs(s, docs, data_dir = DATA_DIR):
	docs = docs or read_shard_docs(s)
	top = '{}/{}'.format(data_dir, s)
	if os.path.isfile('{}/ner/results/pred.txt'.format(top)) and \
		 os.path.isfile('{}/ev/results/test_results.tsv'.format(top)):
		print('\t\tloading ner and ev outputs...')
		processing.add_ner_output(docs, '{}/{}/ner/results/pred.txt'.format(data_dir, s), False)
		processing.add_ev_sent_output(docs, 'test', '{}/{}/ev/'.format(data_dir, s))
		return True
	else:
		return False

def write_phase2_inputs(s, docs, data_dir = DATA_DIR):
	top = '{}/{}'.format(data_dir, s)
	if not os.path.isfile('{}/o_ev/test.tsv'.format(top)):
		print('\t\twriting o_ev inputs...')
		os.makedirs('{}/o_ev'.format(top), exist_ok = True)
		os.makedirs('{}/o_ev/results'.format(top), exist_ok = True)
		writer.write_o_ev_data_pipeline(docs, '{}/o_ev/'.format(top))
	if not os.path.isfile('{}/ic_ev/test.tsv'.format(top)):
		print('\t\twriting ic_ev inputs...')
		os.makedirs('{}/ic_ev'.format(top), exist_ok = True)
		os.makedirs('{}/ic_ev/results'.format(top), exist_ok = True)
		writer.write_i_c_data_pipeline(docs, writer.intro_group_ev, '{}/ic_ev'.format(top))

def load_phase2_outputs(s, docs, data_dir = DATA_DIR):
	top = '{}/{}'.format(data_dir, s)
	if os.path.isfile('{}/o_ev/results/test_results.tsv'.format(top)) and \
		 os.path.isfile('{}/ic_ev/results/test_results.tsv'.format(top)):
		print('\t\tloading o_ev and ic_ev outputs...')
		processing.add_o_ev_output(docs, 'test', '{}/o_ev/'.format(top))
		processing.add_ic_ev_output(docs, 'test', '{}/ic_ev/'.format(top))
		return True
	else:
		return False

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
					'i_mesh': [m['mesh_term'] for m in minimap.get_unique_terms([ev.pred_i.text])],
					'c_mesh': [m['mesh_term'] for m in minimap.get_unique_terms([ev.pred_c.text])],
					'o_mesh': [m['mesh_term'] for m in minimap.get_unique_terms([pred_o.text])],
					'ev': ev.text })
	return icos

def FLATTEN_MESH(m):
	if m in ['Progression-Free Survival', 'Survival Rate', 'Event-Free Survival']:
		return 'Survival'
	return m

def write_trialstreamer_inputs(s, docs, data_dir = DATA_DIR):
	fname = '{}/{}/{}_pipeline_data.json'.format(data_dir, s, s)
	if not os.path.isfile(fname):
		print('\t\tconstructing JSON')
		rows = []
		for d in docs:
			r = {}
			r['text'] = d.text
			r['pmid'] = d.id
			r['title'] = open('{}/{}/{}.title'.format(data_dir, s, d.id)).read()
			r['p_spans'] = [(s.i, s.f) for s in d.labels['NER_p']]
			r['i_spans'] = [(s.i, s.f) for s in d.labels['NER_i']]
			r['o_spans'] = [(s.i, s.f) for s in d.labels['NER_o']]
			r['p_mesh'] = [m['mesh_term'] for m in minimap.get_unique_terms([s.text for s in d.labels['NER_p']])]
			r['i_mesh'] = [m['mesh_term'] for m in minimap.get_unique_terms([s.text for s in d.labels['NER_i']])]
			r['o_mesh'] = [m['mesh_term'] for m in minimap.get_unique_terms([s.text for s in d.labels['NER_o']])]
			r['frames'] = get_icos(d)
			rows.append(r)
		json.dump(rows, open(fname, 'w'))

def print_ico(ico):
	print(ico['ev'].replace('\n', '\\n'))
	for e in 'ico':
		print('\t{}: [{}] {}'.format(e, ico[e], '|'.join(ico[e+'_mesh'])))

def score_cwr_data(docs, verbose = False):
	pred = []
	true = []
	for d in docs:
		targ_i, targ_o, relevant = d.target
		true.append(relevant)
		if not targ_i or not targ_o:
			print('Unable to find mesh terms for docid={}'.format(d.id))
			pred.append(0)
			continue
		assert len(targ_i) == 1
		assert len(targ_o) == 1
		targ_i = targ_i[0]['mesh_term']
		targ_o = targ_o[0]['mesh_term']
		found = False
		icos = get_icos(d)
		for ico in icos:
			t_mesh = ico['i_mesh']+ico['c_mesh']
			o_mesh = [FLATTEN_MESH(m) for m in ico['o_mesh']]
			if targ_i in t_mesh and targ_o in o_mesh:
				found = True
				break
		pred.append(found)
		if verbose and found != relevant:
			print('True: {}, Pred: {}   {} | {}'.format(relevant, int(found), targ_i, targ_o))
			for ico in icos: print_ico(ico)
			input()

	print(processing.precision_recall_fscore_support(true, pred, labels=[1]))
	return true, pred

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
			d = processing.classes.Doc(idx, r.ab)
			d.replace_acronyms()
			open('../data/trialstreamer/{}_{}/{}.text'.format(i,f,idx), 'w').write(d.text)
			open('../data/trialstreamer/{}_{}/{}.title'.format(i,f,idx), 'w').write(r.ti)

def process_all():
	shards = sorted([d.split('/')[-1] for d in glob.glob('../data/trialstreamer/*')])
	for s in shards:
		process_shard(s)

def process_shard(s):
	print('Processing {}'.format(s))
	docs = read_shard_docs(s)
	print('\tp1 inputs...');         write_phase1_inputs(s, docs)
	print('\tp1 outputs..'); ready = load_phase1_outputs(s, docs)
	if not ready:
		print('\twaiting for phase1 outputs...')
		return
	print('\tp2 inputs...');         write_phase2_inputs(s, docs)
	print('\tp2 outputs..'); ready = load_phase2_outputs(s, docs)
	if not ready:
		print('\twaiting for phase2 outputs...')
		return
	print('\tfinal inputs...'); write_trialstreamer_inputs(s, docs)

