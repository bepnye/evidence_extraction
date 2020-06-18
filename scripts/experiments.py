import processing as p
import run_pipeline as run
import numpy as np

import process_coref_annes as pca

top = '/home/ben/Desktop/evidence_extraction/data/tmp/'
def run_coref(group = 'devtest', force_phase1 = False):
	docs = pca.read_docs(True, group)
	for d in docs:
		d.group = 'test'
		d.parse_text()
		_, _ = p.extract_gold_info(d)

	if force_phase1: 
		for d in docs:
			for e in d.entities:
				for m in e.mentions:
					d.labels['NER_{}'.format(e.type)].append(m)
			evs = set()
			for f in d.frames:
				if f.ev.text not in evs:
					evs.add(f.ev.text)
					d.labels['BERT_ev'].append(f.ev)
	else:
		run.dump_phase1_ner(top, docs)
		run.exec_phase1_ner(top, 'ebm_i')
		run.load_phase1_ner(top, docs)
		run.exec_phase1_ner(top, 'ebm_o')
		run.load_phase1_ner(top, docs)

		run.dump_phase1_ev(top, docs)
		run.exec_phase1_ev(top)
		run.load_phase1_ev(top, docs)
	
	run.dump_phase2_o_ev(top, docs)
	run.exec_phase2_o_ev(top)
	run.load_phase2_o_ev(top, docs)

	run.dump_phase2_ic_ev(top, docs)
	run.exec_phase2_ic_ev(top)
	run.load_phase2_ic_ev(top, docs)

	return docs

def run_train(docs):
	for d in docs:
		d.group = 'test'
		d.parse_text()
	run.dump_phase1_ner(top, docs)
	run.exec_phase1_ner(top, 'ebm_i')
	run.load_phase1_ner(top, docs)
	run.exec_phase1_ner(top, 'ebm_o')
	run.load_phase1_ner(top, docs)
	return docs

def check_bert_ev(docs):
	for d_idx, d in enumerate(docs):
		for e_idx, ev in enumerate(d.labels['BERT_ev']):
			try:
				pred_i = ev.pred_i
			except AttributeError:
				print(d_idx, e_idx)
def check_frame_ev(docs):
	for d_idx, d in enumerate(docs):
		for e_idx, f in enumerate(d.frames):
			try:
				pred_i = f.ev.pred_i
			except AttributeError:
				print(d_idx, e_idx)

def eval_coref():
	workers = ['edin', 'lidija', 'daniela']
	worker_docs = {}
	for worker in workers:
		docs = pca.read_docs('../data/ann_coref_repo/{}_*'.format(worker), check_errors = False)
		for d in docs:
			p.extract_gold_info(d)
		worker_docs[worker] = { d.id: d for d in docs }

	for w1 in workers:
		for w2 in workers:
			shared_ids = set(worker_docs[w1].keys()).intersection(worker_docs[w2].keys())
			scores = []
			for i in shared_ids:
				es1 = worker_docs[w1][i].entities
				es2 = worker_docs[w2][i].entities
				scores.append(p.get_coref_scores(es1, es2, avg_metrics = False))
			print(w1, w2, len(shared_ids))
			for m in zip(*scores):
				print('\t{:.2f}'.format(np.mean(m)))
