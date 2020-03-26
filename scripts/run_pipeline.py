import sys, os, json
from imp import reload

import classes
import writer
import processing
import process_trialstreamer

"""

Everything you need to know about running the pipeline!

Big pieces:
	classes.Doc - contains the text and extracted info
	process_*   - code for reading/handling the * dataset
	writer      - generates model inputs from the Docs
	processing  - parses model output and augments the Docs

All of the data tracking/handling happens in the Doc.labels dict. The key bits are:
	Doc.labels['NER_{p|i|o}']: a list of NER spans for each of the labeled PIO elements
	Doc.labels['BERT_ev']: a list of sentence spans for each tagged evidence sents
Phase 1 handles parsing the text and populating these fields
Phase 2 then uses the extracted info to additionally produce:
	Doc.labels['NER_o'][n].label: the efficacy label for each Outcome contained in an evidence span
	Doc.labels['BERT_ev'][n].pred_os: a list of which spans from Doc.labels['NER_o'] are present
	Doc.labels['BERT_ev'][n].pred_{i|c}: the top-ranked Doc.labels['NER_i'] span for the I and C slots

For an example of what to do with this dict once you've finished generating it, see:
	process_trialstreamer.generate_trialstreamer_inputs

"""

"""
BERT / Tensorflow stuff
"""
bert_model_top = '../models/sentence_classifier/'
ner_model_top = '../models/ner_tagger/'
sys.path.append(bert_model_top)
sys.path.append(ner_model_top)
# Absolute path to your BERT checkpoint
BIO_BERT_DIR = '/home/ben/Desktop/biobert_pubmed/'
# We need the smaller scivcoab uncased to get the whole abst to fit in one sequence (512)
SCI_BERT_DIR = '/home/ben/Desktop/scibert_scivocab_uncased/'
DO_LOWER_CASE = True
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
def clear_flags():
	import tensorflow as tf
	f = tf.flags.FLAGS
	f.remove_flag_values(f.flag_values_dict())

"""
Phase 1
	* PIO extraction (per-token sequence labeling)
	* Evidence identification (sentence labeling)
"""

def write_phase1_ner(top, docs):
<<<<<<< HEAD
	print('\t\twriting ner inputs...')
	os.system('mkdir -p {}/ner'.format(top))
	os.system('mkdir -p {}/ner/results'.format(top))
	writer.write_ner_data(docs, writer.dummy_label, '{}/ner/'.format(top), allow_acronyms = True)

def write_phase1_ev(top, docs):
	print('\t\twriting ev inputs...')
	os.system('mkdir -p {}/ev'.format(top))
	os.system('mkdir -p {}/ev/results'.format(top))
	writer.write_sent_data_pipeline(docs, '{}/ev/'.format(top))
=======
	if not os.path.isfile('{}/ner/test.json'.format(top)):
		print('\t\twriting ner inputs...')
		os.system('mkdir -p {}/ner'.format(top))
		os.system('mkdir -p {}/ner/results'.format(top))
		writer.write_ner_data(docs, writer.dummy_label, '{}/ner/'.format(top), allow_acronyms = True)

def write_phase1_ev(top, docs):
	if not os.path.isfile('{}/ev/test.tsv'.format(top)):
		print('\t\twriting ev inputs...')
		os.system('mkdir -p {}/ev'.format(top))
		os.system('mkdir -p {}/ev/results'.format(top))
		writer.write_sent_data_pipeline(docs, '{}/ev/'.format(top))
>>>>>>> 999dac558a813668da7ed7133f7dd0cb1b2cde0f

def run_phase1_ner(top):
	clear_flags(); reload(ner_model)
	ner_model.FLAGS.do_train   = False
	ner_model.FLAGS.do_predict = True
	ner_model.FLAGS.data_dir         = '{}/ner/'.format(top)
	ner_model.FLAGS.output_dir       = '{}/ner/results/'.format(top)
	ner_model.FLAGS.data_config_path = '{}/ner/results/data.conf'.format(top)
	ner_model.FLAGS.model_dir        = '{}/data/ebm_nlp_ab3p/model/'.format(ner_model_top)
	ner_model.FLAGS.do_lower_case    = DO_LOWER_CASE
	ner_model.FLAGS.label_idx = 1 # target label column idx
	ner_model.FLAGS.vocab_file       = '{}/vocab.txt'.format(BIO_BERT_DIR)
	ner_model.FLAGS.bert_config_file = '{}/bert_config.json'.format(BIO_BERT_DIR)
	ner_model.FLAGS.init_checkpoint  = '{}'.format(BIO_BERT_DIR)
	ner_model.main('')

def run_phase1_ev(top):
	import run_classifier as model
	clear_flags(); reload(model)
	model.FLAGS.task_name  = 'ico'
	model.FLAGS.do_train   = False
	model.FLAGS.do_predict = True
	model.FLAGS.data_dir         = '{}/ev/'.format(top)
	model.FLAGS.output_dir       = '{}/ev/results/'.format(top)
	model.FLAGS.model_dir        = '{}/data/ev_sent/model/'.format(bert_model_top)
	model.FLAGS.vocab_file       = '{}/vocab.txt'.format(BIO_BERT_DIR)
	model.FLAGS.bert_config_file = '{}/bert_config.json'.format(BIO_BERT_DIR)
	model.FLAGS.init_checkpoint  = '{}'.format(BIO_BERT_DIR)
	model.main('')

def load_phase1_ner(top, docs):
	if os.path.isfile('{}/ner/results/pred.txt'.format(top)):
		print('\t\tloading ner outputs...')
		processing.add_ner_output(docs, '{}/ner/results/pred.txt'.format(top), False)

def load_phase1_ev(top, docs):
	if os.path.isfile('{}/ev/results/test_results.tsv'.format(top)):
		print('\t\tloading ev outputs...')
		processing.add_ev_sent_output(docs, 'test', '{}/ev/'.format(top))

"""
Phase 2
	* Outcome efficacy (label each [o, ev_sent] pair as +/-/=)
	* I/C slot filling (for each ev_sent, rank all extracted I as I/C)
"""

def write_phase2_o_ev(top, docs):
	print('\t\twriting o_ev inputs...')
	os.makedirs('{}/o_ev'.format(top), exist_ok = True)
	os.makedirs('{}/o_ev/results'.format(top), exist_ok = True)
	writer.write_o_ev_data_pipeline(docs, '{}/o_ev/'.format(top))

def write_phase2_ic_ev(top, docs):
	print('\t\twriting ic_ev inputs...')
	os.makedirs('{}/ic_ev'.format(top), exist_ok = True)
	os.makedirs('{}/ic_ev/results'.format(top), exist_ok = True)
	writer.write_i_c_data_pipeline(docs, writer.ev_abst, '{}/ic_ev'.format(top))

def run_phase2(top):
	run_phase2_o_ev(top)
	run_phase2_ic_ev(top)

def run_phase2_o_ev(top):
	import run_classifier as model
	clear_flags(); reload(model)
	model.FLAGS.do_train   = False
	model.FLAGS.do_predict = True
	model.FLAGS.task_name  = 'ico_ab'
	model.FLAGS.data_dir         = '{}/o_ev/'.format(top)
	model.FLAGS.output_dir       = '{}/o_ev/results/'.format(top)
	model.FLAGS.model_dir        = '{}/data/o_ev_sent/model/'.format(bert_model_top)
	model.FLAGS.vocab_file       = '{}/vocab.txt'.format(BIO_BERT_DIR)
	model.FLAGS.bert_config_file = '{}/bert_config.json'.format(BIO_BERT_DIR)
	model.FLAGS.init_checkpoint  = '{}'.format(BIO_BERT_DIR)
	model.FLAGS.do_lower_case    = DO_LOWER_CASE
	model.main('')

def run_phase2_ic_ev(top):
	import run_classifier as model
	clear_flags(); reload(model)
	# NOTE: important! We want to fit the whole abstact in memory
	model.FLAGS.max_seq_length = 512
	model.FLAGS.do_train   = False
	model.FLAGS.do_predict = True
	model.FLAGS.task_name  = 'ico_ab'
	model.FLAGS.data_dir         = '{}/ic_ev/'.format(top)
	model.FLAGS.output_dir       = '{}/ic_ev/results/'.format(top)
	model.FLAGS.model_dir        = '{}/data/i_c_ev/model/'.format(bert_model_top)
	model.FLAGS.vocab_file       = '{}/vocab.txt'.format(SCI_BERT_DIR)
	model.FLAGS.bert_config_file = '{}/bert_config.json'.format(SCI_BERT_DIR)
	model.FLAGS.init_checkpoint  = '{}'.format(SCI_BERT_DIR)
	model.FLAGS.do_lower_case    = DO_LOWER_CASE
	model.main('')

def load_phase2_o_ev(top, docs):
	if os.path.isfile('{}/o_ev/results/test_results.tsv'.format(top)):
		print('\t\tloading o_ev outputs...')
		processing.add_o_ev_output(docs, 'test', '{}/o_ev/'.format(top))

def load_phase2_ic_ev(top, docs):
	if os.path.isfile('{}/ic_ev/results/test_results.tsv'.format(top)):
		print('\t\tloading ic_ev outputs...')
		processing.add_ic_ev_output(docs, 'test', '{}/ic_ev/'.format(top))

def run_example():
	data_fname = 'sample_data.json'
	data = json.load(open(data_fname))
	docs = process_trialstreamer.process_generic_data(data)
	# NOTE: we need an absolute path here since running models will chdir
	top = os.path.join(os.getcwd(), '..', 'data', 'example')

	# phase 1! only running the ev model and not ner since we assume we have those labels
	write_phase1_ev(top, docs)
	run_phase1_ev(top)
	load_phase1_ev(top, docs)

	# phase 2!
	write_phase2_o_ev(top, docs)
	run_phase2_o_ev(top)
	load_phase2_o_ev(top, docs)

	write_phase2_ic_ev(top, docs)
	run_phase2_ic_ev(top)
	load_phase2_ic_ev(top, docs)

	# easy peasy
	results = process_trialstreamer.generate_trialstreamer_inputs(docs)
	return results

if __name__ == '__main__':
	run_example()
