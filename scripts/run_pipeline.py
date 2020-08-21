import sys, os, json
from imp import reload
import tensorflow as tf
from shutil import copyfile

sys.path.append('..')
import config

import utils
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

NER_OUTPUT = 'pred.txt'

bert_model_top = '../models/sentence_classifier/'
ner_model_top = '../models/ner_tagger/'
sys.path.append(bert_model_top)
sys.path.append(ner_model_top)

DO_LOWER_CASE = True
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
def clear_flags():
	f = tf.flags.FLAGS
	f.remove_flag_values(f.flag_values_dict())

"""
Phase 1
	* PIO extraction (per-token sequence labeling)
	* Evidence identification (sentence labeling)
"""

def dump_phase1_ner(top, docs):
	ner_fname = '{}/ner/'.format(top)
	print('\t\twriting ner inputs to {}...'.format(ner_fname))
	os.system('mkdir -p {}/ner'.format(top))
	os.system('mkdir -p {}/ner/results'.format(top))
	writer.write_ner_data(docs, writer.dummy_label, ner_fname, allow_acronyms = True)

def dump_phase1_ev(top, docs):
	print('\t\twriting ev inputs...')
	os.system('mkdir -p {}/ev'.format(top))
	os.system('mkdir -p {}/ev/results'.format(top))
	writer.write_sent_data_pipeline(docs, '{}/ev/'.format(top))

def exec_phase1_ner(top, cp = 'ebm_nlp_ab3p'):
	clear_flags()
	import bert_lstm_ner as ner_model
	clear_flags(); reload(ner_model)
	ner_model.FLAGS.do_train   = False
	ner_model.FLAGS.do_predict = True
	ner_model.FLAGS.data_dir         = '{}/ner/'.format(top)
	ner_model.FLAGS.output_dir       = '{}/ner/results/'.format(top)
	ner_model.FLAGS.data_config_path = '{}/ner/results/data.conf'.format(top)
	ner_model.FLAGS.model_dir        = '{}/data/{}/model/'.format(ner_model_top, cp)
	ner_model.FLAGS.do_lower_case    = DO_LOWER_CASE
	ner_model.FLAGS.label_idx = 1 # target label column idx
	ner_model.FLAGS.vocab_file       = '{}/vocab.txt'.format(config.BIO_BERT_DIR)
	ner_model.FLAGS.bert_config_file = '{}/bert_config.json'.format(config.BIO_BERT_DIR)
	ner_model.FLAGS.init_checkpoint  = '{}'.format(config.BIO_BERT_DIR)
	ner_model.main('')
	# TODO - push output fname to FLAGS so you don't have to know this
	src_fname = os.path.join(ner_model.FLAGS.output_dir, NER_OUTPUT)
	bak_fname = os.path.join(ner_model.FLAGS.output_dir, '{}.{}'.format(cp, NER_OUTPUT))
	copyfile(src_fname, bak_fname)
	clear_flags()

def exec_phase1_ev(top, cp = 'ev_sent'):
	clear_flags()
	import run_classifier as model;
	clear_flags(); reload(model)
	model.FLAGS.task_name  = 'ico'
	model.FLAGS.do_train   = False
	model.FLAGS.do_predict = True
	model.FLAGS.data_dir         = '{}/ev/'.format(top)
	model.FLAGS.output_dir       = '{}/ev/results/'.format(top)
	model.FLAGS.model_dir        = '{}/data/{}/model/'.format(bert_model_top, cp)
	model.FLAGS.vocab_file       = '{}/vocab.txt'.format(config.BIO_BERT_DIR)
	model.FLAGS.bert_config_file = '{}/bert_config.json'.format(config.BIO_BERT_DIR)
	model.FLAGS.init_checkpoint  = '{}'.format(config.BIO_BERT_DIR)
	model.main('')
	# TODO - push output fname to FLAGS so you don't have to know this
	src_fname = os.path.join(model.FLAGS.output_dir, 'test_results.tsv')
	bak_fname = os.path.join(model.FLAGS.output_dir, '{}.test_results.tsv'.format(cp))
	copyfile(src_fname, bak_fname)

def load_phase1_ner(top, docs, cp = ''):
	fname = NER_OUTPUT
	if cp:
		fname = '{}.{}'.format(cp, NER_OUTPUT)
	fpath = '{}/ner/results/{}'.format(top, fname)
	if os.path.isfile(fpath):
		print('\t\tloading ner outputs...')
		processing.add_ner_output(docs, fpath, False)

def load_phase1_ev(top, docs, label_fn = utils.argmax):
	if os.path.isfile('{}/ev/results/test_results.tsv'.format(top)):
		print('\t\tloading ev outputs...')
		processing.add_ev_sent_output(docs, 'test', '{}/ev/'.format(top), label_fn)

"""
Phase 2
	* Outcome efficacy (label each [o, ev_sent] pair as +/-/=)
	* I/C slot filling (for each ev_sent, rank all extracted I as I/C)
"""

def dump_phase2_o_ev(top, docs):
	print('\t\twriting o_ev inputs...')
	os.makedirs('{}/o_frame'.format(top), exist_ok = True)
	os.makedirs('{}/o_frame/results'.format(top), exist_ok = True)
	writer.write_o_ev_data_pipeline(docs, '{}/o_frame/'.format(top))

def dump_phase2_ic_ev(top, docs):
	print('\t\twriting ic_ev inputs...')
	os.makedirs('{}/ic_frame'.format(top), exist_ok = True)
	os.makedirs('{}/ic_frame/results'.format(top), exist_ok = True)
	writer.write_i_c_data_pipeline(docs, writer.ev_abst, '{}/ic_frame'.format(top))

def exec_phase2(top):
	exec_phase2_o_ev(top)
	exec_phase2_ic_ev(top)

def exec_phase2_o_ev(top, cp = 'o_ev_sent'):
	clear_flags()
	import run_classifier as model
	clear_flags(); reload(model)
	model.FLAGS.do_train   = False
	model.FLAGS.do_predict = True
	model.FLAGS.task_name  = 'ico_ab'
	model.FLAGS.data_dir         = '{}/o_frame/'.format(top)
	model.FLAGS.output_dir       = '{}/o_frame/results/'.format(top)
	model.FLAGS.model_dir        = '{}/data/{}/model/'.format(bert_model_top, cp)
	model.FLAGS.vocab_file       = '{}/vocab.txt'.format(config.BIO_BERT_DIR)
	model.FLAGS.bert_config_file = '{}/bert_config.json'.format(config.BIO_BERT_DIR)
	model.FLAGS.init_checkpoint  = '{}'.format(config.BIO_BERT_DIR)
	model.FLAGS.do_lower_case    = DO_LOWER_CASE
	model.main('')

def exec_phase2_ic_ev(top, cp = 'i_c_abst'):
	clear_flags()
	import run_classifier as model
	clear_flags(); reload(model)
	# NOTE: important! We want to fit the whole abstact in memory
	model.FLAGS.max_seq_length = 512
	model.FLAGS.do_train   = False
	model.FLAGS.do_predict = True
	model.FLAGS.task_name  = 'ico_ab'
	model.FLAGS.data_dir         = '{}/ic_frame/'.format(top)
	model.FLAGS.output_dir       = '{}/ic_frame/results/'.format(top)
	model.FLAGS.model_dir        = '{}/data/{}/model/'.format(bert_model_top, cp)
	model.FLAGS.vocab_file       = '{}/vocab.txt'.format(config.SCI_BERT_DIR)
	model.FLAGS.bert_config_file = '{}/bert_config.json'.format(config.SCI_BERT_DIR)
	model.FLAGS.init_checkpoint  = '{}'.format(config.SCI_BERT_DIR)
	model.FLAGS.do_lower_case    = DO_LOWER_CASE
	model.main('')

def load_phase2_o_ev(top, docs):
	if os.path.isfile('{}/o_frame/results/test_results.tsv'.format(top)):
		print('\t\tloading o_frame outputs...')
		processing.add_o_ev_output(docs, 'test', '{}/o_frame/'.format(top))

def load_phase2_ic_ev(top, docs):
	if os.path.isfile('{}/ic_frame/results/test_results.tsv'.format(top)):
		print('\t\tloading ic_frame outputs...')
		processing.add_ic_ev_output(docs, 'test', '{}/ic_frame/'.format(top))

def run_example():
	data_fname = 'sample_data.json'
	data = json.load(open(data_fname))
	docs = process_trialstreamer.process_generic_data(data)
	# NOTE: we need an absolute path here since running models will chdir
	top = os.path.join(os.getcwd(), '..', 'data', 'example')

	# phase 1! only running the ev model and not ner since we assume we have those labels
	dump_phase1_ev(top, docs)
	exec_phase1_ev(top)
	load_phase1_ev(top, docs)

	# phase 2!
	dump_phase2_o_ev(top, docs)
	exec_phase2_o_ev(top)
	load_phase2_o_ev(top, docs)

	dump_phase2_ic_ev(top, docs)
	exec_phase2_ic_ev(top)
	load_phase2_ic_ev(top, docs)

	# easy peasy
	results = process_trialstreamer.generate_trialstreamer_inputs(docs)
	return results

def load_trialstreamer_shard(top, docs = None):
	docs = docs or process_trialstreamer.read_shard_docs(top)
	load_phase1_ner(top, docs, 'ebm_p')
	load_phase1_ner(top, docs, 'ebm_i')
	load_phase1_ner(top, docs, 'ebm_o')
	load_phase1_ev(top, docs)
	load_phase2_o_ev(top, docs)
	load_phase2_ic_ev(top, docs)
	return docs

def run_eli(docs, top = '../data/tmp/'):
	assert all([d.group == 'test' for d in docs])
	dump_phase1_ner(top, docs)
	exec_phase1_ner(top, 'ebm_p')
	load_phase1_ner(top, docs, 'ebm_p')
	exec_phase1_ner(top, 'ebm_i')
	load_phase1_ner(top, docs, 'ebm_i')
	exec_phase1_ner(top, 'ebm_o')
	load_phase1_ner(top, docs, 'ebm_o')

	dump_phase1_ev(top, docs)
	exec_phase1_ev(top)
	load_phase1_ev(top, docs)
	
	dump_phase2_o_ev(top, docs)
	exec_phase2_o_ev(top)
	load_phase2_o_ev(top, docs)

	dump_phase2_ic_ev(top, docs)
	exec_phase2_ic_ev(top)
	load_phase2_ic_ev(top, docs)

if __name__ == '__main__':
	run_example()
