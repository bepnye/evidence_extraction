from flask import Flask, render_template, send_file
from flask import request
from flask_cors import CORS
#from trialstreamer import dbutil
import os, json, time, sys
from collections import defaultdict
from flask import jsonify
from operator import itemgetter

import run_pipeline
from process_trialstreamer import process_json_data

app = Flask(__name__)
CORS(app)

CWD = os.getcwd()

@app.route('/')
def landing():
	return 'Evidence Extraction endpoint!'

@app.route('/get_ev', methods = ['POST', 'GET'])
def get_ev():
	json_docs = json.loads(request.data.decode('utf-8'))
	docs = process_json_data(json_docs)
	run_dir = os.path.join(CWD, 'server')
	run_pipeline.write_phase1_ev(run_dir, docs)
	run_pipeline.run_phase1_ev(run_dir)
	run_pipeline.load_phase1_ev(run_dir, docs)
	return json.dumps([[s.text for s in d.labels['BERT_ev']] for d in docs])

@app.route('/get_icos', methods = ['POST', 'GET'])
def get_icos():
	json_docs = json.loads(request.data.decode('utf-8'))
	docs = process_json_data(json_docs)
	run_dir = os.path.join(CWD, 'server')
	run_pipeline.write_phase2_o_ev(run_dir, docs)
	run_pipeline.write_phase2_ic_ev(run_dir, docs)
	run_pipeline.run_phase2_o_ev(run_dir)
	run_pipeline.run_phase2_ic_ev(run_dir)
	run_pipeline.load_phase2_o_ev(run_dir, docs)
	run_pipeline.load_phase2_ic_ev(run_dir, docs)
	results = []
	for d in docs:
		icos = []
		print(d.labels['NER_o'])
		for s in d.labels['BERT_ev']:
			i = s.pred_i
			c = s.pred_c
			for o in s.pred_os:
				print(o)
				icos.append((i.text, c.text, o.text, o.label))
		results.append(icos)
	return json.dumps(results)

def main():
	host = os.environ.get('APP_HOST', '0.0.0.0')
	port = int(os.environ.get('APP_PORT', 8000))
	app.run(debug=True, host=host, port=port)

if __name__ == '__main__':
	main()
