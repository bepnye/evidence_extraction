import json, requests
URL = 'http://trialstreamer.ccs.neu.edu'
PORT = 8000

def post_example():
	# API takes a list of json-formatted docs
	docs = json.load(open('sample_data.json'))

	# get_ev
	# returns a list of the ev_spans in each doc
	endpoint = '{}:{}/{}'.format(URL, PORT, 'get_ev')
	per_doc_ev = requests.post(url = endpoint, data = json.dumps(docs))
	for doc, ev in zip(docs, per_doc_ev.json()):
		doc['ev'] = ev

	# get_icos
	# returns a list of (i, c, o, label) tuples for each doc
	endpoint = '{}:{}/{}'.format(URL, PORT, 'get_icos')
	per_doc_icos = requests.post(url = endpoint, data = json.dumps(docs))
	for doc, icos in zip(docs, per_doc_icos.json()):
		doc['icos'] = icos

	return docs
