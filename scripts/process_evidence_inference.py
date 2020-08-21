import sys, random, os, csv
from collections import namedtuple, defaultdict, Counter

import classes
import utils
sys.path.append('..')
import config
sys.path.append(os.path.join(config.EV_INF_DIR, 'evidence_inference', 'preprocess'))
import preprocessor

def extract_raw_abstract(article):
	return article.get_abstract(False).replace('<p>', '')

def init_doc(pmcid, abst_only):
	article = preprocessor.get_article(pmcid)
	if (abst_only):
		# gotta add the same gunk as the preprocessor so it all lines up
		text = "TITLE:\n{}\n\n\n\n{}".format(article.get_title(), extract_raw_abstract(article))
	else:
		text = preprocessor.extract_raw_text(article)
	doc = classes.Doc(pmcid, text)
	return doc

def read_docs(abst_only = False):

	Prompt = namedtuple('Prompt', 'i c o')
	docs = {}
	prompts = {}

	print('Reading prompts + articles')
	for prompt in preprocessor.read_prompts().to_dict('records'):
		pmcid = prompt['PMCID']
		if pmcid not in docs:
			docs[pmcid] = init_doc(pmcid, abst_only)

		pid = prompt['PromptID']
		if pid not in prompts:
			prompts[pid] = Prompt(prompt['Intervention'], prompt['Comparator'], prompt['Outcome'])

	print(len(docs))
	print(len(prompts))

	n_anns = 0
	n_bad_offsets = 0
	print('Processing annotations')
	anns = preprocessor.read_annotations().to_dict('records')
	for ann in anns:
		if abst_only and not ann['In Abstract']:
			continue
		if not ann['Annotations']:
			continue
		ev = classes.Span(ann['Evidence Start'], ann['Evidence End'], ann['Annotations'])
		doc = docs[ann['PMCID']]
		if doc.text[ev.i:ev.f] != ev.text:
			n_bad_offsets += 1
			continue
		n_anns += 1
		prompt = prompts[ann['PromptID']]
		label = ann['Label']
		i = prompt.i.strip()
		c = prompt.c.strip()
		o = prompt.o.strip()
		add_new_frame = True
		for f in doc.frames:
			if f.i.text == i and f.c.text == c and f.o.text == o:
				assert f.label == classes.Frame.get_encoded_label(label)
				if utils.s_overlap(f.ev, ev):
					add_new_frame = False
		if add_new_frame:
			frame = classes.Frame( \
			  classes.Span(-1, -1, i),
			  classes.Span(-1, -1, c),
			  classes.Span(-1, -1, o), ev, label)
			doc.frames.append(frame)


	pmcids_docs = list(docs.items())
	for pmcid, doc in pmcids_docs:
		if not doc.frames:
			del docs[pmcid]
	
	print('Retained {}/{} valid annotations ({} w/ bad offsets)'.format(\
			n_anns, len(anns), n_bad_offsets))
	print('Retained {}/{} docs with nonzero prompts'.format(len(docs), len(pmcids_docs)))
	
	return list(docs.values())
