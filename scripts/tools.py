import sys, os
import subprocess
from collections import namedtuple
	
ab3p_dir = '../tools/Ab3P'

concept_fields = ['Concept_Name', 'Concept_Id', 'Semantic_Types', 'Sources']
Concept = namedtuple('Concept', concept_fields)
Node = namedtuple('Node', 'type parent children data')

def ab3p_text(text):
	tmp_fname = '{}/tmp_input.txt'.format(ab3p_dir)
	with open(tmp_fname, 'w') as fp:
		fp.write(str(text))
		fp.flush()
		return ab3p_fname(tmp_fname)

def ab3p_fname(fname, verbose = False):
	if verbose:
		print('Calling ab3p on:\n\t', fname)
	ret = subprocess.check_output(['{}/identify_abbr'.format(ab3p_dir), fname])
	return parse_ab3p(ret.decode('utf-8'))

def parse_ab3p(ret):
	sf_to_lf = {}
	matches = [m.strip() for m in ret.strip().split('\n')]
	matches = [m for m in matches if m]
	for m in matches:
		sf, lf, acc = m.split('|')
		sf_to_lf[sf] = lf
	return sf_to_lf

def mm_fname(fname):
	return mm_text(open(fname).read())

def mm_text(text, verbose = False):
	if verbose:
		print('Calling metamap on:\n\t', text)
	# semicolon jacks up the mmap command line parse
	if text == ';':
		return parse_mm_raw_output('')
	mm_dir = '../tools/metamap'
	try:
		ret = subprocess.check_output([ \
				'java',
				'-classpath',
				'{}/prologbeans.jar:{}/MetaMapApi.jar'.format(mm_dir, mm_dir),
				'gov.nih.nlm.nls.metamap.MetaMapApiTest',
				'-DG',
				'-R', 'MSH',
				'{}'.format(text.encode('ascii', 'replace').decode('utf-8'))
		])
		return parse_mm_raw_output(ret.decode('utf-8'))
	except Exception:
		print('Metamap choked on input text: "{}"'.format(text))
		return parse_mm_raw_output('')

def get_mm_phrases(text):
	try:
		raw_output = mm_text(text)
		phrases = process_mm_structured_output(raw_output, text)
	except ValueError:
		print('Unable to process METAMAP return value')
		print(text[:30]+'...')
		phrases = []
	return phrases

def get_mm_concepts(text):
	concepts = set()
	phrases = get_mm_phrases(text)
	for phrase in phrases:
		for concept in phrase.concepts:
			concepts.add(concept)
	return concepts

def parse_mm_raw_output(ret):
	node_names = ['Document', 'input text', 'Utterance', 'Phrase', \
								'Mappings', 'Map Score', 'Score']
	child_types = { \
			'Document': 'input text',
			'input text': 'Utterance',
			'Utterance': 'Phrase',
			'Phrase': 'Mappings',
			'Mappings': 'Map Score',
			'Map Score': 'Score',
			'Score': None,
	}

	def create_node(node_type, parent):
		return Node(node_type, parent, [], {})

	mm_structured_output = create_node('Document', None)
	cur_node = mm_structured_output

	lines = ret.split('\n')
	for i, l in enumerate(lines):
		l = l.strip()
		k_idx = l.find(':')
		if k_idx >= 0:
			k = l[:k_idx]
			v = l[k_idx+2:]
			if k in node_names:
				parent_node = cur_node
				while child_types[parent_node.type] != k:
					parent_node = parent_node.parent
				cur_node = create_node(k, parent_node)
				parent_node.children.append(cur_node)
			if v:
				cur_node.data[k] = v

	return mm_structured_output

MM_Phrase = namedtuple('MM_Phrase', 'text i f concepts')
def process_mm_structured_output(mm_structured_output, text):
	span_concepts = []
	cur_u_offset = 0
	# try to match the metamap text preprocessing as much as possible so we can
	# locate the utterances in the source text
	source_text = text.replace('\n', ' ').encode('ascii', 'replace').decode('utf-8')
	# metamap splits doc text in to a series of inputs (roughly on "\n\n")
	# our parser doesn't cleanly extract the text of each input since it can be 2+ lines
	for input_node in mm_structured_output.children:
		for utterance in input_node.children:
			u_text = utterance.data['Utterance text']
			cur_u_offset = source_text.find(u_text, cur_u_offset)
			# can't use utterance.data['Position'] since we don't know the offset of the input
			if cur_u_offset < 0:
				print('Unable to find utterance in source text:\n{}'.format(u_text))
				input()
				continue

			for phrase in utterance.children:
				if len(phrase.children) == 0 or len(phrase.children[0].children) == 0:
					continue
				p_text = phrase.data['text']
				p_i = cur_u_offset + u_text.index(p_text)
				p_f = p_i + len(p_text)

				# due to mmap settings we only keep the top-scoring mapping anyways
				pref_mapping = phrase.children[0]
				# arbitrarily take the first set of mapped concepts that got the top score
				# TODO: make this decision based on semantic types and/or vocab
				pref_map_score = pref_mapping.children[0]
				concepts = set()
				for score in pref_map_score.children:
					concepts.add(Concept(*[score.data[f.replace('_', ' ')] for f in concept_fields]))
				if concepts:
					span_concepts.append(MM_Phrase(p_text, p_i, p_f, concepts))

	return span_concepts
