import glob, json
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
import traceback
from Levenshtein import distance as string_distance

import pandas as pd

import classes

def clean_html_str(s):
	s_clean = s.replace('&nbsp;', ' ').replace('&gt;', '>').replace('&lt;', '<').replace('&amp;', '&')
	return s_clean

def read_complete_docs(glob_str):
	fnames = glob.glob(glob_str)
	docs = []
	for fname in fnames:
		worker, pmid, task, timestamp = fname.split('/')[-1].split('_')
		ann = json.load(open(fname))
		frames = ann['frames']

def read_phase2_docs(glob_str = None, abst_only = True, check_errors = True):
	fnames = glob.glob(glob_str)
	frames = defaultdict(list)
	for idx, frame in pd.read_csv('../data/exhaustive_ico_fixed.csv').iterrows():
		frames[str(frame.RowID)].append(frame)

	n_missing_ico = 0
	n_missing_ev = 0
	n_total = 0

	docs = []
	for fname in fnames:
		worker, pmid, exta = fname.split('/')[-1].split('_')
		text, offsets = extract_text_and_offsets(pmid, abst_only)
		ann = json.load(open(fname))
		doc = classes.Doc(pmid, text)
		doc.max_xml = offsets[-1][1]
		doc.group = 'test'
		doc.parse_text()
		docs.append(doc)

		entity_group_ids = {}
		coref_spans = defaultdict(set)
		for e in 'io':
			for group_id, (html_id, group_data) in enumerate(ann[e].items()):
				group_name = group_data['name']
				name_tokens = group_name.split(' ')
				if name_tokens[0].isdigit():
					group_name = ' '.join(name_tokens[1:])
				group_id = '{}_{}'.format(e, group_name.replace('_', '-'))
				for s in group_data['spans']:
					if s['i'] == '-1' and s['f'] == '-1':
						try:
							assert entity_group_ids.get(s['txt'], group_id) == group_id
						except AssertionError:
							if check_errors:
								print(fname)
								print(s['txt'])
								print(group_id)
								print(entity_group_ids.get(s['txt'], group_id))
								input()
							continue
						entity_group_ids[s['txt']] = group_id
					else:
						text_i, text_f = xml_to_text(offsets, s['i'], s['f'], s['txt'], text)
						if text_i == -1 or text_f == -1:
							continue
						coref_spans[group_id].add(classes.Span(text_i, text_f, s['txt']))
		for group_id, spans in coref_spans.items():
			doc.labels['GOLD_'+group_id] = list(spans)

		for frame in frames[pmid]:
			xml_i, xml_f = frame.xml_offsets.split(':')
			if not (xml_i.isdigit() and xml_f.isdigit()):
				continue
			xml_i, xml_f = int(xml_i), int(xml_f)
			if xml_f > doc.max_xml:
				continue
			n_total += 1
			ev_text = clean_html_str(frame.Reasoning)
			ev_i = text.find(ev_text)
			if ev_i < 0:
				n_missing_ev += 1
				continue
			try:
				i_span = classes.Span(-1, -1, frame.Comparator, entity_group_ids[frame.Comparator])
				c_span = classes.Span(-1, -1, frame.Intervention, entity_group_ids[frame.Intervention])
				o_span = classes.Span(-1, -1, frame.Outcome, entity_group_ids[frame.Outcome])
				if i_span.label == c_span.label:
					print('Warning! PMID {}: I group matches C group [{}] & [{}] = [{}]'.format(doc.id, i_span.text, c_span.text, i_span.label))
			except KeyError:
				n_missing_ico += 1
				continue
			ev_f = ev_i + len(ev_text)
			ev_span = classes.Span(ev_i, ev_f, ev_text)
			frame = classes.Frame(i_span, c_span, o_span, ev_span, frame.Answer)
			doc.frames.append(frame)

	print('Read coref groups for {} docs'.format(len(docs)))
	print('\t{}/{} frames w/ ico missing'.format(n_missing_ico, n_total))
	print('\t{}/{} frames w/ ev  missing'.format(n_missing_ev,  n_total))
	return docs

def extract_text_and_offsets(pmid, abst_only = False):
	fname = '../data/anne_htmls/PMC{}.html'.format(pmid)
	root = ET.parse(fname).getroot()
	if abst_only:
		abst_nodes = [n for n in root.iter() if n.tag == 'abstract']
		root = abst_nodes[0]
	parent_map = { c: p for p in root.iter() for c in p }
	offsets = [n for n in root.iter() if n.tag == 'offsets']
	text = ''
	xml_offsets = []
	text_offsets = []
	for o_idx, o in enumerate(offsets):
		s = o.text.replace('\xa0', ' ') # lord save us all
		xml_i = int(o.attrib['xml_i'])
		xml_offsets.append(xml_i)
		text_offsets.append(len(text))
		# Issue: we want to add some sort of spacing between HTML chunks
		# when appropriate. However, not all tags are <p> type things that
		# should take a space. Some are just footnotes, non-breaking spaces,
		# math markup, etc.
		if parent_map[o].tag == 'title' and o_idx > 0:
			text += '\n'
		text += s
		if parent_map[o].tag == 'title':
			text += '\n'
		elif parent_map[o].tag == 'p':
			if s[-1] == '.' and not s.endswith('vs.'):
				text += ' '
		else:
			pass
	xml_offsets.append(int(o.attrib['xml_f']))
	text_offsets.append(len(text))
	return text, list(zip(text_offsets, xml_offsets))

def fix_offsets(ev, i, f, text, window = 5, max_dist = 3):
	span = text[i:f]
	if ev == span:
		pass
	elif ev in text[i-window:f+window]:
		i = text.index(ev, i-window)
		f = i + len(ev)
	elif string_distance(ev.strip(' '), span.strip(' ')) <= max_dist:
		ev = span.strip(' ')
		i = text.index(ev, i-window)
		f = i + len(ev)
	else:
		i = -1
		f = -1
	return ev, i, f

def xml_to_text(offsets, xml_start, xml_end, span_text, doc_text):
	xml_start = int(xml_start)
	xml_end = int(xml_end)
	for text_i, xml_i in offsets[::-1]:
		if xml_i <= xml_start:
			text_start = text_i + (xml_start - xml_i)
			text_end = text_start + len(span_text)
			span_text, span_i, span_f = fix_offsets(span_text, text_start, text_end, doc_text)
			try:
				assert doc_text[span_i:span_f] == span_text
			except AssertionError:
				print('WARNING: XML -> text SLICE MISMATCH')
				print(doc_text[text_start:text_end])
				print(span_text)
			return span_i, span_f
	return -1, -1

def generate_json(docs):
	def decode_ner(l):
		return {'i': 'intervention', 'o': 'outcome'}[l]
	def decode_rel(l):
		return {-1: 'decreased', 0: 'unaffected', 1: 'increased'}[l]

	rows = []
	for d in docs:
		jd = {}
		jd['PMCID'] = 'PMC{}'.format(d.id)
		jd['text'] = d.text
		jd['entities'] = []
		jd['relations'] = []
		for label, spans in d.labels.items():
			assert label[:4] == 'GOLD'
			assert label[4:7] == '_i_' or label[4:7] == '_o_'
			etype = label[5]
			ename = label[7:]
			mentions = []
			for m in spans:
				mentions.append({'i': m.i, 'f': m.f, 'text': m.text})
			jd['entities'].append({'name': ename, 'label': decode_ner(etype), 'mentions': mentions})
		for f in d.frames:
			r = {}
			r['intervention'] = f.i.label[2:]
			r['comparator'] = f.c.label[2:]
			r['outcome'] = f.o.label[2:]
			r['support'] = {'i': f.ev.i, 'f': f.ev.f, 'text': f.ev.text}
			r['label'] = decode_rel(f.label)
			jd['relations'].append(r)
		rows.append(jd)
	return {'docs': rows}
