import glob, json
import xml.etree.ElementTree as ET
from collections import defaultdict
import traceback

import pandas as pd

import classes

def read_docs(abst_only = True):
	fnames = glob.glob('../data/coref_anns/dev/pretty-edin_*.json')
	frames = defaultdict(list)
	for idx, frame in pd.read_csv('../data/exhaustive_ico_fixed.csv').iterrows():
		frames[str(frame.RowID)].append(frame)

	n_errors = 0
	n_total = 0

	docs = []
	for fname in fnames:
		worker, pmid, exta = fname.split('/')[-1].split('_')
		text, offsets = extract_text_and_offsets(pmid, abst_only)
		ann = json.load(open(fname))
		doc = classes.Doc(pmid, text)
		doc.max_xml = offsets[-1][1]
		docs.append(doc)

		entity_group_ids = {}
		coref_spans = defaultdict(set)
		for e in 'io':
			for group_id, (html_id, group) in enumerate(ann[e].items()):
				group_name = group['name']
				name_tokens = group_name.split(' ')
				if name_tokens[0].isdigit():
					group_name = ' '.join(name_tokens[1:])
				group_id = '{}_{}'.format(e, group_name.replace('_', '-'))
				for s in group['spans']:
					if s['i'] == '-1' and s['f'] == '-1':
						assert entity_group_ids.get(s['txt'], group_id) == group_id
						entity_group_ids[s['txt']] = group_id
					else:
						text_i, text_f = xml_to_text(offsets, s['i'], s['f'], s['txt'], text)
						if text_i == -1 or text_f == -1:
							continue
						coref_spans[group_id].add(classes.Span(text_i, text_f, s['txt']))
		for group_id, spans in coref_spans.items():
			doc.labels['GOLD_'+group_id] = list(spans)

		for frame in frames[pmid]:
			n_total += 1
			xml_i, xml_f = frame.xml_offsets.split(':')
			if not (xml_i.isdigit() and xml_f.isdigit()):
				continue
			xml_i, xml_f = int(xml_i), int(xml_f)
			if xml_f > doc.max_xml:
				continue

			try:
				i_span = classes.Span(-1, -1, frame.Comparator, entity_group_ids[frame.Comparator])
				c_span = classes.Span(-1, -1, frame.Intervention, entity_group_ids[frame.Intervention])
				o_span = classes.Span(-1, -1, frame.Outcome, entity_group_ids[frame.Outcome])
			except KeyError:
				n_errors += 1
				continue
			text_i, text_f = xml_to_text(offsets, xml_i, xml_f, frame.Reasoning, text)
			ev_span = classes.Span(text_i, text_f, frame.Reasoning)
			frame = classes.Frame(i_span, c_span, o_span, ev_span, frame.Answer)
			doc.frames.append(frame)

	print('Read coref groups for {} docs ({}/{} frames missing)'.format(len(docs), n_errors, n_total))
	return docs

def extract_text_and_offsets(pmid, abst_only = False):
	fname = '../data/anne_htmls/PMC{}.html'.format(pmid)
	root = ET.parse(fname).getroot()
	if abst_only:
		abst_nodes = [n for n in root.iter() if n.tag == 'abstract']
		root = abst_nodes[0]
	offsets = [n for n in root.iter() if n.tag == 'offsets']
	text = ''
	xml_offsets = []
	text_offsets = []
	for o in offsets:
		s = o.text
		xml_i = int(o.attrib['xml_i'])
		xml_offsets.append(xml_i)
		text_offsets.append(len(text))
		text += s
		text += ' ' # extra padding never hurts...
	xml_offsets.append(int(o.attrib['xml_f']))
	text_offsets.append(len(text))
	return text, list(zip(text_offsets, xml_offsets))

def xml_to_text(offsets, xml_start, xml_end, span_text, doc_text):
	xml_start = int(xml_start)
	xml_end = int(xml_end)
	for text_i, xml_i in offsets[::-1]:
		if xml_i <= xml_start:
			text_start = text_i + (xml_start - xml_i)
			text_end = text_start + len(span_text)
			try:
				assert doc_text[text_start:text_end] == span_text
			except AssertionError:
				if False:
					print('WARNING: XML -> text SLICE MISMATCH')
					print(doc_text[text_start:text_end])
					print(span_text)
			return text_start, text_end
	return -1, -1
