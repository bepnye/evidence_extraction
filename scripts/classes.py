import os, json
from collections import defaultdict
from itertools import groupby

import numpy as np
import scispacy
import spacy
#from scispacy.abbreviation import AbbreviationDetector
NLP = spacy.load("en_core_sci_lg")
#ABBR_PIPE = AbbreviationDetector(NLP)
#NLP.add_pipe(ABBR_PIPE)

import tools
import utils

def string_to_tokens(string):
	if type(string) is not str:
		print('How do you expect me to NLP this: "{}"'.format(string))
		input()
	nlp = NLP(str(string))
	return [Span(t.idx, t.idx+len(t.text), t.text) for t in nlp]

class Span:
	def __init__(self, i, f, text, label = None, concepts = None):
		self.i = i
		self.f = f
		self.text = text
		self.label = label
		self.concepts = concepts 

	def __repr__(self):
		return self.__str__()

	def __str__(self):
		label_str = ' ({})'.format(self.label) if self.label != None else ''
		s = '["{}" {}:{}{}]'.format(self.text.replace('\n', '\\n'), self.i, self.f, label_str)
		return s

class Entity:
	def __init__(self, span, entity_type, label = None):
		self.text = span.text
		self.type = entity_type
		self.label = label
		self.name = None
		self.concepts = span.concepts
		self.mentions = []
		self.source_texts = [span.text]
		self.relations = []

	def __str__(self):
		return '[ENTITIY: {}]'.format(self.string)

	def pprint(self):
		print('TEXT: {}'.format(self.text))
		for m in self.mentions:
			print('\t', m.text)

class Frame:
	label_encoder = { \
		'No significant difference': 0,
		'no significant difference': 0,
		'Significantly increased': 1,
		'significantly increased': 1,
		'Significantly decreased': -1,
		'significantly decreased': -1
	}

	@classmethod
	def get_encoded_label(cls, raw_label):
		enc_label = None
		if type(raw_label) == str:
			if raw_label in cls.label_encoder:
				enc_label = cls.label_encoder[raw_label]
			elif raw_label.isdigit():
				enc_label = int(raw_label)
		elif type(raw_label) == int and raw_label in cls.label_encoder.values():
			enc_label = raw_label
		if enc_label == None:
			raise Exception('Unable to parse frame label: {}'.format(raw_label))
		return enc_label

	@classmethod
	def init_from_strings(cls, i, c, o, ev, label):
		i = Span(-1, -1, i)
		c = Span(-1, -1, c)
		o = Span(-1, -1, o)
		ev = Span(-1, -1, ev)
		frame = cls(i, c, o, ev, label)
		return frame

	def __init__(self, i, c, o, ev, label):
		self.i = i
		self.c = c
		self.o = o
		self.ev = ev
		self.label = self.get_encoded_label(label)

	def __str__(self):
		return '[I: {}, C: {}, O: {}, label: {}]'.format(self.i.text, self.c.text, self.o.text, self.label)


class Doc:
	def __init__(self, d_id, text):
		self.id = str(d_id)
		# everything that references text offsets need to go here:
		self.labels = defaultdict(list)
		self.text = text
		self.frames = []
		self.coref_groups = []
		self.parsed = False

	def has_sf_lf_map(self):
		return hasattr(self, 'sf_lf_map')

	def parse_text(self):
		nlp = NLP(self.text)
		self.spacy_extra = nlp._
		self.tokens = [Span(t.idx, t.idx+len(t.text), t.text, label = t.tag_) for t in nlp]
		self.tokens = [s for s in self.tokens if not s.text.isspace()]
		self.sents = [Span(s.start_char, s.end_char, s.text) for s in nlp.sents]
		self.parsed = True

	def get_char_labels(self, prefix = None, multi_label = True):
		per_char_labels = [[] for char in self.text]
		for label, label_spans in self.labels.items():
			if prefix and not label.startswith(prefix):
				continue
			for span in label_spans:
				for char_labels in per_char_labels[span.i:span.f]:
					char_labels.append(label)
		if multi_label:
			per_char_labels = [set(ls) for ls in per_char_labels]
		else:
			per_char_labels = [utils.mode(ls) if ls else None for ls in per_char_labels]
		return per_char_labels

	def get_span_labels(self, spans, prefix = None, multi_label = False):
		per_char_labels = self.get_char_labels(prefix)
		if not self.parsed:
			self.parse_text()
		all_span_labels = []
		for span in spans:
			span_labels = list(utils.unioned(per_char_labels[span.i:span.f]))
			if not multi_label:
				span_labels = 0 if not span_labels else span_labels[0]
			all_span_labels.append(span_labels)
		return all_span_labels

	def get_sent_labels(self, prefix = None, multi_label = False):
		if not self.parsed:
			self.parse_text()
		return self.get_span_labels(self.sents, prefix, multi_label)
	
	def get_token_labels(self, prefix = None, multi_label = False):
		if not self.parsed:
			self.parse_text()
		return self.get_span_labels(self.tokens, prefix, multi_label)

	def substitute_string(self, start_str, substitutions):
		char_offsets = [0]*(len(start_str)+1)
		new_str = ''
		cur_idx = 0
		for sf_start, sf_end, lf_text in substitutions:
			new_str += start_str[cur_idx:sf_start]
			new_str += lf_text
			cur_idx = sf_end
			# the first char of the SF remains at the same index, but the rest
			# of the original SF text gets shifted down to accomodate the len
			# of the new LF
			char_offsets[sf_start+1] = len(lf_text) - (sf_end - sf_start)
		new_str += start_str[cur_idx:]
		return new_str, np.cumsum(char_offsets)

	def get_sf_token_substitutions(self, string, tokens = None):
		tokens = tokens or string_to_tokens(string)
		text_substitutions = []
		for token in tokens:
			if token.text in self.sf_lf_map:
				text_substitutions.append((token.i, token.f, self.sf_lf_map[token.text]))
			# Allows partial token matches when it's only off by a trailing "s"
			#   e.g. "RDTs" => "rapid diagnostic tests"
			elif token.text[-1] == 's':
				base_token = token.text[:-1]
				if base_token in self.sf_lf_map:
					text_substitutions.append((token.i, token.f-1, self.sf_lf_map[base_token]))
		return text_substitutions

	def get_sf_substituted_string(self, string):
		subs = self.get_sf_token_substitutions(string)
		new_str, char_offsets = self.substitute_string(string, subs)
		return new_str

	def update_text(self, substitutions):
		new_text, char_offsets = self.substitute_string(self.text, substitutions)
		self.text = new_text
		for label_class in self.labels:
			for span in self.labels[label_class]:
				span.i += char_offsets[span.i]
				span.f += char_offsets[span.f]
				span.text = self.text[span.i:span.f]
		for frame in self.frames:
			frame.ev.i += char_offsets[frame.ev.i]
			frame.ev.f += char_offsets[frame.ev.f]
			frame.ev.text = self.text[frame.ev.i:frame.ev.f]
		self.parse_text()

	def replace_acronyms(self, save_map = False, load_map = False):
		if not self.has_sf_lf_map():
			self.sf_lf_map = tools.ab3p_text(self.text)
		if not self.parsed:
			self.parse_text()
		text_substitutions = self.get_sf_token_substitutions(self.text, self.tokens)
		self.update_text(text_substitutions)
		self.replace_frame_acronyms()

	def replace_frame_acronyms(self):
		for frame in self.frames:
			for sf, lf in self.sf_lf_map.items():
				frame.i.text = self.get_sf_substituted_string(frame.i.text)
				frame.c.text = self.get_sf_substituted_string(frame.c.text)
				frame.o.text = self.get_sf_substituted_string(frame.o.text)

	def metamap_text(self):
		self.metamap_spans = []
		for phrase in tools.get_mm_phrases(self.text):
			span = Span(phrase.i, phrase.f, phrase.text, concepts = phrase.concepts)
			self.metamap_spans.append(span)

	def metamap_frames(self):
		# many duplicated frame elements!
		mm = {}
		for frame in self.frames:
			if frame.i.text not in mm: mm[frame.i.text] = tools.get_mm_concepts(frame.i.text)
			if frame.c.text not in mm: mm[frame.c.text] = tools.get_mm_concepts(frame.c.text)
			if frame.o.text not in mm: mm[frame.o.text] = tools.get_mm_concepts(frame.o.text)
			# lets use copies so we can modify the concepts later (just in case!)
			frame.i.concepts = mm[frame.i.text].copy()
			frame.c.concepts = mm[frame.c.text].copy()
			frame.o.concepts = mm[frame.o.text].copy()

	def metamap_ner_spans(self):
		for ner_class in self.ner:
			for span in self.ner[ner_class]:
				span.concepts = tools.get_mm_concepts(span.text)

	def get_overlap_labels(self, span, attr):
		doc_spans = getattr(self, attr)
		labels = [int(utils.s_overlap(span, s)) for s in doc_spans]
		return labels
