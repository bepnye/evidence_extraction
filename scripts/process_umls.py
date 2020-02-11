import os
from collections import namedtuple

HEADERS = {
	'MRCONSO': 'cui|lang|term_status|lui|str_type|sui|is_pref|aui|saui|scui|sdui|source|term_type|code|str|srl|suppress|cvf',
	'MRDEF': 'cui|aui|atui|satui|source|def|suppress|cvf',
	'MRHIER': 'cui|aui|context|paui|source|rel|ptr|hcd|cvf',
	'MRREL': 'cui1|aui1|stype1|rel|cui2|aui2|stype2|rela|rui|srui|source|sl|rg|dir|suppress|cvf'
}

LENS = {
	'MRCONSO': 7200348,
	'MRDEF': 275902,
}

def assemble_umls(top_dir):
	atoms = {}
	concepts = {}

	def get(l, f, h):
		return l[h[f]]

	i = 0
	fbase = 'MRCONSO'
	header = { s: i for i, s in enumerate(HEADERS[fbase].split('|')) }
	for l in open(os.path.join(top_dir, fbase+'.RRF')):
		i += 1
		if (i+1)%100000 == 0:
			print('{}/{}'.format(i, LENS[fbase]))
		l = l.split('|')
		if get(l, 'lang', header) == 'ENG':
			cui = get(l, 'cui', header)
			aui = get(l, 'aui', header)

			if cui not in concepts:
				concepts[cui] = { 'auis': set() }
			concepts[cui]['auis'].add(aui)

			if aui not in atoms:
				atoms[aui] = {\
						'cui': cui,
						'str': get(l, 'str', header),
						'pref': get(l, 'is_pref', header),
						'source': get(l, 'source', header),
				}

	i = 0
	fbase = 'MRDEF'
	header = { s: i for i, s in enumerate(HEADERS[fbase].split('|')) }
	for l in open(os.path.join(top_dir, fbase+'.RRF')):
		i += 1
		if (i+1)%100000 == 0:
			print('{}/{}'.format(i, LENS[fbase]))
		l = l.split('|')
		aui = get(l, 'aui', header)
		if aui in atoms:
			defn = get(l, 'def', header)
			atoms[aui]['def'] = defn

	return atoms, concepts

