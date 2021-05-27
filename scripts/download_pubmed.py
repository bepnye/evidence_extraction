import wget, lxml
import os, random, glob, pickle
from lxml import etree

top = '../data/pubmed/'

def download():
	os.chdir('{}/xml/'.format(top))
	fnames = ['pubmed20n{:04d}.xml.gz'.format(i) for i in range(1, 1015)]
	fnames = random.sample(fnames, len(fnames))
	for fname in fnames:
		if not os.path.isfile(fname):
			print(fname)
			wget.download('ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/{}'.format(fname))
			print()

def get_child(node, tag, check_unique = True):
	cs = node.findall(tag)
	assert len(cs) > 0
	if check_unique: assert len(cs) == 1
	return cs[0]

def parse():
	fnames = glob.glob('{}/xml/*.xml'.format(top))
	print('Found {} files to parse'.format(len(fnames)))
	for f in fnames:
		data = parse_fname(f)


def parse_fname(f):
	print('Parsing {}...'.format(f))
	tree = etree.parse(f)
	pm_articles = tree.findall('PubmedArticle')
	data = []
	for pm_a in pm_articles:
		try:
			data.append(parse_article(pm_a))
		except:
			pass
	return data

def parse_article(pm_a):
	a = get_child(get_child(pm_a, 'MedlineCitation'), 'Article')

	pt_list = get_child(a, 'PublicationTypeList')
	pts = [pt.text for pt in pt_list]

	abst = get_child(a, 'Abstract')
	abst_text = '\n\n'.join([t.text for t in abst.findall('AbstractText')])

	return {'abst': abst_text, 'pts': pts }
