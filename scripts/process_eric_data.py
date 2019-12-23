
import pandas as pd
import numpy as np

import classes

def read_docs(csv_fname):
	df = pd.read_csv(csv_fname)
	df = df[~df.Relevant.apply(np.isnan)]
	docs = []
	for i, r in df.iterrows():
		if not r.PMID.isdigit():
			continue
		if not type(r.Abstract) is str:
			continue
		doc = classes.Doc.init_from_text(r.PMID, r.Abstract)
		doc.frames.append(classes.Frame.init_from_strings( \
		  r['Matched.Intervention..Word.Embeddings.'],
		  'placebo', # filler
		  r['Matched.Outcome..Word.Embeddings.'],
			'',
			int(r['Relevant'])))
		doc.eric_label = int(r['Relevant'])
		docs.append(doc)
	return docs
