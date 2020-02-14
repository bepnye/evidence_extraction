import sys
import pickle
sys.path.append('../')
from classes import Entity
from load_data import TOKENIZER

def word_piece_to_char_offset(word_pieces, start, end):
    """ 
    Take an offset into word piece text, and convert it into character offsets.
    @param word_piece_text is a list of word pieces corresponding with the given start + end.
    @param start is the word piece token to start on.
    @param end is the word piece token to end on.
    @return a tuple corresponding to the new start/end
    """ 
    text = ''
    ch_start, ch_end = -1, -1
    #word_pieces = word_piece_text.split(' ')
    for idx, wp in enumerate(word_pieces): 
        filtered_wp = wp.replace('##', '')
        if idx == start:
            ch_start = len(text)

        if idx == end:
            ch_end   = len(text) # this may need to be -1 

        # only add a space if '##' does not appear
        text += filtered_wp + ('' if wp != filtered_wp else ' ') 

    return (ch_start, ch_end)


fname = './dump.out'
data = pickle.load(open(fname, 'rb'))
relation_idx = 0
output = []
for i in range(len(data[0])):
    document = data[0][i]
    relations = document['relations']
    how_many = len(relations) 
    preds  = data[1][relation_idx:relation_idx+how_many]
    labels = data[2][relation_idx:relation_idx+how_many]
    word_piece_text = TOKENIZER.tokenize(TOKENIZER.decode(document['text'], clean_up_tokenization_spaces=False))
    text = TOKENIZER.decode(document['text'], clean_up_tokenization_spaces=False)
    for idx, r in enumerate(relations):
        for span in r[0].mentions:
            if type(span.text) == str: continue 
            span.text = TOKENIZER.decode(span.text, clean_up_tokenization_spaces=False)
            new_spans = word_piece_to_char_offset(word_piece_text, span.i, span.f)
            span.i = new_spans[0]
            span.f = new_spans[1]

        for span in r[1].mentions:
            if type(span.text) == str: continue
            span.text = TOKENIZER.decode(span.text, clean_up_tokenization_spaces=False)
            new_spans = word_piece_to_char_offset(word_piece_text, span.i, span.f)
            span.i = new_spans[0]
            span.f = new_spans[1]
        
        output.append({'text': text, 'e1': r[0], 'e2': r[1], 'pred': preds[idx], 'label': labels[idx]})
        relation_idx += 1
    

pickle.dump(output, open('ben_output_format.p', 'wb'))
