import torch
from torch import optim
import torch.nn as nn
from transformers import *
from padded_sequence import PaddedSequence

SCI_BERT_LOCATION = '/home/eric/evidence-inference/evidence_inference/models/structural_attn/scibert_scivocab_uncased/'

class BERTVergaPytorch(nn.Module):
    """
    Represents the updated version of the VERGA model, but this time to work
    with BERT.

    Fields:
        @param bert_dim      what is the output size of BERT?
        @param len_cutoff    how many word pieces can we shove through BERT?
        @param out_dim       range of labels we are predicting.
        @param bert_backprop is whether or not to backprop through BERT.
        @param ner_loss      is whether or not we are learning NER labels.
        @param hard_attention is whether or not to use hard attention for alternate loss.
        @param bert_encoder  is the BERT model used.
    """

    def __init__(self, output_dim, bert_backprop, ner_loss, hard_attention):
        super(BERTVergaPytorch, self).__init__()
        # Make sure data used makes sense
        assert(ner_loss.upper() in set(['NULL', 'JOINT', 'ALTERNATE']))
        assert(not(hard_attention) or ner_loss.upper() == 'ALTERNATE')

        # INIT ANY VARIABLES USED
        self.bert_dim   = 768
        self.len_cutoff = 500
        self.out_dim    = output_dim
        self.ner_loss   = ner_loss.upper()
        self.bert_backprop = bert_backprop
        self.hard_attention = hard_attention
        self.bert_encoder   = BertModel.from_pretrained(SCI_BERT_LOCATION, output_hidden_states = True).cuda()

        # INIT MODEL LAYERS
        if not(bert_backprop):
            self.bert_encoder.eval()

        # init the weight matrix used for final output
        weight_matrix_dim  = torch.empty(self.out_dim, self.bert_dim, self.bert_dim)
        biaffine_transform = nn.init.kaiming_normal_(weight_matrix_dim, nonlinearity = 'relu')
        self.W = torch.nn.Parameter(biaffine_transform)
        self.W.requires_grad = True
        assert(self.bert_encoder.config.output_hidden_states==True)

    def bert_encode(self, text):
        """
        Run BERT over abstract data.
        @param text is the word piece tokens
        @param segments is the word piece token information about which sentence
        it is from.
        @return the encoded BERT representation, complete with ALL layer info.
        """
        if self.bert_backprop:
            return self.bert_encoder(text, None)

        ### NOT BACKPROPING ###
        padded_text = PaddedSequence.autopad(text, batch_first = True, padding_value = 0)
        with torch.no_grad():
            encoded_layers = self.bert_encoder(padded_text.data.cuda(), torch.zeros(padded_text.data.shape).cuda())
    
        word_embeddings = encoded_layers[-1][-2]
        mask = padded_text.mask(on=1, off=0, device='cuda', dtype=torch.float)
        return (word_embeddings * mask.unsqueeze(dim=-1)) #word_embeddings

    def get_entity_mentions(self, word_embeddings, relations):
        """
        Modify the given relations to be soft/hard attention scores, and
        return these predictions.

        @param word_embeddings are the BERT word embeddings used here.
        @param relations is a list of pairs of entities, which will all be
        modified to be new losses.
        """
        if self.ner_loss == 'NULL':
            pass
        elif self.ner_loss == 'JOINT':
            pass
        else: # we are alternating
            pass

    def forward(self, inputs):
        text = [torch.tensor(input_['text'][:self.len_cutoff]).cuda() for input_ in inputs]
        
        # encode the data
        word_embeddings = self.bert_encode(text)
        mention_scores  = self.get_entity_mentions(word_embeddings, inputs)#['relations'])
       
        batch_e1_pieces = []
        batch_e2_pieces = []
        for idx, input_ in enumerate(inputs):
            for entity1, entity2 in input_['relations']:
                entity1_word_pieces = []
                for mention in entity1.mentions:
                    if mention.i > self.len_cutoff: continue
                    entity1_word_pieces.append(word_embeddings[idx][mention.i:mention.f])

                entity2_word_pieces = []
                for mention in entity2.mentions:
                    if mention.i > self.len_cutoff: continue
                    entity2_word_pieces.append(word_embeddings[idx][mention.i:mention.f])

                # ADD TO BIG ARRAY TO DO ONE BIG MULTIPLICATION FOR SPEED (kerchoo)
                batch_e1_pieces.append(torch.cat(entity1_word_pieces))
                batch_e2_pieces.append(torch.cat(entity2_word_pieces))
        
        e1 = PaddedSequence.autopad(batch_e1_pieces, batch_first = True, padding_value = 0)
        e2 = PaddedSequence.autopad(batch_e2_pieces, batch_first = True, padding_value = 0) 
        e1_mask = e1.mask(on=0, off=float('-inf'), size=e1.data.size()[:2])
        e2_mask = e2.mask(on=0, off=float('-inf'), size=e2.data.size()[:2])
        mask = (e1_mask.unsqueeze(dim=1) + e2_mask.unsqueeze(dim=2)).unsqueeze(dim=-1)

        z_head   = torch.matmul(self.W.unsqueeze(dim=1).cuda(), e1.data.transpose(1, 2))
        z        = torch.matmul(z_head.transpose(2, 3), e2.data.transpose(1, 2))
        masked_z = (z.transpose(0, 1) + mask.transpose(1, 3).cuda())
        entity_relation_scores = torch.logsumexp(masked_z.view(*masked_z.size()[:2], -1), dim=2) 
        return entity_relation_scores, mention_scores
