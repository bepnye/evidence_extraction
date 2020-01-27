import torch
from torch import optim
import torch.nn as nn
from transformers import *

class BERTVergaPytorch(nn.Module):
    """ TODO: """

    def __init__(self):
        super(BERTVergaPytorch, self).__init__()
        # INIT ANY VARIABLES USED
        self.out_dim    = 4
        self.bert_dim   = 768
        self.len_cutoff = 500

        # INIT MODEL LAYERS
        self.bert_encoder = BertModel.from_pretrained('/home/eric/evidence-inference/evidence_inference/models/structural_attn/scibert_scivocab_uncased/', output_hidden_states = True).cuda().eval()
        
        # init the weight matrix used for final output 
        weight_matrix_dim  = torch.empty(self.bert_dim, self.bert_dim, self.out_dim)
        biaffine_transform = nn.init.kaiming_normal_(weight_matrix_dim, nonlinearity = 'relu')
        self.W = torch.nn.Parameter(biaffine_transform)
        self.W.requires_grad = True
        self.sm = nn.Softmax(dim = -1)
        assert(self.bert_encoder.config.output_hidden_states==True)

    def forward(self, inputs):
        inputs = inputs[0] # this is just a fake run... so its okay..
        text = torch.tensor(inputs['text'][:self.len_cutoff]).cuda().unsqueeze(0)
        segments = torch.tensor(inputs['segment_ids'][:self.len_cutoff]).cuda().unsqueeze(0)
        with torch.no_grad():
            encoded_layers = self.bert_encoder(text, segments)
        word_embeddings = encoded_layers[-1][-2][0]

        num_head_mentions, num_tail_mentions = 0, 0
        entity_mention_scores = []
        for entity1, entity2 in inputs['relations']:
            head_w = torch.zeros((1, self.bert_dim)).cuda() - 100
            tail_w = torch.zeros((1, self.bert_dim)).cuda() - 100
            
            for mention in entity1.mentions:
                if mention.i > self.len_cutoff: continue 
                spans  = word_embeddings[mention.i:mention.f]
                head_w = torch.max(torch.cat([spans, head_w]), dim = 0)[0].unsqueeze(0)

            for mention in entity2.mentions:
                if mention.i > self.len_cutoff: continue
                spans = word_embeddings[mention.i:mention.f]
                tail_w = torch.max(torch.cat([spans, tail_w]), dim = 0)[0].unsqueeze(0)

            mention_score = self.sm(torch.matmul(tail_w.squeeze(), torch.matmul(head_w.squeeze(), self.W.cuda())))
            entity_mention_scores.append(mention_score) 

        return torch.stack(entity_mention_scores)

        
