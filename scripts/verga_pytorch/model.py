import torch
from torch import optim
import torch.nn as nn
from transformers import *

class BERTVergaPytorch(nn.Module):
    """ TODO: """

    def __init__(self, output_dim = 4):
        super(BERTVergaPytorch, self).__init__()
        # INIT ANY VARIABLES USED
        self.out_dim    = output_dim
        self.bert_dim   = 768
        self.len_cutoff = 500

        # INIT MODEL LAYERS
        self.bert_encoder = BertModel.from_pretrained('/home/eric/evidence-inference/evidence_inference/models/structural_attn/scibert_scivocab_uncased/', output_hidden_states = True).cuda().eval()
        
        # init the weight matrix used for final output 
        weight_matrix_dim  = torch.empty(self.bert_dim, self.out_dim, self.bert_dim)
        biaffine_transform = nn.init.kaiming_normal_(weight_matrix_dim, nonlinearity = 'relu')
        self.W = torch.nn.Parameter(biaffine_transform)
        self.W.requires_grad = True
        #self.sm = nn.Softmax(dim = -1)
        assert(self.bert_encoder.config.output_hidden_states==True)

    def forward(self, inputs):
        inputs = inputs[0] # this is just a fake run... so its okay..
        text = torch.tensor(inputs['text'][:self.len_cutoff]).cuda().unsqueeze(0)
        segments = torch.tensor(inputs['segment_ids'][:self.len_cutoff]).cuda().unsqueeze(0)
        with torch.no_grad():
            encoded_layers = self.bert_encoder(text, segments)
        
        word_embeddings = encoded_layers[-1][-2][0]
        num_head_mentions, num_tail_mentions = 0, 0
        entity_relation_scores = []
        for entity1, entity2 in inputs['relations']:
            entity1_word_pieces = []
            for mention in entity1.mentions:
                if mention.i > self.len_cutoff: continue
                entity1_word_pieces.append(word_embeddings[mention.i:mention.f])

            entity2_word_pieces = []
            for mention in entity2.mentions:
                if mention.i > self.len_cutoff: continue
                entity2_word_pieces.append(word_embeddings[mention.i:mention.f])
            
            entity1_word_pieces = torch.cat(entity1_word_pieces)
            entity2_word_pieces = torch.cat(entity2_word_pieces)

            z_head = torch.matmul(self.W.cuda(), entity1_word_pieces.transpose(0, 1)).transpose(0, 1).transpose(1, 2)
            z = torch.matmul(z_head, entity2_word_pieces.transpose(0, 1))
            # LogSumExp((Weights * E1 (transpose)) * E2 (transpose)) over dim = 0
            flattened_z = z.transpose(0, 2).reshape(-1, self.out_dim)
            entity_relation_scores.append(torch.logsumexp(flattened_z, dim = 0))

        return torch.stack(entity_relation_scores) 
