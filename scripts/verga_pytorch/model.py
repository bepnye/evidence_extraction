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
        self.bert_encoder = BertModel.from_pretrained('/home/eric/evidence-inference/evidence_inference/models/structural_attn/scibert_scivocab_uncased/', output_hidden_states = True).cuda()
        
        # init the weight matrix used for final output 
        empty_weights = torch.empty(self.bert_dim, self.bert_dim, self.out_dim)
        kaiming_normal = nn.init.kaiming_normal_(empty_weights, nonlinearity = 'relu')
        self.W = torch.nn.Parameter(kaiming_normal)
        self.W.requires_grad = True
        self.sm = nn.Softmax(dim = -1)
        assert(self.bert_encoder.config.output_hidden_states==True)

    def forward(self, inputs):
        inputs = inputs[0] # this is just a fake run... so its okay..
        text = torch.tensor(inputs['text'][:self.len_cutoff]).cuda().unsqueeze(0)
        segments = torch.tensor(inputs['segment_ids'][:self.len_cutoff]).cuda().unsqueeze(0)
        encoded_layers = self.bert_encoder(text, segments)
        word_embeddings = encoded_layers[-1][-2][0]

        head = torch.zeros(self.bert_dim).cuda()
        tail = torch.zeros(self.bert_dim).cuda()
        head_mentions, tail_mentions = 0, 0
        relations = []
        for entity1, entity2 in inputs['relations']:
            head = torch.zeros(self.bert_dim).cuda()
            tail = torch.zeros(self.bert_dim).cuda()
            head_mentions, tail_mentions = 0, 0
            
            for mention in entity1.mentions:
                if mention.i > self.len_cutoff: continue 
                spans = word_embeddings[mention.i:mention.f]
                head += torch.sum(spans, dim = 0)
                head_mentions += 1

            for mention in entity2.mentions:
                if mention.i > self.len_cutoff: continue
                spans = word_embeddings[mention.i:mention.f]
                tail += torch.sum(spans, dim = 0)
                tail_mentions += 1
            
            head = head / head_mentions if head_mentions > 0 else head
            tail = tail / tail_mentions if tail_mentions > 0 else tail
            res = self.sm(torch.matmul(tail, torch.matmul(head, self.W.cuda())))
            relations.append(res) 

        return torch.stack(relations)

        
