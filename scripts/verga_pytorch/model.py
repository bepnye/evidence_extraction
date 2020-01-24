import torch
from torch import optim
import torch.nn as nn
from transformers import *

class BERTVergaPytorch(nn.Module):
    """ TODO: """

    def __init__(self):
        super(BERTVergaPytorch, self).__init__()
        self.output = nn.Linear(10, 5)
        self.bert_encoder = BertModel.from_pretrained('/home/eric/evidence-inference/evidence_inference/models/structural_attn/scibert_scivocab_uncased/', output_hidden_states = True).cuda()
        self.W = torch.nn.Parameter(torch.randn(768, 4, 768))
        self.W.requires_grad = True
        self.sm = nn.Softmax(dim = -1)
        assert(self.bert_encoder.config.output_hidden_states==True)

    def forward(self, inputs):
        inputs = inputs[0] # this is just a fake run... so its okay..
        text = torch.tensor(inputs['text'][:500]).cuda().unsqueeze(0)
        segments = torch.tensor(inputs['segment_ids'][:500]).cuda().unsqueeze(0)
        encoded_layers = self.bert_encoder(text, segments)
        word_embeddings = encoded_layers[-1][-2][0]

        head = torch.zeros(768).cuda()
        tail = torch.zeros(768).cuda()
        head_mentions, tail_mentions = 0, 0
        relations = []
        for entity1, entity2 in inputs['relations']:
            head = torch.zeros(768).cuda()
            tail = torch.zeros(768).cuda()
            head_mentions, tail_mentions = 0, 0
            
            for mention in entity1.mentions:
                if mention.i > 500: continue 
                spans = word_embeddings[mention.i:mention.f]
                head += torch.sum(spans, dim = 0)
                head_mentions += 1

            for mention in entity2.mentions:
                if mention.i > 500: continue
                spans = word_embeddings[mention.i:mention.f]
                tail += torch.sum(spans, dim = 0)
                tail_mentions += 1
            
            head = head / head_mentions if head_mentions > 0 else head
            tail = tail / tail_mentions if tail_mentions > 0 else tail
            mid_tail = torch.matmul(self.W.cuda(), tail.unsqueeze(1)).squeeze()
            res = self.sm(torch.matmul(head, mid_tail))
            relations.append(res) 

        return torch.stack(relations)

        
