import os
import torch
from torch import optim
import torch.nn as nn
from transformers import *
from padded_sequence import PaddedSequence

SCI_BERT_LOCATION = '/home/eric/evidence-inference/evidence_inference/models/structural_attn/scibert_scivocab_uncased/'
NER_BERT_LOCATION = '/home/jay/scibert_ner_ebmnlp/' #'/home/jay/scibert_ner_ebmlnlp/' #'/home/jay/scibert_ner/'
CDR_NER_BERT_LOCATION = '/home/jay/cdr_trained_scibert_small/'

class NotQuiteVergaNER(nn.Module):
    def __init__(self,
                 num_classes,
                 bert_dim=64,
                 len_cutoff=512,
                 bert_source=SCI_BERT_LOCATION):
        super(NotQuiteVergaNER, self).__init__()
        self.bert_dim   = bert_dim
        self.len_cutoff = len_cutoff
        with open(os.path.join(bert_source, 'vocab.txt'), 'r') as vf:
            vocab = vf.readlines()
            vocab_size = len(vocab)

        # Make sure data used makes sense
        config = BertConfig(vocab_size=vocab_size,
                            num_attention_heads=8,
                            hidden_size=self.bert_dim,
                            hidden_dropout_prob=0.5,
                            attention_probs_dropout_prob=0.5,
                            num_hidden_layers=1,
                            max_position_embeddings=self.len_cutoff,
                            output_hidden_states=True,
                            num_labels=num_classes)

        self.model = BertForTokenClassification(config)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

class BERTVergaPytorch(nn.Module):
    """
    Represents the updated version of the VERGA model, but this time to work
    with BERT.

    Fields:
        @param bert_dim      what is the output size of BERT?
        @param len_cutoff    how many word pieces can we shove through BERT?
        @param out_dim       range of labels we are predicting.
        @param bert_backprop is whether or not to backprop through BERT.
        @param bert_encoder  is the BERT model used.
    """

    def __init__(self,
                 output_dim,
                 bert_backprop,
                 initialize_bert=True,
                 bert_dim=768,
                 len_cutoff=512,
                 bert_source=SCI_BERT_LOCATION):
        super(BERTVergaPytorch, self).__init__()
        # Make sure data used makes sense

        # INIT ANY VARIABLES USED
        self.bert_dim   = bert_dim
        self.len_cutoff = len_cutoff
        self.out_dim    = output_dim
        self.bert_backprop = bert_backprop
        if initialize_bert and not(bert_backprop):
            self.bert_encoder = BertModel.from_pretrained(bert_source, output_hidden_states = True).cuda().eval()
        elif initialize_bert:
            self.bert_encoder = BertModel.from_pretrained(bert_source, output_hidden_states = True).cuda()
        else:
            self.bert_encoder = None

        # init the weight matrix used for final output
        weight_matrix_dim  = torch.empty(self.out_dim, self.bert_dim, self.bert_dim)
        biaffine_transform = nn.init.kaiming_normal_(weight_matrix_dim, nonlinearity = 'relu')
        self.W = torch.nn.Parameter(biaffine_transform)
        self.W.requires_grad = True

    def bert_encode(self, text):
        """
        Run BERT over abstract data.
        @param text is the word piece tokens
        @param segments is the word piece token information about which sentence
        it is from.
        @return the encoded BERT representation, complete with ALL layer info.
        """
        padded_text = PaddedSequence.autopad(text, batch_first = True, padding_value = 0)
        if self.bert_backprop:
            encoded_layers = self.bert_encoder(padded_text.data.cuda(), torch.zeros(padded_text.data.shape).cuda())
        else:
            with torch.no_grad(): encoded_layers = self.bert_encoder(padded_text.data.cuda(), torch.zeros(padded_text.data.shape).cuda())
    
        word_embeddings = encoded_layers[-1][-2]
        mask = padded_text.mask(on=1, off=0, device='cuda', dtype=torch.float)
        return (word_embeddings * mask.unsqueeze(dim=-1))

    def forward(self, inputs, word_embeddings=None):
        text = [torch.tensor(input_['text'][:self.len_cutoff]).cuda() for input_ in inputs]
        
        # encode the data
        if word_embeddings is None: word_embeddings = self.bert_encode(text)

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
        return entity_relation_scores


class VergaClone(BERTVergaPytorch):

    def __init__(self,
                 output_dim,
                 bert_backprop,
                 initialize_bert=True,
                 bert_source=SCI_BERT_LOCATION,
                 bert_dim=64,
                 len_cutoff=512):
        super(VergaClone, self).__init__(output_dim,
                                         bert_backprop,
                                         initialize_bert=False,
                                         bert_dim=bert_dim,
                                         len_cutoff=len_cutoff)

        self.bert_dim   = bert_dim
        self.len_cutoff = len_cutoff

        # Make sure data used makes sense

        # INIT ANY VARIABLES USED
        self.out_dim    = output_dim
        self.bert_backprop = bert_backprop
        if initialize_bert:
            with open(os.path.join(bert_source, 'vocab.txt'), 'r') as vf:
                vocab = vf.readlines()
                vocab_size = len(vocab)
            config = BertConfig(vocab_size=vocab_size,
                                num_attention_heads=8,
                                hidden_size=self.bert_dim,
                                num_hidden_layers=1,
                                hidden_dropout_prob=0.5,
                                attention_probs_dropout_prob=0.5,
                                max_position_embeddings=self.len_cutoff,
                                output_hidden_states=True)
            bert_encoder = BertModel(config).cuda()
            if not bert_backprop:
                bert_encoder = bert_encoder.eval()
            self.bert_encoder = bert_encoder
        else:
            self.bert_encoder = None

        # init the weight matrix used for final output
        self.projection = nn.Sequential(nn.Linear(self.bert_dim, bert_dim), nn.ReLU())
        weight_matrix_dim  = torch.empty(self.out_dim, self.bert_dim, self.bert_dim)
        biaffine_transform = nn.init.kaiming_normal_(weight_matrix_dim, nonlinearity = 'relu')
        self.W = torch.nn.Parameter(biaffine_transform)
        self.W.requires_grad = True

    def bert_encode(self, text):
        padded_text = PaddedSequence.autopad(text, batch_first = True, padding_value = 0)
        encoded_layers = self.bert_encoder(padded_text.data.cuda(), torch.zeros(padded_text.data.shape).cuda())
        word_embeddings = encoded_layers[-1]
        word_embeddings = self.projection(word_embeddings)
        mask = padded_text.mask(on=1, off=0, device='cuda', dtype=torch.float)
        word_embeddings = word_embeddings * mask
        if not self.bert_backprop:
            word_embeddings = word_embeddings.detach()
        return word_embeddings
