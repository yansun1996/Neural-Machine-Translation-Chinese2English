#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:Darksoul
# datetime:11/24/2018 21:59
# software: PyCharm

# import pytorch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import math
import random
import os


class Encoder(nn.Module):
    '''
    Define encoder and forward process
    '''

    def __init__(self, dim_input, dim_embed, dim_hidden, num_layers, dropout):
        super(Encoder, self).__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_embed = dim_embed
        self.embed = nn.Embedding(dim_input, dim_embed)
        self.cell = nn.GRU(dim_embed, dim_hidden,
                           num_layers, dropout=dropout,
                           bidirectional=True)

    def forward(self, inputs, inputs_lens, hidden=None):
        '''
        We need to sum the outputs since bi-diretional is used
        '''
        embedded = self.embed(inputs)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, inputs_lens)
        outputs, hidden = self.cell(packed, hidden)
        outputs, output_lengths = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.dim_hidden] + \
                  outputs[:, :, self.dim_hidden:]
        return outputs, hidden


class Attention(nn.Module):
    '''
    Define attention mechanism
    '''
    def __init__(self, dim_hidden):
        super(Attention, self).__init__()
        self.dim_hidden = dim_hidden
        # 2*dim_hidden is needed since bi-direction is used
        self.attn = nn.Linear(2*self.dim_hidden, dim_hidden)
        self.v = nn.Parameter(torch.rand(dim_hidden))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        scores = self.score(h, encoder_outputs)
        return F.relu(scores).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        e = F.softmax(self.attn(torch.cat([hidden, encoder_outputs], 2)),dim=1)
        e = e.transpose(1, 2)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        e = torch.bmm(v, e)
        return e.squeeze(1)


class Decoder(nn.Module):
    '''
    Define decoder with attention
    '''
    def __init__(self, dim_embed, dim_hidden, dim_output, num_layers, dropout):
        super(Decoder, self).__init__()
        self.dim_embed = dim_embed
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.num_layers = num_layers

        self.embed = nn.Embedding(dim_output, dim_embed)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(dim_hidden)
        self.cell = nn.GRU(dim_hidden + dim_embed, dim_hidden,
                          num_layers, dropout=dropout)
        self.out = nn.Linear(2*dim_hidden, dim_output)

    def forward(self, inputs, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        embedded = self.embed(inputs).unsqueeze(0)  # (1,B,N)
        embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2)
        #print(embedded.size())
        #print(context.size())
        #print(rnn_input.size())
        #print(last_hidden.size())
        output, hidden = self.cell(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        # For Debug
        #print(output.size())
        #print(context.size())
        torch.cat([output, context], 1)
        output = self.out(torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_len, tgt, tgt_len, teacher_forcing_ratio=0.5):
        batch_size = src.size(1)
        max_len = tgt.size(0)
        vocab_size = self.decoder.dim_output
        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size).cuda())
        # for debug
        # print(src.size())
        # print(src_len.size())
        encoder_output, hidden = self.encoder(src, src_len)
        hidden = hidden[:self.decoder.num_layers]
        # Put <sos> at first position
        output = Variable(tgt.data[0, :])
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(
                output, hidden, encoder_output)
            outputs[t] = output
            # Randomly choose whether to use teacher force or not
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            output = Variable(tgt.data[t].cuda() if is_teacher else top1.cuda())
        return outputs

    def inference(self, src, src_len, max_len=cfg.MAX_LENGTH):
        pred_idx = []
        batch_size = src.size(1)
        vocab_size = self.decoder.dim_output
        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()

        encoder_output, hidden = self.encoder(src, src_len)
        hidden = hidden[:self.decoder.num_layers]
        # Put <sos> at first position
        output = Variable(src.data[0, :])
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(
                output, hidden, encoder_output)
            outputs[t] = output
            top1 = output.data.max(1)[1]
            pred_idx.append(top1.item())
            output = Variable(top1).cuda()
            if top1 == EOS_idx: break
        return outputs, pred_idx


def load_checkpoint(net, cfg):
    # load from required checkpoint
    if cfg.load_checkpoint > 0:
        net.load_state_dict(
            torch.load(os.path.join(cfg.checkpoints_path, 'checkpoint_step_' + str(cfg.load_checkpoint) + '.pth.tar')))
        print('load checkpoint ' + str(cfg.load_checkpoint))
    elif cfg.load_checkpoint == 0:
        print('train from scratch')
    else:
        raise ValueError('load_checkpoint should be larger or equals to 0')

    return net


def save_checkpoint(net, cfg, step, batch_index):
    # save model
    print('checkpoint '+str(cfg.load_checkpoint + step)+'_'+str(batch_index)+' saved')
    torch.save(net.state_dict(), os.path.join(cfg.checkpoints_path,
                                              'checkpoint_step_' + str(cfg.load_checkpoint + step)+'_'+str(batch_index) + '.pth.tar'))