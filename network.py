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

    def forward(self, src, src_len, tgt, tgt_len, teacher_forcing_ratio=cfg.teacher_forcing_ratio):
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
    if cfg.load_checkpoint == 0:
        print('train from scratch')
    elif cfg.load_checkpoint > 0:
        net.load_state_dict(
            torch.load(os.path.join(cfg.checkpoints_path, 'checkpoint_step_' + str(cfg.load_checkpoint) + '.pth.tar')))
        print('load checkpoint ' + str(cfg.load_checkpoint))
    else:
        raise ValueError('load_checkpoint should be larger or equals to 0')

    return net


def save_checkpoint(net, cfg, step):

    # save model
    print('checkpoint_'+str(step) + ' saved')
    torch.save(net.state_dict(), os.path.join(cfg.checkpoints_path,
                                              'checkpoint_step_' + str(step) + '.pth.tar'))


class BeamSearch(nn.Module):
    '''
    Implement BeamSearch for testing
    '''

    def __init__(self, encoder, decoder, width):
        super(BeamSearch, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.width = width

    def forward(self, src, src_len, width, max_len):
        vocab_size = self.decoder.dim_output
        encoder_output, hidden = self.encoder(src, src_len)
        hidden = hidden[:self.decoder.num_layers]

        if USE_CUDA:
            # output is in the shape of batch_size * dict size
            beam_best = Variable(torch.zeros(width, max_len)).cuda()
            prob_best = Variable(torch.zeros(width)).cuda()
            # beam options, all possible width * width options
            beam_options = Variable(torch.zeros(width * width, max_len)).cuda()
            prob_options = Variable(torch.zeros(width * width)).cuda()
            output = Variable(src.data[0, :].cuda())
            temp = Variable(src.data[0, :].cuda())
        else:
            beam_best = Variable(torch.zeros(width, max_len))
            prob_best = Variable(torch.zeros(width))
            beam_options = Variable(torch.zeros(width * width, max_len))
            prob_options = Variable(torch.zeros(width * width))
            temp = Variable(src.data[0, :])
            output = Variable(src.data[0, :])

        output, _, _ = self.decoder(
            output, hidden, encoder_output)

        # first round
        # val: prob, idx: index
        val, idx = output.data.topk(k=width, dim=1)

        # generate beams
        for i in range(width):
            beam_best[i][1] = idx[0][i]
            prob_best[i] = val[0][i].exp()  # since prob log,softmax from -10 to -12

        for t in range(2, max_len):
            cnt = 0
            for i in range(width):
                curr_prob = prob_best[i]
                #
                hidden = self.encoder(src, src_len)[1][:self.decoder.num_layers]

                # feed the entire vector for the training process
                for j in range(t):
                    temp = beam_best[i][j].reshape(1).long()
                    new_output, hidden, _ = self.decoder(
                        temp, hidden, encoder_output)

                val1, idx1 = new_output.data.topk(k=width, dim=1)

                for j in range(width):
                    beam_options[cnt] = beam_best[i]
                    beam_options[cnt][t] = idx1[0][j]
                    prob_options[cnt] = val1[0][j].exp() * curr_prob
                    cnt += 1

            topVal, topInx = prob_options.topk(k=width, dim=0)
            for j in range(topInx.size(0)):
                prob_best[j] = topVal[j]
                beam_best[j] = beam_options[topInx[j]]

        best_index = prob_best.max(0)[1]
        return beam_options[best_index].cpu().numpy()