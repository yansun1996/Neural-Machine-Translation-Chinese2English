#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:Darksoul
# datetime:11/24/2018 22:02
# software: PyCharm
import os

from utils import *
from network import *

from torch import optim
from torch.nn.utils import clip_grad_norm_

# import loss func
import masked_cross_entropy


src, tgt, pairs = prepareData('eng', 'fra', True)
src.trim()
tgt.trim()

batch_size = 2
hidden_size = 8
embed_size = 8
n_layers = 3
grad_clip = 10

encoder_test = Encoder(src.num, embed_size, hidden_size, n_layers, dropout=0.5)
decoder_test = Decoder(embed_size, hidden_size, tgt.num, n_layers, dropout=0.5)

net = Seq2Seq(encoder_test,decoder_test).cuda()
load_checkpoint(net, cfg)

opt = optim.Adam(net.parameters(),lr=0.01)
print(net)

for step in range(1, cfg.iteration):
    total_loss = 0
    input_batches, input_lengths, \
    target_batches, target_lengths = random_batch(src, tgt, pairs, batch_size)
    opt.zero_grad()
    output = net(input_batches, input_lengths, target_batches, target_lengths)

    # For Debug
    # print('target lengths', target_lengths)

    loss = masked_cross_entropy.compute_loss(
        output.transpose(0, 1).contiguous(),
        target_batches.transpose(0, 1).contiguous(),
        target_lengths
    )
    print('loss = ', loss.item())

    clip_grad_norm_(net.parameters(), grad_clip)
    loss.backward()
    opt.step()

    save_checkpoint(net, cfg, step)