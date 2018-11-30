#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:Darksoul
# datetime:11/24/2018 22:02
# software: PyCharm

import os
from network import *

from torch import optim
from torch.nn.utils import clip_grad_norm_

# import loss func
import masked_cross_entropy


def nmt_training(src, tgt, pairs):
    encoder_test = Encoder(src.num, cfg.embed_size, cfg.hidden_size, cfg.n_layers, dropout=0.5)
    decoder_test = Decoder(cfg.embed_size, cfg.hidden_size, tgt.num, cfg.n_layers, dropout=0.5)

    net = Seq2Seq(encoder_test,decoder_test).cuda()
    load_checkpoint(net, cfg)

    opt = optim.Adam(net.parameters(),lr=0.01)
    # print(net)

    for step in range(1, cfg.iteration):
        total_loss = 0
        input_batches, input_lengths, \
        target_batches, target_lengths = random_batch(src, tgt, pairs, cfg.batch_size)
        opt.zero_grad()
        output = net(input_batches, input_lengths, target_batches, target_lengths)

        # For Debug
        # print('target lengths', target_lengths)

        loss = masked_cross_entropy.compute_loss(
            output.transpose(0, 1).contiguous(),
            target_batches.transpose(0, 1).contiguous(),
            target_lengths
        )

        clip_grad_norm_(net.parameters(), cfg.grad_clip)
        loss.backward()
        opt.step()


        if step % cfg.save_iteration == 0:
            print('loss = ', loss.item())
            save_checkpoint(net, cfg, step)
            # print(output)
            # TODO apply decoder to the output

def nmt_testing(sec, tgt, pairs):
    # TODO finish testing
    pass

if __name__ == '__main__':
    if not os.path.exists(cfg.checkpoints_path):
        os.mkdir(cfg.checkpoints_path)

    src, tgt, pairs = prepareData('eng', 'fra', True)
    src.trim()
    tgt.trim()

    if cfg.is_training:
        nmt_training(src, tgt, pairs)
    else:
        nmt_testing(src, tgt, pairs)

