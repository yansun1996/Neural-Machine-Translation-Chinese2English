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
    num_batch = len(pairs) // cfg.batch_size

    encoder_test = Encoder(src.num, cfg.embed_size, cfg.hidden_size, cfg.n_layers_encoder, dropout=cfg.dropout)
    decoder_test = Decoder(cfg.embed_size, cfg.hidden_size, tgt.num, cfg.n_layers_decoder, dropout=cfg.dropout)

    net = Seq2Seq(encoder_test,decoder_test).cuda()
    load_checkpoint(net, cfg)

    opt = optim.Adam(net.parameters(), cfg.lr)
    # print(net)

    if isinstance(cfg.load_checkpoint, str):
        ckt_step = int(cfg.load_checkpoint.split('_')[0])
        ckt_idx = int(cfg.load_checkpoint.split('_')[1])
    else:
        ckt_step = 0
        ckt_idx = 0
    for step in range(1, cfg.iteration):
        total_loss = 0
        tmp_loss = 0

        for batch_index in range(num_batch):
            input_batches, input_lengths, \
            target_batches, target_lengths = random_batch(src, tgt, pairs, cfg.batch_size, batch_index)
            opt.zero_grad()
            output = net(input_batches, input_lengths, target_batches, target_lengths)

            # For Debug
            # print('target lengths', target_lengths)

            # mask loss
            if cfg.loss_type == 'mask':
                loss = masked_cross_entropy.compute_loss(
                    output.transpose(0, 1).contiguous(),
                    target_batches.transpose(0, 1).contiguous(),
                    target_lengths
                )
            else:
                loss = F.nll_loss(output[1:].view(-1, tgt.num),
                                  target_batches[1:].contiguous().view(-1),
                                  ignore_index=cfg.PAD_idx)

            tmp_loss += loss.item()

            if (batch_index + 1 + ckt_idx) % cfg.save_iteration == 0:
                print("Epoch: {}, Batch Num: {}, Loss: {}".format(str(max(step,ckt_step)), batch_index+1+ckt_idx, tmp_loss/cfg.save_iteration))
                tmp_loss = 0
                save_checkpoint(net, cfg, str(max(step,ckt_step)), batch_index+1+ckt_idx)

                _, pred = net.inference(input_batches[:, 1].reshape(input_lengths[0].item(), 1),
                                        input_lengths[0].reshape(1))

                print(' '.join([tgt.idx2w[t] for t in pred]))


            clip_grad_norm_(net.parameters(), cfg.grad_clip)
            loss.backward()
            opt.step()

        print("Epoch {} finished".format(str(step)))
        random.shuffle(pairs)



def nmt_testing(sec, tgt, pairs):
    # TODO finish testing
    pass

if __name__ == '__main__':
    if not os.path.exists(cfg.checkpoints_path):
        os.mkdir(cfg.checkpoints_path)

    src, tgt, pairs = prepareData(cfg.data_path, 'english', 'chinese')
    src.trim()
    tgt.trim()

    if cfg.is_training:
        nmt_training(src, tgt, pairs)
    else:
        nmt_testing(src, tgt, pairs)

