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
    net = load_checkpoint(net, cfg)

    opt = optim.Adam(net.parameters(), cfg.lr)

    total_loss = []

    for step in range(1, cfg.iteration-cfg.load_checkpoint):
        tmp_loss = 0

        for batch_index in range(num_batch):
            input_batches, input_lengths, \
            target_batches, target_lengths = random_batch(src, tgt, pairs, cfg.batch_size, batch_index)
            opt.zero_grad()
            output = net(input_batches, input_lengths, target_batches, target_lengths)

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

            total_loss.append(loss.item())

            clip_grad_norm_(net.parameters(), cfg.grad_clip)
            loss.backward()
            opt.step()

        if (step + cfg.load_checkpoint) % cfg.save_iteration == 0:

            with open('./loss_log.txt', 'w') as outfile:
                for item in total_loss:
                    outfile.write("%s\n" % item)

            print("Epoch: {}, Loss: {}".format(step + cfg.load_checkpoint, tmp_loss/num_batch))
            save_checkpoint(net, cfg, step + cfg.load_checkpoint)

            _, pred = net.inference(input_batches[:, 1].reshape(input_lengths[0].item(), 1),
                                    input_lengths[0].reshape(1))

            try:
                pred = ' '.join([tgt.idx2w[t] for t in pred])
                gt = ' '.join([tgt.idx2w[t.item()] for t in input_batches[:, 1]])
                print("Ground Truth: {}".format(gt))
                print("Prediction: {}".format(pred))
                # print(bleu([pred], [gt], 2))
                # print(' '.join([tgt.idx2w[t] for t in pred]))
                # print(' '.join([tgt.idx2w[t.item()] for t in input_batches[:, 1]]))

            except Exception as e:
                print(e)

        # print("Epoch {} finished".format(str(step)))
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

