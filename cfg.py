#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:Darksoul
# datetime:11/24/2018 22:13
# software: PyCharm

class configuration:
    def __init__(self):

        self.is_training = True
        self.USE_CUDA = True
        self.beam_search = True

        self.batch_size = 128
        self.hidden_size = 64
        self.embed_size = 64
        self.lr = 0.0001
        self.n_layers_encoder = 2
        self.n_layers_decoder = 2
        self.grad_clip = 5
        self.loss_type = 'mask'
        self.dropout = 0.5
        self.teacher_forcing_ratio = 0.5
        self.beam_widths = [1,2]
        self.iteration = 5
        self.save_iteration = 1

        self.load_checkpoint = 0

        self.SOS_idx = 0
        self.EOS_idx = 1
        self.UNK_idx = 2
        self.PAD_idx = 3

        self.MIN_LENGTH = 1
        self.MAX_LENGTH = 50

        self.checkpoints_path = './checkpoints_small_data'
        self.data_path = 'data/small_set/train.txt'
