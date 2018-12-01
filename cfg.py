#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:Darksoul
# datetime:11/24/2018 22:13
# software: PyCharm

class configuration:
    def __init__(self):

        self.is_training = True
        self.USE_CUDA = True

        self.batch_size = 128
        self.hidden_size = 512
        self.embed_size = 256
        self.lr = 0.001
        self.n_layers_encoder = 3
        self.n_layers_decoder = 3
        self.grad_clip = 10
        self.loss_type = 'mask'
        self.dropout = 0.2
        self.iteration = 100000
        self.save_iteration = 200

        # self.load_checkpoint = 0
        self.load_checkpoint = '1_140'

        self.SOS_idx = 0
        self.EOS_idx = 1
        self.UNK_idx = 2
        self.PAD_idx = 3

        self.MIN_LENGTH = 10
        self.MAX_LENGTH = 50

        self.checkpoints_path = './checkpoints_small_data'

        self.data_path = 'data/small_set/train.txt'


