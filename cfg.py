#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:Darksoul
# datetime:11/24/2018 22:13
# software: PyCharm

class configuration:
    def __init__(self):

        self.is_training = True

        self.batch_size = 128
        self.hidden_size = 512
        self.embed_size = 256
        self.lr = 0.0001
        self.n_layers = 3
        self.grad_clip = 10

        self.PAD_idx = 0
        self.SOS_idx = 0
        self.EOS_idx = 2
        self.UNK_idx = 3
        self.USE_CUDA = True
        self.MIN_LENGTH = 10
        self.MAX_LENGTH = 50
        self.eng_prefixes = (
                                "i am ", "i m ",
                                "he is", "he s ",
                                "she is", "she s",
                                "you are", "you re ",
                                "we are", "we re ",
                                "they are", "they re "
                            )

        self.checkpoints_path = './checkpoints_small_data'
        self.iteration = 100000
        self.save_iteration = 500
        self.load_checkpoint = 0
        self.data_path = 'data/small_set/train.txt'

