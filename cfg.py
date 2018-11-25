#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:Darksoul
# datetime:11/24/2018 22:13
# software: PyCharm

class configuration:
    def __init__(self):
        self.PAD_idx = 0
        self.SOS_idx = 0
        self.EOS_idx = 2
        self.UNK_idx = 3
        self.USE_CUDA = True
        self.MIN_LENGTH = 5
        self.MAX_LENGTH = 10
        self.eng_prefixes = (
                                "i am ", "i m ",
                                "he is", "he s ",
                                "she is", "she s",
                                "you are", "you re ",
                                "we are", "we re ",
                                "they are", "they re "
                            )