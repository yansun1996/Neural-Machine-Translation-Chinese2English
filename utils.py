#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:Darksoul
# datetime:11/24/2018 22:02
# software: PyCharm
import unicodedata
import re
import math
from cfg import *
# import pytorch
import torch
from torch.autograd import Variable


cfg = configuration()

PAD_idx = cfg.PAD_idx
SOS_idx = cfg.SOS_idx
EOS_idx = cfg.EOS_idx
UNK_idx = cfg.UNK_idx
USE_CUDA = cfg.USE_CUDA
MIN_LENGTH = cfg.MIN_LENGTH
MAX_LENGTH = cfg.MAX_LENGTH


class Preprocessor:
    '''
    class for preprocessing
    '''

    def __init__(self, name):
        '''
        initialize vocab and counter
        '''
        self.name = name
        self.w2idx = {"<sos>": 0, "<eos>": 1, "<unk>": 2, "<pad>": 3}
        self.counter = {}
        self.idx2w = {0: "<sos>", 1: "<eos>", 2: "<unk>", 3: "<pad>"}
        self.num = 4

    def SentenceAdder(self, sentence):
        '''
        Add a sentence to dataset
        '''
        for word in sentence.split(' '):
            self.WordAdder(word)

    def WordAdder(self, word):
        '''
        Add single word to dataset and update vocab and counter
        '''
        if word in self.w2idx:
            self.counter[word] += 1
        else:
            self.w2idx[word] = self.num
            self.counter[word] = 1
            self.idx2w[self.num] = word
            self.num += 1

    def trim(self, min_count=5):
        '''
        Trim to remove non-frequent word
        '''
        keep = []
        for k, v in self.counter.items():
            if v >= min_count: keep.append(k)
        print(self.name + ':')
        print('Total words', len(self.w2idx))
        print('After Trimming', len(keep))
        print('Keep Ratio %', 100 * len(keep) / len(self.w2idx))
        self.w2idx = {"<sos>": 0, "<eos>": 1, "<unk>": 2, "<pad>": 3}
        self.counter = {}
        self.idx2w = {0: "<sos>", 1: "<eos>", 2: "<unk>", 3: "<pad>"}
        self.num = 4
        for w in keep:
            self.WordAdder(w)


def Uni2Ascii(s):
    '''
    transfer from unicode to ascii
    '''
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                    if unicodedata.category(c) != 'Mn')

def StrCleaner(s):
    '''
    trim, delete non-letter and lowercase string
    '''
    s = Uni2Ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def DataReader(path, lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(path, encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    #pairs = [[StrCleaner(s) for s in l.split('<------>')] for l in lines]
    pairs = [[s.lower() for s in l.split('<------>')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Preprocessor(lang2)
        output_lang = Preprocessor(lang1)
    else:
        input_lang = Preprocessor(lang1)
        output_lang = Preprocessor(lang2)

    return input_lang, output_lang, pairs


def filterPair(p):
    '''
    Filter to get expected pairs with specific length
    '''
    return MIN_LENGTH <= len(p[0].split(' ')) <= MAX_LENGTH and \
        MIN_LENGTH <= len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(path, lang1, lang2, reverse=True):
    input_lang, output_lang, pairs = DataReader(path, lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.SentenceAdder(pair[0])
        output_lang.SentenceAdder(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.num)
    print(output_lang.name, output_lang.num)
    return input_lang, output_lang, pairs


def sentence2idx(preprocessor, sentence):
    '''
    Read sentence and translate into word index plus eos
    '''
    return [SOS_idx] + [preprocessor.w2idx[w] if w in preprocessor.w2idx \
            else UNK_idx for w in sentence.split(' ')] + [EOS_idx]


def pad(seq, max_len):
    '''
    Add padding to sentence with different length
    '''
    seq += [PAD_idx for i in range(max_len - len(seq))]
    return seq



def random_batch(src, tgt, pairs, batch_size, batch_index):
    '''
    Randomly generate batch data
    '''
    inputs, target = [], []

    # Choose batch randomly
    # for _ in range(batch_size):
    #     pair = random.choice(pairs)
    #     inputs.append(sentence2idx(src, pair[0]))
    #     target.append(sentence2idx(tgt, pair[1]))

    # Choose batch
    for s in pairs[batch_index*batch_size:(batch_index+1)*batch_size]:
        inputs.append(sentence2idx(src, s[0]))
        target.append(sentence2idx(tgt, s[1]))

    # Sort by length
    seq_pairs = sorted(zip(inputs, target), key=lambda p: len(p[0]), reverse=True)
    inputs, target = zip(*seq_pairs)

    # Obtain length of each sentence and pad
    input_lens = [len(s) for s in inputs]
    input_max = max(input_lens)
    input_padded = [pad(s, input_max) for s in inputs]
    target_lens = [len(s) for s in target]
    target_max = max(target_lens)
    target_padded = [pad(s, target_max) for s in target]

    # Create Variable
    if USE_CUDA:
        input_vars = Variable(torch.LongTensor(input_padded).cuda()).transpose(0, 1)
        input_lens = Variable(torch.LongTensor(input_lens).cuda())
        target_vars = Variable(torch.LongTensor(target_padded).cuda()).transpose(0, 1)
        target_lens = Variable(torch.LongTensor(target_lens).cuda())
    else:
        input_vars = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
        input_lens = Variable(torch.LongTensor(input_lens))
        target_vars = Variable(torch.LongTensor(target_padded)).transpose(0, 1)
        target_lens = Variable(torch.LongTensor(target_lens))

    return input_vars, input_lens, target_vars, target_lens


def modified_precision(candidate, references, n):
    count = 0
    match_num = 0
    len_c = len(candidate)

    for i in range(len_c):
        ref_temp = []
        for reference in references:
            for k in range(len(reference)):
                ref_sentence = reference[k]
                words_ref_sentence = ref_sentence.strip()
                words_ref_sentence = words_ref_sentence.split()
                num_max = len(words_ref_sentence) - n + 1
                ngram_temp = {}
                for j in range(num_max):
                    ngram = ' '.join(words_ref_sentence[j:j + n])
                    ngram = ngram.lower()
                    if ngram in ngram_temp.keys():
                        ngram_temp[ngram] += 1
                    else:
                        ngram_temp[ngram] = 1
                ref_temp.append(ngram_temp)

        cand_sentence = candidate[i]
        words_cand = cand_sentence.strip()
        words_cand = words_cand.split()
        num_max_cand = len(words_cand) - n + 1
        cand_temp = {}
        for j in range(num_max_cand):
            ngram = ' '.join(words_cand[j:j + n])
            ngram = ngram.lower()
            if ngram in cand_temp.keys():
                cand_temp[ngram] += 1
            else:
                cand_temp[ngram] = 1
        count += num_max_cand
        match_num += match_counts(ref_temp, cand_temp)
    if match_num != 0:
        p = 1. * match_num / count
    else:
        p = 0
    return p


def match_counts(ref_counts, cand_temp):
    num = 0
    for ngram in cand_temp.keys():
        count = cand_temp[ngram]
        max_ref = 0
        for ref in ref_counts:
            if ngram in ref:
                max_ref = max(max_ref, ref[ngram])
        count = min(max_ref, count)
        num = num + count
    return num


def brevity_penalty(candidate, references):
    len_c = len(candidate)
    r = 0
    c = 0
    for i in range(len_c):
        ref_lens = []
        for reference in references:
            for k in range(len(reference)):
                ref_sentence = reference[k]
                words_ref_sentence = ref_sentence.strip()
                words_ref_sentence = words_ref_sentence.split()
                ref_lens.append(len(words_ref_sentence))
        cand_sentence = candidate[i]
        words_cand = cand_sentence.strip().split()
        init_len_diff = abs(len(words_cand) - ref_lens[0])
        best = ref_lens[0]
        for num in ref_lens:
            if (abs(len(words_cand) - num)) < init_len_diff:
                init_len_diff = abs(len(words_cand) - num)
                best = num
        r = r + best
        c = c + len(words_cand)
    if c > r:
        bp = 1
    else:
        bp = math.exp(1 - 1. * r / c)
    return bp


def bleu(candidate, references, n):
    # n is the maximum length of each ngram you set
    weight = 1. / n
    temp = 0
    for i in range(n):
        p = modified_precision(candidate, references, i + 1)
        temp += math.log(p) * weight
    temp = math.exp(temp)
    return brevity_penalty(candidate, references) * temp