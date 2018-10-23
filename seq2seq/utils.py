import numpy as np
from global_utils import *
# unknown_token = '<unk>'
# empty_token = '<EMP>'
# start_token = '<s>'
# end_token = '</s>'
import re


def get_one_hot(idx, vocab_size):
    # print('idx ', idx)
    vec = np.zeros(vocab_size)
    vec[idx] = 1
    return vec


def get_start_token_index(dict_rev):
    return [dict_rev[start_token]]


def sentence_by_id(sen, dic_rev):
    li = []
    for token in sen:
        if token in dic_rev:
            li.append(dic_rev[token])
        else:
            li.append(dic_rev[unknown_token])
    return li

