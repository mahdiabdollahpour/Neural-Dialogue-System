import numpy as np
import tensorflow as tf
import os


def convert_to_one_hot(data_, vocab):
    data = np.zeros((len(data_), len(vocab)))
    cnt = 0
    for s in data_:
        v = [0.0] * len(vocab)
        v[vocab.index(s)] = 1.0
        data[cnt, :] = v
        cnt += 1
    return data

def get_one_hot(idx, vocab):
    vec = np.zeros(len(vocab))
    vec[idx] = 1
    return vec


def data_by_ID(data, vocab):
    data_ = []
    for token in data:
        data_.append(vocab.index(token))
    return data_


def decode_embed(array, vocab):
    return vocab[array.index(1)]


import re


def load_data(input):
    # Load the data
    data_ = ""
    with open(input, 'r') as f:
        line = f.read().lower()
        line = line.replace("'", " ' ")
        line = line.replace("-", " - ")
        data_ += re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", line)
    # data_ = data_.lower()
    data_ = data_.split()
    # Convert to 1-hot coding
    vocab = sorted(list(set(data_)))
    # data = convert_to_one_hot(data_, vocab)
    return data_, vocab
