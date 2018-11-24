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


def decode_embed(array, vocab):
    return vocab[array.index(1)]


def load_data(input):
    # Load the data
    data_ = []
    with open(input, 'r') as f:
        lines = f.readlines()
    for line in lines:
        data_.append(line.lower())

    # data_ = data_.lower()
    # Convert to 1-hot coding
    vocab = sorted(list(set(data_)))
    data = convert_to_one_hot(data_, vocab)
    return data, vocab

