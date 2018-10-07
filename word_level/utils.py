import numpy as np

unknown_token = "<UNK>"
empty_token = "<UNK>"
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


def data_by_ID_and_truncated(data, vocab, time_steps):
    data_ = []
    for sen in data:
        sen2 = []
        for i in range(time_steps + 1):
            if i < len(sen):
                sen2.append(vocab.index(sen[i]))
            else:
                sen2.append(vocab.index(empty_token))

        data_.append(sen2)
    return data_


def decode_embed(array, vocab):
    return vocab[array.index(1)]


import re


def load_data(input):
    # Load the data
    data_ = []
    vocab = []
    lengthes = 0
    with open(input, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.lower()
        line = line.replace("'", " ' ")
        line = line.replace("-", " - ")
        line = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", line)

        line = line.split()
        # print(line)
        lengthes += len(line)
        data_.append(line)
        for x in line:
            vocab.append(x)
    # print('data , ', data_[0])
    vocab.append(unknown_token)
    vocab.append(empty_token)
    print('average line length :', lengthes / len(lines))
    # print('vocab , ', vocab[0])
    vocabb = sorted(list(set(vocab)))
    # vocab = sorted(list(set(x for x in data_)))

    return data_, vocabb


def get_trainable_variables_num():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    return total_parameters