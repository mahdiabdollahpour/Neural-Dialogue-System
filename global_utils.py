import tensorflow as tf
import os
import pandas as pd
import nltk
import numpy as np
from parametrs import *
from collections import Counter
import re


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


def check_restore_parameters(sess, saver, path):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(path))
    if ckpt and ckpt.model_checkpoint_path:
        print('Restoring parameters')
        saver.restore(sess, ckpt.model_checkpoint_path)



def load_data_from_csv(data_path):
    df2 = pd.read_csv(data_path)
    cols2 = df2.columns
    data = []
    for col_name in cols2[1:]:
        data.append(df2[col_name].tolist())
    return data


def load_vocab_from_csv(vocab_path):
    df = pd.read_csv(vocab_path)
    cols = df.columns
    dict = {}
    dict_rev = {}
    for i, token in enumerate(cols[1:]):
        # print(df[token][0] , ' --- ', token)
        # if(df[token][0] == 15575):
        #     exit()
        # print('token',token)
        dict[df[token][0]] = token
        dict_rev[token] = df[token][0]
    return dict, dict_rev


def pad_sentence(sen, length):
    sen_ = [start_token] + sen.split()
    sen_ = sen_[:min(length, len(sen_)) - 1]
    for i in range(len(sen_), length):
        sen_.append(end_token)
    return sen_


def get_sentence_back(sen, vocab):
    sent = ""
    for token in sen:
        # print(token)
        # print(token)
        sent += vocab[token + 1] + " "
        if vocab[token+1]==end_token:
            return sent
    return sent



def BLEU_score(ref_ans, output_ans):
    return nltk.translate.bleu_score.sentence_bleu([ref_ans], output_ans)


def get_one_hot(idx, vocab_size):
    # print('idx ', idx)
    vec = np.zeros(vocab_size)
    vec[idx] = 1
    return vec


# def get_start_token_index(dict_rev):
#     return [dict_rev[start_token]]


def get_token_index(dict_rev, token):
    return [dict_rev[token]]


def sentence_by_id(sen, dic_rev):
    li =[]
    for token in sen:
        if token in dic_rev:
            li.append(dic_rev[token])
        else:
            li.append(dic_rev[unknown_token])
    return li
