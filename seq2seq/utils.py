import numpy as np

unknown_token = '<unk>'
empty_token = '<EMP>'
start_token = '<s>'
end_token = '</s>'
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


# def prepend_start(seq, vocab):
#     return [vocab.index(start_token)] + seq
#
#
# def append_end(seq, vocab):
#     return [vocab.index(start_token)] + seq


def load_data(text_addr, vocab_addr):
    # Load the data
    data_ = []
    vocab = []
    lengthes = 0
    with open(text_addr, 'r', encoding='utf-8') as f:
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
    print('average line length :', lengthes / len(lines))

    vocab.append(empty_token)
    with open(vocab_addr, 'r', encoding='utf-8') as vf:
        lines = vf.readlines()
    for line in lines:
        vocab.append(line.split()[0])
    # vocab = set(vocab)
    return data_, vocab


def load_data_and_create_vocab(text_addr, line_num):
    # Load the data

    data_ = []
    vocab = []
    lengthes = 0
    with open(text_addr, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        cnt = 0
    for line in lines:
        cnt += 1
        if cnt > line_num:
            break
        line = line.lower()
        line = line.replace("'", " ' ")
        line = line.replace("-", " - ")
        line = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", line)

        line = line.split()
        # print(line)
        vocab += line
        lengthes += len(line)
        data_.append(line)
    print('average line length :', lengthes / line_num)
    vocab.append(start_token)
    vocab.append(unknown_token)
    vocab.append(end_token)
    vocab.append(empty_token)
    vocab = list(set(vocab))
    dict = {}
    dict_rev = {}
    for i, token in enumerate(vocab):
        dict[i] = token
        dict_rev[token] = i
    # vocab = set(vocab)
    return data_, dict, dict_rev


def data_by_ID_and_truncated(data, dict_rev, time_steps, append_and_prepend=False):
    unk_idx = dict_rev[unknown_token]
    end_idx = dict_rev[end_token]
    start_idx = dict_rev[start_token]

    data_ = []
    for sen in data:
        sen2 = []
        if append_and_prepend:
            sen2.append(start_idx)
        flag = append_and_prepend
        for i in range(time_steps):
            if i < len(sen):
                if sen[i] in dict_rev:
                    sen2.append(dict_rev[sen[i]])
                elif (not flag):
                    sen2.append(unk_idx)
                else:
                    flag = False
                    sen2.append(end_idx)
            else:
                sen2.append(dict_rev[empty_token])

        data_.append(sen2)
    return data_
