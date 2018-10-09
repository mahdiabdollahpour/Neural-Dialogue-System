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
def get_start_token_index(vocab):
    return [vocab.index(start_token)]

def sentence_by_id(sen, voacb):
    li = []
    for token in sen:
        if token in voacb:
            li.append(voacb.index(token))
        else:
            li.append(voacb.index(unknown_token))
    return li

def prepend_start(seq, vocab):
    return [vocab.index(start_token)] + seq


def append_end(seq, vocab):
    return [vocab.index(start_token)] + seq


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


def data_by_ID_and_truncated(data, vocab, time_steps, append_and_prepend=False):
    unk_idx = vocab.index(unknown_token)
    end_idx = vocab.index(end_token)
    start_idx = vocab.index(start_token)

    data_ = []
    for sen in data:
        sen2 = []
        if append_and_prepend:
            sen2.append(start_idx)
        flag = append_and_prepend
        for i in range(time_steps):
            if i < len(sen):
                if sen[i] in vocab:
                    sen2.append(vocab.index(sen[i]))
                elif (not flag):
                    sen2.append(unk_idx)
                else:
                    flag = False
                    sen2.append(end_idx)
            else:
                sen2.append(vocab.index(empty_token))

        data_.append(sen2)
    return data_
