import tensorflow as tf
import os
import pandas as pd

unknown_token = '<unk>'
empty_token = '<EMP>'
start_token = '<s>'
end_token = '</s>'
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


def create_vocab_and_data_file(src, vocab_des, data_des, vocab_len, trunc_length, src2=None, data_des2=None,
                               trunc_length2=None):
    data_ = []
    vocab_list = []
    lengthes = 0
    # text = ""
    with open(src, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines[:300]:
        line = line.lower()
        line = line.replace("'", " ' ")
        line = line.replace("-", " - ")
        line = line.replace("(", " ( ")
        line = line.replace(")", " ) ")
        line = line.replace(".", " . ")
        line = line.replace(":", " : ")
        line = line.replace(";", " ; ")
        line = line.replace("]", " ] ")
        line = line.replace("]", " ] ")
        line = line.replace(",", " , ")
        line = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", line)

        line = line.split()
        # data_.append(line)
        vocab_list += line
        lengthes += len(line)
        data_.append(line)
    data_2 = []
    print('Average line length :', lengthes / len(lines))
    lengthes2 = 0
    if src2 is not None:
        with open(src2, 'r', encoding='utf-8') as f2:
            lines = f2.readlines()
        for line in lines[:300]:
            line = line.lower()
            line = line.replace("'", " ' ")
            line = line.replace("-", " - ")
            line = line.replace("(", " ( ")
            line = line.replace(")", " ) ")
            line = line.replace(".", " . ")
            line = line.replace(":", " : ")
            line = line.replace(";", " ; ")
            line = line.replace("]", " ] ")
            line = line.replace("]", " ] ")
            line = line.replace(",", " , ")
            line = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", line)

            line = line.split()
            # data_2.append(line)
            vocab_list += line
            lengthes2 += len(line)
            data_2.append(line)
    print('Average line length :', lengthes2 / len(lines))

    # vocab_list = list(set(vocab_list))
    counter = Counter(vocab_list)
    most_occure = counter.most_common(vocab_len)
    # dict = {}
    most_occure.append((start_token, 1))
    most_occure.append((unknown_token, 1))
    most_occure.append((end_token, 1))
    # vocab_list.append(empty_token)
    dict_rev = {}
    counter = 0
    for (token, i) in most_occure:
        # dict[i] = token
        dict_rev[token] = counter
        counter += 1
    df = pd.DataFrame(dict_rev, index=[0])
    # print(df)
    df.to_csv(vocab_des)

    data_by_index = {}
    unk_idx = dict_rev[unknown_token]
    end_idx = dict_rev[end_token]
    start_idx = dict_rev[start_token]

    for idx, seq in enumerate(data_):
        seq2 = []
        seq2.append(start_idx)

        for token in seq:
            if len(seq2) == trunc_length - 1:
                seq2.append(end_idx)
                break
            if token in dict_rev:
                seq2.append(dict_rev[token])
            else:
                seq2.append(unk_idx)
        while len(seq2) < trunc_length:
            seq2.append(end_idx)
        data_by_index['sen' + str(idx)] = seq2
    df2 = pd.DataFrame.from_dict(data_by_index)
    data_by_index2 = {}
    df2.to_csv(data_des)
    if data_des2 is not None:
        for idx, seq in enumerate(data_2):
            seq2 = []
            seq2.append(start_idx)

            for token in seq:
                if len(seq2) == trunc_length2 - 1:
                    seq2.append(end_idx)
                    break
                if token in dict_rev:
                    seq2.append(dict_rev[token])
                else:
                    seq2.append(unk_idx)
            while len(seq2) < trunc_length2:
                seq2.append(end_idx)
            data_by_index2['sen' + str(idx)] = seq2
        df3 = pd.DataFrame.from_dict(data_by_index2)
        df3.to_csv(data_des2)


def load_vocab_and_data_from_csv(vocab_path, data_path):
    df = pd.read_csv(vocab_path)
    df2 = pd.read_csv(data_path)

    cols = df.columns
    cols2 = df2.columns
    # for tok in cols2[:10]:
    #     print(tok)
    # print('first')
    # for token in df2[cols2[13]]:
    #     print(token)
    # # print('second', df2[cols2[1]])
    dict = {}
    dict_rev = {}
    # token_num = {}
    data = []
    for col_name in cols2[1:]:
        data.append(df2[col_name].tolist())
    # print(data[0])
    # print(data[1])
    # print(data[2])
    for i, token in enumerate(cols):
        dict[df[token][0]] = token
        dict_rev[token] = df[token][0]
    # print('end idx', dict_rev[end_token])
    # print('unknown idx', dict_rev[unknown_token])
    return dict, dict_rev, data


# create_vocab_and_data_file('datasets/it-en/en.txt', 'vocab-en.csv', 'data-en.csv', 100, 30)
# load_vocab_and_data_from_csv('vocab-en.csv', 'data-en.csv')


def load_SQuAD(src, Q_des, A_des):
    import json
    with open(src, encoding='utf-8') as f:
        data = f.read()
        print(len(data))
        j = json.loads(data)
    quesion_list = []
    answer_list = []
    j = j['data']
    for dat in j:
        for da in dat['paragraphs']:
            for dd in da['qas']:
                for ans in dd['answers']:
                    quesion_list.append(dd['question'])
                    answer_list.append(ans['text'])
    print(len(quesion_list))
    with open(Q_des, encoding='utf-8', mode='w') as f2:
        for line in quesion_list:
            f2.write(line + '\n')
    with open(A_des, encoding='utf-8', mode='w') as f3:
        for line in answer_list:
            f3.write(line + '\n')


# load_SQuAD('datasets/SQuAD/train-v2.0.json', 'datasets/SQuAD/train_Q.txt','datasets/SQuAD/train_A.txt')
def load_DailyDialog(src, Q_des, A_des):
    with open(src, mode='r', encoding='utf-8') as f:
        data = f.read()
        data = data.split('__eou__')
    quesion_list = []
    answer_list = []
    for i in range(len(data) - 1):
        line = data[i]
        if line.__contains__('?'):
            quesion_list.append(line.strip())
            answer_list.append(data[i + 1].strip())
    print(len(quesion_list))
    print(len(answer_list))
    with open(Q_des, encoding='utf-8', mode='w') as f2:
        for line in quesion_list:
            f2.write(line + '\n')
    with open(A_des, encoding='utf-8', mode='w') as f3:
        for line in answer_list:
            f3.write(line + '\n')

#
# load_DailyDialog('datasets/DailyDialog/train/dialogues_train.txt', 'datasets/DailyDialog/train/Q_train.txt',
#                  'datasets/DailyDialog/train/A_train.txt')
