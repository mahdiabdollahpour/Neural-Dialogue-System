from global_utils import *

from parametrs import DialogQAParams as dqap, DialyDialogParams as ddp


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


def create_vocab_file(src, vocab_des, max_vocab_len, src2=None):
    vocab_list = []
    lengthes = 0
    with open(src, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
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

    print('Average line length :', lengthes / len(lines))
    lengthes2 = 0
    if src2 is not None:
        with open(src2, 'r', encoding='utf-8') as f2:
            lines = f2.readlines()
        for line in lines:
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

    print('Average line length :', lengthes2 / len(lines))

    # vocab_list = list(set(vocab_list))
    counter = Counter(vocab_list)
    most_occure = counter.most_common(max_vocab_len)
    # dict = {}
    most_occure.append((start_token, 1))
    most_occure.append((unknown_token, 1))
    most_occure.append((end_token, 1))
    # vocab_list.append(empty_token)
    dict_rev = {}
    print('Vocab length :', len(most_occure))
    counter = 0
    for (token, i) in most_occure:
        dict_rev[token] = counter
        counter += 1
    df = pd.DataFrame(dict_rev, index=[0])
    print("Vocab created, size :", len(dict_rev))
    df.to_csv(vocab_des)


def create_data_file(src, dict_rev, data_des, trunc_length):
    data_ = []
    vocab_list = []
    lengthes = 0
    # text = ""
    with open(src, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
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
    df2.to_csv(data_des)


def preprocess_DailyDialog():
    print('preprocessing Data...')
    load_DailyDialog(ddp.dataset_path + '/test/dialogues_test.txt', ddp.test_questions, ddp.test_answers)
    load_DailyDialog(ddp.dataset_path + '/train/dialogues_train.txt', ddp.train_questions, ddp.train_answers)
    load_DailyDialog(ddp.dataset_path + '/validation/dialogues_validation.txt', ddp.validation_questions,
                     ddp.validation_answers)


def Diaalog_QA():
    print('preprocessing Data...')
    create_vocab_file(dqap.all_questions, dqap.vocab_path, 1000000)
    vocab, dict_rev = load_vocab_from_csv(dqap.vocab_path)
    # print()
    create_data_file(dqap.all_questions, dict_rev, dqap.all_questions_csv, dqap.source_sequence_length)
    create_data_file(dqap.all_answers, dict_rev, dqap.all_answers_csv, dqap.decoder_length)


def preprocess():
    print('preprocessing Data...')

    create_vocab_file(src=ddp.all_questions, vocab_des=ddp.vocab_path, max_vocab_len=20000, src2=ddp.all_answers)
    vocab, dict_rev = load_vocab_from_csv(ddp.vocab_path)
    ## validation
    create_data_file(src=ddp.validation_questions,
                     dict_rev=dict_rev, data_des=ddp.validation_questions_csv,
                     trunc_length=ddp.source_sequence_length + 2)

    create_data_file(src=ddp.validation_answers,
                     dict_rev=dict_rev, data_des=ddp.validation_answers_csv,
                     trunc_length=ddp.decoder_length + 1)

    ## test
    create_data_file(src=ddp.test_questions,
                     dict_rev=dict_rev, data_des=ddp.test_questions_csv,
                     trunc_length=ddp.source_sequence_length + 2)

    create_data_file(src=ddp.test_answers,
                     dict_rev=dict_rev, data_des=ddp.test_answers_csv,
                     trunc_length=ddp.decoder_length + 1)
    ##train
    create_data_file(src=ddp.train_questions,
                     dict_rev=dict_rev, data_des=ddp.train_questions_csv,
                     trunc_length=ddp.source_sequence_length + 2)

    create_data_file(src=ddp.train_answers,
                     dict_rev=dict_rev, data_des=ddp.train_answers_csv,
                     trunc_length=ddp.decoder_length + 1)


def generate_data_from_clusters(cluster_add, src_add, dest_add):
    src = []
    dest = []

    for f in os.listdir(cluster_add):
        print(f)
        with open(cluster_add + '/' + f, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for i in range(len(lines)):
            if (lines[i] not in src):
                for j in range(i, len(lines)):
                    src.append(lines[i])
                    dest.append(lines[j])
    print("Writing Files...")

    with open(src_add, 'w', encoding='utf-8') as f:
        for line in src:
            f.write(line)
    with open(dest_add, 'w', encoding='utf-8') as f:
        for line in dest:
            f.write(line)
