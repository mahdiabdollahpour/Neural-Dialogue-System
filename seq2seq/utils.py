import numpy as np

unknown_token = '<unk>'
empty_token = '<EMP>'
start_token = '<s>'
end_token = '</s>'
import re


def get_batch(seq2, vocab2):
    tgt_inp = [vocab2.index(start_token)].append(seq2)
    tgt_out = seq2.append([vocab2.index(start_token)])

    return tgt_inp, tgt_out


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


def data_by_ID_and_truncated(data, vocab, time_steps):
    unk_idx = vocab.index(unknown_token)

    data_ = []
    for sen in data:
        sen2 = []
        for i in range(time_steps):
            if i < len(sen):
                if sen[i] in vocab:
                    sen2.append(vocab.index(sen[i]))
                else:
                    sen2.append(unk_idx)
            else:
                sen2.append(vocab.index(empty_token))

        data_.append(sen2)
    return data_


def batch(inputs, max_sequence_length=None):
    """
    Args:
        inputs:
            list of sentences (integer lists)
        max_sequence_length:
            integer specifying how large should `max_time` dimension be.
            If None, maximum sequence length would be used

    Outputs:
        inputs_time_major:
            input sentences transformed into time-major matrix
            (shape [max_time, batch_size]) padded with 0s
        sequence_lengths:
            batch-sized list of integers specifying amount of active
            time steps in each input sequence
    """

    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)

    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)

    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)  # == PAD

    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    # [batch_size, max_time] -> [max_time, batch_size]
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_time_major, sequence_lengths


def random_sequences(length_from, length_to,
                     vocab_lower, vocab_upper,
                     batch_size):
    """ Generates batches of random integer sequences,
        sequence length in [length_from, length_to],
        vocabulary in [vocab_lower, vocab_upper]
    """
    if length_from > length_to:
        raise ValueError('length_from > length_to')

    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)

    while True:
        yield [
            np.random.randint(low=vocab_lower,
                              high=vocab_upper,
                              size=random_length()).tolist()
            for _ in range(batch_size)
        ]
