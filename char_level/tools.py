import numpy as np

# <start> = 0, <end> = 1, <unk> = 2
# main characters: 3-127 = 3-127
# --persian characters: (char code) 1560-1751 <-> (id) 128-319
START = 0
END = 1
UNK = 2
SIZE_OF_VOCAB = 128


def map_char_to_id(c):
    """

    :return: a number between 0 to 127
    """
    code = ord(c)
    if 2 < code < 128:
        return code
    # if 1560 <= code <= 1751:
    #     return code - 1432
    return UNK


def map_id_to_char(code):
    """

    :param code: a number between 0 to SIZE_OF_VOCAB-1
    """
    if code == START: return '<START>'
    if code == END: return '<END>'
    if code == UNK: return '<UNK>'
    if code < 128: return chr(code)
    # if 128 <= code <= 319: return chr(code + 1432)
    return '<?>'


def convert_to_one_hot(sentence, char_num):
    """

    :param char_num: char_num_of_sentence-1
    :return: it's the one-hot array of the sentence without start and end token
    """
    s_vector = np.zeros((char_num, SIZE_OF_VOCAB))
    for char_iter, char_c in enumerate(sentence):
        v = get_char_vector(char_c)
        s_vector[char_iter] = v

    return s_vector


def get_char_vector(char):
    v = [0.0] * SIZE_OF_VOCAB
    v[map_char_to_id(char)] = 1.0
    return v


def get_one_hot_vector(one_id, size=SIZE_OF_VOCAB):
    v = [0.0] * size
    v[one_id] = 1.0
    return v


def load_data(input, char_num_of_sentence):
    """

    :param input: data file address
    :param char_num_of_sentence: fix char size of any sentences
    :return: a numpy array with the size (len(lines), char_num_of_sentence-1, SIZE_OF_VOCAB)
    """
    # Load the data
    with open(input, 'r') as f:
        lines = f.readlines()
    data = np.zeros((len(lines), char_num_of_sentence-1, SIZE_OF_VOCAB))
    for line_id, line in enumerate(lines):
        line = line.lower()[0:char_num_of_sentence-1]
        line += ' ' * (char_num_of_sentence-1 - len(line))

        # Convert to 1-hot coding
        data[line_id] = convert_to_one_hot(line, char_num_of_sentence-1)
    return data
