import tensorflow as tf
import numpy as np
unknown = '<UNK>'
start_token = '<SOS>'
end_token = '<EOS>'




batch_size = 500
num_steps = 1
lr = 0.001
report_each = 10
class Config():
    INPUT_SIZE = 0
    OUTPUT_SIZE = 0
    USE_LSTM = True
    hidden_size = 20


class Vocab():

    def __init__(self, listt):
        words = set(listt)
        words.add(unknown)
        self.list2 = list(words)
        self.size = len(words)

    def get_one_hot(self, word):
        vec = np.zeros(self.size)
        if word in self.list2:
            vec[self.list2.index(word)] = 1
        else:
            vec[self.list2.index(unknown)] = 1

        assert sum(vec) == 1
        return vec

    def get_idx(self, word):
        if word in self.list2:
            return self.list2.index(word)
        else:
            return self.list2.index(unknown)

    def get_word(self, idx):
        return self.list2[idx]


class NetworkModel():

    def __init__(self, RNN_HIDDEN, data):
        self.data = data
        self.vocab = Vocab(self.data)
        num_features = self.vocab.size

        Config.INPUT_SIZE = num_features  # 2 bits per timestep

        Config.OUTPUT_SIZE = num_features  # 1 bit per timestep
        TINY = 1e-6  # to avoid NaNs in logs
        self.inputs = tf.placeholder(tf.float32, (None, None, Config.INPUT_SIZE), name='input')  # (time, batch, in)
        self.outputs = tf.placeholder(tf.int64, (None, Config.OUTPUT_SIZE), name='labels')  # (time, batch, out)
        # self.outputs = tf.placeholder(tf.int64, (None, Config.OUTPUT_SIZE + 1), name='labels')  # (time, batch, out)
        self.init_state = tf.placeholder(tf.float32, (None, 2 * RNN_HIDDEN), name='init_value')

        if Config.USE_LSTM:
            self.cell = tf.nn.rnn_cell.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=False, activation=tf.nn.relu,
                                                     name='LSTM_CELL')
        else:
            self.cell = tf.nn.rnn_cell.BasicRNNCell(RNN_HIDDEN)

        # self.batch_size = tf.shape(self.inputs)[1]

        self.zerostate = self.cell.zero_state(batch_size, tf.float32)
        # init_state_tupled
        # print('init state :', self.init_state)
        # print('zero state :', self.zerostate)
        rnn_outputs, self.new_states = tf.nn.dynamic_rnn(cell=self.cell, inputs=self.inputs,
                                                         initial_state=self.init_state,
                                                         time_major=True)
        # print('new state :', self.new_states)
        self.softmax_w = tf.get_variable('W', [RNN_HIDDEN, num_features], trainable=True)
        self.softmax_b = tf.get_variable('B', [num_features], trainable=True)

        # The LSTM output can be used to make next word predictions

        self.pred = tf.matmul(rnn_outputs[-1], self.softmax_w) + self.softmax_b

        # self.loss = -(self.outputs * tf.log(self.pred + TINY) + (1.0 - self.outputs) * tf.log(1.0 - self.pred + TINY))
        # self.loss = tf.reduce_mean(self.loss)
        print('pred :', self.pred)
        print('outputs :', self.outputs)
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.outputs, logits=self.pred))
        self.op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

    def train(self):

        with tf.Session() as sess:
            writer = tf.summary.FileWriter('board')
            writer.add_graph(sess.graph)
            writer.close()
            sess.run(tf.global_variables_initializer())
            # for each batch
            # data = prepare_data()
            # print(len(self.data))
            iter = -1
            print('there are ', int(len(self.data) / batch_size), ' batches')
            # last_state = sess.run([self.zerostate], feed_dict={})
            # last_state = last_state[0]
            last_state = np.random.rand(batch_size, 2 * Config.hidden_size)
            print('not started', np.shape(last_state))
            print('feature_num :', Config.INPUT_SIZE)
            for i in range(0, int(len(self.data) / batch_size)):
                iter += 1
                # print('hi')
                batch_X = np.zeros((1, batch_size, Config.INPUT_SIZE))
                batch_Y = np.zeros((batch_size, Config.OUTPUT_SIZE))
                # batch_Y = np.zeros((batch_size, Config.OUTPUT_SIZE + 1))
                for j in range(i * batch_size, batch_size * (i + 1)):
                    # print(np.shape(batch_X))
                    # print(np.shape(self.vocab.get_one_hot(self.data[j])))
                    batch_X[0][j - i * batch_size][:] = self.vocab.get_one_hot(self.data[j])
                    batch_Y[j - i * batch_size][:] = self.vocab.get_one_hot(self.data[j + 1])
                    # batch_Y[j - i * batch_size][:-1] = self.vocab.get_idx(self.data[j + 1])
                    # batch_Y[j - i * batch_size][-1] = Config.OUTPUT_SIZE
                feed_dic = {self.inputs: batch_X,
                            self.outputs: batch_Y,
                            self.init_state: last_state}
                # print('last state', last_state)
                # print('laststate[0] shape', np.shape(last_state[0]))
                loss, op, next_states, pred = sess.run([self.loss, self.op, self.new_states, self.pred],
                                                       feed_dict=feed_dic)
                # print('pred is :', pred)
                last_state = next_states
                if iter % report_each == 0:
                    print('Doing batch ', i, '. loss is :', loss)

    def predict(self, words_in_dataset):
        with tf.Session() as sess:
            feed_dic = {self.inputs: words_in_dataset}
            pred = sess.run(self.pred, feed_dict=feed_dic)
            idx = np.argmax(pred)
            return self.vocab.get_word(idx)


def prepare_data():
    data = []
    negaeshi = ['?', '!', '.', ':', ';']
    negaeshispaced = [' ? ', ' ! ', ' . ', ' : ', ' ; ']
    with open('data.txt', mode='r') as f:
        line = f.readline()
        cnt = 0
        # while line and cnt < 1000:
        while line:
            cnt += 1
            line = line.lower()
            for i, x in enumerate(negaeshi):
                # print(negaeshi[i])
                # print(negaeshispaced[i])
                line = line.replace(negaeshi[i], negaeshispaced[i])
                # print([start_token] + line.split(' ') + [end_token])
                # exit()
            line = line.strip()
            line = [start_token] + line.split(' ') + [end_token]
            if '' in line:
                line.remove('')
            data = data + line
            # print('reading')
            # print(line)
            line = f.readline()
    print('Data Was Read')
    return data


model = NetworkModel(Config.hidden_size, prepare_data())
model.train()
