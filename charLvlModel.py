import os
import random
import time
import numpy as np
import tensorflow as tf
from utils import *

input_file = 'data.txt'
data, vocab = load_data(input_file)
in_size = out_size = len(vocab)
lstm_size = 256  # 128
num_layers = 2
batch_size = 64  # 128
time_steps = 100  # 50
NUM_TRAIN_BATCHES = 20000


def restore_if_possible(sess, saver, check_point):
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(check_point))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)


class LSTM_NN:

    def __init__(self, in_size, hidden_size, num_layers, out_size, session,
                 lr=0.003, ScopeName="RNN"):
        self.scope = ScopeName
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_size = out_size
        self.session = session
        self.lr = tf.constant(lr)

        ## Defining the computational graph

        self.lstm_last_state = np.zeros(
            (self.num_layers * 2 * self.hidden_size,)
        )
        with tf.variable_scope(self.scope):
            self.input_placeholder = tf.placeholder(
                tf.float32,
                shape=(None, None, self.in_size),
                name="input"
            )
            self.lstm_init_value = tf.placeholder(
                tf.float32,
                shape=(None, self.num_layers * 2 * self.hidden_size),
                name="lstm_init_value"
            )
            # LSTM
            self.lstm_cells = [
                tf.contrib.rnn.BasicLSTMCell(
                    self.hidden_size,
                    forget_bias=1.0,
                    state_is_tuple=False
                ) for i in range(self.num_layers)
            ]
            self.lstm = tf.contrib.rnn.MultiRNNCell(
                self.lstm_cells,
                state_is_tuple=False
            )
            outputs, self.lstm_new_state = tf.nn.dynamic_rnn(
                self.lstm,
                self.input_placeholder,
                initial_state=self.lstm_init_value,
                dtype=tf.float32
            )
            self.W = tf.Variable(
                tf.random_normal(
                    (self.hidden_size, self.out_size),
                    stddev=0.01
                )
            )
            self.B = tf.Variable(
                tf.random_normal(
                    (self.out_size,), stddev=0.01
                )
            )
            outputs_reshaped = tf.reshape(outputs, [-1, self.hidden_size])
            network_output = tf.matmul(
                outputs_reshaped,
                self.W
            ) + self.B
            batch_time_shape = tf.shape(outputs)
            self.final_outputs = tf.reshape(
                tf.nn.softmax(network_output),
                (batch_time_shape[0], batch_time_shape[1], self.out_size)
            )

            self.y_batch = tf.placeholder(
                tf.float32,
                (None, None, self.out_size)
            )
            y_batch_long = tf.reshape(self.y_batch, [-1, self.out_size])
            self.cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=network_output,
                    labels=y_batch_long
                )
            )
            self.train_op = tf.train.RMSPropOptimizer(
                self.lr,
                0.9
            ).minimize(self.cost)

    def run_step(self, x, init_zero_state=True):
        if init_zero_state:
            init_value = np.zeros((self.num_layers * 2 * self.hidden_size,))
        else:
            init_value = self.lstm_last_state
        out, next_lstm_state = self.session.run(
            [self.final_outputs, self.lstm_new_state],
            feed_dict={
                self.input_placeholder: [x],
                self.lstm_init_value: [init_value]
            }
        )
        self.lstm_last_state = next_lstm_state[0]
        return out[0][0]

    def train_on_batch(self, xbatch, ybatch):
        init_value = np.zeros(
            (xbatch.shape[0], self.num_layers * 2 * self.hidden_size)
        )
        cost, _ = self.session.run(
            [self.cost, self.train_op],
            feed_dict={
                self.input_placeholder: xbatch,
                self.y_batch: ybatch,
                self.lstm_init_value: init_value
            }
        )
        return cost


def load_model(check_point):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    net = LSTM_NN(
        in_size=in_size,
        hidden_size=lstm_size,
        num_layers=num_layers,
        out_size=out_size,
        session=sess,
        lr=0.003,
        ScopeName="char_rnn_network"
    )
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, check_point)
    return net


def train(check_point, save=True, continue_training=False):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    net = LSTM_NN(
        in_size=in_size,
        hidden_size=lstm_size,
        num_layers=num_layers,
        out_size=out_size,
        session=sess,
        lr=0.003,
        ScopeName="char_rnn_network"
    )
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    ## saving the config to a file to know the config when loading
    ## you should fill in the config in code by your self and the text file
    ## is only for you to read
    if save and (not continue_training):
        addr = os.path.abspath(os.path.join(check_point, os.pardir))
        os.makedirs(addr)
        addr += '\config.txt'
        with open(addr, 'w') as f:
            print('writing config to ', addr)
            f.write('in_size : ' + str(in_size) + '\n')
            f.write('out_size : ' + str(out_size) + '\n')
            f.write('hidden_size : ' + str(lstm_size) + '\n')
            f.write('layer num : ' + str(num_layers) + '\n')

    if continue_training:
        restore_if_possible(sess, saver, check_point)
    last_time = time.time()
    batch = np.zeros((batch_size, time_steps, in_size))
    batch_y = np.zeros((batch_size, time_steps, in_size))
    possible_batch_ids = range(data.shape[0] - time_steps - 1)

    for i in range(NUM_TRAIN_BATCHES):
        # Sample time_steps consecutive samples from the dataset text file
        batch_id = random.sample(possible_batch_ids, batch_size)

        for j in range(time_steps):
            ind1 = [k + j for k in batch_id]
            ind2 = [k + j + 1 for k in batch_id]

            batch[:, j, :] = data[ind1, :]
            batch_y[:, j, :] = data[ind2, :]

        cst = net.train_on_batch(batch, batch_y)

        if (i % 100) == 0:
            new_time = time.time()
            diff = new_time - last_time
            last_time = new_time
            print("batch: {}  loss: {}  speed: {} batches / s".format(
                i, cst, 100 / diff
            ))
            if save:
                saver.save(sess, check_point)


def predict(prefix, model, generate_len=500):
    prefix = prefix.lower()
    for i in range(len(prefix)):
        out = model.run_step(convert_to_one_hot(prefix[i], vocab), i == 0)

    print("Sentence:")
    gen_str = prefix
    for i in range(generate_len):
        element = np.random.choice(range(len(vocab)), p=out)
        gen_str += vocab[element]
        out = model.run_step(convert_to_one_hot(vocab[element], vocab), False)

        print(gen_str)


# model = load_model("saved/model.ckpt")
# predict("I am", model=model)
# train(continue_training=True)
train('test/model.ckpt', True, False)
