import time
from LSTM_tensorflow.tools import *
import tensorflow as tf
import os


class LSTM_NN:

    def __init__(self, session, check_point_dir, hidden_size=256, num_layers=2, lr=0.003,
                 scope_name="RNN"):
        self.scope = scope_name
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.session = session
        self.lr = tf.constant(lr)
        self.check_point_dir = check_point_dir

        # Defining the computational graph

        # type 0 -> is not tuple (use in non-tuple mode)
        # self.lstm_last_state = np.zeros(
        #     (self.num_layers * 2 * self.hidden_size,)
        #     # num_layer * 2 (one for h and one for c) * hidden_size
        # )

        # type 1 -> wrong
        # self.lstm_last_state = [
        #     np.zeros(
        #         (self.num_layers * self.hidden_size,)),
        #     np.zeros(
        #         (self.num_layers * self.hidden_size,))
        #
        # ]

        # type 2
        self.lstm_last_state = tuple(
            [
                tf.nn.rnn_cell.LSTMStateTuple(
                    np.zeros((1, self.hidden_size)),
                    np.zeros((1, self.hidden_size))
                ) for _ in range(self.num_layers)
            ]
        )

        with tf.variable_scope(self.scope):
            self.x_batch = tf.placeholder(
                tf.float32,
                shape=(None, None, SIZE_OF_VOCAB),
                # None, None -> number of sentences in this batch, CHAR_NUM_OF_SENTENCE
                name="input"
            )
            # type 0 -> is not tuple (use in non-tuple mode)
            # self.lstm_init_value = tf.placeholder(
            #     tf.float32,
            #     shape=(None, self.num_layers * 2 * self.hidden_size),
            #     # None -> number of sentences in this batch
            #     name="lstm_init_value"
            # )

            # type 1 -> wrong
            # self.c_layer = tf.placeholder(
            #     tf.float32,
            #     shape=[None, self.num_layers * self.hidden_size],
            #     name='c_layer'
            # )
            #
            # self.h_layer = tf.placeholder(
            #     tf.float32,
            #     shape=[None, self.num_layers * self.hidden_size],
            #     name='h_layer'
            # )
            #
            # self.lstm_init_value = tf.nn.rnn_cell.LSTMStateTuple(self.c_layer, self.h_layer)

            # type 2
            self.lstm_init_value = tuple(
                [
                    tf.nn.rnn_cell.LSTMStateTuple(
                        tf.placeholder(tf.float32, shape=[None, self.hidden_size], name='c_layer' + str(l_id)),
                        tf.placeholder(tf.float32, shape=[None, self.hidden_size], name='h_layer' + str(l_id)),
                    ) for l_id in range(self.num_layers)
                ]
            )

            # LSTM
            self.lstm_cells = [
                tf.contrib.rnn.BasicLSTMCell(
                    self.hidden_size,
                    forget_bias=1.0,
                    state_is_tuple=True
                ) for _ in range(self.num_layers)
            ]
            self.lstm = tf.contrib.rnn.MultiRNNCell(
                self.lstm_cells,
                state_is_tuple=True
            )

            outputs, self.lstm_new_state = tf.nn.dynamic_rnn(
                self.lstm,
                self.x_batch,
                initial_state=self.lstm_init_value,
                dtype=tf.float32
            )
            # fc layer at the end
            self.W = tf.Variable(
                tf.random_normal(
                    (self.hidden_size, SIZE_OF_VOCAB),
                    stddev=0.01
                )
            )
            self.B = tf.Variable(
                tf.random_normal(
                    (SIZE_OF_VOCAB,), stddev=0.01
                    # size : SIZE_OF_VOCAB (1d)
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
                (batch_time_shape[0], batch_time_shape[1], SIZE_OF_VOCAB)
            )

            self.y_batch = tf.placeholder(
                tf.float32,
                (None, None, SIZE_OF_VOCAB)
            )
            y_batch_long = tf.reshape(self.y_batch, [-1, SIZE_OF_VOCAB])
            self.cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=network_output,
                    labels=y_batch_long
                )
            )
            # minimizing the error
            self.train_op = tf.train.RMSPropOptimizer(
                self.lr,
                0.9
            ).minimize(self.cost)

    def train_on_batch(self, x_batch, y_batch):
        """

        :param x_batch: size=(batch_size, char_num_of_sentence, SIZE_OF_VOCAB)
        :param y_batch: size=(batch_size, char_num_of_sentence, SIZE_OF_VOCAB)
        :return:
        """

        # type 0 -> is not tuple (use in non-tuple mode)
        # init_value = np.zeros(
        #     (x_batch.shape[0], self.num_layers * 2 * self.hidden_size)
        # )

        # type 1 -> wrong
        # init_c_layer = np.zeros((
        #     x_batch.shape[0], self.num_layers * self.hidden_size
        # ))
        # init_h_layer = np.zeros((
        #     x_batch.shape[0], self.num_layers * self.hidden_size
        # ))
        #
        # init_value = tf.nn.rnn_cell.LSTMStateTuple(self.c_layer, self.h_layer)

        # type 2
        init_value = tuple(
            [
                tf.nn.rnn_cell.LSTMStateTuple(
                    np.zeros((x_batch.shape[0], self.hidden_size)),
                    np.zeros((x_batch.shape[0], self.hidden_size))
                ) for _ in range(self.num_layers)
            ]
        )

        cost, _ = self.session.run(
            [self.cost, self.train_op],
            feed_dict={
                self.x_batch: x_batch,
                self.y_batch: y_batch,
                self.lstm_init_value: init_value
            }
        )
        return cost

    def run_step(self, x, init_zero_state=False):
        if init_zero_state:
            # type 0 -> is not tuple (use in non-tuple mode)
            # init_value = [np.zeros((self.num_layers * 2 * self.hidden_size,))]

            # type 1 -> wrong
            # init_value = [
            #     np.zeros(
            #         (self.num_layers * self.hidden_size,)),
            #     np.zeros(
            #         (self.num_layers * self.hidden_size,))
            #
            # ]

            # type 2
            init_value = tuple(
                [
                    tf.nn.rnn_cell.LSTMStateTuple(
                        np.zeros((1, self.hidden_size)),
                        np.zeros((1, self.hidden_size))
                    ) for _ in range(self.num_layers)
                ]
            )
            # x_batch.shape[0] is 1

        else:
            init_value = self.lstm_last_state
        out, next_lstm_state = self.session.run(
            [self.final_outputs, self.lstm_new_state],
            feed_dict={
                self.x_batch: [x],  # -> x_batch.shape[0] is 1
                self.lstm_init_value: init_value
            }
        )
        self.lstm_last_state = next_lstm_state#[0]

        return out[0][0]  # it shows the next character that is predicted


def load_model(check_point_dir):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)

    net = LSTM_NN(
        check_point_dir=check_point_dir,
        session=sess,
        scope_name="char_rnn_network"
    )
    check_point = check_point_dir + '/model.ckpt'
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    if os.path.exists(check_point):
        saver.restore(sess, check_point)
    return net


def train(model, data, mini_batch_size=64, num_train_batches=20000, check_point_period=100):
    print("'''TRAINING started'''")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = model.session
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    check_point = model.check_point_dir + '/model.ckpt'

    last_time = time.time()

    np_start_token = [0.0] * SIZE_OF_VOCAB
    np_start_token[START] = 1.0
    np_end_token = [0.0] * SIZE_OF_VOCAB
    np_end_token[END] = 1.0

    batch_size = len(data)
    whole_train_iter = 0
    mini_batch_iter = 0
    train_iter = 0

    while whole_train_iter < num_train_batches:
        x_batch = np.zeros((mini_batch_size, CHAR_NUM_OF_SENTENCE, SIZE_OF_VOCAB))
        y_batch = np.zeros((mini_batch_size, CHAR_NUM_OF_SENTENCE, SIZE_OF_VOCAB))

        for mini_batch_id in range(0, mini_batch_size):
            batch_id = (mini_batch_id + mini_batch_iter) % batch_size
            x_batch[mini_batch_id] = np.append([np_start_token], data[batch_id], axis=0)
            y_batch[mini_batch_id] = np.append(data[batch_id], [np_end_token], axis=0)

        batch_cost = model.train_on_batch(x_batch, y_batch)
        train_iter += 1

        print("     ---whole train iteration: {}    mini batch iteration: {}".format(whole_train_iter, mini_batch_iter))

        if train_iter % check_point_period == 0:
            new_time = time.time()
            diff = new_time - last_time
            last_time = new_time
            print("train iteration: {}  loss: {}  speed: {} batches / s".format(
                whole_train_iter, batch_cost, check_point_period / diff
            ))
            saver.save(sess, check_point)
            print('saved in this path: ', check_point)

        mini_batch_iter += mini_batch_size
        if mini_batch_iter >= batch_size:
            mini_batch_iter -= batch_size
            whole_train_iter += 1


def predict(prefix, model, generate_len=100):
    prefix = prefix.lower()
    if len(prefix) == 0:
        first_char = map_id_to_char(np.random.choice(range(SIZE_OF_VOCAB)))
    else:
        first_char = prefix[0]
    out = model.run_step([get_char_vector(first_char)], True)
    for i in range(1, len(prefix)):
        out = model.run_step([get_char_vector(prefix[i])])

    print("Sentence:")
    gen_str = prefix
    for i in range(generate_len):
        element_id = np.random.choice(range(SIZE_OF_VOCAB), p=out)
        element = map_id_to_char(element_id)
        gen_str += element
        out = model.run_step([get_one_hot_vector(element_id)])

    print(gen_str)


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    saver_directory = os.path.abspath(os.path.join(os.getcwd(), '../.saved'))

    CHAR_NUM_OF_SENTENCE = 100

    data = load_data('../datasets/shakespeare -all.txt', CHAR_NUM_OF_SENTENCE)

    test = True
    if not test:
        model = LSTM_NN(sess, saver_directory)
        train(model, data)
    else:
        model = load_model(saver_directory)
        predict('I am ', model, 500)
