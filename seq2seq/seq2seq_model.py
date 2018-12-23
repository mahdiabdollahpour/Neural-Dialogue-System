import tensorflow as tf
import numpy as np

from seq2seq.utils import *
import os
import global_utils

how_many_lines = 7441
# how_many_lines = 50
data1string, dict1, dict_rev1 = load_data_and_create_vocab('../datasets/it-en/en.txt', how_many_lines)
data2string, dict2, dict_rev2 = load_data_and_create_vocab('../datasets/it-en/it.txt', how_many_lines)
print('Data is read and vocab is created')
source_sequence_length = 23
decoder_lengths = 26
print('vocab len', len(dict1))
print('vocab len target', len(dict2))
data1 = data_by_ID_and_truncated(data1string, dict_rev1, source_sequence_length)
data2 = data_by_ID_and_truncated(data2string, dict_rev2, decoder_lengths + 1,
                                 append_and_prepend=True)
print('sequences are made by indexes ')


class Seq2Seq:

    def __init__(self, path='saved_model'):

        self.src_vocab_size = len(dict_rev1)
        self.tgt_vocab_size = len(dict_rev2)
        # print(vocab1[:10])
        # print(vocab2[:10])
        self.batch_size = 8

        embedding_size = 100
        hidden_num = 128
        self.encoder_inputs_placeholder = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
        self.decoder_inputs_placeholder = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
        self.decoder_targets_placeholder = tf.placeholder(shape=(None, None, self.tgt_vocab_size), dtype=tf.int32,
                                                          name='decoder_targets')
        self.decoder_init_state_placeholder0 = tf.placeholder(shape=(None), dtype=tf.float32,
                                                              name='decoder_init_sate0')
        self.decoder_init_state_placeholder1 = tf.placeholder(shape=(None), dtype=tf.float32,
                                                              name='decoder_init_sate1')

        initial_state_test_mode = tf.nn.rnn_cell.LSTMStateTuple(self.decoder_init_state_placeholder0,
                                                                self.decoder_init_state_placeholder1)

        embeddings1 = tf.Variable(tf.random_uniform([self.src_vocab_size, embedding_size], -1.0, 1.0), dtype=tf.float32)
        embeddings2 = tf.Variable(tf.random_uniform([self.tgt_vocab_size, embedding_size], -1.0, 1.0), dtype=tf.float32)

        encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings1, self.encoder_inputs_placeholder)
        decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings2, self.decoder_inputs_placeholder)

        encoder_cell = tf.contrib.rnn.LSTMCell(hidden_num)

        encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(
            encoder_cell, encoder_inputs_embedded,
            dtype=tf.float32, time_major=True,
        )
        del encoder_outputs

        decoder_cell = tf.contrib.rnn.LSTMCell(hidden_num)
        # self.decoder_init_state = None
        #
        # def train_mode(): return self.decoder_init_state.assign(self.encoder_final_state)
        #
        # def test_mode(): return self.decoder_init_state.assign(initial_state_test_mode)

        # one = tf.constant(1)

        # r = tf.cond(tf.less(self.mode_placholder, one), train_mode(), test_mode())

        self.decoder_outputs, self.decoder_final_state = tf.nn.dynamic_rnn(
            decoder_cell, decoder_inputs_embedded,

            initial_state=self.encoder_final_state,

            dtype=tf.float32, time_major=True, scope="plain_decoder",
        )

        self.decoder_outputs_test, self.decoder_final_state_test = tf.nn.dynamic_rnn(
            decoder_cell, decoder_inputs_embedded,

            initial_state=initial_state_test_mode,

            dtype=tf.float32, time_major=True, scope="plain_decoder_test",
        )

        # print(self.encoder_final_state)
        # exit()

        decoder_logits = tf.contrib.layers.linear(self.decoder_outputs, self.tgt_vocab_size)
        decoder_logits_test = tf.contrib.layers.linear(self.decoder_outputs_test, self.tgt_vocab_size)

        self.decoder_prediction = tf.argmax(decoder_logits_test, 2)
        # labels = tf.one_hot(decoder_targets, depth=tgt_vocab_size, dtype=tf.float32)
        labels = self.decoder_targets_placeholder
        stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels,
            logits=decoder_logits,
        )

        self.loss_op = tf.reduce_mean(stepwise_cross_entropy)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss_op)
        self.path = path
        if not os.path.exists(path):
            print('Model is being created')
            os.makedirs(path)
            self.load = False
        else:
            self.load = True

    def translate(self, src_sen):
        length = 30
        by_index = sentence_by_id(src_sen, dict_rev1)
        sequence = np.asarray(by_index)
        sequence = np.reshape(sequence, [-1, 1])
        # sequence[0][:] = by_index
        feed_dict = {
            self.encoder_inputs_placeholder: sequence,
        }
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())

            if self.load:
                print('loading')
                global_utils.check_restore_parameters(sess, saver, self.path + '\model.ckpt')
            thought_vector = sess.run([self.encoder_final_state], feed_dict)
            # thought_vector = thought_vector[-1]
            inp = get_start_token_index(dict_rev2)
            # print('fisr thought vector :',thought_vector)
            # print('-------------------------------------------')
            thought_vector = thought_vector[0]
            for i in range(length):
                # print(thought_vector)
                # print('input word :',inp)
                feed_dict = {
                    self.decoder_inputs_placeholder: [inp],
                    self.decoder_init_state_placeholder0: thought_vector.c,
                    self.decoder_init_state_placeholder1: thought_vector.h
                }
                word_predicted, thought_vector = sess.run([self.decoder_prediction, self.decoder_final_state_test],
                                                          feed_dict)
                # print(word_predicted[0])
                print(dict2[word_predicted[0][0]])
                inp = word_predicted[0]

    def train(self):

        display_each = 1000

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())
            if self.load:
                global_utils.check_restore_parameters(sess, saver, self.path + '\model.ckpt')

            num = global_utils.get_trainable_variables_num()
            print('Number of trainable variables ', num)
            iter_num = 200
            number_of_batches = int(len(data1) / self.batch_size)
            print('There are ', number_of_batches, ' batches')
            import time
            for i in range(iter_num):
                iter_loss = 0
                iter_time = time.time()
                for j in range(number_of_batches):
                    enc_inp = np.zeros((source_sequence_length, self.batch_size), dtype='int')
                    dec_inp = np.zeros((decoder_lengths + 1, self.batch_size), dtype='int')
                    dec_tgt_dummy = np.zeros((decoder_lengths + 1, self.batch_size), dtype='int')
                    dec_tgt = np.zeros((decoder_lengths + 1, self.batch_size, self.tgt_vocab_size), dtype='int')

                    for idx in range(self.batch_size):
                        # print('input :', data2[j * batch_size + idx])
                        dec_inp[:, idx] = data2[j * self.batch_size + idx][:-1]
                        dec_tgt_dummy[:, idx] = data2[j * self.batch_size + idx][1:]
                        enc_inp[:, idx] = data1[j * self.batch_size + idx]
                        for t in range(decoder_lengths):
                            dec_tgt[t, idx, :] = get_one_hot(dec_tgt_dummy[t, idx], self.tgt_vocab_size)

                    feed_dict = {
                        self.encoder_inputs_placeholder: enc_inp,
                        self.decoder_inputs_placeholder: dec_inp,
                        self.decoder_targets_placeholder: dec_tgt
                    }
                    # print(np.shape(enc_inp))
                    # print(np.shape(dec_inp))
                    # print(np.shape(dec_tgt))

                    _, loss = sess.run([self.train_op, self.loss_op], feed_dict=feed_dict)
                    # print(np.max(labe))
                    iter_loss += np.sum(loss)
                    if j % display_each == 0:
                        print('Mini batch loss is ', loss)
                print('Average loss in iteration ', i, ' is ', iter_loss / number_of_batches)
                print("It took ", time.time() - iter_time)
                print('Saving model')
                saver.save(sess, self.path + '\model.ckpt')


model = Seq2Seq()
model.train()
# model.translate("good morning")
