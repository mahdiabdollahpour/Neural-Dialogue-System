import tensorflow as tf
import numpy as np

from seq2seq.utils import *
import os
import global_utils
import json

# how_many_lines = 7441
# how_many_lines = 50
source_sequence_length = 16
decoder_lengths = 20
# create_vocab_and_data_file('../datasets/DailyDialog/train/A_train.txt', '../datasets/DailyDialog/vocab_low.csv',
#                            '../datasets/DailyDialog/train/data_A_low.csv', 4500, decoder_lengths,
#                            '../datasets/Dailydialog/train/Q_train.txt',
#                            '../datasets/DailyDialog/train/data_Q_low.csv', source_sequence_length)

# create_vocab_and_data_file('../datasets/SQuAD/train_Q.txt', '../datasets/SQuAD/vocab_Q.csv',
#                            '../datasets/SQuAD/data_Q.csv', 5000, source_sequence_length)


print('Loading Data...')
# vocab1, dict_rev1, data1 = load_vocab_and_data_from_csv('../datasets/DailyDialog/vocab_low.csv',
#                                                         '../datasets/DailyDialog/train/data_Q_low.csv')
# vocab2, dict_rev2, data2 = load_vocab_and_data_from_csv('../datasets/DailyDialog/vocab_low.csv',
#                                                         '../datasets/DailyDialog/train/data_A_low.csv')


vocab1, dict_rev1, data1 = load_vocab_and_data_from_csv('../datasets/DailyDialog/vocab.csv',
                                                        '../datasets/DailyDialog/train/data_Q.csv')
vocab2, dict_rev2, data2 = load_vocab_and_data_from_csv('../datasets/DailyDialog/vocab.csv',
                                                        '../datasets/DailyDialog/train/data_A.csv')

print('Data is loaded')
print('Vocab length is', len(dict_rev1))
print('Number of training examples', len(data1))
print('Number of training examples', len(data2))


class Seq2Seq:

    def __init__(self, path='saved_model', QA=True):
        self.j = None
        self.path = path
        if not os.path.exists(path):
            print('Model is being created')
            os.makedirs(path)
            conf = {}
            with open(path + '\config.json', 'w', encoding='utf-8') as f:
                conf['iteration'] = 0
                conf['embedding_size'] = 100
                conf['hidden_num'] = 128
                print(conf)
                self.j = json.dumps(conf)
                print(self.j)
                f.write(self.j)
                f.flush()
                self.j = json.loads(self.j)
            self.load = False
        else:
            with open(path + '\config.json', 'r') as f:
                stri = f.read()
                print('content loaded :', stri)
                self.j = json.loads(stri)
            self.load = False
            self.load = True

        self.src_vocab_size = len(dict_rev1)
        self.tgt_vocab_size = len(dict_rev2)
        # print(vocab1[:10])
        # print(vocab2[:10])
        self.batch_size = 64

        embedding_size = self.j['embedding_size']
        hidden_num = self.j['hidden_num']

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
        if not QA:
            embeddings2 = tf.Variable(tf.random_uniform([self.tgt_vocab_size, embedding_size], -1.0, 1.0),
                                      dtype=tf.float32)
            decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings2, self.decoder_inputs_placeholder)
        else:
            decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings1, self.decoder_inputs_placeholder)

        encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings1, self.encoder_inputs_placeholder)

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
        ###TODO: inja bayad bayaye test_decoder ro linear bezani baraye test
        W = tf.get_variable(shape=(hidden_num, self.tgt_vocab_size), dtype=tf.float32, trainable=True, name='W',
                            initializer=tf.initializers.random_uniform)
        B = tf.get_variable(shape=(self.tgt_vocab_size), dtype=tf.float32, trainable=True, name='B',
                            initializer=tf.initializers.random_uniform)

        # print(self.decoder_outputs)
        # decoder_logits_no_dropout = tf.contrib.layers.linear(self.decoder_outputs, self.tgt_vocab_size)
        batch_time_shape = tf.shape(self.decoder_outputs_test)
        decoder_outputs_reshaped = tf.reshape(self.decoder_outputs, [-1, hidden_num])
        decoder_outputs_reshaped_test = tf.reshape(self.decoder_outputs_test, [-1, hidden_num])

        decoder_logits_no_dropout = tf.matmul(decoder_outputs_reshaped, W) + B
        decoder_logits_Test = tf.matmul(decoder_outputs_reshaped_test, W) + B
        # adding dropout
        decoder_logits = tf.layers.dropout(decoder_logits_no_dropout, 0.5, training=True, name='Dropout')
        # decoder_logits_test = tf.contrib.layers.linear(self.decoder_outputs_test, self.tgt_vocab_size)
        '''
            is this computed only when needed???
        '''
        softmaxed = tf.nn.softmax(decoder_logits_Test)
        self.decoder_prediction = tf.argmax(
            tf.reshape(softmaxed, (batch_time_shape[0], batch_time_shape[1], self.tgt_vocab_size))
            , 2)
        # labels = tf.one_hot(decoder_targets, depth=tgt_vocab_size, dtype=tf.float32)
        labels = self.decoder_targets_placeholder
        stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.reshape(labels, [-1, self.tgt_vocab_size]),
            logits=decoder_logits,
        )
        self.iter_loss_placeholder = tf.placeholder(tf.float32, name='iter_loss')
        self.loss_op = tf.reduce_mean(stepwise_cross_entropy)
        tf.summary.scalar("loss", self.iter_loss_placeholder)
        self.merged_summary = tf.summary.merge_all()
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss_op)
        self.path = path

    def translate(self, src_sen):
        length = 30
        by_index = sentence_by_id(src_sen, dict_rev1)
        sequence = np.asarray(by_index)
        sequence = np.reshape(sequence, [-1, 1])
        # sequence[0][:] = by_index
        feed_dict = {
            self.encoder_inputs_placeholder: sequence,
        }
        answer = ""
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
                the_word = vocab2[word_predicted[0][0]]
                answer += the_word + " "
                inp = word_predicted[0]
                if the_word == global_utils.end_token:
                    break
            print(src_sen)
            print(answer)

    def train(self, logs_dir):

        display_each = 200

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())
            if self.load:
                global_utils.check_restore_parameters(sess, saver, self.path + '\model.ckpt')

            num = global_utils.get_trainable_variables_num()
            print('Number of trainable variables ', num)
            iter_nums = 200
            number_of_batches = int(len(data1) / self.batch_size)
            print('There are ', number_of_batches, ' batches')
            import time
            start = self.j['iteration']

            writer = tf.summary.FileWriter(logs_dir)

            writer.add_graph(sess.graph)

            for i in range(start, iter_nums):
                iter_loss = 0
                iter_time = time.time()
                for j in range(number_of_batches):
                    enc_inp = np.zeros((source_sequence_length, self.batch_size), dtype='int')
                    dec_inp = np.zeros((decoder_lengths - 1, self.batch_size), dtype='int')
                    dec_tgt_dummy = np.zeros((decoder_lengths - 1, self.batch_size), dtype='int')
                    dec_tgt = np.zeros((decoder_lengths - 1, self.batch_size, self.tgt_vocab_size), dtype='int')

                    for idx in range(self.batch_size):
                        # print('input :', data2[j * batch_size + idx])
                        dec_inp[:, idx] = data2[j * self.batch_size + idx][:-1]
                        dec_tgt_dummy[:, idx] = data2[j * self.batch_size + idx][1:]
                        enc_inp[:, idx] = data1[j * self.batch_size + idx]
                        for t in range(decoder_lengths - 1):
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
                tf.summary.scalar('loss', iter_loss)
                s = sess.run(self.merged_summary, feed_dict={self.iter_loss_placeholder: iter_loss / number_of_batches})
                print(self.merged_summary)
                writer.add_summary(s, i)
                print('Saving model')
                self.j['iteration'] = i
                with open(self.path + '\config.json', 'w', encoding='utf-8') as f:
                    f.write(json.dumps(self.j))
                    f.flush()

                saver.save(sess, self.path + '\model.ckpt')


# model = Seq2Seq(path='saved_seq2seq')


model = Seq2Seq(path='Daily_QA')
model.train(logs_dir='QA_Board')


# input_string = ' '
# while input_string != 'q':
#     input_string = input()
#     if input_string == 'q':
#         break
#     model.translate(input_string)
