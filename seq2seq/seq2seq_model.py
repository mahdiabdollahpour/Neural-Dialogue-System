import tensorflow as tf
import numpy as np

from seq2seq.utils import *

data1string, vocab1 = load_data('../datasets/train.en.txt', '../datasets/vocab.en.txt')
data2string, vocab2 = load_data('../datasets/train.vi.txt', '../datasets/vocab.vi.txt')
src_vocab_size = len(vocab1)
tgt_vocab_size = len(vocab2)
# print(vocab1[:10])
# print(vocab2[:10])
batch_size = 64
source_sequence_length = 23
decoder_lengths = 26

how_many_lines = 10000
data1 = data_by_ID_and_truncated(data1string[:how_many_lines], vocab1, source_sequence_length)
data2 = data_by_ID_and_truncated(data2string[:how_many_lines], vocab2, decoder_lengths + 1, append_and_prepend=True)

embedding_size = 100
hidden_num = 128
encoder_inputs_placeholder = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
decoder_inputs_placeholder = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
decoder_targets_placeholder = tf.placeholder(shape=(None, None, tgt_vocab_size), dtype=tf.int32, name='decoder_targets')

embeddings1 = tf.Variable(tf.random_uniform([src_vocab_size, embedding_size], -1.0, 1.0), dtype=tf.float32)
embeddings2 = tf.Variable(tf.random_uniform([tgt_vocab_size, embedding_size], -1.0, 1.0), dtype=tf.float32)

encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings1, encoder_inputs_placeholder)
decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings2, decoder_inputs_placeholder)

encoder_cell = tf.contrib.rnn.LSTMCell(hidden_num)

encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
    encoder_cell, encoder_inputs_embedded,
    dtype=tf.float32, time_major=True,
)

del encoder_outputs

decoder_cell = tf.contrib.rnn.LSTMCell(hidden_num)

decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
    decoder_cell, decoder_inputs_embedded,

    initial_state=encoder_final_state,

    dtype=tf.float32, time_major=True, scope="plain_decoder",
)

decoder_logits = tf.contrib.layers.linear(decoder_outputs, tgt_vocab_size)

decoder_prediction = tf.argmax(decoder_logits, 2)
# labels = tf.one_hot(decoder_targets, depth=tgt_vocab_size, dtype=tf.float32)
labels = decoder_targets_placeholder
stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=labels,
    logits=decoder_logits,
)

loss_op = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss_op)

display_each = 100
import global_utils

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num = global_utils.get_trainable_variables_num()
    print('Number of trainable variables ', num)
    iter_num = 200
    number_of_batches = int(len(data1) / batch_size)
    print('There are ', number_of_batches, ' batches')
    for i in range(iter_num):
        iter_loss = 0

        for j in range(number_of_batches):
            enc_inp = np.zeros((source_sequence_length, batch_size), dtype='int')
            dec_inp = np.zeros((decoder_lengths + 1, batch_size), dtype='int')
            dec_tgt_dummy = np.zeros((decoder_lengths + 1, batch_size), dtype='int')
            dec_tgt = np.zeros((decoder_lengths + 1, batch_size, tgt_vocab_size), dtype='int')

            for idx in range(batch_size):
                # print('input :', data2[j * batch_size + idx])
                dec_inp[:, idx] = data2[j * batch_size + idx][:-1]
                dec_tgt_dummy[:, idx] = data2[j * batch_size + idx][1:]
                enc_inp[:, idx] = data1[j * batch_size + idx]
                for t in range(decoder_lengths):
                    dec_tgt[t, idx, :] = get_one_hot(dec_tgt_dummy[t, idx], tgt_vocab_size)

            feed_dict = {
                encoder_inputs_placeholder: enc_inp,
                decoder_inputs_placeholder: dec_inp,
                decoder_targets_placeholder: dec_tgt
            }
            # print(np.shape(enc_inp))
            # print(np.shape(dec_inp))
            # print(np.shape(dec_tgt))

            _, loss = sess.run([train_op, loss_op], feed_dict=feed_dict)
            # print(np.max(labe))
            iter_loss += np.sum(loss)
            if j % display_each == 0:
                print('Mini batch loss is ', loss)
        print('Average loss in iteration ', i, ' is ', iter_loss / number_of_batches)
