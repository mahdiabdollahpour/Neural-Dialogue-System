import tensorflow as tf
import numpy as np

from seq2seq.utils import *

data1string, vocab1 = load_data('../datasets/train.en.txt', '../datasets/vocab.en.txt')
data2string, vocab2 = load_data('../datasets/train.vi.txt', '../datasets/vocab.vi.txt')
src_vocab_size = len(vocab1)
tgt_vocab_size = len(vocab2)
print(vocab1[:10])
print(vocab2[:10])
batch_size = 64
source_sequence_length = 23
decoder_lengths = 26

data1 = data_by_ID_and_truncated(data1string[:1000], vocab1, source_sequence_length)
data2 = data_by_ID_and_truncated(data2string[:1000], vocab2, decoder_lengths)

# tf.reset_default_graph()
sess = tf.InteractiveSession()

# PAD = 0
# EOS = 1
#
# vocab_size = 10
# input_embedding_size = 20
#
# encoder_hidden_units = 20
# decoder_hidden_units = encoder_hidden_units
embedding_size = 100
hidden_num = 128
encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')

embeddings1 = tf.Variable(tf.random_uniform([src_vocab_size, embedding_size], -1.0, 1.0), dtype=tf.float32)
embeddings2 = tf.Variable(tf.random_uniform([tgt_vocab_size, embedding_size], -1.0, 1.0), dtype=tf.float32)

encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings1, encoder_inputs)
decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings2, decoder_inputs)

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

stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=tgt_vocab_size, dtype=tf.float32),
    logits=decoder_logits,
)

loss_op = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss_op)

sess.run(tf.global_variables_initializer())

display_each = 100
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    iter_num = 200
    for i in range(iter_num):
        iter_loss = 0
        enc_inp = np.zeros((source_sequence_length, batch_size))
        dec_inp = np.zeros((decoder_lengths, batch_size))
        dec_tgt = np.zeros((decoder_lengths, batch_size))
        number_of_batches = int(len(data1) / batch_size)
        for j in range(number_of_batches):
            for idx in range(batch_size):
                dec_inp[:, idx], dec_tgt[:, idx] = get_batch(data2[j * batch_size + idx], vocab2)
                enc_inp[:, idx] = data1[j * batch_size + idx]

            feed_dict = {
                encoder_inputs: enc_inp,
                decoder_inputs: dec_inp,
                decoder_outputs: dec_tgt
            }

            _, loss = sess.run([train_op, loss_op], feed_dict=feed_dict)
            iter_loss += np.sum(loss)
            if j % display_each == 0:
                print('Mini batch loss is ', loss)
        print('Average loss in iteration ', i, ' is ', iter_loss / batch_size)
