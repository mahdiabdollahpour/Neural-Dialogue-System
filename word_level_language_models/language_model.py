""" Recurrent Neural Network.

A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.

"""

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import random
import time
import numpy as np
from word_level_language_models.utils import *


def check_restore_parameters(sess, saver, path):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(path))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)


# Training Parameters

learning_rate = 0.005
batch_size = 64
display_step = 200

# Network Parameters
timesteps = 10  # timesteps
num_hiddens = [128, 128]  # hidden layer num of features
num_fc_hiddens = []
data_string, vocab = load_data("../datasets/data.txt")
data = data_by_ID_and_truncated(data_string, vocab, timesteps)
print("vocab len :", len(vocab))

vocab_size = len(vocab)

X = tf.placeholder("int32", [None, timesteps])
Y = tf.placeholder("float", [None, timesteps, vocab_size])

weights = {
    'out': tf.Variable(tf.random_normal([num_hiddens[-1], vocab_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size]))
}


def RNN(x, weights, biases):
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)\

    embedding = tf.get_variable(
        "embedding", [vocab_size, num_hiddens[0]], dtype=tf.float32)
    x = tf.nn.embedding_lookup(embedding, x)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cells = [rnn.BasicLSTMCell(i, forget_bias=1.0, activation=tf.nn.leaky_relu) for i in num_hiddens]
    cells = tf.contrib.rnn.MultiRNNCell(lstm_cells)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(cells, x, dtype=tf.float32)
    outputs_reshaped = tf.reshape(outputs, [-1, num_hiddens[-1]])
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs_reshaped, weights['out']) + biases['out']


logits = RNN(X, weights, biases)
# prediction = tf.nn.softmax(logits)

# Define loss and optimizer
y_batch_long = tf.reshape(Y, [-1, vocab_size])
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=y_batch_long))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
# correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print('number of trainable variables : ',get_trainable_variables_num())
iter_num = 200
with tf.Session() as sess:
    # Initialize the variables (i.e. assign their default value)
    sess.run(tf.global_variables_initializer())
    # saver = tf.train.Saver(tf.global_variables())
    saved_file = "multilayer_with_embedding/model.ckpt"

    # check_restore_parameters(sess, saver, saved_file)
    last_time = time.time()
    possible_batch_ids = range(len(data) - timesteps - 1)
    for cnt in range(iter_num):
        iter_loss = 0
        number_of_batches = int(len(data) / batch_size)
        # number_of_batches = 1
        for i in range(number_of_batches):
            # Sample time_steps consecutive samples from the dataset text file
            batch_x = np.zeros((batch_size, timesteps))
            batch_y = np.zeros((batch_size, timesteps, vocab_size))
            for kk in range(batch_size):
                batch_x[kk, :] = data[i * batch_size + kk][:-1]
                for zz in range(timesteps):
                    batch_y[kk, zz, :] = get_one_hot(data[i * batch_size + kk][zz + 1], vocab)

            _, loss = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
            iter_loss += np.sum(loss)
            if i % display_step == 0:
                print('Mini batch loss of batch ', i, ' is :', np.sum(loss))

        new_time = time.time()
        diff = new_time - last_time
        last_time = new_time
        # saver.save(sess, saved_file)
        print('Average loss of all batches of iteration ', cnt, ' is ', iter_loss / number_of_batches, ' speed ',
              number_of_batches / diff,
              ' batches / s')
        # print("batch_x: {}  loss: {}  speed: {} batches / s".format(
        #     i, loss, display_step / diff
        # ))
