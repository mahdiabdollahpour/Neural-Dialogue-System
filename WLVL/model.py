""" Recurrent Neural Network.

A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import random
import time
import numpy as np
from WLVL.utils import *

# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''


def check_restore_parameters(sess, saver, path):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(path))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)


data_string, vocab = load_data("../data.txt")
data = data_by_ID(data_string, vocab)
print("vocab len :", len(vocab))
# Training Parameters
learning_rate = 0.005
# training_steps = 1000
batch_size = 64
display_step = 200

# Network Parameters
# num_input = 28 # MNIST data input (img shape: 28*28)
timesteps = 64  # timesteps
num_hiddens = [128,128,128]  # hidden layer num of features
num_fc_hiddens =[]

vocab_size = len(vocab)

# tf Graph input
X = tf.placeholder("int32", [None, timesteps])
Y = tf.placeholder("float", [None, timesteps, vocab_size])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hiddens[-1], vocab_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size]))
}


def RNN(x, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)\

    embedding = tf.get_variable(
        "embedding", [vocab_size, num_hiddens[0]], dtype=tf.float32)
    x = tf.nn.embedding_lookup(embedding, x)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cells = [rnn.BasicLSTMCell(i, forget_bias=1.0,activation=tf.nn.leaky_relu) for i in num_hiddens]
    cells = tf.contrib.rnn.MultiRNNCell(lstm_cells)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(cells, x, dtype=tf.float32)
    batch_time_shape = tf.shape(outputs)

    # final_outputs = tf.reshape(
    #     tf.nn.softmax(outputs),
    #     (batch_time_shape[0], batch_time_shape[1], vocab_size)
    # )
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

# Initialize the variables (i.e. assign their default value)
NUM_TRAIN_BATCHES = 20000
# data, vocab = load_data(input_file)
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # saver = tf.train.Saver(tf.global_variables())
    # check_point = model.check_point_dir + '\model.ckpt'
    saver = tf.train.Saver(tf.global_variables())
    saved_file = "multilayer_with_embedding/model.ckpt"

    check_restore_parameters(sess,saver,saved_file)
    last_time = time.time()
    batch_x = np.zeros((batch_size, timesteps))
    batch_y = np.zeros((batch_size, timesteps, vocab_size))
    possible_batch_ids = range(len(data) - timesteps - 1)

    for i in range(NUM_TRAIN_BATCHES):
        # Sample time_steps consecutive samples from the dataset text file
        batch_id = random.sample(possible_batch_ids, batch_size)

        for j in range(timesteps):
            ind1 = [k + j for k in batch_id]
            ind2 = [k + j + 1 for k in batch_id]
            for zz in range(len(ind1)):
                batch_x[zz, j] = data[ind1[zz]]
            for zz in range(len(ind2)):
                batch_y[zz, j, :] = get_one_hot(data[ind2[zz]], vocab)

        loss = sess.run([loss_op], feed_dict={X: batch_x,
                                              Y: batch_y})

        if (i % display_step) == 0:
            new_time = time.time()
            diff = new_time - last_time
            last_time = new_time
            saver.save(sess, saved_file)
            print("batch_x: {}  loss: {}  speed: {} batches / s".format(
            i, loss, display_step / diff
            ))
