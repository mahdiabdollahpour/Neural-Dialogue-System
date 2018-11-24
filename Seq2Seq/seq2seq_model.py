import tensorflow as tf
import numpy as np

from global_utils import *
import time
import json
from tensorflow.python.layers import core as layers_core
from parametrs import *

global data1, data2, vocab, dict_rev, data1_validation, data2_validation, test1, test2


def load_data(length=None):
    print('Loading Data...')
    global data1, data2, vocab, dict_rev, data1_validation, data2_validation, test1, test2

    # dataset_path = '../datasets/DailyDialog'
    vocab, dict_rev = load_vocab_from_csv(vocab_path)
    if length is not None:
        data1 = load_data_from_csv(train_questions_csv)[:length]
        data2 = load_data_from_csv(train_answers_csv)[:length]
    else:
        data1 = load_data_from_csv(train_questions_csv)
        data2 = load_data_from_csv(train_answers_csv)

    data1_validation = load_data_from_csv(validation_questions_csv)
    data2_validation = load_data_from_csv(validation_answers_csv)
    data1_test = load_data_from_csv(test_questions_csv)
    data2_test = load_data_from_csv(test_answers_csv)

    print('Data is Loaded')
    print('Vocab Length is', len(dict_rev))
    print('Number of Training Examples', len(data1))
    print('Number of Training Examples', len(data2))


global json_loaded_data
global path
global train_op
global loss
global encoder_inputs_placeholder
global decoder_inputs_placeholder
global decoder_lengths
global translations
global decoder_targets_placeholder
global load
global merged_summary


def define_graph(address='saved_model', QA=True):
    global train_op
    global loss_op
    global encoder_inputs_placeholder
    global decoder_inputs_placeholder
    global decoder_lengths
    global translations
    global decoder_targets_placeholder
    global load
    global merged_summary
    global path
    global json_loaded_data

    path = address
    if not os.path.exists(path):
        os.makedirs(path)
        print('Model is being created at :', path + config_file)
        conf = {}
        with open(path + config_file, 'w', encoding='utf-8') as f:
            # global json_loaded_data
            conf['iteration'] = 0
            conf['embedding_size'] = embed_size
            conf['hidden_num'] = hidden_size
            print(conf)
            json_string = json.dumps(conf)
            print(json_string)
            f.write(json_string)
            f.flush()
            json_loaded_data = json.loads(json_string)
        load = False
    else:
        with open(path + config_file, 'r') as f:
            # global json_loaded_data

            stri = f.read()
            print('content loaded :', stri)
            json_loaded_data = json.loads(stri)
        # load = False
        load = True

    src_vocab_size = len(dict_rev)
    tgt_vocab_size = len(dict_rev)

    print(json_loaded_data)
    embedding_size = json_loaded_data['embedding_size']
    hidden_num = json_loaded_data['hidden_num']

    encoder_inputs_placeholder = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
    decoder_inputs_placeholder = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')

    embeddings1 = tf.Variable(tf.random_uniform([src_vocab_size, embedding_size], -1.0, 1.0))
    embeddings2 = None
    if not QA:
        embeddings2 = tf.Variable(tf.random_uniform([tgt_vocab_size, embedding_size], -1.0, 1.0))
        decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings2, decoder_inputs_placeholder)
    else:  # QA
        decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings1, decoder_inputs_placeholder)
        embeddings2 = embeddings1

    encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings1, encoder_inputs_placeholder)

    encoder_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(hidden_num)
                                                for i in range(layer_num)])

    encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
        encoder_cell, encoder_inputs_embedded,
        dtype=tf.float32, time_major=True,
    )
    # del encoder_outputs
    attention_states = tf.transpose(encoder_outputs, [1, 0, 2])

    # Create an attention mechanism
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        hidden_num, attention_states,
        memory_sequence_length=None)

    decoder_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(hidden_num)
                                                for i in range(layer_num)])
    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
        decoder_cell, attention_mechanism,
        attention_layer_size=hidden_num)

    decoder_lengths = tf.placeholder(tf.int32, shape=batch_size, name="decoder_length")
    print('decoder length ', decoder_lengths)
    # batch_size = tf.shape( decoder_lengths)[0]

    helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs_embedded, decoder_lengths, time_major=True)

    projection_layer = layers_core.Dense(
        tgt_vocab_size, use_bias=False)

    initial_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_final_state)

    decoder = tf.contrib.seq2seq.BasicDecoder(
        decoder_cell, helper, initial_state,
        output_layer=projection_layer)

    final_outputs, _final_state, _final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder)
    logits = final_outputs.rnn_output
    decoder_targets_placeholder = tf.placeholder(dtype="int32", shape=(batch_size, decoder_length))

    # Loss
    loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=decoder_targets_placeholder, logits=logits)

    # Train
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Calculate and clip gradients
    params = tf.trainable_variables()
    gradients = tf.gradients(loss_op, params)
    clipped_gradients, _ = tf.clip_by_global_norm(
        gradients, max_gradient_norm)

    # Optimization
    ## TODO: use specified learning rate
    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.apply_gradients(
        zip(clipped_gradients, params), global_step=global_step)

    # Inference
    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings2,
                                                                tf.fill([batch_size],
                                                                        int(dict_rev[start_token])),
                                                                dict_rev[end_token])

    # Inference Decoder
    inference_decoder = tf.contrib.seq2seq.BasicDecoder(
        decoder_cell, inference_helper, initial_state,
        output_layer=projection_layer)

    # We should specify maximum_iterations, it can't stop otherwise.
    # source_sequence_length = encoder_length
    maximum_iterations = tf.round(tf.reduce_max(source_sequence_length) * 2)

    # Dynamic decoding
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
        inference_decoder, maximum_iterations=maximum_iterations)

    translations = outputs.sample_id


def translate(src_sen):
    src_sen_ = pad_sentence(src_sen.lower(), source_sequence_length)

    by_index = sentence_by_id(src_sen_, dict_rev)
    sequence = np.asarray(by_index)
    print('seq ', sequence)

    inference_encoder_inputs = np.empty((source_sequence_length, batch_size))
    for i in range(0, batch_size):
        inference_encoder_inputs[:, i] = sequence

    feed_dict = {
        encoder_inputs_placeholder: inference_encoder_inputs,
        decoder_lengths: np.ones(batch_size, dtype=int) * decoder_length

    }
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())

        if load:
            print('loading')
            check_restore_parameters(sess, saver, path + model_file_name)
        answer = sess.run([translations], feed_dict)

        print(src_sen)
        answer = np.reshape(answer, [-1, batch_size])
        print(get_sentence_back(answer[0], vocab))


def compute_blue(questions, score_func, session=None):
    with open(validation_answers, encoding='utf-8') as f:
        A_lines = f.readlines()

    answers_given = []
    number_of_batches = int(len(A_lines) / batch_size)
    if session is None:
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        if load:
            print('loading Model')
            check_restore_parameters(session, saver, path + model_file_name)

    with session as sess:

        for j in range(0, number_of_batches):
            enc_inp = np.zeros((source_sequence_length, batch_size), dtype='int')

            for idx in range(batch_size):
                enc_inp[:, idx] = questions[j * batch_size + idx][1:-1]

            feed_dict = {
                encoder_inputs_placeholder: enc_inp,
                decoder_lengths: np.ones((batch_size), dtype=int) * (decoder_length)
            }
            trans = sess.run([translations], feed_dict=feed_dict)
            # print('trans_shape :', np.shape(trans))
            trans = np.reshape(trans, [-1, batch_size])
            for sen in trans:
                # print('sen', sen)
                answers_given.append(get_sentence_back(sen, vocab))

    score = score_func(A_lines[:number_of_batches * batch_size], answers_given)

    return score


def train(logs_dir, use_tensorBoard=False):
    global loss_op
    global train_op
    global iter_loss_placeholder
    display_each = 200
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        global path
        if load:
            check_restore_parameters(sess, saver, path + model_file_name)

        num = get_trainable_variables_num()
        print('Number of trainable variables ', num)
        number_of_batches = int(len(data1) / batch_size)
        print('There are ', number_of_batches, ' batches')
        global json_loaded_data
        start = json_loaded_data['iteration']
        if use_tensorBoard:
            writer = tf.summary.FileWriter(logs_dir)
            writer.add_graph(sess.graph)

        ## TODO: do Early Stopping
        for i in range(start, epoch_num):
            iter_loss = 0
            iter_time = time.time()
            for j in range(number_of_batches):
                enc_inp = np.zeros((source_sequence_length, batch_size), dtype='int')
                dec_inp = np.zeros((decoder_length, batch_size), dtype='int')
                dec_tgt = np.zeros((decoder_length, batch_size), dtype='int')
                # dec_tgt = np.zeros((decoder_length - 1,  batch_size,  tgt_vocab_size), dtype='int')

                for idx in range(batch_size):
                    # print('input :', data2[j * batch_size + idx])
                    dec_inp[:, idx] = data2[j * batch_size + idx][:-1]
                    dec_tgt[:, idx] = data2[j * batch_size + idx][1:]
                    enc_inp[:, idx] = data1[j * batch_size + idx][1:-1]

                feed_dict = {
                    encoder_inputs_placeholder: enc_inp,
                    decoder_inputs_placeholder: dec_inp,
                    decoder_targets_placeholder: np.transpose(dec_tgt),
                    decoder_lengths: np.ones((batch_size), dtype=int) * (decoder_length)
                }

                _, loss = sess.run([train_op, loss_op], feed_dict=feed_dict)
                # print(np.max(labe))
                iter_loss += np.mean(loss)
                if j % display_each == 0:
                    print('Mini batch loss is ', np.mean(loss))
            average_loss = iter_loss / number_of_batches
            print('Average loss in iteration ', i, ' is ', average_loss)
            print("It took ", time.time() - iter_time)
            print('Saving model...')
            if use_tensorBoard:
                tf.summary.scalar('loss', iter_loss)
                iter_loss_placeholder = tf.placeholder(tf.float32, name='iter_loss')
                tf.summary.scalar("loss", iter_loss_placeholder)
                merged_summary = tf.summary.merge_all()

                s = sess.run(merged_summary, feed_dict={iter_loss_placeholder: iter_loss / number_of_batches})
                print(merged_summary)
                writer.add_summary(s, i)
            else:
                # print(logs_dir + log_file_name)
                with open(logs_dir + ".txt", 'a', encoding='utf-8') as f:
                    f.write(str(average_loss) + ',' + str(i) + '\n')
                    f.flush()

            json_loaded_data['iteration'] = i + 1
            with open(path + config_file, 'w', encoding='utf-8') as f:
                f.write(json.dumps(json_loaded_data))
                f.flush()

            saver.save(sess, path + model_file_name)
            print('Model saved')
