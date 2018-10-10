import tensorflow as tf
import os
def get_trainable_variables_num():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    return total_parameters


def check_restore_parameters(sess, saver, path):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(path))
    if ckpt and ckpt.model_checkpoint_path:
        print('Restoring parameters')
        saver.restore(sess, ckpt.model_checkpoint_path)

