#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf


def softmax_with_len(inputs, length, max_len):
    inputs = tf.cast(inputs, tf.float32)
    # max_axis = tf.reduce_max(inputs, -1, keep_dims=True)
    # inputs = tf.exp(inputs - max_axis)
    inputs = tf.exp(inputs)
    length = tf.reshape(length, [-1])
    mask = tf.reshape(tf.cast(tf.sequence_mask(length, max_len), tf.float32), tf.shape(inputs))
    inputs *= mask
    _sum = tf.reduce_sum(inputs, reduction_indices=-1, keep_dims=True) + 1e-9
    # with tf.variable_scope("attention_sentence"):
    #     alpha = tf.div(inputs, _sum, name='attention_sen')
    #     # alpha = tf.Variable(alpha, name='attention_sen')
    #     # alpha = tf.Variable(alpha1, tf.float32)
    return inputs / _sum


def bilinear_attention_layer(inputs, attend, length, n_hidden, l2_reg, random_base, layer_id=1):
    """
    :param inputs: batch * max_len * n_hidden
    :param attend: batch * n_hidden
    :param length:
    :param n_hidden:
    :param l2_reg:
    :param random_base:
    :param layer_id:
    :return:
    """
    batch_size = tf.shape(inputs)[0]
    max_len = tf.shape(inputs)[1]
    w = tf.get_variable(
        name='att_w_' + str(layer_id),
        shape=[n_hidden, n_hidden],
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    # [None*max_len, 2*hidden]
    # [batch_size*max_sentence_len, 2*hidden+embedding+position_embedding]
    inputs = tf.reshape(inputs, [-1, n_hidden])
    # [None, max_len, 2*hidden+embedding+position_embedding]
    # inputs_tmp = tf.tanh(tf.matmul(inputs, w) + b)
    tmp = tf.reshape(tf.matmul(inputs, w), [-1, max_len, n_hidden])
    # [None, 2*hidden, 1]
    attend = tf.expand_dims(attend, 2)
    # [None, 1, max_len]
    tmp = tf.reshape(tf.matmul(tmp, attend), [batch_size, 1, max_len])
    # M = tf.expand_dims(tf.matmul(attend, w), 2)
    # tmp = tf.reshape(tf.batch_matmul(inputs, M), [batch_size, 1, max_len])
    return softmax_with_len(tmp, length, max_len)


def position_attention_layer(inputs, attend, input_position, position_embedding_dim, length, n_hidden, l2_reg,
                             random_base, layer_id=1):
    batch_size = tf.shape(inputs)[0]
    max_len = tf.shape(inputs)[1]

    w = tf.get_variable(
        name='att_w_' + str(layer_id),
        shape=[n_hidden, n_hidden],
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    b = tf.get_variable(
        name='att_b' + str(layer_id),
        shape=[n_hidden],
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    w1 = tf.get_variable(
        name='att_p1_' + str(layer_id),
        shape=[position_embedding_dim, 1],
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )

    # [None*max_len, 2*hidden]
    inputs = tf.reshape(inputs, [-1, n_hidden])
    # [batch*max_len, position_embedding]

    inputs_position = tf.reshape(input_position, [-1, position_embedding_dim])

    tmp_position = tf.reshape(tf.matmul(inputs_position, w1), [batch_size, 1, max_len])

    # [None, max_len, 2*hidden]
    inputs_tmp = tf.tanh(tf.matmul(inputs, w) + b)
    tmp = tf.reshape(inputs_tmp, [-1, max_len, n_hidden])
    # tmp = tf.reshape(tf.matmul(inputs, w), [-1, max_len, n_hidden])
    # [None, max_len, 2*hidden]
    # tmp = tf.concat([tmp, tmp_position], 2)
    # tmp = tf.reshape(tmp, [-1, n_hidden + position_embedding_dim])
    # tmp = tf.reshape(tf.matmul(tmp, w2), [-1, max_len, n_hidden])

    # tmp = tf.tanh(tmp)

    # [None, 2*hidden, 1]
    attend = tf.expand_dims(attend, 2)
    # [None, 1, max_len]
    tmp = tf.reshape(tf.matmul(tmp, attend), [batch_size, 1, max_len])
    tmp += tmp_position

    # M = tf.expand_dims(tf.matmul(attend, w), 2)
    # tmp = tf.reshape(tf.batch_matmul(inputs, M), [batch_size, 1, max_len])
    return softmax_with_len(tmp, length, max_len)


# [batch, max_sen_len, 2*hidden]
def mlp_attention_layer(inputs, length, n_hidden, l2_reg, random_base, layer_id=1):
    """
    :param inputs: batch * max_len * n_hidden
    :param length: batch * 1
    :param n_hidden:
    :param l2_reg:
    :param random_base:
    :param layer_id: layer's identical id
    :return: batch * 1 * max_len
    """
    batch_size = tf.shape(inputs)[0]
    max_len = tf.shape(inputs)[1]
    w = tf.get_variable(
        name='att_w_' + str(layer_id),
        shape=[n_hidden, n_hidden],
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    u = tf.get_variable(
        name='att_u_' + str(layer_id),
        shape=[n_hidden, 1],
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    inputs = tf.reshape(inputs, [-1, n_hidden])
    tmp = tf.matmul(inputs, w)
    tmp = tf.reshape(tf.matmul(tmp, u), [batch_size, 1, max_len])
    alpha = softmax_with_len(tmp, length, max_len)
    return alpha
