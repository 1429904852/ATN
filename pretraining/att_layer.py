#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf


def softmax_with_len(inputs, length, max_len, layer_id):
    inputs = tf.cast(inputs, tf.float32)
    # max_axis = tf.reduce_max(inputs, -1, keep_dims=True)
    # inputs = tf.exp(inputs - max_axis)
    inputs = tf.exp(inputs)
    length = tf.reshape(length, [-1])
    mask = tf.reshape(tf.cast(tf.sequence_mask(length, max_len), tf.float32), tf.shape(inputs))
    inputs *= mask
    _sum = tf.reduce_sum(inputs, reduction_indices=-1, keep_dims=True) + 1e-9
    with tf.variable_scope("attention_sentence"):
        alpha = tf.div(inputs, _sum, name='attention_sen' + str(layer_id))
        print(alpha.name)
    return alpha


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
    return softmax_with_len(tmp, length, max_len, layer_id)


