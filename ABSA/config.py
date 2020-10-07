#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('position_embedding_dim', 200, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('batch_size', 25, 'number of example per batch')
tf.app.flags.DEFINE_integer('n_hidden', 300, 'number of hidden unit')
tf.app.flags.DEFINE_float('learning_rate', 0.1, 'learning rate')
tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
tf.app.flags.DEFINE_integer('plority', 2, 'number of distinct polarity')
tf.app.flags.DEFINE_integer('max_sentence_len', 80, 'max number of tokens per sentence')
tf.app.flags.DEFINE_integer('max_target_len', 10, 'max target length')
tf.app.flags.DEFINE_float('l2_reg', 0.00001, 'l2 regularization')
tf.app.flags.DEFINE_float('random_base', 0.1, 'initial random base')
tf.app.flags.DEFINE_float('Auxiliary_loss', 0.2, 'Auxiliary loss')
tf.app.flags.DEFINE_integer('display_step', 4, 'number of test display step')
tf.app.flags.DEFINE_integer('n_iter', 50, 'number of train iter')
tf.app.flags.DEFINE_float('keep_prob1', 0.5, 'dropout keep prob')
tf.app.flags.DEFINE_float('keep_prob2', 0.5, 'dropout keep prob')
tf.app.flags.DEFINE_string('t1', 'last', 'type of hidden output')
tf.app.flags.DEFINE_string('t2', 'last', 'type of hidden output')
tf.app.flags.DEFINE_integer('n_layer', 3, 'number of stacked rnn')
# tf.app.flags.DEFINE_string('saver_checkpoint', 'data/restaurant/checkpoint', 'prob')
# tf.app.flags.DEFINE_string('saver_checkpoint', 'data/laptop/checkpoint', 'prob')
tf.app.flags.DEFINE_string('saver_checkpoint', 'data/twitter/checkpoint', 'prob')

tf.app.flags.DEFINE_string('pre_trained_path', 'data/yelp/checkpoint8/', 'pretrain file')
tf.app.flags.DEFINE_string('pre_trained', 'sentence_tranfer', 'pre_trained model')

# tf.app.flags.DEFINE_string('train_file_path', 'data/restaurant/rest_2014_dmn_train_new_1.txt', 'training file')
# tf.app.flags.DEFINE_string('validate_file_path', 'data/restaurant/rest_2014_dmn_test_new_1.txt', 'validating file')
# tf.app.flags.DEFINE_string('test_file_path', 'data/restaurant/rest_2014_dmn_test_new_1.txt', 'testing file')
# tf.app.flags.DEFINE_string('embedding_file_path', 'data/restaurant/restaurant_2014_840b_300.txt',
#                            'embedding file')

# tf.app.flags.DEFINE_string('train_file_path', 'data/restaurant/rest_2014_dmn_position_distill_train.txt', 'training file')
# tf.app.flags.DEFINE_string('validate_file_path', 'data/restaurant/rest_2014_dmn_position_distill_test.txt', 'validating file')
# tf.app.flags.DEFINE_string('test_file_path', 'data/restaurant/rest_2014_dmn_position_distill_test.txt', 'testing file')
# tf.app.flags.DEFINE_string('embedding_file_path', 'data/restaurant/restaurant_2014_840b_300.txt', 'embedding file')

# tf.app.flags.DEFINE_string('train_file_path', 'data/restaurant/rest_2014_multitop_train_new_1.txt', 'training file')
# tf.app.flags.DEFINE_string('validate_file_path', 'data/restaurant/rest_2014_multitop_test_new_1.txt', 'validating file')
# tf.app.flags.DEFINE_string('test_file_path', 'data/restaurant/rest_2014_multitop_test_new_1.txt', 'testing file')
# tf.app.flags.DEFINE_string('embedding_file_path', 'data/restaurant/restaurant_2014_840b_300.txt', 'embedding file')

# tf.app.flags.DEFINE_string('train_file_path', 'data/restaurant/80_rest_2014_multitop_train_new_1.txt', 'training file')
# tf.app.flags.DEFINE_string('validate_file_path', 'data/restaurant/80_rest_2014_multitop_test_new_1.txt', 'validating file')
# tf.app.flags.DEFINE_string('test_file_path', 'data/restaurant/80_rest_2014_multitop_test_new_1.txt', 'testing file')
# tf.app.flags.DEFINE_string('embedding_file_path', 'data/restaurant/restaurant_2014_840b_300.txt', 'embedding file')

# tf.app.flags.DEFINE_string('train_file_path', 'data/laptop/laptop_2014_position_train.txt', 'training file')
# tf.app.flags.DEFINE_string('validate_file_path', 'data/laptop/laptop_2014_position_test.txt', 'validating file')
# tf.app.flags.DEFINE_string('test_file_path', 'data/laptop/laptop_2014_position_test.txt', 'testing file')
# tf.app.flags.DEFINE_string('embedding_file_path', 'data/laptop/laptop_2014_840b_300.txt',
#                            'embedding file')

# tf.app.flags.DEFINE_string('train_file_path', 'data/laptop/laptop_2014_dmn_position_distill_train.txt', 'training file')
# tf.app.flags.DEFINE_string('validate_file_path', 'data/laptop/laptop_2014_dmn_position_distill_test.txt', 'validating file')
# tf.app.flags.DEFINE_string('test_file_path', 'data/laptop/laptop_2014_dmn_position_distill_test.txt', 'testing file')
# tf.app.flags.DEFINE_string('embedding_file_path', 'data/laptop/laptop_2014_840b_300.txt', 'embedding file')

# tf.app.flags.DEFINE_string('train_file_path', 'data/laptop/laptop_2014_multitop_train_new_1.txt', 'training file')
# tf.app.flags.DEFINE_string('validate_file_path', 'data/laptop/laptop_2014_multitop_test_new_1.txt', 'validating file')
# tf.app.flags.DEFINE_string('test_file_path', 'data/laptop/laptop_2014_multitop_test_new_1.txt', 'testing file')
# tf.app.flags.DEFINE_string('embedding_file_path', 'data/laptop/laptop_2014_840b_300.txt', 'embedding file')

# tf.app.flags.DEFINE_string('train_file_path', 'data/laptop/80_laptop_2014_multitop_train_new_1.txt', 'training file')
# tf.app.flags.DEFINE_string('validate_file_path', 'data/laptop/80_laptop_2014_multitop_test_new_1.txt', 'validating file')
# tf.app.flags.DEFINE_string('test_file_path', 'data/laptop/80_laptop_2014_multitop_test_new_1.txt', 'testing file')
# tf.app.flags.DEFINE_string('embedding_file_path', 'data/laptop/laptop_2014_840b_300.txt', 'embedding file')

# tf.app.flags.DEFINE_string('train_file_path', 'data/twitter/twitter_2014_position_train.txt', 'training file')
# tf.app.flags.DEFINE_string('validate_file_path', 'data/twitter/twitter_2014_position_test.txt', 'validating file')
# tf.app.flags.DEFINE_string('test_file_path', 'data/twitter/twitter_2014_position_test.txt', 'testing file')
# tf.app.flags.DEFINE_string('embedding_file_path', 'data/twitter/twitter_2014_840b_300.txt',
#                            'embedding file')

# tf.app.flags.DEFINE_string('train_file_path', 'data/twitter/twitter_2014_dmn_position_distill_train.txt', 'training file')
# tf.app.flags.DEFINE_string('validate_file_path', 'data/twitter/twitter_2014_dmn_position_distill_test.txt', 'validating file')
# tf.app.flags.DEFINE_string('test_file_path', 'data/twitter/twitter_2014_dmn_position_distill_test.txt', 'testing file')
# tf.app.flags.DEFINE_string('embedding_file_path', 'data/twitter/twitter_2014_840b_300.txt', 'embedding file')

tf.app.flags.DEFINE_string('train_file_path', 'data/twitter/twitter_2014_multitop_train_new_1.txt', 'training file')
tf.app.flags.DEFINE_string('validate_file_path', 'data/twitter/twitter_2014_multitop_test_new_1.txt', 'validating file')
tf.app.flags.DEFINE_string('test_file_path', 'data/twitter/twitter_2014_multitop_test_new_1.txt', 'testing file')
tf.app.flags.DEFINE_string('embedding_file_path', 'data/twitter/twitter_2014_840b_300.txt', 'embedding file')


tf.app.flags.DEFINE_string('prob_file', 'prob1.txt', 'prob')
tf.app.flags.DEFINE_string('att_s_file_1', 'att_s_1.txt', 'prob1')
tf.app.flags.DEFINE_string('att_s_file_2', 'att_s_2.txt', 'prob2')
tf.app.flags.DEFINE_string('att_s_file_3', 'att_s_3.txt', 'prob3')


def print_config():
    FLAGS.flag_values_dict()
    print('\nParameters:')
    for k, v in sorted(FLAGS.__flags.items()):
        print('{}={}'.format(k, v))


def kl(x, y):
    X = tf.distributions.Categorical(probs=x)
    Y = tf.distributions.Categorical(probs=y)
    return tf.distributions.kl_divergence(X, Y, allow_nan_stats=False)


def loss_func(y, prob):
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prob, y))
    loss = - tf.reduce_mean(y * tf.log(prob)) + sum(reg_loss)
    return loss


def acc_func(y, prob):
    correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1))
    acc_num = tf.reduce_sum(tf.cast(correct_pred, tf.int32), name='acc_number')
    acc_prob = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return acc_num, acc_prob


def train_func(loss, r, global_step, optimizer=None):
    # global_step = tf.Variable(0, name="tr_global_step", trainable=False)
    if optimizer:
        return optimizer(learning_rate=r).minimize(loss, global_step=global_step)
    else:
        return tf.train.AdamOptimizer(learning_rate=r).minimize(loss, global_step=global_step)
        # optimizer = tf.train.AdagradOptimizer(learning_rate=r).minimize(loss, global_step=global_step)


def summary_func(loss, acc, test_loss, test_acc, _dir, title, sess):
    summary_loss = tf.summary.scalar('loss' + title, loss)
    summary_acc = tf.summary.scalar('acc' + title, acc)
    test_summary_loss = tf.summary.scalar('loss' + title, test_loss)
    test_summary_acc = tf.summary.scalar('acc' + title, test_acc)
    train_summary_op = tf.summary.merge([summary_loss, summary_acc])
    validate_summary_op = tf.summary.merge([summary_loss, summary_acc])
    test_summary_op = tf.summary.merge([test_summary_loss, test_summary_acc])
    train_summary_writer = tf.summary.FileWriter(_dir + '/train', sess.graph)
    test_summary_writer = tf.summary.FileWriter(_dir + '/test')
    validate_summary_writer = tf.summary.FileWriter(_dir + '/validate')
    return train_summary_op, test_summary_op, validate_summary_op, \
           train_summary_writer, test_summary_writer, validate_summary_writer


def saver_func(_dir):
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=1000)
    import os
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    return saver
