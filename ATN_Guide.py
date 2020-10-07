#!/usr/bin/env python
# encoding: utf-8

from sklearn.metrics import precision_score, recall_score, f1_score
from ABSA.nn_layer import softmax_layer, bi_dynamic_rnn, reduce_mean_with_len
from ABSA.att_layer import bilinear_attention_layer
from ABSA.config import *
from ABSA.utils import load_w2v, batch_index, load_inputs_distill
import os


def ATN_Guide(inputs, sen_len, target, sen_len_tr, keep_prob1, keep_prob2, _id='all'):
    cell = tf.contrib.rnn.LSTMCell
    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob1)

    hiddens_s = bi_dynamic_rnn(cell, inputs, FLAGS.n_hidden, sen_len, FLAGS.max_sentence_len, 'sen12' + _id, 'all')

    target = tf.nn.dropout(target, keep_prob=keep_prob1)
    hiddens_t = bi_dynamic_rnn(cell, target, FLAGS.n_hidden, sen_len_tr, FLAGS.max_target_len, 't' + _id, 'all')
    pool_t_1 = reduce_mean_with_len(hiddens_t, sen_len_tr)

    att_s_1 = bilinear_attention_layer(hiddens_s, pool_t_1, sen_len, 2 * FLAGS.n_hidden, FLAGS.l2_reg,
                                       FLAGS.random_base, 'sen1')
    outputs_s_1 = tf.squeeze(tf.matmul(att_s_1, hiddens_s))
    pool_t_2 = outputs_s_1 + pool_t_1

    prob = softmax_layer(pool_t_2, 2 * FLAGS.n_hidden, FLAGS.random_base, keep_prob2, FLAGS.l2_reg, FLAGS.n_class)
    return prob, att_s_1


def main(_):
    word_id_mapping, w2v = load_w2v(FLAGS.embedding_file_path, FLAGS.embedding_dim)
    word_embedding = tf.constant(w2v, name='word_embedding')

    tr_x, tr_sen_len, tr_target_word, tr_tar_len, tr_y, tr_position, tr_attention_1 = load_inputs_distill(
        FLAGS.train_file_path,
        word_id_mapping,
        FLAGS.max_sentence_len,
        FLAGS.max_target_len
    )
    te_x, te_sen_len, te_target_word, te_tar_len, te_y, te_position, te_attention_1 = load_inputs_distill(
        FLAGS.test_file_path,
        word_id_mapping,
        FLAGS.max_sentence_len,
        FLAGS.max_target_len
    )

    keep_prob1 = tf.placeholder(tf.float32, name='input_keep_prob1')
    keep_prob2 = tf.placeholder(tf.float32, name='input_keep_prob2')
    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len], name='input_x')
        y = tf.placeholder(tf.float32, [None, FLAGS.n_class], name='input_y')
        sen_len = tf.placeholder(tf.int32, [None], name='input_sen_len')
        target_words = tf.placeholder(tf.int32, [None, FLAGS.max_target_len], name='input_target')
        tar_len = tf.placeholder(tf.int32, [None], name='input_tar_len')
        position = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len], name='position')
        attention1 = tf.placeholder(tf.float32, [None, FLAGS.max_sentence_len], name='attention_parameter_1')
    inputs_s = tf.nn.embedding_lookup(word_embedding, x)

    position_embeddings = tf.get_variable(
        name='position_embedding',
        shape=[FLAGS.max_sentence_len, FLAGS.position_embedding_dim],
        initializer=tf.random_uniform_initializer(-FLAGS.random_base, FLAGS.random_base),
        regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg)
    )
    input_position = tf.nn.embedding_lookup(position_embeddings, position)
    inputs_s = tf.concat([inputs_s, input_position], 2)

    u_t = tf.cast(position, tf.float32) / tf.expand_dims(tf.cast(sen_len, tf.float32), -1)
    w_t = 1.0 - tf.abs(u_t)
    atten1 = w_t * attention1

    atten1 = tf.cast(atten1, tf.float32) / tf.expand_dims(tf.cast(tf.reduce_sum(atten1, 1), tf.float32), -1)

    target = tf.nn.embedding_lookup(word_embedding, target_words)
    prob, att_1_s = ATN_Guide(inputs_s, sen_len, target, tar_len, keep_prob1, keep_prob2, FLAGS.t1)

    att_s_11 = tf.reshape(tf.squeeze(att_1_s), [-1, FLAGS.max_sentence_len])

    loss3 = FLAGS.Auxiliary_loss * tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=atten1, logits=att_s_11))

    loss1 = loss_func(y, prob)
    loss = loss1 + loss3

    acc_num, acc_prob = acc_func(y, prob)

    global_step = tf.Variable(0, name='tr_global_step', trainable=False)
    optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate, momentum=0.9).minimize(loss,
                                                                                                     global_step=global_step)
    true_y = tf.argmax(y, 1, name='true_y_1')
    pred_y = tf.argmax(prob, 1, name='pred_y_1')

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        init = tf.global_variables_initializer()
        sess.run(init)

        def get_batch_data(x_f, sen_len_f, yi, target, tl, x_poisition, x_attention_1,
                           batch_size, kp1, kp2, is_shuffle=True):
            for index in batch_index(len(yi), batch_size, 1, is_shuffle):
                feed_dict = {
                    x: x_f[index],
                    y: yi[index],
                    sen_len: sen_len_f[index],
                    target_words: target[index],
                    tar_len: tl[index],
                    position: x_poisition[index],
                    attention1: x_attention_1[index],
                    keep_prob1: kp1,
                    keep_prob2: kp2,
                }
                yield feed_dict, len(index)

        checkpoint_dir = os.path.abspath(FLAGS.saver_checkpoint)
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        max_acc = 0.
        max_ty, max_py = None, None
        for i in range(FLAGS.n_iter):
            for train, _ in get_batch_data(tr_x, tr_sen_len, tr_y, tr_target_word, tr_tar_len, tr_position,
                                           tr_attention_1, FLAGS.batch_size, FLAGS.keep_prob1, FLAGS.keep_prob2):
                _, step = sess.run([optimizer, global_step], feed_dict=train)

            acc, cost, cnt = 0., 0., 0
            s1, ty, py = [], [], []
            p = []
            for test, num in get_batch_data(te_x, te_sen_len, te_y, te_target_word, te_tar_len, te_position,
                                            te_attention_1, 2000, 1.0, 1.0, False):
                _loss, _acc, _s1, _ty, _py, _p = sess.run([loss, acc_num, att_1_s, true_y, pred_y, prob],
                                                          feed_dict=test)
                s1 += list(_s1)
                ty += list(_ty)
                py += list(_py)
                p += list(_p)
                acc += _acc
                cost += _loss * num
                cnt += num
            print('all samples={}, correct prediction={}'.format(cnt, acc))
            acc = acc / cnt
            cost = cost / cnt
            print('Iter {}: mini-batch loss={:}, test acc={:}'.format(i, cost, acc))

            current_step = tf.train.global_step(sess, global_step)
            if acc > max_acc:
                max_acc = acc
                max_ty = ty
                max_py = py
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
        P = precision_score(max_ty, max_py, average=None)
        R = recall_score(max_ty, max_py, average=None)
        F1 = f1_score(max_ty, max_py, average=None)
        print('P:', P, 'avg=', sum(P) / FLAGS.n_class)
        print('R:', R, 'avg=', sum(R) / FLAGS.n_class)
        print('F1:', F1, 'avg=', sum(F1) / FLAGS.n_class)
        print('Optimization Finished! Max acc={}'.format(max_acc))


if __name__ == '__main__':
    tf.app.run()
