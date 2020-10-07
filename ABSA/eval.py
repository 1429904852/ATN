#! /usr/bin/env python

import tensorflow as tf
from ABSA.utils import load_w2v, load_inputs_twitter_position, batch_index
from sklearn.metrics import precision_score, recall_score, f1_score

tf.app.flags.DEFINE_string('test_file_path', 'data/restaurant/rest_2014_dmn_test_new_1.txt', 'testing file')
tf.app.flags.DEFINE_string('embedding_file_path', 'data/restaurant/rest_2014_word_embedding_300_new.txt',
                           'embedding file')
tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')

tf.flags.DEFINE_string("checkpoint_dir", "data/restaurant/checkpoint", "Checkpoint directory from training run")
tf.app.flags.DEFINE_integer('max_sentence_len', 80, 'max number of tokens per sentence')
tf.app.flags.DEFINE_integer('max_target_len', 10, 'max target length')
tf.app.flags.DEFINE_string('result_file', 'MF1833103.txt', 'prob')
tf.app.flags.DEFINE_string('prob_file', 'prob1.txt', 'prob')
tf.app.flags.DEFINE_string('att_s_file', 'att_s.txt', 'prob1')
tf.app.flags.DEFINE_string('att_t_file', 'att_t.txt', 'prob2')
tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

word_id_mapping = load_w2v(FLAGS.embedding_file_path, FLAGS.embedding_dim)
te_x, te_sen_len, te_target_word, te_tar_len, te_y, te_position = load_inputs_twitter_position(FLAGS.test_file_path,
                                                                                               word_id_mapping,
                                                                                               FLAGS.max_sentence_len,
                                                                                               FLAGS.max_target_len)

print("\nTest...\n")
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        x = graph.get_operation_by_name("inputs/input_x").outputs[0]
        sen_len = graph.get_operation_by_name("inputs/input_sen_len").outputs[0]
        target_words = graph.get_operation_by_name("inputs/input_target").outputs[0]
        tar_len = graph.get_operation_by_name("inputs/input_tar_len").outputs[0]
        position = graph.get_operation_by_name("inputs/position").outputs[0]
        keep_prob1 = graph.get_operation_by_name("input_keep_prob1").outputs[0]
        keep_prob2 = graph.get_operation_by_name("input_keep_prob2").outputs[0]
        att_s = graph.get_operation_by_name("att_w_sen1").outputs[0]
        y = graph.get_operation_by_name("inputs/input_y").outputs[0]
        true_y = graph.get_operation_by_name("true_y_1").outputs[0]
        pred_y = graph.get_operation_by_name("pred_y_1").outputs[0]
        acc_num = graph.get_operation_by_name("acc_number").outputs[0]


        def get_batch_data(x_f, sen_len_f, yi, target, tl, x_poisition, batch_size, kp1, kp2, is_shuffle=True):
            for index in batch_index(len(yi), batch_size, 1, is_shuffle):
                feed_dict = {
                    x: x_f[index],
                    y: yi[index],
                    sen_len: sen_len_f[index],
                    target_words: target[index],
                    tar_len: tl[index],
                    position: x_poisition[index],
                    keep_prob1: kp1,
                    keep_prob2: kp2,
                }
                yield feed_dict, len(index)


        acc, cnt = 0., 0
        s, ty, py = [], [], []

        for test1, num in get_batch_data(te_x, te_sen_len, te_y, te_target_word, te_tar_len, te_position, 2000, 1.0,
                                         1.0, False):
            _acc, _ty, _py, _s = sess.run([acc_num, true_y, pred_y, att_s], feed_dict=test1)
            ty += list(_ty)
            py += list(_py)
            s += list(_s)
            acc += _acc
            cnt += num
        print('all samples={}, correct prediction={}'.format(cnt, acc))
        acc = acc / cnt
        print('test acc={:.6f}'.format(acc))
        P = precision_score(ty, py, average=None)
        R = recall_score(ty, py, average=None)
        F1 = f1_score(ty, py, average=None)
        print('P:', P, 'avg=', sum(P) / FLAGS.n_class)
        print('R:', R, 'avg=', sum(R) / FLAGS.n_class)
        print('F1:', F1, 'avg=', sum(F1) / FLAGS.n_class)

        fp = open(FLAGS.att_s_file, 'w')
        for y1, y2, ws in zip(ty, py, s):
            fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0] if w != 0.0]) + '\n')
