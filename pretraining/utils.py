#!/usr/bin/env python
# encoding: utf-8
# @author: zhaof


import numpy as np


def batch_index(length, batch_size, n_iter=100, is_shuffle=True):
    index = list(range(length))
    for j in range(n_iter):
        if is_shuffle:
            np.random.shuffle(index)
        for i in range(int(length / batch_size) + (1 if length % batch_size else 0)):
            yield index[i * batch_size:(i + 1) * batch_size]


def load_word_id_mapping(word_id_file, encoding='utf8'):
    """
    :param word_id_file: word-id mapping file path
    :param encoding: file's encoding, for changing to unicode
    :return: word-id mapping, like hello=5
    """
    word_to_id = dict()
    for line in open(word_id_file):
        line = line.encode(encoding, 'ignore').decode(encoding, 'ignore').lower().split()
        word_to_id[line[0]] = int(line[1])
    print('\nload word-id mapping done!\n')
    return word_to_id


def load_w2v(w2v_file, embedding_dim, is_skip=False):
    fp = open(w2v_file)
    w2v = []
    word_dict = dict()
    # [0,0,...,0] represent absent words
    w2v.append([0.] * embedding_dim)
    cnt = 0
    # a_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for line in fp:
        line = line.encode('utf8', 'ignore').decode('utf8', 'ignore').split()
        # line = line.split()
        if len(line) != embedding_dim + 1:
            print(u'a bad word embedding: {}'.format(line[0]))
            continue
        cnt += 1
        # if line[0] ==
        w2v.append([float(v) for v in line[1:]])
        word_dict[line[0]] = cnt
    w2v = np.asarray(w2v, dtype=np.float32)
    w2v = np.row_stack((w2v, np.sum(w2v, axis=0) / cnt))
    print(np.shape(w2v))
    word_dict['$t$'] = (cnt + 1)
    print(word_dict['$t$'], len(w2v))
    return word_dict, w2v


def change_y_to_onehot(y):
    class_set = set(y)
    n_class = len(class_set)
    y_onehot_mapping = dict(zip(class_set, range(n_class)))
    onehot = []
    for label in y:
        tmp = [0] * n_class
        tmp[y_onehot_mapping[label]] = 1
        onehot.append(tmp)
    return np.asarray(onehot, dtype=np.int32)


def load_inputs(input_file, word_id_file, sentence_len):
    word_to_id = word_id_file
    print('load word-to-id done!')
    x, y, sen_len = [], [], []
    lines = open(input_file).readlines()
    for i in range(len(lines)):
        words_2 = lines[i].lower().strip().split('\t')
        y.append(words_2[0])
        words = []
        for word in words_2[1].split():
            word = ''.join(word)
            if word in word_to_id:
                words.append(word_to_id[word])
            else:
                words.append(word_to_id['$t$'])
        sen_len.append(len(words))
        words = words[:sentence_len]
        x.append(words + [0] * (sentence_len - len(words)))
    y = change_y_to_onehot(y)
    return np.asarray(x), np.asarray(sen_len), np.asarray(y)


def load_inputs_test(input_file, word_id_file, sentence_len):
    word_to_id = word_id_file
    x, sen_len = [], []
    lines = open(input_file).readlines()
    for i in range(len(lines)):
        words_2 = lines[i].lower().strip()
        words = []
        for word in words_2.split():
            if word in word_to_id:
                words.append(word_to_id[word])
            else:
                words.append(word_to_id['$t$'])
        sen_len.append(len(words))
        words = words[:sentence_len]
        x.append(words + [0] * (sentence_len - len(words)))
    return np.asarray(x), np.asarray(sen_len)
