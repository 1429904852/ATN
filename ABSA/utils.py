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
    if is_skip:
        fp.readline()
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
    # w2v -= np.mean(w2v, axis=0)
    # w2v /= np.std(w2v, axis=0)
    print(word_dict['$t$'], len(w2v))
    return word_dict, w2v


def change_y_to_onehot(y):
    from collections import Counter
    print(Counter(y))
    class_set = set(y)
    n_class = len(class_set)
    y_onehot_mapping = dict(zip(class_set, range(n_class)))
    print(y_onehot_mapping)
    onehot = []
    for label in y:
        tmp = [0] * n_class
        tmp[y_onehot_mapping[label]] = 1
        onehot.append(tmp)
    return np.asarray(onehot, dtype=np.int32)


def load_inputs_twitter_position(input_file, word_id_file, sentence_len, target_len=10, encoding='utf8'):
    if type(word_id_file) is str:
        word_to_id = load_word_id_mapping(word_id_file)
    else:
        word_to_id = word_id_file
    print('load word-to-id done!')
    x, y, sen_len = [], [], []
    target_words = []
    tar_len = []
    position = []
    lines = open(input_file).readlines()
    for i in range(0, len(lines), 3):
        words_2 = lines[i].encode(encoding).decode(encoding).lower().split()
        target_word = []
        for w in words_2:
            if w in word_to_id:
                target_word.append(word_to_id[w])
        l = min(len(target_word), target_len)
        tar_len.append(l)
        target_words.append(target_word[:l] + [0] * (target_len - l))

        y.append(lines[i + 1].strip().split()[0])
        words_1 = lines[i + 2].encode(encoding).decode(encoding).lower().split()

        words, pp = [], []
        for word in words_1:
            t = word.split('/')
            ind = int(t[-1])
            word = ''.join(t[:-1])
            if word in word_to_id:
                words.append(word_to_id[word])
                pp.append(ind)
            else:
                words.append(word_to_id['$t$'])
                pp.append(ind)

        sen_len.append(len(words))
        words = words[:sentence_len]
        pp = pp[:sentence_len]
        x.append(words + [0] * (sentence_len - len(words)))
        position.append(pp + [0] * (sentence_len - len(pp)))
    y = change_y_to_onehot(y)
    return np.asarray(x), np.asarray(sen_len), np.asarray(target_words), \
           np.asarray(tar_len), np.asarray(y), np.asarray(position)


def load_inputs_distill(input_file, word_id_file, sentence_len, target_len=10, encoding='utf8'):
    if type(word_id_file) is str:
        word_to_id = load_word_id_mapping(word_id_file)
    else:
        word_to_id = word_id_file
    print('load word-to-id done!')
    ploritylabel = {
        0: [1, 0],
        1: [0, 1]
    }
    tag2label = {
        "1": [0, 0, 1],
        "0": [0, 1, 0],
        "-1": [1, 0, 0]
    }

    x, y, sen_len, attention_label_1 = [], [], [], []
    target_words = []
    tar_len = []
    position = []
    attention_sum_1, attention_sum_2, attention_sum_3 = [], [], []
    lines = open(input_file).readlines()
    for i in range(0, len(lines), 7):
        # target
        words_2 = lines[i].encode(encoding).decode(encoding).lower().split()
        target_word = []
        for w in words_2:
            if w in word_to_id:
                target_word.append(word_to_id[w])
        l = min(len(target_word), target_len)
        tar_len.append(l)
        target_words.append(target_word[:l] + [0] * (target_len - l))

        # y
        y.append(lines[i + 1].strip().split()[0])

        # sentence
        words_1 = lines[i + 2].encode(encoding).decode(encoding).lower().split()
        words, pp = [], []
        for word in words_1:
            t = word.split('/')
            ind = int(t[-1])
            word = ''.join(t[:-1])
            if word in word_to_id:
                words.append(word_to_id[word])
                pp.append(ind)
        sen_len.append(len(words))
        words = words[:sentence_len]
        pp = pp[:sentence_len]
        x.append(words + [0] * (sentence_len - len(words)))
        position.append(pp + [0] * (sentence_len - len(pp)))

        # attention_1
        attention1 = []
        attention_words_1 = lines[i + 3].encode(encoding).decode(encoding).lower().split()
        for attention_1 in attention_words_1:
            attention1.append(attention_1)
        attention1 = attention1[:sentence_len]
        attention_sum_1.append(attention1 + [0] * (sentence_len - len(attention1)))

        # attention_2
        attention2 = [] 
        attention_words_2 = lines[i + 4].encode(encoding).decode(encoding).lower().split()
        for attention_2 in attention_words_2:
            attention2.append(attention_2)
        attention2 = attention2[:sentence_len]
        attention_sum_2.append(attention2 + [0] * (sentence_len - len(attention2)))

        # attention_3
        attention3 = []
        attention_words_3 = lines[i + 5].encode(encoding).decode(encoding).lower().split()
        for attention_3 in attention_words_3:
            attention3.append(attention_3)
        attention3 = attention3[:sentence_len]
        attention_sum_3.append(attention3 + [0] * (sentence_len - len(attention3)))

        # attention label
        attention1 = []
        attention_words_1 = lines[i + 6].split()
        for attention_1 in attention_words_1:
            attention1.append(int(attention_1))
        attention1 = attention1[:sentence_len]
        attention2 = attention1 + [0] * (sentence_len - len(attention1))
        attention3 = [ploritylabel[tag] for tag in attention2]
        attention_label_1.append(attention3)

    # y = change_y_to_onehot(y)
    y = [tag2label[tag] for tag in y]
    return np.asarray(x), np.asarray(sen_len), np.asarray(target_words), \
           np.asarray(tar_len), np.asarray(y), np.asarray(position), np.asarray(attention_sum_1), \
           np.asarray(attention_sum_2), np.asarray(attention_sum_3), np.asarray(attention_label_1)


def load_inputs_distill1(input_file, word_id_file, sentence_len, target_len=10, encoding='utf8'):
    if type(word_id_file) is str:
        word_to_id = load_word_id_mapping(word_id_file)
    else:
        word_to_id = word_id_file
    print('load word-to-id done!')
    x, y, sen_len = [], [], []
    target_words = []
    tar_len = []
    position = []
    attention_sum = []
    lines = open(input_file).readlines()
    for i in range(0, len(lines), 4):
        # target
        words_2 = lines[i].encode(encoding).decode(encoding).lower().split()
        target_word = []
        for w in words_2:
            if w in word_to_id:
                target_word.append(word_to_id[w])
        l = min(len(target_word), target_len)
        tar_len.append(l)
        target_words.append(target_word[:l] + [0] * (target_len - l))

        # y
        y.append(lines[i + 1].strip().split()[0])

        # sentence
        words_1 = lines[i + 2].encode(encoding).decode(encoding).lower().split()
        words, pp = [], []
        for word in words_1:
            t = word.split('/')
            ind = int(t[-1])
            word = ''.join(t[:-1])
            if word in word_to_id:
                words.append(word_to_id[word])
                pp.append(ind)
        sen_len.append(len(words))
        words = words[:sentence_len]
        pp = pp[:sentence_len]
        x.append(words + [0] * (sentence_len - len(words)))
        position.append(pp + [0] * (sentence_len - len(pp)))
        
        # attention
        attention = []
        attention_words_1 = lines[i + 3].encode(encoding).decode(encoding).lower().split()
        for attention_1 in attention_words_1:
            attention.append(attention_1)
        attention = attention[:sentence_len]
        attention_sum.append(attention + [0] * (sentence_len - len(attention)))

    y = change_y_to_onehot(y)
    return np.asarray(x), np.asarray(sen_len), np.asarray(target_words), \
           np.asarray(tar_len), np.asarray(y), np.asarray(position), np.asarray(attention_sum)

