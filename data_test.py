# -*- coding: utf-8 -*-
# @Time    : 2019-01-17 10:21
# @Author  : Maximus
# @Site    : 
# @File    : data_test.py
# @Software: PyCharm
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np
import logging
import tensorflow as tf

sys.path.append(os.path.normpath('{}/'.format(os.path.dirname(os.path.abspath(__file__)))))
logger = logging.getLogger(__name__)
from predict_elmo import *


def loadtfrecord(record_file):
    def parse_func(example_proto):
        feature = {'embedding': tf.VarLenFeature(tf.float32),
                   'text': tf.FixedLenFeature([], tf.string),
                   'label_str': tf.FixedLenFeature([], tf.string),
                   'qid': tf.FixedLenFeature([], tf.int64),
                   'label': tf.FixedLenFeature([], tf.int64),
                   }
        parse_example = tf.parse_single_example(example_proto, feature)
        embedding_tensor = tf.reshape(parse_example['embedding'].values, [-1, 1024])
        text_tensor = parse_example['text']
        label_str_tensor = parse_example['label_str']
        qid_tensor = parse_example['qid']
        label_tensor = parse_example['label']
        return embedding_tensor, text_tensor, label_str_tensor, qid_tensor, label_tensor

    return tf.data.TFRecordDataset(record_file).map(parse_func).make_one_shot_iterator().get_next()


if __name__ == '__main__':
    sess = tf.Session()
    embedding_tensor, text_tensor, label_str_tensor, qid_tensor, label_tensor = loadtfrecord('../deep-lib/sentence_embedding/elmo/tf_record_-1_dq_amq_20181106_train_test.pb')
    for i in range(40000):
        embedding, text, label_str, qid, label = sess.run([embedding_tensor, text_tensor, label_str_tensor, qid_tensor, label_tensor])
        if i % 1000 == 0:
            logger.info(f'embedding shape: {embedding.shape}, {text.decode("utf8")}, {label_str.decode("utf8")}, {qid}, {label}')
