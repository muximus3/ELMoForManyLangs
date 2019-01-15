# -*- coding: utf-8 -*-
# @Time    : 2019-01-14 15:11
# @Author  : Maximus
# @Site    : 
# @File    : predict_elmo.py
# @Software: PyCharm
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np
import logging
from typing import *
from batch_generator_old import OldBatchGenMultiNoneInf
import pandas as pd
import tensorflow as tf
import collections
import json

sys.path.append(os.path.normpath('{}/'.format(os.path.dirname(os.path.abspath(__file__)))))
logger = logging.getLogger(__name__)

from elmoformanylangs import Embedder

flags = tf.flags

FLAGS = flags.FLAGS
flags.DEFINE_string("input_file", None, "")


def save_json(save_path: str, json_obj):
    assert isinstance(json_obj, (dict, list))
    with open(save_path, "w", encoding='utf8') as openfile:
        json.dump(json_obj, openfile)


def create_bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_float_feature(value: list):
    """Returns a float_list from a float / double.
    Args:
      value: 1-d list
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class ElmoEmbedding(object):
    def __init__(self):
        pass

    def predict(self, sentences, layer_index) -> np.ndarray:
        """predict sentences embedding

        Args:
          sentences:

        Returns:
          embedding matrix
        """
        raise NotImplementedError()


def label_str_qid_process(all_qids, all_label_strs=None):
    """针对无序数据也可以使用"""
    qid_index_count = np.unique(all_qids, return_index=True, return_counts=True)
    sample_index = list(map(int, qid_index_count[1]))
    count = list(map(int, qid_index_count[2]))
    # 传入的qid就是label_str
    if all_label_strs is None and isinstance(all_qids[0], str):
        label_strs = qid_index_count[0]
        qids = label_strs
    # 传入的qid是qid，
    else:
        label_strs = None if all_label_strs is None else all_label_strs[sample_index]
        qids = list(map(int, qid_index_count[0]))
    labels = list(range(len(sample_index)))
    qid_sample_index = dict(zip(qids, sample_index))
    qid_sample_count = dict(zip(qids, count))
    qid_labels = dict(zip(qids, labels))
    if label_strs is not None:
        label_str_qid = dict(zip(label_strs, qids))
    else:
        label_str_qid = {}
    return qid_labels, qid_sample_count, qid_sample_index, label_str_qid


def pd_reader(file_dir, header: Union[int, type(None)] = 0, usecols: Union[list, type(None)] = None, drop_value: Union[list, str, type(None)] = None,
              drop_axis: Union[int, type(None)] = None, drop_dup_axis: Union[int, type(None)] = None, sep='\t', drop_na_axis: Union[int, type(None)] = 0) -> pd.DataFrame:
    """preprocess files drop special data, and keep useful columns"""
    if file_dir.endswith('.csv'):
        df = pd.read_csv(open(file_dir, 'rU'), engine='c', sep=sep, na_filter=True, skipinitialspace=True, header=header, usecols=usecols, lineterminator='\n')
        # df = pd.read_csv(open(file_dir, 'rU'), engine='python', sep=sep, na_filter=True, skipinitialspace=True, header=header, usecols=usecols)
    elif file_dir.endswith('.xlsx'):
        df = pd.read_excel(file_dir, header=header, usecols=usecols).dropna(axis=drop_na_axis, how='any')
    else:
        raise AssertionError('not supportted type')
    logger.info('=========> source data:{} shape:{}'.format(file_dir[file_dir.rindex('/') + 1:], df.shape))
    # 自定义去掉某些行
    if drop_value is not None and drop_axis is not None:
        col_key = df.keys()[drop_axis]
        # 过滤多个值：多个语义或者多个语义id
        if isinstance(drop_value, (list, set, Set)):
            df = df[~df[col_key].isin(drop_value)]
        else:
            df = df[df[col_key] != drop_value]
        logger.info('=========> after drop value:{}'.format(df.shape))
    if drop_dup_axis is not None:
        df = df.drop_duplicates(subset=[df.keys()[drop_dup_axis]])
        logger.info('=========> after drop dup:{}'.format(df.shape))
    return df


def file2tfrecord(output_file: str, input_file: str, embedder: ElmoEmbedding, layer_index):
    writer = tf.python_io.TFRecordWriter(output_file)
    train_df = pd_reader(input_file, usecols=[0, 1, 2])
    train_sentences = train_df.values[:, 2]
    train_label_str = train_df.values[:, 1]
    train_qids = train_df.values[:, 0]
    qid_labels, qid_sample_count, qid_sample_index, label_str_qid = label_str_qid_process(train_qids, train_label_str)
    train_labels = [qid_labels[qid] for qid in train_qids]
    batch_gen = OldBatchGenMultiNoneInf([train_sentences, train_label_str, train_qids, train_labels], _batch_size=512)
    for batch in batch_gen:
        batch_sentences, batch_label_str, batch_qids, batch_labels = batch[0], batch[1], batch[2], batch[3]
        datas = embedder.predict(batch_sentences, layer_index=layer_index)
        for i, data in enumerate(datas):
            if i % 100 == 0:
                print(data.shape)
            assert len(data.shape) == 2
            text = batch_sentences[i]
            label_str = batch_label_str[i]
            qid = batch_qids[i]
            label = batch_labels[i]
            feature = collections.OrderedDict()
            feature['embedding'] = create_float_feature(data.reshape((data.shape[0] * data.shape[1],)))
            feature['text'] = create_bytes_feature(text.encode('utf8'))
            feature['label_str'] = create_bytes_feature(label_str.encode('utf8'))
            feature['qid'] = create_int64_feature(qid)
            feature['label'] = create_int64_feature(label)
            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(tf_example.SerializeToString())


class PretrainEmbedder(ElmoEmbedding):
    def __init__(self, model_dir, batch_size=64):
        super(PretrainEmbedder, self).__init__()
        self.embedder = Embedder(model_dir, batch_size)

    def predict(self, sentences, layer_index=-1):
        return self.embedder.sents2elmo(sentences, layer_index)


if __name__ == '__main__':
    inout_file = FLAGS.input_file
    logger.info('======> input_file:{}'.format(os.path.basename(inout_file)))
    e = PretrainEmbedder('zhs.model/', 64)
    layer_index = -1
    file2tfrecord('tf_record_{}_{}.pb'.format(layer_index, '_'.join(os.path.basename(inout_file).split('_')[:2])), inout_file, e, layer_index=layer_index)
    '../pre-training/global_data/train_test_data/dq_amq_20181106_train_test.csv'
