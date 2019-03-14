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
import jieba
import re
from functools import partial

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


def label_str_qid_process(df_train, qid_axis, label_str_axis) -> (dict, dict, dict):
    """处理qid label_str label之间的对应关系"""
    grouped = df_train.groupby(df_train.keys()[qid_axis])
    qids = []
    sample_count = []
    label_strs = []
    for grouped_id, group in grouped:
        qids.append(grouped_id)
        sample_count.append(len(group))
        label_str = (group.iloc[0][label_str_axis])
        label_strs.append(label_str.strip() if isinstance(label_str, str) else label_str)
    labels = list(range(len(qids)))
    qid_sample_count = dict(zip(qids, sample_count))
    qid_labels = dict(zip(qids, labels))
    label_str_qid = dict(zip(label_strs, qids))
    return qid_labels, qid_sample_count, label_str_qid


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
    logger.info('=========> source data:{} shape:{}'.format(os.path.basename(file_dir), df.shape))
    # 自定义去掉某些行
    if drop_value is not None and drop_axis is not None:
        col_key = df.keys()[drop_axis]
        # 过滤多个值：多个语义或者多个语义id
        if isinstance(drop_value, (list, set, Set, np.ndarray)):
            df = df[~df[col_key].isin(drop_value)]
        else:
            df = df[df[col_key] != drop_value]
        logger.info('=========> after drop value:{}'.format(df.shape))
    if drop_dup_axis is not None:
        df = df.drop_duplicates(subset=[df.keys()[drop_dup_axis]])
        logger.info('=========> after drop dup:{}'.format(df.shape))
    return df


def sentence2token(sentence, msl):
    sentence = jieba.cut_for_search(sentence)
    tokens = [token for token in sentence if token.isalnum()]
    return tokens[:msl]


def load_vocab() -> dict:
    #
    pass


def token2ids(tokens, word_index: dict):
    return list(map(lambda token: word_index.get(token, 0), tokens))


def file2tfrecord(output_file: str, input_file: str, embedder: ElmoEmbedding, layer_index, msl=35):
    writer = tf.python_io.TFRecordWriter(output_file)
    train_df = pd_reader(input_file)
    data_axis = 1
    qid_axis = 3
    label_str_axis = 0
    domain_label_axis = 5
    train_sentences = train_df.values[:, data_axis]
    # must split into token list!!!!!!!
    train_sentences = list(map(partial(sentence2token, msl=msl), train_sentences))
    train_label_str = train_df.values[:, label_str_axis]
    train_qids = train_df.values[:, qid_axis]
    domain_label = train_df.values[:, domain_label_axis]
    assert len(train_sentences) == len(train_label_str) == len(train_qids)
    batch_gen = OldBatchGenMultiNoneInf([train_sentences, train_qids, domain_label], _batch_size=500)
    for batch in batch_gen:
        batch_sentences, batch_qids, batch_labels = batch[0], batch[1], batch[2]
        datas = embedder.predict(batch_sentences, layer_index=layer_index)
        for i, data in enumerate(datas):
            assert len(data.shape) == 2
            qid = batch_qids[i]
            label = batch_labels[i]
            feature = collections.OrderedDict()
            feature['embedding'] = create_float_feature(data.reshape(-1))
            feature['qid'] = create_int64_feature(qid)
            feature['domain_label'] = create_int64_feature(label)
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
    e = PretrainEmbedder('zhs.model/', 300)
    layer_index = -1
    file2tfrecord('tf_record_{}_{}.pb'.format(layer_index, os.path.basename(inout_file).split('.')[0]), inout_file, e, layer_index=layer_index)
