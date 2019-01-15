# -*- coding: utf-8 -*-
# @Time    : 2018/10/28 10:56 AM
# @Author  : Muximus
# @Site    :
# @File    : batch_generator.py
# @Software: PyCharm
import os
import sys

_root = os.path.normpath("%s/.." % os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_root)
import numpy as np
import abc
import math


class OldBatchGenerator(abc.ABC):
    def __init__(self, *args):
        pass

    @abc.abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError


class OldBatchGenMultiInf(OldBatchGenerator):
    def __init__(self, _all_x, _all_y, _batch_size: int = 320,
                 _shuffle: bool = True):
        super(OldBatchGenMultiInf, self).__init__()
        self._all_x = _all_x
        self._all_y = _all_y
        self._batch_size = _batch_size
        self._shuffle = _shuffle
        self.x_size = len(self._all_y[0])
        for i in range(len(_all_x) - 1):
            assert len(_all_x[i]) == len(_all_y[i])
        self.step_per_epoch = int(math.ceil(self.x_size / self._batch_size))

    def __len__(self):
        return self.step_per_epoch

    def __iter__(self):
        while True:
            if self._shuffle:
                shuffle_indices = np.random.permutation(np.arange(self.x_size))
                for _j in range(len(self._all_y)):
                    self._all_y[_j] = self._all_y[_j][shuffle_indices]
                    self._all_x[_j] = self._all_x[_j][shuffle_indices]
            for batch_num in range(self.step_per_epoch):
                # print('=======EPOCH START BATCH:{}/{}============'.format(batch_num,step_per_epoch))
                start_index = batch_num * self._batch_size
                end_index = min((batch_num + 1) * self._batch_size, self.x_size)
                datas = []
                labels = []
                for _j in range(len(self._all_y)):
                    datas_per_batch = self._all_x[_j][start_index:end_index]
                    label_per_batch = self._all_y[_j][start_index:end_index]
                    datas.append(datas_per_batch)
                    labels.append(label_per_batch)
                yield datas, labels


class OldBatchGen2InputsInf(OldBatchGenerator):
    def __init__(self, _x_train0, _x_train1, _y_train0, _batch_size=320,
                 _shuffle=True):
        super(OldBatchGen2InputsInf, self).__init__()
        self._x_train0 = _x_train0
        self._y_train0 = _y_train0
        self._x_train1 = _x_train1
        self._batch_size = _batch_size
        self._shuffle = _shuffle
        self.x_size0, y_size0 = len(self._x_train0), len(self._y_train0)
        if self.x_size0 != y_size0:
            raise AssertionError(
                'length wrong len x0 {} y0 {}'.format(self.x_size0, y_size0))
        self.step_per_epoch = int(math.ceil(self.x_size0 / self._batch_size))

    def __len__(self):
        return self.step_per_epoch

    def __iter__(self):
        while True:
            if self._shuffle:
                shuffle_indices = np.random.permutation(np.arange(self.x_size0))
                self._x_train0 = self._x_train0[shuffle_indices]
                self._x_train1 = self._x_train1[shuffle_indices]
                self._y_train0 = self._y_train0[shuffle_indices]
            for batch_num in range(self.step_per_epoch):
                start_index = batch_num * self._batch_size
                end_index = min((batch_num + 1) * self._batch_size, self.x_size0)
                labels = self._y_train0[start_index:end_index]
                datas0 = self._x_train0[start_index:end_index]
                datas1 = self._x_train1[start_index:end_index]
                yield [datas0, datas1], labels


class OldBatchGenBasic(OldBatchGenerator):
    def __init__(self, _x_train, _y_train, _batch_size=320, _shuffle=False):
        super(OldBatchGenerator, self).__init__()
        self._x_train = _x_train
        self._y_train = _y_train
        self._batch_size = _batch_size
        self._shuffle = _shuffle
        self.x_size, self.y_size = len(self._x_train), len(self._y_train)
        if self.x_size != self.y_size:
            raise AssertionError(
                'length wrong len x {} y {}'.format(self.x_size, self.y_size))
        self.step_per_epoch = int(math.ceil(self.x_size / self._batch_size))

    def __len__(self):
        return self.step_per_epoch

    def __iter__(self):
        while True:
            if self._shuffle:
                shuffle_indices = np.random.permutation(np.arange(self.x_size))
                _x_train = self._x_train[shuffle_indices]
                _y_train = self._y_train[shuffle_indices]
            else:
                _x_train = self._x_train
                _y_train = self._y_train
            for batch_num in range(self.step_per_epoch):
                start_index = batch_num * self._batch_size
                end_index = min((batch_num + 1) * self._batch_size, self.x_size)
                labels = _y_train[start_index:end_index]
                datas = _x_train[start_index:end_index]
                yield datas, labels


class OldBatchGenOneNoneInf(OldBatchGenerator):
    def __init__(self, _x_train, _batch_size=320, _shuffle=False):
        super(OldBatchGenOneNoneInf, self).__init__()
        self.x_size = len(_x_train)
        self._x_train = _x_train
        self._batch_size = _batch_size
        self._shuffle = _shuffle
        self.num_batches_per_epoch = int(math.ceil(self.x_size / self._batch_size))

    def __len__(self):
        return self.num_batches_per_epoch

    def __iter__(self):
        if self._shuffle:
            shuffle_indices = np.random.permutation(np.arange(self.x_size))
            _x_train = self._x_train[shuffle_indices]
        else:
            _x_train = self._x_train
        for batch_num in range(self.num_batches_per_epoch):
            start_index = batch_num * self._batch_size
            end_index = min((batch_num + 1) * self._batch_size, self.x_size)
            datas = _x_train[start_index:end_index]
            yield datas


class OldBatchGenTwoNoneInf(OldBatchGenerator):
    def __init__(self, _x_train, _y_train, _batch_size=320, _shuffle=False):
        super(OldBatchGenTwoNoneInf, self).__init__()
        self._x_train = _x_train
        self._y_train = _y_train
        self._batch_size = _batch_size
        self._shuffle = _shuffle
        self.x_size, self.y_size = len(self._x_train), len(self._y_train)
        if self.x_size != self.y_size:
            raise AssertionError(
                'length wrong len x {} y {}'.format(self.x_size, self.y_size))
        self.step_per_epoch = int(math.ceil(self.x_size / self._batch_size))

    def __len__(self):
        return self.step_per_epoch

    def __iter__(self):
        if self._shuffle:
            shuffle_indices = np.random.permutation(np.arange(self.x_size))
            self._x_train = self._x_train[shuffle_indices]
            self._y_train = self._y_train[shuffle_indices]
        for batch_num in range(self.step_per_epoch):
            start_index = batch_num * self._batch_size
            end_index = min((batch_num + 1) * self._batch_size, self.x_size)
            labels = self._y_train[start_index:end_index]
            datas = self._x_train[start_index:end_index]
            yield datas, labels


class OldBatchGenMultiNoneInf(OldBatchGenerator):
    def __init__(self, _all_x, _batch_size: int = 320,
                 _shuffle: bool = False):
        super(OldBatchGenMultiNoneInf, self).__init__()
        self._all_x = _all_x
        self._batch_size = _batch_size
        self._shuffle = _shuffle
        self.x_size = len(self._all_x[0])
        self.step_per_epoch = int(math.ceil(self.x_size / self._batch_size))

    def __len__(self):
        return self.step_per_epoch

    def __iter__(self):
        if self._shuffle:
            shuffle_indices = np.random.permutation(np.arange(self.x_size))
            for _j in range(len(self._all_x)):
                self._all_x[_j] = self._all_x[_j][shuffle_indices]
        for batch_num in range(self.step_per_epoch):
            # print('=======EPOCH START BATCH:{}/{}============'.format(batch_num,step_per_epoch))
            start_index = batch_num * self._batch_size
            end_index = min((batch_num + 1) * self._batch_size, self.x_size)
            datas = []
            for _j in range(len(self._all_x)):
                datas_per_batch = self._all_x[_j][start_index:end_index]
                datas.append(datas_per_batch)
            yield datas


if __name__ == '__main__':
    a = ['s', 'y', 'z']
    print(a[:2])
    gen = OldBatchGenMultiNoneInf([a], 100)
    for i in gen:
        print(i)
    print(len(gen))
