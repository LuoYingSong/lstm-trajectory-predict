from my_util import *
import ujson
import os
import numpy as np
import time
from data_process import LINE_LENGTH, HAS_COL
from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler(feature_range=(0, 1))
class Vocab:
    def __init__(self,save_path,is_test):
        self._num_step = 0
        self._is_test = is_test
        self._inputs, self._labels = self._parse_saver(save_path)
        self._random_shuffle()
        self._max_length = len(self._inputs)


    def _parse_saver(self,file_path):
        datas= [] ; labels=[]
        with open(file_path) as f:
            data = ujson.load(f)
        slice = int(len(data) * 0.8)
        if self._is_test:
            data = data[slice:]
        else:
            data = data[:slice]
        for each_data in data:
            datas.append(each_data[0])
            labels.append(each_data[1])
        datas,labels = np.array(datas), np.array(labels)
        # datas = scaler.fit_transform(datas.reshape(-1,HAS_COL)).reshape(-1,LINE_LENGTH,HAS_COL)
        # labels = scaler.fit_transform(labels.reshape(-1,2)).reshape(-1,2)
        return datas, labels

    def _random_shuffle(self):
        p = np.random.permutation(len(self._inputs))
        print(len(self._inputs),len(self._labels))
        self._inputs = self._inputs[p]
        self._labels = self._labels[p]

    def next_batch(self,batch_size):
        if batch_size > self._max_length:
            raise Exception("too large to use")
        end_slice = self._num_step + batch_size
        if end_slice > len(self._inputs):
            return [] , []
        batch_inputs = self._inputs[self._num_step:end_slice]
        batch_labels = self._labels[self._num_step:end_slice]
        self._num_step = end_slice
        np.array(batch_labels)
        return np.array(batch_inputs,dtype=np.float32)\
            , np.array(batch_labels,dtype=np.float32)

    @property
    def length(self):
        return len(self._labels)


if __name__ == '__main__':
    # for i in range(10):
    sender = Vocab(os.path.join(SAVE_PATH, "processed_data.json"), False)
    while True:
        a,b = sender.next_batch(10)
        print(a[0],b[0])
        break