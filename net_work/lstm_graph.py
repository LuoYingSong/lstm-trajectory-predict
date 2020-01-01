import tensorflow as tf
import numpy




class Model():
    def __init__(self):
        pass

    def input(self):
        inputs = tf.placeholder(tf.float32,shape=[None,15,2])
        drop_rate = tf.placeholder(tf.float32)
        output = tf.placeholder(tf.float32,shape=[None,2])

    def network(self):
        for i in range(num_lstm_layers):  # rnn 构建
            cell = tf.contrib.rnn.BasicLSTMCell(
                hiden_node,
                state_is_tuple=True,
            )
            cell = tf.contrib.rnn.DropoutWrapper(
                cell,
                output_keep_prob=keep_prob)
            cells.append(cell)