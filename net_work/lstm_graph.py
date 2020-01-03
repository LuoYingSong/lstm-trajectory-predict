import tensorflow as tf
import numpy

parameter = {'layer':2,
             'lstm_hidden_node_num':30
             }


class Model():
    def __init__(self):
        self.inputs,self.most_like_node,self.drop_rate,self.output = self.inputs()

    @staticmethod
    def input_():
        inputs = tf.placeholder(tf.float32,shape=[None,15,2])
        most_like_node = tf.placeholder(tf.float32,shape=[None,2])
        drop_rate = tf.placeholder(tf.float32)
        output = tf.placeholder(tf.float32,shape=[None,2])
        return inputs, most_like_node, drop_rate, output

    def network(self):
        dtw_layer = tf.layers.dense(self.most_like_node,2,activation=tf.nn.relu)
        cells = []
        for i in range(parameter['layer']):  # rnn 构建
            cell = tf.contrib.rnn.BasicLSTMCell(
                parameter['lstm_hidden_node_num'],
                forget_bias=0.0,
                state_is_tuple=True,
            )
            cell = tf.contrib.rnn.DropoutWrapper(
                cell,
                output_keep_prob=self.drop_rate)
            cells.append(cell)
        pre_date_layer = tf.nn.dynamic_rnn(cells,self.inputs)

