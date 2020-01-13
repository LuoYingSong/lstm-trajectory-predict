import tensorflow as tf
import numpy as np
from sender2 import Sender
import os
import sys
sys.path.append('..')
from util import haversine

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parameter = {'layer':4,
             'lstm_hidden_node_num':64,
             'hideen_layer':[100,500,300,64],
             'first_layer':32,
             'learning_rate':0.00005,
             'batch_size':100,
             'drop':1,
             'epoch':100,
             'dtw_layer':96
             }



class Model():
    def __init__(self):
        self.inputs,self.most_like_node,self.drop_rate,self.output = self.input_()
        self.op,self.loss,self.network_output = self.network()
    @staticmethod
    def input_():
        inputs = tf.placeholder(tf.float32,shape=[None,15,2])
        most_like_node = tf.placeholder(tf.float32,shape=[None,2])
        drop_rate = tf.placeholder(tf.float32)
        output = tf.placeholder(tf.float32,shape=[None,2])
        return inputs, most_like_node, drop_rate, output

    def network(self):
        dtw_layer = tf.layers.dense(self.most_like_node,parameter['dtw_layer'],activation=tf.nn.relu)
        cells = []
        for i in range(parameter['layer']):  # rnn 构建
            cell = tf.contrib.rnn.BasicLSTMCell(
                parameter['lstm_hidden_node_num'],
                forget_bias=0.3,
                state_is_tuple=True,


            )
            cell = tf.contrib.rnn.DropoutWrapper(
                cell,
                output_keep_prob=self.drop_rate)
            cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells)
        pre_data_layer,_ = tf.nn.dynamic_rnn(cell,inputs=self.inputs,dtype=tf.float32,time_major=False)
        print(pre_data_layer)
        pre_data_layer = tf.layers.dense(pre_data_layer[:,-1,:],
                                         units=parameter['first_layer'],activation=tf.nn.relu)
        concat_layer = tf.concat([dtw_layer,pre_data_layer],1)
        print(dtw_layer,pre_data_layer,concat_layer,'!!!!!!!!!!!!!!!!!!!!!!')
        for layer_num in parameter['hideen_layer']:
            concat_layer = tf.layers.dense(concat_layer,layer_num,activation=tf.nn.relu)
        output = tf.layers.dense(concat_layer, units=2, activation=tf.nn.relu)  # relu
        loss = tf.losses.mean_squared_error(self.output,output)
        op = tf.train.AdamOptimizer(learning_rate=parameter['learning_rate'],beta1=0.9).minimize(loss)
        return op,loss,output

    def train(self):
        sender = Sender()
        count = 0
        epoch_count = 0
        with tf.Session() as sess:
            writer = tf.summary.FileWriter("logs/", sess.graph)
            init = tf.global_variables_initializer()
            sess.run(init)
            saver = tf.train.Saver()
            while True:
                try:
                    input_data, label, histort_ptr = sender.send(parameter['batch_size'])
                except Exception:
                    saver.save(sess, 'ckpt/mnist.ckpt',global_step=epoch_count)
                    sender.restart()
                    input_data, label, histort_ptr = sender.send(parameter['batch_size'])
                    epoch_count += 1
                    if epoch_count == parameter['epoch']:
                        break
                    else:
                        continue
                count += 1

                loss,_,net_work_output = sess.run([self.loss,self.op,self.network_output],feed_dict={self.drop_rate:parameter['drop'],
                                                                 self.inputs:input_data,
                                                                 self.output:label,
                                                                 self.most_like_node:label})
                if not count % 500:

                    print(loss,np.array([haversine(i,j) for i,j in zip(net_work_output,label)]).mean())

    def rev_net(self):
        pass


if __name__ == '__main__':
    model = Model()
    model.train()



