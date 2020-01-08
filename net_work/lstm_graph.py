import tensorflow as tf
import numpy
from sender2 import Sender
parameter = {'layer':2,
             'lstm_hidden_node_num':30,
             'hideen_layer':[100,300,200],
             'first_layer':32,
             'learning_rate':0.001,
             'batch_size':33,
             'drop':0.8,
             'epoch':100,

             }



class Model():
    def __init__(self):
        self.inputs,self.most_like_node,self.drop_rate,self.output = self.inputs()
        self.op,self.loss,self.network_output = self.network()
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
        pre_data_layer,_ = tf.nn.dynamic_rnn(cells,self.inputs)
        print(pre_data_layer)
        pre_data_layer = tf.layers.dense(pre_data_layer[:,-1,:],
                                         units=parameter['first_layer'],activation=tf.nn.relu)
        concat_layer = tf.concat([dtw_layer,pre_data_layer],1)
        output = tf.layers.dense(concat_layer,units=2,activation=tf.nn.tanh) #relu
        for layer_num in parameter['hideen_layer']:
            concat_layer = tf.layers.dense(concat_layer,layer_num,activation=tf.nn.relu)
        loss = tf.losses.mean_squared_error(self.output,output)
        op = tf.train.AdadeltaOptimizer(learning_rate=parameter['learning_rate']).minimize(loss)
        return op,loss,output

    def train(self):
        sender = Sender()
        epoch_count = 0
        with tf.Session() as sess:
            writer = tf.summary.FileWriter("logs/", sess.graph)
            init = tf.global_variables_initializer()
            sess.run(init)
            saver = tf.train.Saver()
            while True:
                try:
                    input_data, label, histort_ptr = sender.send(parameter['batch_size'])
                except IndexError:
                    saver.save(sess, 'ckpt/mnist.ckpt',global_step=epoch_count)
                    sender.restart()
                    input_data, label, histort_ptr = sender.send(parameter['batch_size'])
                    epoch_count += 1
                    if epoch_count == parameter['epoch']:
                        break
                loss,_ = sess.run([self.loss,self.op],feed_dict={self.drop_rate:parameter['drop'],
                                                                 self.inputs:input_data,
                                                                 self.output:label,
                                                                 self.most_like_node:histort_ptr})



