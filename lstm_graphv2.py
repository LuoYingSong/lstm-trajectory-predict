import tensorflow as tf
from data_sender import Vocab
from my_util import hps
from data_process import GET_DATA_COL_INDEX, SAVER_PATH, LINE_LENGTH, MAP_SLICE_Y, MAP_SLICE_X
import numpy as np
import os
from util import haversine,recover,equal

sender = Vocab(os.path.join(SAVER_PATH,"processed_data.json"),False)

INPUT_LENGTH = len(GET_DATA_COL_INDEX) #输入单个数值维度 2
NUM_BATCH = hps.batch_size  #单次喂入数值batch数
hiden_node = 64 #隐藏层数


inputs = tf.placeholder(tf.float32, (None, LINE_LENGTH , INPUT_LENGTH),name='inputs') # batch, input_size 20 * input_line 必须要归一化
outputs = tf.placeholder(tf.float32, (None,2),name='outputs')
print(inputs,outputs)
keep_prob = tf.placeholder(tf.float32, name='keep_prob') #损失
global_step = tf.Variable(
    0, trainable=False,name='global_step'
)
train_keep_prob_value = 1


# lstm_init = tf.zeros((NUM_BATCH, LINE_NUM, INPUT_LENGTH))
cells = []
for i in range(hps.num_lstm_layers): #rnn 构建
    cell = tf.contrib.rnn.BasicLSTMCell(
        hiden_node,
        state_is_tuple=True,
    )
    cell = tf.contrib.rnn.DropoutWrapper(
        cell,
        output_keep_prob=keep_prob)
    cells.append(cell)
cell = tf.contrib.rnn.MultiRNNCell(cells)
#rnn运行
rnn_outputs, _ = tf.nn.dynamic_rnn(
        cell,inputs=inputs ,dtype=tf.float32,time_major=False
    )
output_1 = tf.layers.dense(inputs=rnn_outputs[:, -1, :],units=32,activation=tf.nn.relu)
output = tf.layers.dense(inputs=output_1,units=2)
# output2 = tf.map_fn(lambda x:tf.stack([tf.rint(x[0] / (1 / MAP_SLICE_X)) * (1 / MAP_SLICE_X),tf.rint(x[1] / (1 / MAP_SLICE_Y)) * (1 / MAP_SLICE_Y)]),output)
loss =  tf.losses.mean_squared_error(outputs,output)
print(loss)
train_op = tf.train.AdamOptimizer(learning_rate=0.0003,beta1=0.9).minimize(loss,global_step=global_step)
saver=tf.train.Saver()
with tf.Session() as sess:
    writer = tf.summary.FileWriter("logs/", sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)
    # test_batch_inputs, test_batch_labels = sender.next_batch(
    #     NUM_BATCH)
    while True:
        batch_inputs, batch_labels = sender.next_batch(
            NUM_BATCH)
        try:
            outputs_val = sess.run([loss,output,train_op],feed_dict={
                inputs:batch_inputs,outputs:batch_labels,keep_prob:train_keep_prob_value
            })
        except ValueError:
            print('end early')
            break
        i = sess.run(global_step)
        if not (i+1) % 100:
            predict = recover(outputs_val[1])
            input_data = recover(batch_labels)
            aver_dist = np.mean([haversine(i,j) for i,j in zip(predict,input_data)])
            print(outputs_val[0],aver_dist,np.mean(np.abs(predict[:,0]-input_data[:,0])),np.mean(np.abs(predict[:,1]-input_data[:,1])),equal(batch_labels,outputs_val[1]))
            # if i > 500 and aver_dist > 120:
            #     print(batch_inputs,predict)
        if not (i+1) % 1000:
            saver.save(sess, 'ckpt/mnist.ckpt', global_step=global_step)