import tensorflow as tf
from data_sender import Vocab, scaler
from my_util import hps
from data_process import GET_DATA_COL_INDEX, SAVER_PATH, LINE_LENGTH
import numpy as np
import os

sender = Vocab(os.path.join(SAVER_PATH, "processed_data.json"), False)

INPUT_LENGTH = len(GET_DATA_COL_INDEX)  # 输入单个数值维度 2
LINE_NUM = hps.num_timesteps  # 单个序列   含有点数值输入数 15
NUM_BATCH = hps.batch_size  # 单次喂入数值batch数
hiden_node = 64  # 隐藏层数
INPUT_TYPES = 8  # 有几个不同的寓意

inputs = tf.placeholder(tf.float32, (None, LINE_LENGTH, INPUT_LENGTH),
                        name='inputs')  # batch, input_size 20 * input_line 必须要归一化
outputs = tf.placeholder(tf.float32, (None, 2), name='outputs')

old_inputs = tf.slice(inputs, [0, 0, 0], [15, LINE_LENGTH, 2])
if inputs.shape[-1].value != 2:
    new_inputs = tf.cast(tf.slice(inputs, [0, 0, 2], [15, LINE_LENGTH, INPUT_LENGTH - 2]),tf.int32)
    embedding_init = tf.random_uniform_initializer(-1.0, 1.0)
    with tf.variable_scope(
            'embedding',
            initializer=embedding_init
    ):
        embeddings = tf.get_variable(
            'embedding',
            [INPUT_TYPES],
            tf.float32
        )
        embedding_inputs = tf.nn.embedding_lookup(embeddings, new_inputs)
        print(old_inputs,embedding_inputs)
        total_inputs = tf.concat([old_inputs, embedding_inputs], 2)
        print(total_inputs)
else:
    total_inputs = old_inputs

keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # 损失
global_step = tf.Variable(
    0, trainable=False, name='global_step'
)
train_keep_prob_value = 0.8
# lstm_init = tf.zeros((NUM_BATCH, LINE_NUM, INPUT_LENGTH))
cells = []
for i in range(hps.num_lstm_layers):  # rnn 构建
    cell = tf.contrib.rnn.BasicLSTMCell(
        hiden_node,
        state_is_tuple=True,
    )
    cell = tf.contrib.rnn.DropoutWrapper(
        cell,
        output_keep_prob=keep_prob)
    cells.append(cell)
cell = tf.contrib.rnn.MultiRNNCell(cells)
# rnn运行
rnn_outputs, _ = tf.nn.dynamic_rnn(
    cell, inputs=inputs, dtype=tf.float32, time_major=False
)
output = tf.layers.dense(inputs=rnn_outputs[:, -1, :], units=2)
print(output)
loss = tf.sqrt(tf.losses.mean_squared_error(outputs, output), name='loss')
print(loss)
train_op = tf.train.AdamOptimizer(hps.learning_rate).minimize(loss, global_step=global_step)
saver = tf.train.Saver()
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
            outputs_val = sess.run([loss, output, train_op], feed_dict={
                inputs: batch_inputs, outputs: batch_labels, keep_prob: train_keep_prob_value
            })
        except ValueError:
            break
        i = sess.run(global_step)
        if not (i + 1) % 10:
            predict = scaler.inverse_transform(outputs_val[1])
            input_data = scaler.inverse_transform(batch_labels)
            print(outputs_val[0], np.mean(np.abs(predict[:, 0] - input_data[:, 0])),
                  np.mean(np.abs(predict[:, 1] - input_data[:, 1])))
        if not (i + 1) % 30:
            saver.save(sess, 'ckpt/mnist.ckpt', global_step=global_step)
