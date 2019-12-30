import tensorflow as tf
import os
from data_preprocess.data_process import SAVER_PATH
from net_work.data_sender import Vocab,scaler
import numpy as np
import ujson
from util import haversine
from my_util import hps

sender = Vocab(os.path.join(SAVER_PATH,"processed_data.json"),True)

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('ckpt/mnist.ckpt-8999.meta')
    model_file=tf.train.latest_checkpoint('ckpt/')
    new_saver.restore(sess,model_file)
    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name("inputs:0")
    w2 = graph.get_tensor_by_name("outputs:0")
    loss = graph.get_tensor_by_name('mean_squared_error/value:0')
    outputs = graph.get_tensor_by_name("dense/BiasAdd:0")
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    train_op = graph.get_tensor_by_name('Adam:0')
    dist_list = []
    counter = 0
    while True:
        try:
            counter += 1
            datas, labels = sender.next_batch(hps.num_timesteps)
            losses, output,_ = sess.run([loss,outputs,train_op],feed_dict={w1:datas,w2:labels,keep_prob:1})
            predict = scaler.inverse_transform(output)
            input_data = scaler.inverse_transform(labels)
            dist_list_min = [haversine(i,j) for i,j in zip(predict,input_data)]
            dist_list.extend(dist_list_min)
            if not counter % 200:
                print(losses, np.mean(dist_list_min))
        except ValueError:
            break
    with open('reason.txt',"w") as f:
        ujson.dump(dist_list,f)