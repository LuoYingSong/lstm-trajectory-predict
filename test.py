import tensorflow as tf
import os
from data_process import SAVER_PATH
from data_sender import Vocab
import numpy as np
import ujson
from util import haversine,recover

sender = Vocab(os.path.join(SAVER_PATH,"processed_data.json"),True)

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('ckpt/mnist.ckpt-6999.meta')
    model_file=tf.train.latest_checkpoint('ckpt/')
    new_saver.restore(sess,model_file)
    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name("inputs:0")
    w2 = graph.get_tensor_by_name("outputs:0")
    loss = graph.get_tensor_by_name('mean_squared_error/value:0')
    outputs = graph.get_tensor_by_name("dense_1/BiasAdd:0")
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    datas, labels = sender.next_batch(sender.length-1)
    loss, outputes = sess.run([loss,outputs],feed_dict={w1:datas,w2:labels,keep_prob:1})
    predict = recover(outputes)
    input_data = recover(labels)
    dist_list = [haversine(i,j) for i,j in zip(predict,input_data)]
    print(loss, np.mean(dist_list))
    with open('reason.txt',"w") as f:
        ujson.dump(dist_list,f)