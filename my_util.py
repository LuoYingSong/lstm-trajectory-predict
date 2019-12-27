import tensorflow as tf
from data_preprocess import data_process


def get_default_params():
    return tf.contrib.training.HParams(
        num_timesteps = 15,
        num_lstm_nodes = [64, 64,],
        num_lstm_layers = 4,
        num_fc_nodes = 32,
        batch_size = 25  ,
        clip_lstm_grads = 1.0,
        learning_rate = 0.0005,
        num_word_threshold = 10,
        trainstep = 6000,
    )

SAVE_PATH = data_process.SAVER_PATH
hps = get_default_params()