
"""
Based on https://github.com/igormq/ctc_tensorflow_example/blob/master/ctc_tensorflow_example.py
"""

import os
import time
import tensorflow as tf
from asr_lstm_ctc import model, input_data
from asr_lstm_ctc.input_data import FIRST_INDEX, SEP_INDEX, SPACE_INDEX, N_FEATURES
from typing import List

PATH = "F:/for learn/Python/asr_ctc/"
audio_path = os.path.join(PATH, "audios")
n_classes = 28  # 26 + 1 no_label + 1 blank
n_epochs = 800
n_rnn_units = 40
n_rnn_layers = 1
batch_size = 1
initial_learning_rate = 0.01
momentum = 0.9


def get_audio_infos():
    audio_files = []
    labels = []
    for file in os.listdir(audio_path):
        if file.endswith(".wav"):
            audio_files.append(os.path.join(audio_path, file))

    with open(os.path.join(audio_path, "label.txt"), "r") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    return audio_files, labels, max(map(len, labels)) * 2 + 1


def build_session(graph):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 程序最多只能占用指定gpu80%的显存
    config.gpu_options.allow_growth = True  # 程序按需申请内存
    sess = tf.Session(graph=graph, config=config)
    return sess


def decoded(encoded_values: List[int]):
    res = ""
    for w in encoded_values:
        if w == SPACE_INDEX:
            char = ""
        elif w == SEP_INDEX:
            char = " "
        else:
            w += FIRST_INDEX
            char = chr(w) if w <= ord("z") else ""
        res += char
    return res


if __name__ == '__main__':
    ctc_model = model.Model(
        n_features=N_FEATURES, n_rnn_units=n_rnn_units, n_rnn_layers=n_rnn_layers, n_classes=n_classes,
        bidirectional=True, learning_rate=initial_learning_rate, momentum=momentum
    )

    audio_files, labels, max_label_len = get_audio_infos()
    assert len(audio_files) == len(labels), "num of wav files doesn't match num of labels"
    total_num = len(audio_files)

    with build_session(ctc_model.graph) as sess:
        sess.run(ctc_model.init)
        for epoch in range(n_epochs):
            train_cost, train_ler = 0, 0
            start = time.time()
            for x_batch, seq_len_batch, y_indices_batch, y_values_batch in input_data.get_batch(audio_files, labels, batch_size):
                _batch_size = len(x_batch)
                feed_dict = {
                    ctc_model.inputs: x_batch,
                    ctc_model.seq_len: seq_len_batch,
                    ctc_model.labels: (y_indices_batch, y_values_batch, (_batch_size, max_label_len))
                }

                batch_cost, _, ler = sess.run([ctc_model.loss, ctc_model.train_op, ctc_model.ler], feed_dict=feed_dict)
                train_cost += batch_cost * _batch_size
                train_ler += ler * _batch_size
            end = time.time()
            train_cost /= total_num
            train_ler /= total_num
            log = "Epoch {} / {}, train cost: {:.4f}, train ler: {:.4f}, time using: {:.2f}"
            print(log.format(epoch + 1, n_epochs, train_cost, train_ler, end - start))

        # use first item as test sample
        x_batch, seq_len_batch, y_indices_batch, y_values_batch =\
            input_data.get_batch([audio_files[0]], [labels[0]], batch_size=1).__next__()
        feed_dict = {
            ctc_model.inputs: x_batch,
            ctc_model.seq_len: seq_len_batch,
            ctc_model.labels: (y_indices_batch, y_values_batch, (1, max_label_len))
        }
        values = sess.run([ctc_model.result], feed_dict=feed_dict)
        #print(values[0][1])
        decoded_str = decoded(values[0][1])  # # single element list: element is indice, value, shape
        print("Decoded: %s" % decoded_str)







