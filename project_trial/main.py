
import os
import time
import sys
import platform
sys.path.append("F:/Code projects/Python/SpeechRecognition" if platform.system().lower() == "windows"
                else "/data/zhaochengming/projects/SpeechRecognition")
import pandas as pd
import tensorflow as tf
from _utils.tf.util import get_ckpt_global_step
from project_trial.constant import PATH, PURED_FILE, TRAIN_AUDIO_PATH, TEST_AUDIO_PATH, INVERSE_LABEL_MAP, N_CLASSES, SPACE_INDEX, SEP_INDEX
from project_trial.input_data import N_FEATURES, get_batch, get_expected_label_length
from project_trial.model import Model
from typing import Dict, List

n_epochs = 800
n_rnn_units = 40
n_rnn_layers = 1
batch_size = 64
initial_learning_rate = 0.05
momentum = 0.9
ckpt_path = os.path.join(PATH, "model")
os.makedirs(ckpt_path, exist_ok=True)
log_path = os.path.join(PATH, "log")


def get_audio_id_mapping():
    df = pd.read_table(PURED_FILE)
    return dict(zip(df["id"], df["pinyin"]))


def get_audio_infos(id_mapping: Dict[int, str], partition: str="train") -> (List[str], List[str]):
    audio_path = TRAIN_AUDIO_PATH if partition == "train" else TEST_AUDIO_PATH
    audio_files, labels = [], []
    max_label_len = 0
    for file in os.listdir(audio_path)[:1000]:
        audio_files.append(os.path.join(audio_path, file))
        filename, _ = os.path.splitext(file)
        label = id_mapping[int(filename)]
        labels.append(label)
        max_label_len = max(max_label_len, get_expected_label_length(label))
    return audio_files, labels, max_label_len


def build_session(graph):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 程序最多只能占用指定gpu80%的显存
    config.gpu_options.allow_growth = True  # 程序按需申请内存
    sess = tf.Session(graph=graph, config=config)
    return sess


def decoded(encoded_values: List[int]):
    res = []
    for w in encoded_values:
        if w == SPACE_INDEX:
            char = ""
        elif w == SEP_INDEX:
            char = " "
        else:
            char = INVERSE_LABEL_MAP.get(w, "")
        res.append(char)
    return "-".join(res)


def eval_test(test_audio_files: List[str], test_labels: List[str], max_label_len: int, model: Model):
    test_cost, test_ler = 0, 0
    for x_batch, seq_len_batch, y_indices_batch, y_values_batch in get_batch(test_audio_files, test_labels, 128):
        _batch_size = len(x_batch)
        feed_dict = {
            model.inputs: x_batch,
            model.seq_len: seq_len_batch,
            model.labels: (y_indices_batch, y_values_batch, (_batch_size, max_label_len))
        }
        batch_cost, _, ler = sess.run([model.loss, model.train_op, model.ler], feed_dict=feed_dict)
        test_cost += batch_cost * _batch_size
        test_ler += ler * _batch_size
    test_cost /= total_num
    test_ler /= total_num
    test_log = "** Test cost: {:.4f}, Test ler: {:.4f} **".format(test_cost, test_ler)

    # select the first example to output actual pinyin
    x_batch, seq_len_batch, y_indices_batch, y_values_batch = \
        get_batch([test_audio_files[0]], [test_labels[0]], batch_size=1).__next__()
    feed_dict = {
        model.inputs: x_batch,
        model.seq_len: seq_len_batch,
        model.labels: (y_indices_batch, y_values_batch, (1, max_label_len))
    }
    values, summary = sess.run([model.result, model.merge_summary], feed_dict=feed_dict)
    decoded_str = decoded(values[0])  # # single element list: element is indice, value, shape for tf 11

    print("*** Test Eval ***")
    print(test_log)
    print("%s:  %s" % (test_labels[0], decoded_str))
    print("\n\n")
    return summary


if __name__ == "__main__":
    audio_id_mapping = get_audio_id_mapping()
    train_audio_files, train_labels, max_label_len_train = get_audio_infos(audio_id_mapping, partition="train")
    test_audio_files, test_labels, max_label_len_test = get_audio_infos(audio_id_mapping, partition="test")
    total_num = len(train_audio_files)

    model = Model(
        n_features=N_FEATURES, n_rnn_units=n_rnn_units, n_rnn_layers=n_rnn_layers, n_classes=N_CLASSES + 1,  # for null label
        bidirectional=True, learning_rate=initial_learning_rate, momentum=momentum
    )

    with build_session(model.graph) as sess:
        train_writer = tf.summary.FileWriter(logdir=os.path.join(log_path, "train"), graph=sess.graph)
        test_writer = tf.summary.FileWriter(logdir=os.path.join(log_path, "test"))
        ckpt_state = tf.train.get_checkpoint_state(ckpt_path)

        if ckpt_state:
            model.saver.restore(sess, ckpt_state.model_checkpoint_path)
            old_epoch = get_ckpt_global_step(ckpt_state)
        else:
            sess.run(model.init)
            old_epoch = 0
        iterations = 0
        for epoch in range(old_epoch, n_epochs):
            train_cost, train_ler = 0., 0.
            start = time.time()
            for x_batch, seq_len_batch, y_indices_batch, y_values_batch in get_batch(train_audio_files, train_labels, batch_size):
                _batch_size = len(x_batch)
                feed_dict = {
                    model.inputs: x_batch,
                    model.seq_len: seq_len_batch,
                    model.labels: (y_indices_batch, y_values_batch, (_batch_size, max_label_len_train))
                }

                batch_cost, _, ler, train_summary =\
                    sess.run([model.loss, model.train_op, model.ler, model.merge_summary], feed_dict=feed_dict)
                iterations += 1
                train_writer.add_summary(train_summary, iterations)
                train_cost += batch_cost * _batch_size
                train_ler += ler * _batch_size
            end = time.time()
            train_cost /= total_num
            train_ler /= total_num
            log = "Epoch {} / {}, train cost: {:.4f}, train ler: {:.4f}, time using: {:.2f}"
            print(log.format(epoch + 1, n_epochs, train_cost, train_ler, end - start))

            if epoch % 10 == 0 and epoch != 0:
                test_summary = eval_test(test_audio_files, test_labels, max_label_len_test, model)
                test_writer.add_summary(test_summary, epoch)
                model.saver.save(sess, save_path=os.path.join(ckpt_path, "model.ckpt"), global_step=epoch)
    train_writer.close()
    test_writer.close()

