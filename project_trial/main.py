
import os
import time
import pandas as pd
import tensorflow as tf
from project_trial.constant import PURED_FILE, TRAIN_AUDIO_PATH, TEST_AUDIO_PATH, INVERSE_LABEL_MAP, N_CLASSES, SPACE_INDEX, SEP_INDEX
from project_trial.input_data import N_FEATURES, get_batch
from project_trial.model import Model
from typing import Dict, List


n_epochs = 800
n_rnn_units = 40
n_rnn_layers = 1
batch_size = 1
initial_learning_rate = 0.01
momentum = 0.9


def get_audio_id_mapping():
    df = pd.read_table(PURED_FILE)
    return dict(zip(df["id"], df["pinyin"]))


def get_audio_infos(id_mapping: Dict[int, str], partition: str="train") -> (List[str], List[str]):
    audio_path = TRAIN_AUDIO_PATH if partition == "train" else TEST_AUDIO_PATH
    audio_files, labels = [], []
    for file in os.listdir(audio_path):
        audio_files.append(os.path.join(audio_path, file))
        filename, _ = os.path.splitext(file)
        labels.append(id_mapping[int(filename)])
    return audio_files, labels


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
            char = INVERSE_LABEL_MAP.get(w, "")
        res += char
    return res


if __name__ == "__main__":
    audio_id_mapping = get_audio_id_mapping()
    train_audio_files, train_labels = get_audio_infos(audio_id_mapping, partition="train")
    test_audio_files, test_labels = get_audio_infos(audio_id_mapping, partition="test")

    ctc_model = Model(
        n_features=N_FEATURES, n_rnn_units=n_rnn_units, n_rnn_layers=n_rnn_layers, n_classes=N_CLASSES,
        bidirectional=True, learning_rate=initial_learning_rate, momentum=momentum
    )