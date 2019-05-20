
import numpy as np
from project_trial.constant import LABEL_MAP
from project_trial.preprocessing.data_generator import SPLITTER
from librosa.feature import mfcc
from scipy.io import wavfile
from typing import List


N_FEATURES = 13


def _get_audio_feature(audio_file: str) -> np.ndarray:
    """
    读取单个音频文件，并采用 mfcc数据作为特征
    :param audio_file: 音频文件地址
    :return: [n_mfcc, time]
    """
    sample_rate, audio = wavfile.read(audio_file)
    feature = mfcc(y=audio.astype(np.float32), sr=sample_rate, n_mfcc=N_FEATURES)  # [n_mfcc, time]
    feature = (feature - np.mean(feature)) / np.std(feature)  # normalize
    return feature


def _get_audio_label(label: str, index: int=0):
    """将单个字符串pinyin 转换成整数序列，再转换成稀疏向量
    空格转换为 space_index
    :param label: 待转换的label, 用 data_generator的 gen_pinyin生成
    :param index: 序号，主要为了生成满足 tf.sparse_placeholder要求的稀疏三元组

    输出 稀疏向量的主要数据：indices, values, shape
    """
    indices = []
    values = []
    i = 1
    for w in label.split(SPLITTER):
        indices.append((index, i))
        values.append(LABEL_MAP[w])
        i += 2
    return indices, values  # indices and values must have same length


def get_batch(audio_files: List[str], labels: List[str], batch_size: int):
    x_batch = []
    seq_length_batch = []  # 记录输入序列的帧数， 计算ctc时需要
    # 构建 Sparse Label所需的两个向量
    y_indices_batch: List[(int, int)] = []
    y_values_batch: List[int] = []
    for i, (audio_file, label) in enumerate(zip(audio_files, labels)):
        i %= batch_size  # 保证在一个batch中，索引正确
        feature = _get_audio_feature(audio_file)  # [N_FEATURES, time]
        x_batch.append(feature.T)  # [batch_size, time, N_FEATURES]
        seq_length_batch.append(feature.shape[-1])  # [batch_size]

        indices, values = _get_audio_label(label, i)
        y_indices_batch += indices  # [valid_dense_values, 2]
        y_values_batch += values  # [valid_dense_values]

        if i + 1 == batch_size:
            yield x_batch, seq_length_batch, y_indices_batch, y_values_batch
            x_batch, seq_length_batch, y_indices_batch, y_values_batch = [], [], [], []

    if x_batch:
        yield x_batch, seq_length_batch, y_indices_batch, y_values_batch
