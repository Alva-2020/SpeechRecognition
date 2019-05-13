
import os
import re
import numpy as np
from librosa.feature import melspectrogram, mfcc
from librosa import power_to_db
from asr_lstm_ctc.constant import PATH
from scipy.io import wavfile
from typing import List


N_MELS = 128
N_FEATURES = 13
N_CLASSES = 28  # 26 + 1 no_label + 1 blank

# SPACE_TOKEN = "<space>"
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1
audio_path = os.path.join(PATH, "audios")


def get_audio_feature(audio_file: str) -> np.ndarray:
    """
    读取音频文件，并采用 mfcc数据作为特征
    :param audio_file: 音频文件地址
    :return: [n_mfcc, time]
    """
    sample_rate, audio = wavfile.read(audio_file)
    S = melspectrogram(y=audio.astype(np.float32), sr=sample_rate, n_mels=N_MELS)
    log_S = power_to_db(S=S, ref=np.max)
    features = mfcc(S=log_S, n_mfcc=N_FEATURES)  # [n_mfcc, time]
    features = (features - np.mean(features, axis=1)) / np.std(features, axis=1)
    return features


def get_audio_label(label: str) -> (List[int], List[int], int):
    """将文本转换成整数序列，再转换成稀疏三元组
    空格转换为 space_index

    输出 稀疏向量的主要数据：indices, values, shape
    """
    label = label.lower().replace(".", "")
    label = re.sub("\s+", " ", label)  # 多个空格替换成一个空格
    indices = []
    values = []
    for i, w in enumerate(label):
        if w != " ":
            indices.append(i)
            values.append(w)
    return indices, values, len(values)
