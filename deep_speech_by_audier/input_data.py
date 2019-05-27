
import os
import numpy as np
import pandas as pd
from scipy.fftpack import fft
# from librosa.feature import mfcc
from scipy.io import wavfile
from python_speech_features import mfcc
from sklearn.utils import shuffle
from typing import Optional, List



N_FEATURES = 26

# mfcc 特征
def _compute_mfcc(file):
    sample_rate, audio = wavfile.read(file)
    mfcc_feat = mfcc(signal=audio, samplerate=sample_rate, numcep=N_FEATURES)  # shape: [n_frames, numcep]
    mfcc_feat = mfcc_feat[::3]
    return mfcc_feat.T


# 时频图
def compute_fbank(file, time_window=400, time_step: int = 10):
    w = np.hamming(time_window)
    fs, wav_arr = wavfile.read(file)
    #    range0_end = int(len(wav_arr) / fs * 1000) // time_step + 1 # 计算循环终止的位置，也就是最终生成的窗数
    data_input = []
    i = 0
    while True:
        p_start = int(i * fs * time_step / 1000)
        p_end = p_start + 400
        if p_end > len(wav_arr) - 1:
            break
        data_line = wav_arr[p_start: p_end]
        data_line = data_line * w  # 加窗
        data_line = np.abs(fft(data_line))
        data_input.append(data_line[:int(time_window / 2)])  # 设置为400除以2的值（即200）是取一半数据，因为是对称的
        i += 1
    return np.log(np.vstack(data_input))



class DataGenerator(object):
    def __init__(self, data_source: str, data_type: str, is_shuffle: bool=False, data_length: Optional[int]=None):
        """
        :param data_source: 结构化标注数据源位置
        :param data_type: 指明取哪部分，[train, test, dev]
        :param is_shuffle: 是否shuffle
        :param data_length: 限制读入的数据条数
        """
        self.data = pd.read_csv(data_source, sep="\t", encoding="utf-8", header=None,
                                names=["src", "content", "pinyin", "data_type"]).query("data_type == %s" % data_type)
        if data_length is not None:
            self.data = self.data[: data_length]
        if is_shuffle:
            self.data = shuffle(self.data)


        self.am_vocab, self.png_vocab, self.han_vocab = {}, {}, {}

    def get_am_batch(self):
        pass

    def _wav_padding(self, wav_data_list: List[np.ndarray]):
        n_features = wav_data_list[0].shape[1]
        wav_lens = np.array([len(data) for data in wav_data_list])
        wav_max_len = max(wav_lens)
        wav_lens //= 8
        new_wav_data_list = np.zeros(shape=(len(wav_data_list), wav_max_len, n_features))
        for i in range(len(wav_data_list)):
            new_wav_data_list[i, :len(wav_data_list[i]), :] = wav_data_list[i]
        return new_wav_data_list[:, np.newaxis], wav_lens











