
import numpy as np
import pandas as pd
from scipy.fftpack import fft
# from librosa.feature import mfcc
from scipy.io import wavfile
from python_speech_features import mfcc
from tensorflow.python.keras.utils import Sequence
from typing import Optional, List, Dict, Tuple


# mfcc 特征
def compute_mfcc(file, n_features: int=26):
    sample_rate, audio = wavfile.read(file)
    mfcc_feat = mfcc(signal=audio, samplerate=sample_rate, numcep=n_features)  # shape: [n_frames, numcep]
    return mfcc_feat  # shape: [n_frames, numcep]


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
    return np.log(np.vstack(data_input) + 1)


class DataGenerator(Sequence):
    def __init__(self, data_source: str, pinyin_sep: str, data_type: str, batch_size: int, feature_type: str, n_features: int,
                 model_type: str, shuffle: bool=False, data_length: Optional[int]=None, vocab: Optional[Dict[str, int]]=None):
        """
        :param data_source: 结构化标注数据源位置
        :param data_type: 指明取哪部分，[train, test, dev]
        :param shuffle: 是否shuffle
        :param data_length: 限制读入的数据条数
        """
        self._data = pd.read_csv(
            data_source, sep="\t", encoding="utf-8", header=None, engine="python",
            names=["src", "content", "pinyin", "data_type"]
        ).query("data_type == '%s'" % data_type)
        self.data = self._data[["src", "pinyin"]].values

        if data_length > 0:
            self.data = self.data[: data_length]

        self.batch_size = batch_size
        self.indexes = np.arange(len(self.data))
        self.feature_type = feature_type
        self.n_features = n_features
        self.model_type = model_type
        self.shuffle = shuffle
        self.png_vocab, self.han_vocab = [], []
        self.am_vocab: Dict[str, int] = vocab if vocab else self._make_am_vocab(self.data["pinyin"], sep=pinyin_sep)
        self.pinyin_sep = pinyin_sep

    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch_data = self.data[batch_indexes]
        wav_data_list, label_data_list, input_length, label_length = self._data_process(batch_data)
        input_length = input_length // 8 if self.model_type.upper() == "DFCNN" else input_length

        inputs = {
            "the_inputs": wav_data_list,
            "the_labels": label_data_list,
            "the_input_length": input_length,
            "the_label_length": label_length
        }
        outputs = np.zeros(shape=(len(input_length),))  # 一个空的输出用来占位满足keras.Model的fit_generator 输入API
        return inputs, outputs

    def on_epoch_end(self):
        """在每次epoch结束时进行何种操作"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _data_process(self, batch_data):
        """ 生成训练数据 """
        wav_data_list, label_data_list = [], []
        for src, pnys in batch_data:
            features = compute_fbank(src, time_window=self.n_features * 2, time_step=10)\
                if self.feature_type == "fbank" else compute_mfcc(src, n_features=self.n_features)
            pad_num = (len(features) // 8 + 1) * 8 - len(features)
            features = np.pad(features, ((0, pad_num), (0, 0)), mode="constant", constant_values=0.)

            label = [self.am_vocab[pny] for pny in pnys.split(self.pinyin_sep)]
            label_ctc_len = self._ctc_len(label)

            if len(features) // 8 >= label_ctc_len:  # 考虑 DFCNN的输入特征大小
                wav_data_list.append(features)
                label_data_list.append(label)

        pad_wav_data, input_length = self._wav_padding(wav_data_list)
        pad_label_data, label_length = self._label_padding(label_data_list)
        return pad_wav_data, pad_label_data, input_length, label_length

    @staticmethod
    def _make_am_vocab(pny_data_list, sep=" ") -> Dict[str, int]:
        all_pny = sorted(list(set(sep.join(pny_data_list).split(sep))))
        all_pny.insert(0, "_")
        return {pny: i for i, pny in enumerate(all_pny)}

    @staticmethod
    def _wav_padding(wav_data_list: List[np.ndarray]) -> (np.ndarray, np.ndarray):
        """
        对全部语音做 长度对齐 padding
        :param wav_data_list: 语音基础特征数据 [ndarray(seq_1, n_features), ndarray(seq_2, n_features), ...]
        :return:
            - 对齐的特征数据 [ndarray(max_len, n_features), ndarray(max_len, n_features), ...]
            - 有效的序列长度
        """
        n_features = wav_data_list[0].shape[1]
        wav_lens = np.array([len(data) for data in wav_data_list])
        wav_max_len = max(wav_lens)
        new_wav_data_list = np.zeros(shape=(len(wav_data_list), wav_max_len, n_features))  # padding完毕的容器
        for i, wav_data in enumerate(wav_data_list):
            new_wav_data_list[i, :len(wav_data), :] = wav_data  # 塞入数据
        return new_wav_data_list[..., np.newaxis], wav_lens

    @staticmethod
    def _label_padding(label_data_list):
        label_lens = np.array([len(label) for label in label_data_list])
        max_label_len = max(label_lens)
        new_label_data_list = np.zeros(shape=(len(label_data_list), max_label_len))
        for i, label_data in enumerate(label_data_list):
            new_label_data_list[i, :len(label_data)] = label_data
        return new_label_data_list, label_lens

    @staticmethod
    def _ctc_len(label: List[int]) -> int:
        add_len = 0
        label_len = len(label)
        for i in range(label_len - 1):
            if label[i] == label[i + 1]:
                add_len += 1
        return label_len + add_len
