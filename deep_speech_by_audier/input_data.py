
import numpy as np
import pandas as pd
from scipy.fftpack import fft
# from librosa.feature import mfcc
from scipy.io import wavfile
from python_speech_features import mfcc
from sklearn.utils import shuffle
from collections import namedtuple
from typing import Optional, List, Dict


BatchData = namedtuple("BatchData", ["inputs", "labels", "input_length", "label_length"])
Outputs = namedtuple("Outputs", ["ctc"])


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


class DataGenerator(object):
    def __init__(self, data_source: str, pinyin_sep: str, data_type: str,
                 is_shuffle: bool=False, data_length: Optional[int]=None, vocab: Optional[Dict[str, int]]=None):
        """
        :param data_source: 结构化标注数据源位置
        :param data_type: 指明取哪部分，[train, test, dev]
        :param is_shuffle: 是否shuffle
        :param data_length: 限制读入的数据条数
        """
        self.data = pd.read_table(
            data_source, encoding="utf-8", header=None, names=["src", "content", "pinyin", "data_type"]
        ).query("data_type == %s" % data_type)

        if data_length is not None and data_length > 0:
            self.data: pd.DataFrame = self.data[: data_length]

        self.is_shuffle = is_shuffle
        self.png_vocab, self.han_vocab = [], []
        self.am_vocab: Dict[str, int] = vocab if vocab else self._make_am_vocab(self.data["pinyin"], sep=pinyin_sep)
        self.pinyin_sep = pinyin_sep

    def get_am_batch(self, feature_type: str, n_features: int, batch_size: int=64):
        """ 生成训练数据 """
        self.data: pd.DataFrame = shuffle(self.data) if self.is_shuffle else self.data
        wav_data_list, label_data_list = [], []
        while True:  # 持续输出
            for _, row in self.data.iterrows():
                features = compute_fbank(row["src"], time_window=n_features * 2, time_step=10)\
                    if feature_type == "fbank" else compute_mfcc(row["src"], n_features=n_features)
                pad_num = (len(features) // 8 + 1) * 8 - len(features)
                features = np.pad(features, ((0, pad_num), (0, 0)), mode="constant", constant_values=0.)
                label = [self.am_vocab[pny] for pny in row["pinyin"].split(self.pinyin_sep)]
                label_ctc_len = self._ctc_len(label)
                if len(features) // 8 >= label_ctc_len:
                    wav_data_list.append(features)
                    label_data_list.append(label)

                    if len(wav_data_list) == batch_size:
                        pad_wav_data, input_length = self._wav_padding(wav_data_list)
                        pad_label_data, label_length = self._label_padding(label_data_list)
                        inputs = BatchData(
                            inputs=pad_wav_data, labels=pad_label_data,
                            input_length=input_length, label_length=label_length)
                        outputs = Outputs(ctc=np.zeros(len(pad_wav_data), ))
                        yield inputs, outputs  # 一个空的输出用来占位满足keras.Model的fit_generator 输入API
                        wav_data_list, label_data_list = [], []

    @staticmethod
    def _make_am_vocab(pny_data_list, sep=" ") -> Dict[str, int]:
        all_pny = sorted(list(set(sep.join(pny_data_list).split(sep))))
        all_pny.insert(0, "_")
        return {pny: i for i, pny in enumerate(all_pny)}

    @staticmethod
    def _wav_padding(wav_data_list: List[np.ndarray]):
        n_features = wav_data_list[0].shape[1]
        wav_lens = np.array([len(data) for data in wav_data_list])
        wav_max_len = max(wav_lens)
        wav_lens //= 8
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
