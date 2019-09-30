"""DataGenerator for model feeding."""

import os
import numpy as np
import pandas as pd
from utils.audio import read_audio
from random import Random
from tensorflow.python.keras.utils import Sequence
from .featurizer import AudioFeaturizer, TextFeaturizer
from ..config import MODELKEYS
from typing import List, Tuple, Dict


class AudioConfig(object):
    """Configs for spectrogram extraction from audio."""

    def __init__(self, sample_rate: int, window_ms: int, stride_ms: int, normalize: bool = False):
        """Initialize the AudioConfig class.

        :param sample_rate: an integer denoting the sample rate of the input waveform.
        :param window_ms: an integer for the length of a spectrogram frame, in ms.
        :param stride_ms: an integer for the frame stride, in ms.
        :param normalize: a boolean for whether apply normalization on the audio feature.
        """

        self.sample_rate = sample_rate
        self.window_ms = window_ms
        self.stride_ms = stride_ms
        self.normalize = normalize


class DatasetConfig(object):
    """Config class for generating the DeepSpeechDataset."""

    def __init__(self, audio_config: AudioConfig, data_path: str, vocab_file_path: str, sortagrad: bool, batch_size: int):
        """Initialize the configs for deep speech dataset.

        :param audio_config: AudioConfig object specifying the audio-related configs.
        :param data_path: a string denoting the full path of a manifest file.
        :param vocab_file_path: a string specifying the vocabulary file path.
        :param sortagrad: a boolean, if set to true, audio sequences will be fed by increasing length
                          in the first training epoch, which will expedite network convergence.
        :param batch_size: The batch size.

        Raises:
          IOError: file path not exist.
        """
        if not os.path.exists(data_path):
            raise IOError(f"`data_path` {data_path} not exists.")
        if not os.path.exists(vocab_file_path):
            raise IOError(f"`vocab_file_path` {vocab_file_path} not exists.")

        self.audio_config = audio_config
        self.vocab_file = vocab_file_path
        self.data_path = data_path
        self.sortagrad = sortagrad
        self.batch_size = batch_size


class DataGenerator(Sequence):
    def __init__(self, partition: str, config: DatasetConfig, seed: int = 0):
        self.audio_featurizer = AudioFeaturizer(
            sample_rate=config.audio_config.sample_rate,
            window_ms=config.audio_config.window_ms,
            stride_ms=config.audio_config.stride_ms,
            normalize=config.audio_config.normalize
        )
        self.text_featurizer = TextFeaturizer.from_file(config.vocab_file)
        self.n_features = self.audio_featurizer.n_features
        self.n_labels = self.text_featurizer.n_labels
        self.batch_size = config.batch_size
        self._rng = Random(seed)
        self.entries = self._read_data(config.data_path, partition=partition, sortagrad=config.sortagrad)
        self._max_buckets = len(self.entries) // self.batch_size
        self._indexes = np.arange(self._max_buckets)  # the indexes of batch

    @staticmethod
    def _read_data(file_path: str, partition: str, sortagrad: bool) -> np.ndarray:
        """
        Generate a list of tuples (wav_filename, transcript).
        Each dataset file contains three columns: "src", "duration", and "transcript".
        This function parses the txt file and stores each example by the increasing order of audio length.
        AS the waveforms are ordered in increasing length, audio samples in a mini-batch have similar length.

        :param file_path: a string specifying the csv file path for a dataset.
        :param partition: part of data to select, possible choices are ["train", "dev", "test"]
        :param sortagrad: whether sorting by audio length.
        :return: A list of tuples (wav_filename, transcript) sorted by duration.
        """
        if partition not in ["train", "dev", "test"]:
            raise ValueError(f"Invalid partition '{partition}', possible choices are 'train', 'dev' or 'test'.")
        data = pd.read_csv(file_path, sep="\t").query("partition == '%s'" % partition)
        if sortagrad:
            data.sort_values("duration", inplace=True)
        return data[["src", "transcript"]].values

    def __len__(self):
        return self._max_buckets

    def __getitem__(self, index) -> Tuple[Dict, np.ndarray]:
        batch_index = self._indexes[index]
        batch_data = self.entries[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]
        wav_data_list, label_data_list, input_length, label_length = self.process_on_batch(batch_data)
        inputs = {
            MODELKEYS.INPUT: wav_data_list,
            MODELKEYS.LABELS: label_data_list,
            MODELKEYS.INPUT_LENGTH: input_length,
            MODELKEYS.LABEL_LENGTH: label_length,
        }
        outputs = np.zeros_like(input_length)
        return inputs, outputs

    def process_on_batch(self, batch_data: np.ndarray):
        wav_data_list, label_data_list, input_length, label_length = [], [], [], []
        for src, transcript in batch_data:
            fs, samples = read_audio(src)
            feature = self.audio_featurizer.transform(samples)
            remain = len(feature) % 8
            pad_num = 8 - remain if remain > 0 else 0
            if pad_num:  # pad features with 8X -> fit model's requirement
                feature = np.pad(feature, [(0, pad_num), (0, 0), (0, 0)], mode="constant", constant_values=0.)

            label = self.text_featurizer.transform(transcript.split("-"))
            if len(feature) // 8 < self._ctc_len(label):  # ctc length check
                continue

            wav_data_list.append(feature)
            label_data_list.append(label)
            input_length.append(len(feature))
            label_length.append(len(label))

        wav_data_list = self._padding(wav_data_list)
        label_data_list = self._padding(label_data_list)
        return wav_data_list, label_data_list, input_length, label_length

    @staticmethod
    def _ctc_len(label: List[int]) -> int:
        add_len = 0
        temp = label[0]
        for x in label[1:]:
            if x == temp:
                add_len += 1
            temp = x
        return len(label) + add_len

    @staticmethod
    def _padding(data: List[np.ndarray]) -> np.ndarray:
        max_lens = max([len(feature) for feature in data])
        pad_shape = (len(data), max_lens, *data[0].shape[1:])
        res = np.zeros(shape=pad_shape)
        for i, x in enumerate(data):
            res[i, :len(x)] = x
        return res

    def on_epoch_end(self):
        """Shuffled by batch instead of by element"""
        self._rng.shuffle(self._indexes)
