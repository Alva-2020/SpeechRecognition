"""Generate tf.data.Dataset object for deep speech training/evaluation."""

import random
import os
import numpy as np
import pandas as pd
import soundfile as sf
import deep_speech2_udf_librispeech.data.featurizer as featurizer
from typing import List


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

    def __init__(self, audio_config: AudioConfig, data_path: str, vocab_file_path: str, sortagrad: bool):
        """Initialize the configs for deep speech dataset.

        :param audio_config: AudioConfig object specifying the audio-related configs.
        :param data_path: a string denoting the full path of a manifest file.
        :param vocab_file_path: a string specifying the vocabulary file path.
        :param sortagrad: a boolean, if set to true, audio sequences will be fed by increasing length
                          in the first training epoch, which will
                    expedite network convergence.

        Raises:
          RuntimeError: file path not exist.
        """
        if not os.path.exists(data_path):
            raise IOError(f"`data_path` {data_path} not exists.")
        if not os.path.exists(vocab_file_path):
            raise IOError(f"`vocab_file_path` {vocab_file_path} not exists.")

        self.audio_config = audio_config
        self.data_path = data_path
        self.vocab_file_path = vocab_file_path
        self.sortagrad = sortagrad


class DeepSpeechDataset(object):
    """Dataset class for training/evaluation of DeepSpeech model."""

    def __init__(self, dataset_config: DatasetConfig, partition: str, batch_size: int,
                 seed: int, label_padding_value: int):
        self.config = dataset_config
        self.partition = partition
        self.batch_size = batch_size
        self.label_padding_value = label_padding_value
        self.audio_featurizer = featurizer.AudioFeaturizer(
            sample_rate=self.config.audio_config.sample_rate,
            window_ms=self.config.audio_config.window_ms,
            stride_ms=self.config.audio_config.stride_ms,
            normalize=self.config.audio_config.normalize)
        self.text_featurizer = featurizer.TextFeaturizer(vocab_file=self.config.vocab_file_path)

        self.speech_labels = self.text_featurizer.speech_labels
        self.entries = self._read_data(self.config.data_path, partition, sortagrad=self.config.sortagrad)
        self.rng = random.Random(seed)
        self.n_features = self.audio_featurizer.n_features  # 161 by default
        self.num_classes = self.text_featurizer.num_classes  # 29 by default

    @staticmethod
    def _read_data(file_path: str, partition: str, sortagrad: bool) -> np.ndarray:
        """
        Generate a list of tuples (wav_filename, wav_filesize, transcript).
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
        return len(self.entries) // self.batch_size

    def __getitem__(self, index: int):
        batch_features, batch_labels = [], []
        for audio_file, transcript in self.entries[index * self.batch_size: (index + 1) * self.batch_size]:
            samples, _ = sf.read(audio_file)
            features = self.audio_featurizer.transform(samples)
            labels = self.text_featurizer.transform(transcript)
            batch_features.append(features)
            batch_labels.append(labels)

        batch_features, batch_input_length = self._wav_padding(batch_features)
        batch_labels, batch_label_length = self._label_padding(batch_labels, self.label_padding_value)
        return {
            "features": batch_features,
            "labels": batch_labels,
            "input_length": batch_input_length.reshape(-1, 1),
            "label_length": batch_label_length.reshape(-1, 1)
        }

    def batch_wise_shuffle(self) -> None:
        """Shuffled by batch instead of by element"""
        shuffled_entries = np.empty_like(self.entries)
        max_buckets = len(self.entries) // self.batch_size
        ids = list(range(max_buckets))
        self.rng.shuffle(ids)
        for i, index in enumerate(ids):
            shuffled_entries[i * self.batch_size: (i + 1) * self.batch_size] =\
                self.entries[i * self.batch_size: (i + 1) * self.batch_size]

        shuffled_entries[max_buckets * self.batch_size:] = self.entries[max_buckets * self.batch_size:]
        self.entries = shuffled_entries

    @staticmethod
    def _wav_padding(wav_data_list: List[np.ndarray]) -> (np.ndarray, np.ndarray):
        n_features = wav_data_list[0].shape[1]
        wav_lens = np.array([len(data) for data in wav_data_list])
        wav_max_len = max(wav_lens)
        new_wav_data_list = np.zeros(shape=(len(wav_data_list), wav_max_len, n_features, 1))  # padding完毕的容器
        for i, wav_data in enumerate(wav_data_list):
            new_wav_data_list[i, :len(wav_data), :, :] = wav_data  # 塞入数据
        return new_wav_data_list, wav_lens

    @staticmethod
    def _label_padding(label_data_list, label_padding_value: int):
        label_lens = np.array([len(label) for label in label_data_list])
        max_label_len = max(label_lens)
        new_label_data_list = np.ones(shape=(len(label_data_list), max_label_len)) * label_padding_value
        for i, label_data in enumerate(label_data_list):
            new_label_data_list[i, :len(label_data)] = label_data
        return new_label_data_list, label_lens
