"""Generate tf.data.Dataset object for deep speech training/evaluation."""

import random
import os
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
import deep_speech2_tf_research_librispeech.data.featurizer as featurizer


class AudioConfig(object):
    """Configs for spectrogram extraction from audio."""

    def __init__(self, sample_rate: int, window_ms: int, stride_ms: int, normalize: bool=False):
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

    def __init__(self, dataset_config: DatasetConfig, partition: str, seed: int):
        self.config = dataset_config
        self.partition = partition
        self.audio_featurizer = featurizer.AudioFeaturizer(
            sample_rate=self.config.audio_config.sample_rate,
            window_ms=self.config.audio_config.window_ms,
            stride_ms=self.config.audio_config.stride_ms,
            normalize=self.config.audio_config.normalize)
        self.text_featurizer = featurizer.TextFeaturizer(vocab_file=self.config.vocab_file_path)

        self.speech_labels = self.text_featurizer.speech_labels
        self.entries = self._preprocess_data(self.config.data_path, partition, sortagrad=self.config.sortagrad)
        self.rng = random.Random(seed)
        self.n_features = self.audio_featurizer.n_features  # 161 by default

    def _gen_data(self):
        for audio_file, transcript in self.entries:
            samples, _ = sf.read(audio_file)
            features = self.audio_featurizer.transform(samples)
            labels = self.text_featurizer.transform(transcript)
            input_length = [len(features)]
            label_length = [len(labels)]

            yield {
                "features": features,
                "input_length": input_length,
                "label_length": label_length,
                "labels": labels
            }

    @staticmethod
    def _preprocess_data(file_path: str, partition: str, sortagrad: bool) -> np.ndarray:
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

    def batch_wise_shuffle(self, batch_size: int) -> None:
        """Shuffled by batch instead of by element"""
        shuffled_entries = np.empty_like(self.entries)
        max_buckets = len(self.entries) // batch_size
        ids = list(range(max_buckets))
        self.rng.shuffle(ids)
        for i, index in enumerate(ids):
            shuffled_entries[i * batch_size: (i + 1) * batch_size] = self.entries[i * batch_size: (i + 1) * batch_size]

        shuffled_entries[max_buckets * batch_size:] = self.entries[max_buckets * batch_size:]
        self.entries = shuffled_entries

    def input_fn(self, batch_size: int, repeat: int = 1) -> tf.data.Dataset:
        """Input function for model training and evaluation.

        :param batch_size: an integer denoting the size of a batch.
        :param repeat: how many times to repeat the dataset.
        :return: a tf.data.Dataset object for model to consume.
        """
        # output is a tuple which is the requirement of tf.estimator.Estimator's `model_fn`
        dataset = tf.data.Dataset.from_generator(
            generator=self._gen_data,
            output_types=(
                {
                    "features": tf.float32,
                    "input_length": tf.int32,
                    "label_length": tf.int32
                },
                tf.int32),
            output_shapes=(
                {
                    "features": tf.TensorShape([None, self.n_features, 1]),
                    "input_length": tf.TensorShape([1]),
                    "label_length": tf.TensorShape([1])
                },
                tf.TensorShape([None])))

        dataset = dataset.repeat(repeat)

        # Padding the features to its max length dimensions.
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=(
                {
                    "features": tf.TensorShape([None, self.n_features, 1]),
                    "input_length": tf.TensorShape([1]),
                    "label_length": tf.TensorShape([1])
                },
                tf.TensorShape([None])))

        dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
        return dataset
