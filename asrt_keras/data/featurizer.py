"""Featurize the audio and text"""

import numpy as np
from typing import Dict


class AudioFeaturizer(object):
    """Class to extract spectrogram features from the audio input."""

    def __init__(self, sample_rate: int = 16000, window_ms: int = 25, stride_ms: int = 10, normalize: bool = True):
        """Initialize the audio featurizer class according to the configs.

        :param sample_rate: an integer specifying the sample rate of the input waveform.
        :param window_ms: an integer for the length of a spectrogram frame, in ms.
        :param stride_ms: an integer for the frame stride, in ms.
        :param normalize: Whether normalize features.
        """
        self.sample_rate = sample_rate
        self.window_ms = window_ms
        self.stride_ms = stride_ms
        self.normalize = normalize
        self.n_features = (self.window_ms / 1000 * self.sample_rate) // 2  # 200 by default
        self.shape = [None, self.n_features, 1]

    def transform(self, samples: np.ndarray):
        feature = self.compute_fbank(
            samples, fs=self.sample_rate, time_stride_ms=self.stride_ms, time_window_ms=self.window_ms)
        if self.normalize:
            feature = self._normalize_audio_feature(feature)

        # Adding Channel dimension for conv2D input.
        feature = np.expand_dims(feature, axis=2)
        return feature

    @staticmethod
    def compute_fbank(samples: np.ndarray, fs: int, time_window_ms: int = 25, time_stride_ms: int = 10):
        """Compute FBank feature of audio.
        :param samples: The original wave data read from file.
        :param fs: The sample rate.
        :param time_window_ms: The time-span of window, ms.
        :param time_stride_ms: The time-span of stride, ms.
        :return: The FBank feature of audio, shape: [?, time_window_ms * fs / 1000 // 2]
        """
        window_size = int(fs * time_window_ms / 1000)
        stride_size = int(fs * time_stride_ms / 1000)
        w = np.hamming(window_size)

        truncate_size = (len(samples) - window_size) % stride_size
        samples = samples[: (len(samples) - truncate_size)]
        n_windows = (len(samples) - window_size) // stride_size + 1
        nbytes = samples.strides[0]

        data = np.lib.stride_tricks.as_strided(samples,
                                               shape=(n_windows, window_size),
                                               strides=(stride_size * nbytes, nbytes))

        data_input = np.abs(np.fft.fft(data * w, axis=1))
        data_input = data_input[:, :window_size // 2]
        return np.log(data_input + 1)

    @staticmethod
    def _normalize_audio_feature(audio_feature: np.ndarray):
        """Perform mean and variance normalization on the spectrogram feature.

        :param audio_feature: a numpy array for the spectrogram feature.
        :return: a numpy array of the normalized spectrogram.
        """
        mean = np.mean(audio_feature, axis=0)
        std = np.std(audio_feature, axis=0)
        return (audio_feature - mean) / (std + 1e-6)


class TextFeaturizer(object):
    """Extract text feature based on char-level granularity.

    By looking up the vocabulary table, each input string (one line of transcript)
    will be converted to a sequence of integer indexes.
    """

    def __init__(self, vocab: Dict[str, int]):
        self.token_to_index = vocab
        self.index_to_token = {index: word for word, index in vocab.items()}
        self.n_labels = len(self.token_to_index)

    @classmethod
    def from_file(cls, vocab_file: str):
        index = 0
        vocab: Dict[str, int] = {}
        with open(vocab_file, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip("\n")
                if line.startswith("#") or not line:
                    continue
                vocab[line] = index
                index += 0
        return cls(vocab)

    def transform(self, text: str):
        """Convert string to a list of integers."""
        feats = [self.token_to_index[token] for token in text.strip().lower()]
        return feats
