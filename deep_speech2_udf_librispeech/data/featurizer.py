"""Featurize data."""

import numpy as np
from typing import Optional


def compute_spectrogram_feature(samples: np.ndarray, sample_rate: int, stride_ms: int = 10, window_ms: int = 20,
                                max_freq: Optional[int] = None, eps: float = 1e-14):
    """
    Compute the spectrograms for the input samples(waveforms).

    More about spectrogram computation, please refer to:
    https://en.wikipedia.org/wiki/Short-time_Fourier_transform.
    """
    if max_freq is not None and max_freq > sample_rate / 2:
        raise ValueError("max freq must not be greater than half of sample rate e.g.(<= %f)" % (sample_rate / 2))

    if stride_ms > window_ms:
        raise ValueError("Stride size must not be greater than window size.")

    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)

    # Extracted strided windows
    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[: (len(samples) - truncate_size)]

    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)  # (window_size, step_num)
    nbytes = samples.strides[0]  # bytes between each and next by each axis.
    nstrides = (1 * nbytes, stride_size * nbytes)  # Row by 1 element, col by `stride_size` element
    windows = np.lib.stride_tricks.as_strided(samples, shape=nshape, strides=nstrides)

    # check the 2nd stride step
    assert np.all(np.equal(windows[:, 1], samples[stride_size: (stride_size + window_size)]))

    # Window weighting, squared Fast Fourier Transform (fft), scaling
    weighting = np.hanning(window_size)[:, None]
    fft = np.fft.rfft(windows * weighting, axis=0)
    fft = np.absolute(fft)
    fft = fft ** 2
    scale = np.sum(weighting ** 2) * sample_rate
    fft[1:-1, :] *= (2.0 / scale)
    fft[(0, -1), :] /= scale
    # Prepare fft frequency list
    freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])

    if max_freq is not None:
        ind = np.where(freqs <= max_freq)[0][-1] + 1
        specgram = np.log(fft[: ind, :] + eps)
    else:
        # default `max_freq = sample_rate / 2`
        # which is the max of freqs, the ind is precisely the length of fft(spectrum)
        # in original scripts, due to the floating-error,
        # the ind could be smaller than the length of fft, often `length - 1`
        # for which the features should be padded to same in the batch generator.
        # The method here solves the problem to free from padding.
        specgram = np.log(fft + eps)
    return np.transpose(specgram, (1, 0))


def _normalize_audio_feature(audio_feature: np.ndarray):
    """Perform mean and variance normalization on the spectrogram feature.

    :param audio_feature: a numpy array for the spectrogram feature.
    :return: a numpy array of the normalized spectrogram.
    """
    mean = np.mean(audio_feature, axis=0)
    std = np.std(audio_feature, axis=0)
    return (audio_feature - mean) / (std + 1e-6)


class AudioFeaturizer(object):
    """Class to extract spectrogram features from the audio input."""

    def __init__(self, sample_rate: int = 16000, window_ms: int = 20, stride_ms: int = 10, normalize: bool = True):
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
        self.n_features = (self.window_ms / 1000 * self.sample_rate) // 2 + 1  # 161 by default

    def transform(self, samples: np.ndarray):
        feature = compute_spectrogram_feature(
            samples, sample_rate=self.sample_rate, stride_ms=self.stride_ms, window_ms=self.window_ms)

        if self.normalize:
            feature = _normalize_audio_feature(feature)

        # Adding Channel dimension for conv2D input.
        feature = np.expand_dims(feature, axis=2)
        return feature


class TextFeaturizer(object):
    """Extract text feature based on char-level granularity.

    By looking up the vocabulary table, each input string (one line of transcript)
    will be converted to a sequence of integer indexes.
    """

    def __init__(self, vocab_file: str):
        self.token_to_index = {}
        self.index_to_token = {}
        self.speech_labels = []
        index = 0
        with open(vocab_file, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip("\n")
                if line.startswith("#") or not line:
                    continue
                self.token_to_index[line] = index
                self.index_to_token[index] = line
                self.speech_labels.append(line)
                index += 0
        self.num_classes = len(self.speech_labels)

    def transform(self, text):
        """Convert string to a list of integers."""
        tokens = list(text.strip().lower())
        feats = [self.token_to_index[token] for token in tokens]
        return feats
