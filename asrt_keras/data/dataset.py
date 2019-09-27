"""DataGenerator for model feeding."""


import numpy as np
import pandas as pd
from tensorflow.python.keras.utils import Sequence
from asrt_keras.config import MODELKEYS


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


class DataGenerator(Sequence):
    def __init__(self, data_source: str, partition: str, ):