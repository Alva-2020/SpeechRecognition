
import numpy as np
from deep_speech2.data_utils.segments import AudioSegment, SpeechSegment
from python_speech_features import mfcc, delta
from typing import Optional, Union


class AudioFeaturizer(object):
    """
    Audio featurizer, for extracting features from audio contents of AudioSegment or SpeechSegment.
    Currently, it supports feature types of `linear` spectrogram and `mfcc`.

    :param specgram_type: Specgram feature type. Options: 'linear', 'mfcc'.
    :param stride_ms: Striding size (in milliseconds) for generating frames.
    :param window_ms: Window size (in milliseconds) for generating frames.
    :param max_freq: If`specgram_type == 'linear'`, only FFT bins corresponding to frequencies between [0, max_freq] are returned;
                     If `specgram_type == 'mfcc'`, max_feq is the highest band edge of mel filters.
                     If not provided, `max_freq = target_sample_rate / 2`
    :param target_sample_rate: Audio are resampled (if upsampling or downsampling is allowed) to this before extracting spectrogram features.
    :param use_dB_normalization: Whether to normalize the audio to a certain dB before extracting the features.
    :param target_dB: Target audio decibels for normalization.
    """
    def __init__(self,
                 specgram_type: str="linear",
                 stride_ms: float=10.,
                 window_ms: float=20.,
                 max_freq: Optional[float]=None,
                 target_sample_rate: int=16000,
                 use_dB_normalization: bool=True,
                 target_dB: int=-20):

        if stride_ms > window_ms:
            raise ValueError("Stride must be not be greater than window.")
        if max_freq is None:
            max_freq = target_sample_rate / 2
        if max_freq > target_sample_rate / 2:
            raise ValueError("max freq must not be greater than half of sample rate e.g.(<= %f)" % (target_sample_rate / 2))

        self._specgram_type = specgram_type
        self._stride_ms = stride_ms
        self._window_ms = window_ms
        self._max_freq = max_freq
        self._target_sample_rate = target_sample_rate
        self._use_dB_normalization = use_dB_normalization
        self._target_dB = target_dB

    @property
    def n_features(self):
        """The expected `n_features` of the featurized audio specgram"""
        return (self._window_ms / 1000 * self._target_sample_rate) // 2 + 1

    def featurize(self, segment: Union[AudioSegment, SpeechSegment],
                  allow_downsampling: bool=True, allow_upsampling: bool=True) -> np.ndarray:
        """
        Extract audio features from AudioSegment or SpeechSegment.
        :param segment: Audio/speech segment to extract features from.
        :param allow_downsampling: Whether to allow audio downsampling before featurizing.
        :param allow_upsampling: Whether to allow audio upsampling before featurizing.
        :return: Spectrogram audio feature in 2d array.
        :raises ValueError: If audio sample rate is not supported.
        """
        if (segment.sample_rate > self._target_sample_rate and allow_downsampling) or \
                (segment.sample_rate < self._target_sample_rate and allow_upsampling):
            # resample the base signal to the target sample rate.
            segment.resample(self._target_sample_rate)

        if segment.sample_rate != self._target_sample_rate:
            raise ValueError("Segment's rate (%d Hz) doesn't match target rate (%d Hz). "
                             "Could turn `allow_downsampling` or `allow_upsampling` on." %
                             (segment.sample_rate, self._target_sample_rate))
        if self._use_dB_normalization:
            segment.normalize(target_db=self._target_dB)
        return self._compute_specgram(segment.samples, segment.sample_rate)

    def _compute_specgram(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract various audio features."""
        if self._specgram_type == "linear":
            return self._compute_linear_specgram(samples, sample_rate, self._stride_ms, self._window_ms, self._max_freq)
        elif self._specgram_type == "mfcc":
            return self._compute_mfcc(samples, sample_rate, self._stride_ms, self._window_ms, self._max_freq)
        else:
            raise ValueError("Unknown specgram_type %s" % self._specgram_type)

    @staticmethod
    def _compute_mfcc(samples: np.ndarray, sample_rate: int, stride_ms: float=10.,
                      window_ms: float=20., max_freq: Optional[float]=None) -> np.ndarray:
        """Compute mfcc features from samples."""
        if max_freq is None:
            max_freq = sample_rate / 2
        if max_freq > sample_rate / 2:
            raise ValueError("max freq must not be greater than half of sample rate e.g.(<= %f)" % (sample_rate / 2))
        if stride_ms > window_ms:
            raise ValueError("Stride must be not be greater than window.")

        mfcc_feat = mfcc(signal=samples, samplerate=sample_rate, winlen=0.001 * window_ms,
                         winstep=0.001 * stride_ms, highfreq=max_freq)  # n_frames * n_features
        # deltas
        d1_mfcc_feat = delta(mfcc_feat, 2)  # n_frames * n_features
        d2_mfcc_feat = delta(d1_mfcc_feat, 2)  # n_frames * n_features

        concat_mfcc_feat = np.concatenate([np.transpose(feat) for feat in [mfcc_feat, d1_mfcc_feat, d2_mfcc_feat]])

        # in original script, no transpose, e.g. the shape is: [3 * n_features, n_frames]
        # here the shape is: [n_frames, 3 * n_features]
        return np.transpose(concat_mfcc_feat, (1, 0))

    @staticmethod
    def _specgram_real(samples: np.ndarray, stride_size: int, window_size: int, sample_rate: int):
        """Compute the spectrogram for samples from a real signal."""
        # length = stride_size * K + window_size + truncate_size (K is as max as possible, truncate_size < stride_size)
        truncate_size = (len(samples) - window_size) % stride_size  # length of samples to truncate from the tail
        samples = samples[: (len(samples) - truncate_size)]  # len(samples) = length - truncate_size
        nbytes = samples.strides[0]  # bytes between each and next by each axis.

        nshape = (window_size, (len(samples) - window_size) // stride_size + 1)  # (window_size, step_num)
        nstrides = (nbytes, stride_size * nbytes)  # Row by 1 element, col by `stride_size` element

        windows = np.lib.stride_tricks.as_strided(samples, shape=nshape, strides=nstrides)

        # check the 2nd stride step
        assert np.all(np.equal(windows[:, 1], samples[stride_size: (stride_size + window_size)]))

        # window weighting, squared FFT, scaling
        weighting = np.hanning(window_size)[:, np.newaxis]
        fft = np.abs(np.fft.rfft(windows * weighting, axis=0)) ** 2  # shape: (window_size // 2 + 1, step_num)
        scale = np.sum(weighting ** 2) * sample_rate
        fft[1: -1, :] /= (scale / 2.)
        fft[(0, -1), :] /= scale

        # length: window_size // 2 + 1
        # max element is sample_rate / 2
        freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])
        return fft, freqs

    def _compute_linear_specgram(self, samples: np.ndarray, sample_rate: int, stride_ms: float = 10.,
                                 window_ms: float = 20., max_freq: Optional[float] = None, eps: float = 1e-14) -> np.ndarray:
        """Compute the linear spectrogram from FFT energy."""
        if max_freq is not None and max_freq > sample_rate / 2:
            raise ValueError("max freq must not be greater than half of sample rate e.g.(<= %f)" % (sample_rate / 2))
        if stride_ms > window_ms:
            raise ValueError("Stride must be not be greater than window.")

        stride_size = int(0.001 * stride_ms * sample_rate)
        window_size = int(0.001 * window_ms * sample_rate)
        spectrum, freqs = self._specgram_real(
            samples,
            stride_size=stride_size,
            window_size=window_size,
            sample_rate=sample_rate)

        if max_freq is not None:
            ind = np.where(freqs <= max_freq)[0][-1] + 1
            specgram = np.log(spectrum[: ind, :] + eps)
        else:
            # default `max_freq = sample_rate / 2`
            # which is the max of freqs, the ind is precisely the length of fft(spectrum)
            # in original scripts, due to the floating-error,
            # the ind could be smaller than the length of fft, often `length - 1`
            # for which the features should be padded to same in the batch generator.
            # The method here solves the problem to free from padding.
            specgram = np.log(spectrum + eps)

        # in original script, no transpose, e.g. the shape is same as fft result: [window_size // 2 + 1, step_num]
        # here the shape is: [step_num, window_size // 2 + 1]
        return np.transpose(specgram, (1, 0))
