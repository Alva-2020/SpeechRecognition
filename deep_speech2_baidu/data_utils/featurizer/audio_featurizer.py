
import numpy as np
from deep_speech2_baidu.data_utils.data import read_labeled_data
from python_speech_features import mfcc, delta
from typing import Optional


class AudioFeaturizer(object):
    """Audio featurizer, for extracting features from audio contents of
    AudioSegment or SpeechSegment.

    Currently, it supports feature types of linear spectrogram and mfcc.

    :param specgram_type: Specgram feature type. Options: 'linear', 'mfcc'.
    :param stride_ms: Striding size (in milliseconds) for generating frames.
    :param window_ms: Window size (in milliseconds) for generating frames.
    :param max_freq: When specgram_type is 'linear', only FFT bins
                     corresponding to frequencies between [0, max_freq] are
                     returned; when specgram_type is 'mfcc', max_feq is the
                     highest band edge of mel filters.
    :param target_sample_rate: Audio are resampled (if upsampling or
                               downsampling is allowed) to this before
                               extracting spectrogram features.
    :param use_dB_normalization: Whether to normalize the audio to a certain
                                 decibels before extracting the features.
    :param target_dB: Target audio decibels for normalization.
    """
    def __init__(self,
                 specgram_type: str="linear",
                 stride_ms: int=10,
                 window_ms: int=20,
                 max_freq: Optional[float]=None,
                 target_sample_rate: int=16000,
                 use_dB_normalization: bool=True,
                 target_dB: int=-20):
        self._specgram_type = specgram_type
        self._stride_ms = stride_ms
        self._window_ms = window_ms
        self._max_freq = max_freq
        self._target_sample_rate = target_sample_rate
        self._use_dB_normalization = use_dB_normalization
        self._target_dB = target_dB

    def feature(self, audio_segment, allow_downsampling: bool=True, allow_upsampling: bool=True) -> np.ndarray:
        """Extract audio features from AudioSegment or SpeechSegment.
        :param audio_segment: Audio/speech segment to extract features from.
        :type audio_segment: AudioSegment|SpeechSegment
        :param allow_downsampling: Whether to allow audio downsampling before
                                   featurizing.
        :param allow_upsampling: Whether to allow audio upsampling before
                                 featurizing.
        :return: Spectrogram audio feature in 2darray.
        :rtype: ndarray
        :raises ValueError: If audio sample rate is not supported.
        """
        
