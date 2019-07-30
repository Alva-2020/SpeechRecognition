"""Contains the speech featurizer class."""

import numpy as np
from deep_speech2.data_utils.featurizer.audio_featurizer import AudioFeaturizer
from deep_speech2.data_utils.featurizer.text_featurizer import TextFeaturizer
from deep_speech2.data_utils.segments import SpeechSegment
from typing import Optional, List


class SpeechFeaturizer(object):
    """
    Speech featurizer, for extracting features from both audio and transcript contents of SpeechSegment.
    for audio parts, it supports feature types of linear spectrogram and mfcc;
    for transcript parts, it only supports char-level tokenizing and conversion into a list of token indices.
    Note that the token indexing order follows the given vocabulary file.

    :param vocab_filepath: Filepath to load vocabulary for token indices conversion.
    :param specgram_type: Specgram feature type. Options: 'linear', 'mfcc'.
    :param stride_ms: Striding size (in milliseconds) for generating frames.
    :param window_ms: Window size (in milliseconds) for generating frames.
    :param max_freq: If specgram_type is 'linear', only FFT bins corresponding to frequencies between [0, max_freq] are
                     returned;
                     If specgram_type is 'mfcc', `max_freq` is the highest band edge of mel filters.
    :param target_sample_rate: Speech are resampled (if upsampling or downsampling is allowed) to this.
    :param use_dB_normalization: Whether to normalize the audio to a certain dB before extracting the features.
    :param target_dB: Target audio dB for normalization.
    """
    def __init__(self,
                 vocab_filepath: str,
                 specgram_type: str="linear",
                 stride_ms: float=10.,
                 window_ms: float=20.,
                 max_freq: Optional[float]=None,
                 target_sample_rate: int=16000,
                 use_dB_normalization: bool=True,
                 target_dB: int=-20):
        self._audio_featurizer = AudioFeaturizer(
            specgram_type=specgram_type,
            stride_ms=stride_ms,
            window_ms=window_ms,
            max_freq=max_freq,
            target_sample_rate=target_sample_rate,
            use_dB_normalization=use_dB_normalization,
            target_dB=target_dB
        )
        self._text_featurizer = TextFeaturizer(
            vocab_filepath=vocab_filepath
        )

    def featurize(self, speech_segment: SpeechSegment, keep_transcription_text: bool) -> (np.ndarray, List):
        """
        Extract features for speech segment.

        1. For audio parts, extract the audio features.
        2. For transcript parts, keep the original text or convert text string to a list of token indices in char-level.

        :param speech_segment: Speech segment to extract features from.
        :param keep_transcription_text: whether to return original text or token indices.
        :return: A tuple 1) spectrogram audio feature in 2d array.
                         2) If `keep_transcription_text = False`: list of token indices.
                            If `keep_transcription_text = True`
        """
        audio_feature = self._audio_featurizer.featurize(speech_segment)
        if keep_transcription_text:
            return audio_feature, speech_segment.transcript
        text_ids = self._text_featurizer.featurize(speech_segment.transcript)
        return audio_feature, text_ids

    @property
    def vocab_size(self):
        return self._text_featurizer.vocab_size

    @property
    def vocab_list(self):
        return self._text_featurizer.vocab_list

    @property
    def vocab_dict(self):
        return self._text_featurizer.vocab_dict
