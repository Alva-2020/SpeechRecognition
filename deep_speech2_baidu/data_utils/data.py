"""
Contains data generator for orgnaizing various audio data preprocessing pipeline
and offering data reader interface of tensorflow requirements.
"""

import random
import multiprocessing
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils import Sequence
from deep_speech2_baidu.data_utils.utility import read_data
from deep_speech2_baidu.data_utils.augmentor import AugmentationPipeline
from deep_speech2_baidu.data_utils.featurizer.speech_featurizer import SpeechFeaturizer
from deep_speech2_baidu.data_utils.segments import SpeechSegment
from deep_speech2_baidu.data_utils.normalizer import FeatureNormalizer
from typing import Optional, List, Tuple


class DataGenerator(Sequence):
    """
    DataGenerator provides basic audio data preprocessing pipeline, and offers data reader interfaces for Keras.

    :param data_file: The Data source path to read.
    :param data_type: The Data partition to generate, possible choices are 'train', 'dev' or 'test'.
    :param batch_size: The batch size to generate batch.
    :param vocab_filepath: Vocabulary file path for indexing tokenized transcripts.
    :param mean_std_filepath: File containing the pre-computed mean and stddev.
    :param augmentation_config: Augmentation configuration in json string. Details see AugmentationPipeline.__doc__.
    :param max_duration: Audio with duration (in seconds) greater than this will be discarded.
    :param min_duration: Audio with duration (in seconds) smaller than this will be discarded.
    :param stride_ms: Striding size (in milliseconds) for generating frames.
    :param window_ms: Window size (in milliseconds) for generating frames.
    :param max_freq: Used when specgram_type is 'linear', only FFT bins corresponding to frequencies
                     between [0, max_freq] are returned.
    :param specgram_type: Specgram feature type. Options: 'linear'.
    :param use_dB_normalization: Whether to normalize the audio to -20 dB before extracting the features.
    :param num_threads: Number of CPU threads for processing data.
    :param random_seed: Random seed.
    :param keep_transcription_text: If set to True, transcription text will be passed forward directly without
                                    converting to index sequence.
    :type keep_transcription_text: bool
    """
    def __init__(self, data_file: str, data_type: str, batch_size: int, vocab_filepath: str, mean_std_filepath: str,
                 augmentation_config: str="{}", max_duration: float=float("inf"), min_duration: float=0.,
                 stride_ms: float=10., window_ms: float=20., max_freq: Optional[float]=None, specgram_type: str="linear",
                 use_dB_normalization: bool=True, num_threads: int=multiprocessing.cpu_count() // 2, random_seed: int=0,
                 keep_transcription_text: bool=False):
        self.data = read_data(data_file=data_file, to_dict=False)
        self._max_duration = max_duration
        self._min_duration = min_duration
        self._normalizer = FeatureNormalizer(mean_std_filepath)
        self._augmentation_pipeline = AugmentationPipeline(
            augmentation_config=augmentation_config, random_seed=random_seed)
        self._speech_featurizer = SpeechFeaturizer(
            vocab_filepath=vocab_filepath,
            specgram_type=specgram_type,
            stride_ms=stride_ms,
            window_ms=window_ms,
            max_freq=max_freq,
            use_dB_normalization=use_dB_normalization
        )
        self._num_threads = num_threads
        self._rng = random.Random(random_seed)
        self._keep_transcription_text = keep_transcription_text

    def process_utterance(self, audio_file: str, transcript: str) -> (np.ndarray, List):
        """
        Load, augment, featurize and normalize for speech data.

        :param audio_file: File path of audio file.
        :param transcript: Transcription text.
        :return: Tuple of audio feature tensor and data of transcription part,
                 where transcription part could be token ids or text.
        """
        speech_segment = SpeechSegment.from_file(audio_file, transcript)
        self._augmentation_pipeline.transform_audio(speech_segment)
        specgram, transcript_part = self._speech_featurizer.featurize(speech_segment, self._keep_transcription_text)
        specgram = self._normalizer.apply(specgram)
        return specgram, transcript_part

    @staticmethod
    def _read_data(data_file: str, data_tag: Optional[str]="labeled_data", data_type: str="train",
                  max_duration: float=float("inf"), min_duration: float=0.0):
        data = read_data(data_file=data_file, data_tag=data_tag, to_dict=False)

        # use query with @ external parameter
        data = data.query("(@min_duration <= duration <= @max_duration) and (data_type == @data_type)")






    def batch_reader_creator(self, data_path: str, batch_size: int, min_batch_size: int=1,
                             padding_to: int=-1, flatten: bool=False, sortagrad: bool=False):



