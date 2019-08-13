"""
Contains data generator for orgnaizing various audio data preprocessing pipeline
and offering data reader interface of tensorflow requirements.
"""

import random
import numpy as np
import _utils.tensorflow as utf
from deep_speech2.tools.build_vocab import get_vocab_column
from deep_speech2.data_utils.utility import read_data
from deep_speech2.data_utils.augmentor import AugmentationPipeline
from deep_speech2.data_utils.featurizer.speech_featurizer import SpeechFeaturizer
from deep_speech2.data_utils.segments import SpeechSegment
from deep_speech2.data_utils.normalizer import FeatureNormalizer
from collections import namedtuple
from typing import Optional, List, Tuple

BatchData = namedtuple("BatchData", ["features", "labels", "input_length", "label_length"])


class DataGenerator(object):
    """
    DataGenerator provides basic audio data preprocessing pipeline, and offers data reader interfaces for TensorFlow.

    :param data_file: The Data source path to read.
    :param partition: The Data partition to generate, possible choices are 'train', 'dev' or 'test'.
    :param vocab_file: Vocabulary file path for indexing tokenized transcripts.
    :param vocab_type: The type of content which will be tokenized. possible choices are 'pny' or 'han'.
    :param mean_std_file: File containing the pre-computed mean and stddev.
    :param augmentation_config: Augmentation configuration in json string. Details see AugmentationPipeline.__doc__.
    :param max_duration: Audio with duration (in seconds) greater than this will be discarded.
    :param min_duration: Audio with duration (in seconds) smaller than this will be discarded.
    :param stride_ms: Striding size (in milliseconds) for generating frames.
    :param window_ms: Window size (in milliseconds) for generating frames.
    :param max_freq: Used when specgram_type is 'linear', only FFT bins corresponding to frequencies
                     between [0, max_freq] are returned.
    :param sample_rate: target sample rate to featurize.
    :param specgram_type: Specgram feature type. Options: 'linear'.
    :param use_dB_normalization: Whether to normalize the audio to -20 dB before extracting the features.
    :param random_seed: Random seed.
    :param keep_transcription_text: If set to True, transcription text will be passed forward directly without
                                    converting to index sequence.
    :type keep_transcription_text: bool
    """
    def __init__(self, data_file: str, partition: str, vocab_file: str, vocab_type: str,
                 mean_std_file: str, augmentation_config: str="{}", max_duration: float=float("inf"),
                 min_duration: float=0., stride_ms: float=10., window_ms: float=20., max_freq: Optional[float]=None,
                 sample_rate: int=16000, specgram_type: str="linear", use_dB_normalization: bool=True, random_seed: int=0,
                 keep_transcription_text: bool=False):

        self._augmentation_pipeline = \
            AugmentationPipeline(augmentation_config=augmentation_config, random_seed=random_seed)
        self._max_duration = max_duration
        self._min_duration = min_duration
        self._normalizer = FeatureNormalizer(mean_std_file)

        self._speech_featurizer = SpeechFeaturizer(
            vocab_filepath=vocab_file,
            specgram_type=specgram_type,
            stride_ms=stride_ms,
            window_ms=window_ms,
            max_freq=max_freq,
            target_sample_rate=sample_rate,
            use_dB_normalization=use_dB_normalization
        )

        self._rng = random.Random(random_seed)
        self._sep = "-" if vocab_type == "pny" else None
        self._keep_transcription_text = keep_transcription_text
        self._data = self.read_source(
            data_file=data_file, partition=partition, vocab_type=vocab_type, max_duration=max_duration,
            min_duration=min_duration)

        self.partition = partition

    @property
    def num_classes(self):
        """Indicates the total classes."""
        return self._speech_featurizer.vocab_size

    @property
    def n_features(self):
        return self._speech_featurizer.n_features

    @property
    def __len__(self):
        return len(self._data)

    @staticmethod
    def read_source(data_file: str, data_tag: Optional[str]="labeled_data", partition: str="train",
                    vocab_type: str="pny", max_duration: float=float("inf"), min_duration: float=0.0) -> np.ndarray:
        """Read data info from given source file, not process just the data infos."""

        if min_duration < 0:
            raise ValueError("The min duration should be greater than 0.")
        if max_duration < min_duration:
            raise ValueError("The max duration should be no smaller than min duration.")
        if partition not in {'train', 'dev', 'test'}:
            raise ValueError("Invalid data_type '%s', possible choices are ['train', 'dev', 'test'.]" % partition)

        data = read_data(data_file=data_file, data_tag=data_tag, to_dict=False)
        transcript_column = get_vocab_column(vocab_type)

        # use query with @ external parameter
        # no need add '' for string variables.
        data = data.query("(@min_duration <= duration <= @max_duration) and (data_type == @partition)")
        if len(data) == 0:
            raise ValueError("The %s data for %s is empty!" % (data_tag, partition))

        return data[["src", transcript_column]].values

    def write_to_record(self, file: str):
        """Write the data into tf record"""

        # The processing func on each element of data, which is also the important parameter of saver.
        def _gen_data(element: Tuple):
            src, transcript = element
            specgram, tokens = self.process_utterance(src, transcript, self._sep)
            true_length = len(specgram)  # original specgram length without padding, needed by calculating ctc loss
            label_length = len(tokens)  # the label length without padding
            features = np.expand_dims(specgram, 2)  # shape: [n_frames, n_features, 1]
            return features, tokens, true_length, label_length

        print("Saving {partition} file as record into {file}".format(partition=self.partition, file=file))
        saver = utf.record.RecordSaver(file)
        saver.save(self._data, length=len(self._data), desc="{} Saving".format(self.partition.upper()),
                   keys=["features", "labels", "true_length", "label_length"], element_func=_gen_data)
        print("Saved successfully!")

    def process_utterance(self, audio_file: str, transcript: str, text_sep: Optional[str]=None) -> (np.ndarray, List):
        """
        Load, augment, featurize and normalize for speech data.

        :param audio_file: File path of audio file.
        :param transcript: Transcription text.
        :param text_sep: The sep string of text. if `None`, use `list` to convert text to list.
        :return: Tuple of audio feature tensor and data of transcription part,
                 where transcription part could be token ids or text.
                 If `keep_transcription_text` is True: the transcription part is text
                 else(which is default) the transcription part is token ids.
        """
        speech_segment = SpeechSegment.from_file(audio_file, transcript)
        self._augmentation_pipeline.transform_audio(speech_segment)
        specgram, transcript_part =\
            self._speech_featurizer.featurize(speech_segment, self._keep_transcription_text, text_sep)
        specgram = self._normalizer.apply(specgram)
        return specgram, transcript_part
