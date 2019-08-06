"""
Contains data generator for orgnaizing various audio data preprocessing pipeline
and offering data reader interface of tensorflow requirements.
"""

import random
import numpy as np
import tensorflow as tf
from deep_speech2.tools.build_vocab import get_vocab_column
from deep_speech2.data_utils.utility import read_data
from deep_speech2.data_utils.augmentor import AugmentationPipeline
from deep_speech2.data_utils.featurizer.speech_featurizer import SpeechFeaturizer
from deep_speech2.data_utils.segments import SpeechSegment
from deep_speech2.data_utils.normalizer import FeatureNormalizer
from collections import namedtuple
from typing import Optional, List

BatchData = namedtuple("BatchData", ["features", "labels", "input_length", "label_length"])


class DataGenerator(object):
    """
    DataGenerator provides basic audio data preprocessing pipeline, and offers data reader interfaces for TensorFlow.

    :param data_file: The Data source path to read.
    :param data_type: The Data partition to generate, possible choices are 'train', 'dev' or 'test'.
    :param batch_size: The batch size to generate batch.
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
    def __init__(self, data_file: str, data_type: str, batch_size: int, vocab_file: str, vocab_type: str,
                 mean_std_file: str, augmentation_config: str="{}", max_duration: float=float("inf"),
                 min_duration: float=0., stride_ms: float=10., window_ms: float=20., max_freq: Optional[float]=None,
                 sample_rate: int=16000, specgram_type: str="linear", use_dB_normalization: bool=True, random_seed: int=0,
                 keep_transcription_text: bool=False):
        self._data = self._process_data(data_file=data_file, data_tag="labeled_data", data_type=data_type,
                                        vocab_type=vocab_type, max_duration=max_duration, min_duration=min_duration)

        self._indexes = np.arange(len(self._data))
        self._batch_size = batch_size
        self._max_duration = max_duration
        self._min_duration = min_duration
        self._normalizer = FeatureNormalizer(mean_std_file)
        self._augmentation_pipeline = AugmentationPipeline(
            augmentation_config=augmentation_config, random_seed=random_seed)
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
        self._keep_transcription_text = keep_transcription_text

    @property
    def num_classes(self):
        """Indicates the total classes."""
        return self._speech_featurizer.vocab_size

    @property
    def n_batches(self):
        return len(self._data) // self._batch_size

    @property
    def n_features(self):
        return self._speech_featurizer.n_features

    def __getitem__(self, index) -> BatchData:
        batch_data = self._data[index * self._batch_size: (index + 1) * self._batch_size]
        batch_data = self._padding_batch(batch_data)
        features, labels, input_length, label_length = zip(*batch_data)
        return BatchData(features=features, labels=labels, input_length=input_length, label_length=label_length)

    def process_utterance(self, audio_file: str, transcript: str) -> (np.ndarray, List):
        """
        Load, augment, featurize and normalize for speech data.

        :param audio_file: File path of audio file.
        :param transcript: Transcription text.
        :return: Tuple of audio feature tensor and data of transcription part,
                 where transcription part could be token ids or text.
                 If `keep_transcription_text` is True: the transcription part is text
                 else(which is default) the transcription part is token ids.
        """
        speech_segment = SpeechSegment.from_file(audio_file, transcript)
        self._augmentation_pipeline.transform_audio(speech_segment)
        specgram, transcript_part = self._speech_featurizer.featurize(speech_segment, self._keep_transcription_text)
        specgram = self._normalizer.apply(specgram)
        return specgram, transcript_part

    def _process_data(self, data_file: str, data_tag: Optional[str]="labeled_data", data_type: str="train",
                      vocab_type: str="pny", max_duration: float=float("inf"), min_duration: float=0.0) -> List:
        if min_duration < 0:
            raise ValueError("The min duration should be greater than 0.")
        if max_duration < min_duration:
            raise ValueError("The max duration should be no smaller than min duration.")
        data = read_data(data_file=data_file, data_tag=data_tag, to_dict=False)
        vocab_column = get_vocab_column(vocab_type)

        # use query with @ external parameter
        data = data.query("(@min_duration <= duration <= @max_duration) and (data_type == '@data_type')")
        if len(data) == 0:
            raise ValueError("The %s data for %s is empty!" % (data_tag, data_type))

        res = []
        for src, transcript in data[["src", vocab_column]].values:
            specgram, tokens = self.process_utterance(src, transcript)
            res.append((specgram, tokens))
        return res

    @staticmethod  # not used in TensorFlow research, but used in baidu original scripts.
    def _padding_batch(batch_data: List, padding_to: int=-1, flatten: bool=False) -> List:
        """
        Padding audio features with zeros to make them have the same shape (or a user-defined shape) within one bach.

        If ``padding_to`` is -1, the maximum shape in the batch will be used as the target shape for padding.
        Otherwise, `padding_to` will be the target shape (only refers to the second axis).
        If `flatten` is True, features will be flatten to 1d array.
        """
        new_batch = []
        max_length = max([len(audio) for audio, text in batch_data])  # audio is of shape [#n_frames, #n_features]
        if padding_to != -1:
            if padding_to < max_length:
                raise ValueError("If `padding_to` != -1, it should be larger than any instance's n_frames in the batch.")
            max_length = padding_to

        for audio, tokens in batch_data:
            true_length = len(audio)  # the original length
            padded_audio = np.pad(audio, pad_width=[(0, max_length - true_length), (0, 0)], mode="constant")
            if flatten:
                padded_audio = padded_audio.flatten()
            padded_instance = [padded_audio, tokens, true_length, len(tokens)]
            new_batch.append(padded_instance)
        return new_batch

    def shuffle(self):
        self._rng.shuffle(self._data)

    def input_fn(self):
        """A constructor for `Input_fn`"""

        def _gen_data():
            for specgram, tokens in self._data:
                features = np.expand_dims(specgram, axis=2)  # shape: [n_frames, n_features, 1]
                input_length = [len(specgram)]
                label_length = [len(tokens)]
                yield (
                    {
                        "features": features,
                        "input_length": input_length,
                        "label_length": label_length
                    },
                    tokens
                )

        data_set = tf.data.Dataset.from_generator(
            generator=_gen_data,
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
                tf.TensorShape([None])
            ))

        data_set = data_set.repeat(1)
        data_set = data_set.padded_batch(
            batch_size=self._batch_size,
            padded_shapes=(
                {
                    "features": tf.TensorShape([None, self.n_features, 1]),
                    "input_length": tf.TensorShape([1]),
                    "label_length": tf.TensorShape([1])
                },
                tf.TensorShape([None])
            )
        )

        data_set = data_set.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)  # available when tf >= 1.8.0
        return data_set
