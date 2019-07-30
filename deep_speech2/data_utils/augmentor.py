
import random
import json
from deep_speech2.data_utils.utility import read_data
from deep_speech2.data_utils.segments import AudioSegment, SpeechSegment
from abc import ABCMeta, abstractmethod
from typing import Union, Dict, List


class AugmentorBase(metaclass=ABCMeta):
    """
    Abstract base class for augmentation model (augmentor) class.
    All augmentor classes should inherit from this class, and implement the following abstract methods.
    """
    @abstractmethod
    def transform_audio(self, segment: Union[AudioSegment, SpeechSegment]) -> None:
        """
        Adds various effects to the input audio segment. Such effects will augment the training data to make the model
        invariant to certain types of perturbations in the real world, improving model's generalization ability.

        Note that this is an in-place transformation.

        :param segment: Audio segment to add effects to.
        :type segment: AudioSegmenet|SpeechSegment
        """
        pass


class VolumePerturbAugmentor(AugmentorBase):
    """
    Augmentation model for adding random volume perturbation.
    This is used for multi-loudness training of PCEN. See `https://arxiv.org/pdf/1607.05666v1.pdf` for more details.

    :param rng: Random generator object.
    :param min_gain_dBFS: Minimal gain in dBFS.
    :param max_gain_dBFS: Maximal gain in dBFS.
    """
    def __init__(self, rng: random.Random, min_gain_dBFS: float, max_gain_dBFS: float):
        if max_gain_dBFS < min_gain_dBFS:
            raise ValueError("The max gain must not be smaller than the min gain.")
        self._rng = rng
        self._min_gain_dBFS = min_gain_dBFS
        self._max_gain_dBFS = max_gain_dBFS

    def transform_audio(self, segment: Union[AudioSegment, SpeechSegment]) -> None:
        """
        Change audio loadness.
        Note that this is an in-place transformation.
        """
        gain = self._rng.uniform(self._min_gain_dBFS, self._max_gain_dBFS)
        segment.gain_db(gain)


class SpeedPertubAugmentor(AugmentorBase):
    """
    Augmentation model for adding speed perturbation.

    See reference paper here: http://www.danielpovey.com/files/2015_interspeech_augmentation.pdf

    :param rng: Random generator object.
    :param min_speed_rate: Lower bound of new speed rate to sample and should not be smaller than 0.9.
    :param max_speed_rate: Upper bound of new speed rate to sample and should not be larger than 1.1.
    """
    def __init__(self, rng: random.Random, min_speed_rate: float, max_speed_rate: float):
        if min_speed_rate < 0.9:
            raise ValueError("Sampling speed below 0.9 can cause unnatural effects.")
        if max_speed_rate > 1.1:
            raise ValueError("Sampling speed above 1.1 can cause unnatural effects.")
        if max_speed_rate < min_speed_rate:
            raise ValueError("The max speed must not be smaller than the min speed.")

        self._rng = rng
        self._min_speed_rate = min_speed_rate
        self._max_speed_rate = max_speed_rate

    def transform_audio(self, segment: Union[AudioSegment, SpeechSegment]) -> None:
        """
        Sample a new speed rate from the given range and changes the speed of the given audio clip.
        Note that this is an in-place transformation.
        """
        sampled_speed = self._rng.uniform(self._min_speed_rate, self._max_speed_rate)
        segment.change_speed(sampled_speed)


class ShiftPertubAugmentor(AugmentorBase):
    """
    Augmentation model for adding random shift perturbation.

    :param rng: Random generator object.
    :param min_shift_ms: Minimal shift in milliseconds.
    :param max_shift_ms: Maximal shift in milliseconds.
    """
    def __init__(self, rng: random.Random, min_shift_ms: float, max_shift_ms: float):
        if max_shift_ms < min_shift_ms:
            raise ValueError("The max shift time must not be smaller than the min shift time.")
        self._rng = rng
        self._min_shift_ms = min_shift_ms
        self._max_shift_ms = max_shift_ms

    def transform_audio(self, segment: Union[AudioSegment, SpeechSegment]) -> None:
        """Shift audio"""
        shift_ms = self._rng.uniform(self._min_shift_ms, self._max_shift_ms)
        segment.shift(shift_ms)


class ResampleAugmentor(AugmentorBase):
    """
    Augmentation model for resampling. See more info here: `https://ccrma.stanford.edu/~jos/resample/index.html`

    :param rng: Random generator object.
    :param new_sample_rate: New sample rate in Hz.
    """
    def __init__(self, rng: random.Random, new_sample_rate: int):
        self._rng = rng
        self._new_sample_rate = new_sample_rate

    def transform_audio(self, segment: Union[AudioSegment, SpeechSegment]) -> None:
        """Resamples the input audio to a target sample rate"""
        segment.resample(self._new_sample_rate)


class OnlineBayesianNormalizationAugmentor(AugmentorBase):
    """
    Augmentation model for adding online bayesian normalization.

    :param rng: Random generator object.
    :param target_db: Target RMS value in decibels.
    :param prior_db: Prior RMS estimate in decibels.
    :param prior_samples: Prior strength in number of samples.
    :param startup_delay: Default 0.0s. If provided, this function will accrue statistics for the first startup_delay
                          seconds before applying online normalization.
    """
    def __init__(self, rng: random.Random, target_db: float, prior_db: float, prior_samples: int, startup_delay: float):
        self._target_db = target_db
        self._prior_db = prior_db
        self._prior_samples = prior_samples
        self._rng = rng
        self._startup_delay = startup_delay

    def transform_audio(self, segment: Union[AudioSegment, SpeechSegment]) -> None:
        """Normalizes the input audio using the online Bayesian approach."""
        segment.normalize_online_bayesian(self._target_db, self._prior_db, self._prior_samples, self._startup_delay)


class NoisePerturbAugmentor(AugmentorBase):
    """
    Augmentation model for adding background noise.

    :param rng: Random generator object.
    :param min_snr_dB: Minimal signal noise ratio, in decibels.
    :param max_snr_dB: Maximal signal noise ratio, in decibels.
    :param noise_file_path: file path for noise audio data.
    """
    def __init__(self, rng: random.Random, min_snr_dB: float, max_snr_dB: float, noise_file_path: str):
        self._min_snr_dB = min_snr_dB
        self._max_snr_dB = max_snr_dB
        self._rng = rng
        self._noise_data = read_data(noise_file_path, data_tag="noise", to_dict=True)

    def transform_audio(self, segment: Union[AudioSegment, SpeechSegment]) -> None:
        """Add background noise audio."""
        noise_data = self._rng.sample(self._noise_data, 1)[0]
        if noise_data["duration"] < segment.duration:
            raise RuntimeError("The duration of sampled noise audio is smaller than the audio segment.")
        diff_duration = noise_data["duration"] - segment.duration
        start = self._rng.uniform(0, diff_duration)
        end = start + segment.duration
        noise_seg = AudioSegment.from_slice_file(noise_data["src"], start=start, end=end)
        snr_dB = self._rng.uniform(self._min_snr_dB, self._max_snr_dB)
        segment.add_noise(noise_seg, snr_dB=snr_dB, allow_downsampling=True, rng=self._rng)


class ImpulseResponseAugmentor(AugmentorBase):
    """
    Augmentation model for adding impulse response effect.

    :param rng: Random generator object.
    :param impulse_file_path: file path for impulse audio data.
    """
    def __init__(self, rng: random.Random, impulse_file_path: str):
        self._rng = rng
        self._impulse_data = read_data(impulse_file_path, data_tag="impulse", to_dict=True)

    def transform_audio(self, segment: Union[AudioSegment, SpeechSegment]) -> None:
        """Add impulse response effect."""
        impulse_data = self._rng.sample(self._impulse_data, 1)[0]
        impulse_segment = AudioSegment.from_file(impulse_data["src"])
        segment.convolve(impulse_segment=impulse_segment, allow_resample=True)


_AUGMENT_TYPE_ALIAS = {
    "volume": VolumePerturbAugmentor,
    "shift": ShiftPertubAugmentor,
    "speed": SpeedPertubAugmentor,
    "resample": ResampleAugmentor,
    "bayesian_normal": OnlineBayesianNormalizationAugmentor,
    "noise": NoisePerturbAugmentor,
    "impulse": ImpulseResponseAugmentor
}


class AugmentationPipeline(object):
    """
    Build a pre-processing pipeline with various augmentation models.Such a data augmentation pipeline is often
    leveraged to augment the training samples to make the model invariant to certain types of perturbations in the
    real world, improving model's generalization ability.

    The pipeline is built according the the augmentation configuration in json string, e.g.

    .. code-block::

        [ {
                "type": "noise",
                "params": {"min_snr_dB": 10,
                           "max_snr_dB": 20,
                           "noise_manifest_path": "datasets/manifest.noise"},
                "prob": 0.0
            },
            {
                "type": "speed",
                "params": {"min_speed_rate": 0.9,
                           "max_speed_rate": 1.1},
                "prob": 1.0
            },
            {
                "type": "shift",
                "params": {"min_shift_ms": -5,
                           "max_shift_ms": 5},
                "prob": 1.0
            },
            {
                "type": "volume",
                "params": {"min_gain_dBFS": -10,
                           "max_gain_dBFS": 10},
                "prob": 0.0
            },
            {
                "type": "bayesian_normal",
                "params": {"target_db": -20,
                           "prior_db": -20,
                           "prior_samples": 100},
                "prob": 0.0
            }
        ]

    This augmentation configuration inserts two augmentation models into the pipeline,
    with one is `VolumePerturbAugmentor` and the other `SpeedPerturbAugmentor`.
    `prob` indicates the probability of the current augmentor to take effect.
    If "prob" is zero, the augmentor does not take effect.

    :param augmentation_config: Augmentation configuration in json string.
    :param random_seed: Random seed.
    :raises ValueError: If the augmentation json config is in incorrect format".
    """
    def __init__(self, augmentation_config: str, random_seed: int=0):
        self._rng = random.Random(random_seed)
        self._augmentors, self._rates = self._parse_pipeline_from(augmentation_config)

    def transform_audio(self, segment: Union[AudioSegment, SpeechSegment]) -> None:
        """Run the pre-processing pipeline for data augmentation."""
        for augmentor, rate in zip(self._augmentors, self._rates):
            if self._rng.uniform(0., 1.) < rate:
                augmentor.transform_audio(segment)

    def _parse_pipeline_from(self, config_json: str):
        """Parse the config json to build a augmentation pipelien."""
        try:
            configs: List[Dict] = json.loads(config_json)
            augmentors = [
                self._get_augmentor(config["type"], config["params"])
                for config in configs
            ]
            rates = [config["prob"] for config in configs]
        except Exception as e:
            raise ValueError("Failed to parse the augmentation config json: %s" % str(e))
        return augmentors, rates

    def _get_augmentor(self, augmentor_type: str, params: Dict):
        """Return an augmentation model by the type name, and pass in params."""
        params["rng"] = self._rng
        if augmentor_type not in _AUGMENT_TYPE_ALIAS:
            raise ValueError("Unknown augumentor type [%s]" % augmentor_type)
        return _AUGMENT_TYPE_ALIAS[augmentor_type](**params)
