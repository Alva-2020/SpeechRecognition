
"""Contains the several segment class"""

import numpy as np
import resampy
import random
import copy
from scipy.io import wavfile
from scipy import signal
from typing import Optional


class AudioSegment(object):
    """
    Monaural audio segment abstraction.
    :param samples: Audio samples [num_samples x num_channels].
    :type samples: ndarray.float32
    :param sample_rate: Audio sample rate.
    :raises TypeError: If the sample data type is not float or int.
    """
    def __init__(self, samples: np.ndarray, sample_rate: int):
        self.samples: np.ndarray = self._convert_samples_to_float32(samples)
        if self.samples.ndim > 1:
            self.samples = np.mean(self.samples, axis=1)
        self.sample_rate = sample_rate
        self.num_samples: int = len(samples)
        self.duration: float = self.num_samples / sample_rate

    @property
    def rms_db(self) -> float:
        """
        :return: Return root mean square energy of the audio in dB.
        """
        mean_square = np.mean(self.samples ** 2)
        return 10 * np.log10(mean_square)

    def __eq__(self, other: "AudioSegment"):
        """Whether two objects are equal"""
        if isinstance(other, type(self)):
            return False
        if self.sample_rate != other.sample_rate:
            return False
        if self.samples.shape != other.samples.shape:
            return False
        if np.any(self.samples != other.samples):
            return False
        return True

    def __ne__(self, other: "AudioSegment"):
        """Whether two objects are unequal"""
        return not self.__eq__(other)

    def __str__(self):
        """Return readable representation of object"""
        return "%s: num_samples=%d, sample_rate=%d, duration=%.2fsec, rms=%.2fdB" %\
               (type(self), self.num_samples, self.sample_rate, self.duration, self.rms_db)

    @classmethod
    def from_file(cls, file: str) -> "AudioSegment":
        """
        Create audio segment from audio file.
        :param file: Filepath to audio file.
        :return: Audio segment instance.
        :rtype: AudioSegment
        """

        sample_rate, samples = wavfile.read(file)
        return cls(samples, sample_rate)

    @classmethod
    def from_slice_file(cls, file, start: Optional[float]=None, end: Optional[float]=None) -> "AudioSegment":
        """
        Loads a small section of an audio without having to load the entire file.
        :param file: Input audio filepath or file object.
        :param start: Start time in seconds. If start is negative, it wraps around from the end.
                      If not provided, this function reads from the very beginning.
        :param end: End time in seconds. If end is negative, it wraps around from the end.
                    If not provided, the default behvaior is to read to the end of the file.
        :return: AudioSegment instance of the specified slice of the input audio file.
        :rtype: AudioSegment
        :raises ValueError: If the start and end is invalid to slice from file.
        """
        sample_rate, samples = wavfile.read(file)
        duration = float(len(samples) / sample_rate)
        start = 0. if start is None else start
        end = 0. if end is None else end
        if start < 0.:
            start += duration
        if end < 0.:
            end += duration
        if start < 0.:
            raise ValueError("The start position (%f s) is out of bounds." % start)
        if end < 0.:
            raise ValueError("The end position (%f s) is out of bounds." % end)
        if start > end:
            raise ValueError("The start position (%f s) is later than the end position (%f s)." % (start, end))
        if end > duration:
            raise ValueError("The end position (%f s) is out of bounds (> %f s)" % (end, duration))

        start_frame = int(start * sample_rate)
        end_frame = int(end * sample_rate)
        data = samples[start_frame: end_frame]
        return cls(data, sample_rate)

    @classmethod
    def concatenate(cls, *segments: "AudioSegment") -> "AudioSegment":
        """
        Concatenate an arbitrary number of audio segments together.
        :param segments: Input audio segments to be concatenated.
        :type segments: tuple of AudioSegment
        :return: Audio segment instance as concatenating results.
        :rtype: AudioSegment
        :raises ValueError: If the number of segments is zero, or if the sample_rate of any segments does not match.
        :raises TypeError: If any segment is not AudioSegment instance.
        """
        if not segments:
            raise ValueError("No audio segments are given.")
        sample_rate = segments[0].sample_rate
        for seg in segments:
            if not isinstance(seg, cls):
                raise TypeError("Only audio segments of the same type can be concatenated.")
            if seg.sample_rate != sample_rate:
                raise ValueError("Can't concatenate segments with different sample rate.")
        samples = np.concatenate([seg.samples for seg in segments])
        return cls(samples, sample_rate)

    @classmethod
    def make_silence(cls, duration: float, sample_rate: int) -> "AudioSegment":
        """
        Creates a silent audio segment of the given duration and sample rate.
        :param duration: Length of silence in seconds.
        :param sample_rate: Sample rate.
        :return: Silent AudioSegment instance of the given duration.
        :rtype: AudioSegment
        """
        samples = np.zeros(int(duration * sample_rate))
        return cls(samples, sample_rate)

    def to_wav_file(self, filepath: str, dtype: str="float32") -> None:
        """
        Save audio segment to disk as wav file.
        :param filepath: WAV file path to save the audio segment.
        :param dtype: Subtype for audio file. Options: 'uint8', 'int16', 'int32', 'float32',. Default is 'float32'.
        :raises TypeError: If dtype is not supported.
        """
        samples: np.ndarray = self._convert_samples_from_float32(self.samples, dtype)
        wavfile.write(filepath, self.sample_rate, samples)

    @staticmethod
    def _convert_samples_to_float32(samples: np.ndarray) -> np.ndarray:
        """
        Convert sample type to float32.
        Audio sample type is usually integer or float-point.
        Integers will be scaled to [-1, 1] in float32.
        """
        float32_samples = samples.astype("float32")
        if samples.dtype in np.sctypes["int"]:
            bits = np.iinfo(samples.dtype).bits
            float32_samples /= 2**(bits - 1)
        elif samples.dtype in np.sctypes["float"]:
            pass
        else:
            raise TypeError("Unsupported sample type: %s" % samples.dtype)
        return float32_samples

    @staticmethod
    def _convert_samples_from_float32(samples: np.ndarray, dtype: str) -> np.ndarray:
        """
        Convert sample type from float32 to `dtype`.
        Audio sample type is usually integer or float-point.
        For integer type, float32 will be rescaled from [-1, 1] to the maximum range supported by the integer type.
        This is for writing a audio file.
        """
        dtype = np.dtype(dtype)
        output_samples = samples.copy()
        if dtype in np.sctypes["int"]:
            bits = np.iinfo(dtype).bits
            output_samples *= (2**(bits - 1))
            min_val = np.iinfo(dtype).min
            max_val = np.iinfo(dtype).max
        elif dtype in np.sctypes["float"]:
            min_val = np.finfo(dtype).min
            max_val = np.finfo(dtype).max
        else:
            raise TypeError("Unsupported target type: %s" % dtype)
        return np.clip(output_samples, min_val, max_val).astype(dtype)

    def superimpose(self, other: "AudioSegment") -> None:
        """
        Add samples from another segment to those of this segment (sample-wise addition, not segment concatenation).
        Note that this is an in-place transformation.

        :param other: Segment containing samples to be added in.
        :type other: AudioSegments
        :raise TypeError: If type of two segments don't match.
        :raise ValueError:
        If the sample rates of the two segments are not equal, or if the lengths of segments don't match.
        """
        if not isinstance(other, type(self)):
            raise TypeError("Can't add segments of different types: %s and %s" % (type(self), type(other)))
        if self.sample_rate != other.sample_rate:
            raise ValueError("Can't add segments of different sample rate: %d and %d" % (self.sample_rate, other.sample_rate))
        if len(self.samples) != len(other.samples):
            raise ValueError("Can't add segments of different length: %d and %d" % (len(self.samples), len(other.samples)))
        self.samples += other.samples

    def to_bytes(self, dtype: str="float32") -> bytes:
        """
        Create a byte string containing the audio content.

        :param dtype: Data type for export samples. Options: 'uint8', 'int16', 'int32', 'float32'. Default is 'float32'.
        :return: Byte string containing audio content.
        """
        samples = self._convert_samples_from_float32(self.samples, dtype)
        return samples.tostring()

    def gain_db(self, gain: float) -> None:
        """
        Apply gain in decibels to samples.
        Note that this is an in-place transformation.

        :param gain: Gain in decibels to apply to samples.
        """
        self.samples *= 10.**(gain / 20.)

    def change_speed(self, speed_rate: float) -> None:
        """
        Change the audio speed by linear interpolation.
        Note that this is an in-place transformation.

        :param speed_rate: Rate of speed change:
                           speed_rate > 1.0, speed up the audio;
                           speed_rate = 1.0, unchanged;
                           speed_rate < 1.0, slow down the audio;
                           speed_rate <= 0.0, not allowed, raise ValueError.
        :type speed_rate: float
        :raises ValueError: If speed_rate <= 0.0.
        """
        if speed_rate <= 0:
            raise ValueError("speed rate should be greater than zero.")
        old_length = len(self.samples)
        new_length = int(old_length / speed_rate)
        old_indices = np.arange(old_length)
        new_indices = np.linspace(start=0, stop=old_length, num=new_length)
        self.samples = np.interp(x=new_indices, xp=old_indices, fp=self.samples)

    def normalize(self, target_db: float = -20., max_gain_db: float = 300.) -> None:
        """
        Normalize audio to be of the desired RMS value in decibels.
        Note that this is an in-place transformation.

        :param target_db: Target RMS value in decibels. This value should be less than 0.0 as 0.0 is full-scale audio.
        :param max_gain_db: Max amount of gain in dB that can be applied for normalization. This is to prevent nans when
                            attempting to normalize a signal consisting of all zeros.
        :raises ValueError: If the required gain to normalize the segment to the target_db value exceeds max_gain_db.
        """
        gain = target_db - self.rms_db
        if gain > max_gain_db:
            raise ValueError("Unable to normalize segment to %f dB"
                             " because the probable gain exceeds max_gain_db(%f dB)" % (target_db, max_gain_db))
        self.gain_db(gain)

    # todo: not understand yet
    def normalize_online_bayesian(self, target_db: float, prior_db: float, prior_samples: float, startup_delay: float=0.0) -> None:
        """
        Normalize audio using a production-compatible online/causal algorithm.
        This uses an exponential likelihood and gamma prior to make online estimates of the RMS
        even when there are very few samples.
        Note that this is an in-place transformation.

        :param target_db: Target RMS value in dB.
        :param prior_db: Prior RMS estimate in dB.
        :param prior_samples: Prior strength in number of samples.
        :param startup_delay: Default 0.0s. If provided, this function will accrue statistics for the first startup_delay
                              seconds before applying online normalization.
        """
        startup_sample_idx = min(self.num_samples - 1, int(self.sample_rate * startup_delay))
        prior_mean_squared = 10.**(prior_db / 10.)
        prior_sum_of_squares = prior_mean_squared * prior_samples
        cumsum_of_squares = np.cumsum(self.samples ** 2)
        sample_count = np.arange(self.num_samples) + 1
        if startup_sample_idx > 0:
            cumsum_of_squares[:startup_sample_idx] = cumsum_of_squares[startup_sample_idx]
            sample_count[:startup_sample_idx] = sample_count[startup_sample_idx]
        mean_squared_estimate = (cumsum_of_squares + prior_sum_of_squares) / (sample_count + prior_samples)
        rms_estimate_db = 10 * np.log10(mean_squared_estimate)
        # Compute required time-varying gain.
        gain = target_db - rms_estimate_db
        self.gain_db(gain)

    def resample(self, target_sample_rate: int, filter: str="kaiser_best") -> None:
        """
        Resample the audio to a target sample rate.
        Note that this is an in-place transformation.

        :param target_sample_rate: Target sample rate.
        :param filter: The resampling filter to use one of {'kaiser_best', 'kaiser_fast'}.
        """
        self.samples = resampy.resample(self.samples, sr_orig=self.sample_rate, sr_new=target_sample_rate, filter=filter)
        self.sample_rate = target_sample_rate

    def pad_silence(self, duration: float, sides: str="both") -> None:
        """
        Pad this audio sample with a period of silence.
        Note that this is an in-place transformation.

        :param duration: Length of silence in seconds to pad.
        :param sides: Position for padding:
                     'beginning' - adds silence in the beginning;
                     'end' - adds silence in the end;
                     'both' - adds silence in both the beginning and the end.
        :raises ValueError: If sides is not supported.
        """
        if duration > 0.:
            cls = type(self)
            silence = self.make_silence(duration, self.sample_rate)
            if sides == "beginning":
                padded = cls.concatenate(silence, self)
            elif sides == "end":
                padded = cls.concatenate(self, silence)
            elif sides == "both":
                padded = cls.concatenate(silence, self, silence)
            else:
                raise ValueError("Unknown value for the sides %s" % sides)
            self.samples = padded

    def shift(self, shift_ms: float) -> None:
        """
        Shift the audio in time. If `shift_ms` is positive, shift with time advance;
        if negative, shift with time delay. Silence are padded to keep the duration unchanged.
        Note that this is an in-place transformation.

        :param shift_ms: Shift time in milliseconds.
                         If positive, shift with time advance;
                         If negative, shift with time delay.
        :raises ValueError: If shift_ms is longer than audio duration.
        """
        if abs(shift_ms) / 1000. > self.duration:
            raise ValueError("Abs value of shift_ms should be smaller than audio duration.")
        shift_samples = int(shift_ms * self.sample_rate / 1000)
        if shift_samples > 0:
            # time advance
            self.samples[:-shift_samples] = self.samples[shift_samples:]
            self.samples[-shift_samples:] = 0.
        elif shift_samples < 0:
            # time delay
            self.samples[-shift_samples:] = self.samples[:shift_samples]
            self.samples[:-shift_samples] = 0.

    def subsegment(self, start_sec: Optional[float]=None, end_sec: Optional[float]=None) -> None:
        """
        Cut the AudioSegment between given boundaries.
        Note that this is an in-place transformation.

        :param start_sec: Beginning of subsegment in seconds.
        :param end_sec: End of subsegment in seconds.
        :raise ValueError: If start_sec or end_sec is incorrectly set, e.g. out of bounds in time.
        """
        start_sec = 0. if start_sec is None else start_sec
        end_sec = self.duration if end_sec is None else end_sec
        if start_sec < 0.:
            start_sec += self.duration
        if end_sec < 0.:
            end_sec += self.duration

        if start_sec < 0.:
            raise ValueError("The slice start position (%f s) is out of bounds." % start_sec)
        if end_sec < 0.:
            raise ValueError("The slice end position (%f s) is out of bounds." % end_sec)
        if start_sec > end_sec:
            raise ValueError("The slice start position (%f s) is later than the end position (%f s)." % (start_sec, end_sec))
        if end_sec > self.duration:
            raise ValueError("The slice end position (%f s) is out of bounds (> %f s)" % (end_sec, self.duration))

        start_sample = int(round(start_sec * self.sample_rate))
        end_sample = int(round(end_sec * self.sample_rate))
        self.samples = self.samples[start_sample: end_sample]

    def random_subsegment(self, subsegment_length: float, rng: Optional[random.Random]=None) -> None:
        """
        Cut the specified length of the audiosegment randomly.
        Note that this is an in-place transformation.

        :param subsegment_length: Subsegment length in seconds.
        :param rng: Random number generator state.
        :raises ValueError: If the length of subsegment is greater than the origineal segemnt.
        """
        if subsegment_length > self.duration:
            raise ValueError("Length of subsegment must not be greater than original segment (%d)" % self.duration)
        rng = random.Random() if rng is None else rng
        start_time = rng.uniform(0., self.duration - subsegment_length)
        self.subsegment(start_time, start_time + subsegment_length)

    def convolve(self, impulse_segment: "AudioSegment", allow_resample: bool=False) -> None:
        """
        Convolve this audio segment with the given impulse segment.
        Note that this is an in-place transformation.

        :param impulse_segment: Impulse response segments.
        :param allow_resample: Indicates whether resampling is allowed when the impulse_segment has a different sample
                               rate from this signal.
        :raises ValueError: If the sample rate is not match between two audio segments when resample is not allowed.
        """
        if self.sample_rate != impulse_segment.sample_rate:
            if not allow_resample:
                raise ValueError("Impulse segment's rate (%d Hz) doesn't match base signal's rate (%d Hz)."
                                 "Could set `allow_resample=True` to resample impulse segment." %
                                 (impulse_segment.sample_rate, self.sample_rate))
            impulse_segment.resample(self.sample_rate)
        samples = signal.fftconvolve(in1=self.samples, in2=impulse_segment.samples, mode="full")
        self.samples = samples

    def convolve_and_normalize(self, impulse_segment: "AudioSegment", allow_resample: bool=False) -> None:
        """
        Convolve and normalize the resulting audio segment so that it
        has the same average power as the input signal.

        Note that this is an in-place transformation.
        :param impulse_segment: Impulse response segments.
        :param allow_resample: Indicates whether resampling is allowed when the impulse_segment has a different sample
                               rate from this signal.
        """
        target_db = self.rms_db
        self.convolve(impulse_segment, allow_resample=allow_resample)
        self.normalize(target_db)

    def add_noise(self, noise: "AudioSegment", snr_dB: float, allow_downsampling: bool=False,
                  max_gain_db: float=300., rng: Optional[random.Random]=None) -> None:
        """
        Add the given noise segment at a specific signal-to-noise ratio.
        If the noise segment is longer than this segment, a random subsegment of matching length is sampled from it
        and used instead.

        Note that this is an in-place transformation.

        :param noise: Noise signal to add.
        :param snr_dB: Signal-to-Noise Ratio, in dB.
        :param allow_downsampling: Whether to allow the noise to be downsampled to match the base signal sample rate.
        :param max_gain_db: Maximum amount of gain to apply to noise signal before adding it in.
                            This is to prevent attempting to apply infinite gain to a zero signal.
        :param rng: Random number generator state.
        :raises ValueError: If the sample rate does not match between noise and base signal and downsampling is not allowed,
                            or if the duration of noise segments is shorter than original audio segments.
        """
        if noise.sample_rate < self.sample_rate:
            raise ValueError("Noise sample rate (%d Hz) doesn't match base signal's rate (%d Hz)" %
                             (noise.sample_rate, self.sample_rate))

        if noise.sample_rate > self.sample_rate:
            if not allow_downsampling:
                raise ValueError("Noise sample rate (%d Hz) doesn't match base signal's rate (%d Hz). "
                                 "Could set `allow_downsampling=True` to resample noise." %
                                 (noise.sample_rate, self.sample_rate))
            noise.resample(self.sample_rate)

        if noise.duration < self.duration:
            raise ValueError("Noise signal (%d sec) must be not shorter than base siganl (%d sec)" %
                             (noise.duration, self.duration))

        noise_gain_db = min(self.rms_db - noise.rms_db - snr_dB, max_gain_db)
        noise_new = copy.deepcopy(noise)
        noise_new.random_subsegment(self.duration, rng=rng)
        noise_new.gain_db(noise_gain_db)
        self.superimpose(noise_new)


class SpeechSegment(AudioSegment):
    """
    Speech segment abstraction, a subclass of AudioSegment, with an additional transcript.

    :param samples: Audio samples [num_samples x num_channels].
    :param sample_rate: Audio sample rate.
    :param transcript: Transcript text for the speech.
    :raises TypeError: If the sample data type is not float or int.
    """
    def __init__(self, samples: np.ndarray, sample_rate: int, transcript: str):
        super(SpeechSegment, self).__init__(samples, sample_rate)
        self.transcript = transcript

    def __eq__(self, other: "SpeechSegment"):
        if not super(SpeechSegment, self).__eq__(other):
            return False
        if self.transcript != other.transcript:
            return False
        return True

    def __ne__(self, other: "SpeechSegment"):
        return not self.__eq__(other)

    @classmethod
    def from_audio(cls, audio: AudioSegment, transcript: str) -> "SpeechSegment":
        return cls(audio.samples, audio.sample_rate, transcript)

    @classmethod
    def from_file(cls, filepath: str, transcript: str) -> "SpeechSegment":
        """
        Create speech segment from audio file and corresponding transcript.
        :param filepath: Filepath or file object to audio file.
        :param transcript: Transcript text for the speech.
        :return: Speech segment instance.
        """
        audio = AudioSegment.from_file(filepath)
        return cls.from_audio(audio, transcript)

    @classmethod
    def concatenate(cls, *segments: "SpeechSegment") -> "SpeechSegment":
        """
        Concatenate an arbitrary number of speech segments together, both audio and transcript will be concatenated.

        :param *segments: Input speech segments to be concatenated.
        :return: Speech segment instance.
        :raises ValueError: If the number of segments is zero, or if the sample_rate of any two segments does not match.
        :raises TypeError: If any segment is not SpeechSegment instance.
        """
        if not segments:
            raise ValueError("No speech segments are given to concatenate.")
        sample_rate = segments[0].sample_rate
        transcript = ""
        for seg in segments:
            if not isinstance(seg, cls):
                raise TypeError("Only speech segments of the same type can be concatenated.")
            if sample_rate != seg.sample_rate:
                raise ValueError("Can't concatenate segments with different sample rate.")
            transcript += seg.transcript

        samples = np.concatenate([seg.samples for seg in segments])
        return cls(samples, sample_rate, transcript)

    @classmethod
    def from_slice_file(cls, filepath: str, transcript: str, start: Optional[float]=None, end: Optional[float]=None) -> "SpeechSegment":
        """
        Loads a small section of an speech without having to load the entire file.

        :param filepath: Filepath or file object to audio file.
        :param start: Start time in seconds.
                      If start is negative, it wraps around from the end.
                      If not provided, this function reads from the very beginning.
        :param end: End time in seconds.
                    If end is negative, it wraps around from the end.
                    If not provided, the default behvaior is to read to the end of the file.
        :param transcript: Transcript text for the speech.
        :return: SpeechSegment instance of the specified slice of the input speech file.
        """
        audio = AudioSegment.from_slice_file(filepath, start, end)
        return cls.from_audio(audio, transcript)

    @classmethod
    def make_silence(cls, duration: float, sample_rate: int) -> "SpeechSegment":
        """
        Creates a silent speech segment of the given duration and sample rate, transcript will be an empty string.

        :param duration: Length of silence in seconds.
        :param sample_rate: Sample rate.
        :return: Silence of the given duration.
        """
        audio = AudioSegment.make_silence(duration, sample_rate)
        return cls.from_audio(audio, transcript="")
