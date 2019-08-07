

import numpy as np
import random
from deep_speech2.data_utils.utility import read_data
from deep_speech2.data_utils.segments import AudioSegment
from typing import Optional, Callable, List, Dict


class FeatureNormalizer(object):
    """
    Feature normalizer. Normalize features to be of zero mean and unit stddev.

    if `mean_std_filepath` is provided (not None), the normalizer will directly initialize from the file.
    Otherwise, both `data_path` and `featurize_func` should be given for on-the-fly mean and stddev computing.

    :param mean_std_filepath: File containing the pre-computed mean and stddev.
    :param data_path: File of instances for computing mean and stddev.
    :param featurize_func: Function to extract features. It should be callable with `featurize_func(audio_segment)`.
    :param num_samples: Number of random samples for computing mean and stddev.
    :param random_seed: Random seed for sampling instances.
    :raises ValueError: When `mean_std_filepath`  is None e.g. not loading from the file,
                        `data_path` or and `featurize_func` which is necessary for computing is None.
    """

    def __init__(self,
                 mean_std_filepath: Optional[str]=None,
                 data_path: Optional[str]=None,
                 featurize_func: Optional[Callable]=None,
                 num_samples: int=500,
                 random_seed: int=0):
        if not mean_std_filepath:
            if not (data_path and featurize_func):
                raise ValueError("If mean_std_filepath is None, data_path and featurize_func should not be None.")
            self._rng = random.Random(random_seed)
            self._compute_mean_std(data_path, featurize_func, num_samples)
        else:
            self._read_mean_std_from_file(mean_std_filepath)

    def apply(self, features: np.ndarray, eps: float=1e-14) -> np.ndarray:
        """Normalize features to be of zero mean and unit stddev."""
        return (features - self._mean) / (self._std + eps)

    def _compute_mean_std(self, data_path: str, featurize_func: Callable, num_samples: int):
        """Compute mean and std from randomly sampled instances."""
        data: List[Dict] = read_data(data_path, data_tag="labeled_data", to_dict=True)
        sampled_data = self._rng.sample(data, num_samples)
        features = []
        for instance in sampled_data:
            feature = featurize_func(AudioSegment.from_file(instance["src"]))  # [N_frames, N_features]
            features.append(feature)
        features = np.vstack(features)  # [Total_frames, N_features]
        self._mean = np.mean(features, axis=0, keepdims=True)  # [1, N_features]
        self._std = np.std(features, axis=0, keepdims=True)  # [1, N_features]

    def write_to_file(self, filepath: str):
        """Write the mean and std to the file"""
        np.savez(filepath, mean=self._mean, std=self._std)

    def _read_mean_std_from_file(self, filepath: str):
        """Load mean and std from file."""
        npzfile = np.load(filepath)
        self._mean = npzfile["mean"]
        self._std = npzfile["std"]
