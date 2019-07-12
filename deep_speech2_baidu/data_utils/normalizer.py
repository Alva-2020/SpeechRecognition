

import numpy as np
import random
from deep_speech2_baidu.data_utils.data import read_data
from deep_speech2_baidu.data_utils.segments import AudioSegment
from typing import Optional, Callable


class FeatureNormalizer(object):
    """
    Feature normalizer. Normalize features to be of zero mean and unit stddev.

    if `mean_std_filepath` is provided (not None), the normalizer will directly initilize from the file.
    Otherwise, both `data_path` and `featurize_func` should be given for on-the-fly mean and stddev computing.

    :param mean_std_filepath: File containing the pre-computed mean and stddev.
    :param data_path: File of instances for computing mean and stddev.
    :param featurize_func: Function to extract features. It should be callable with `featurize_func(audio_segment)`.
    :param num_samples: Number of random samples for computing mean and stddev.
    :param random_seed: Random seed for sampling instances.
    :raises ValueError: If both `mean_std_filepath` and `data_path`
                        or both `mean_std_filepath` and `featurize_func` are None.
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
        data = read_data(data)