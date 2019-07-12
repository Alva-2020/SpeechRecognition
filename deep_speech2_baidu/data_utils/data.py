"""
Contains data generator for orgnaizing various audio data preprocessing pipeline
and offering data reader interface of tensorflow requirements.
"""

import random
import numpy as np
import tensorflow as tf
from deep_speech2_baidu.data_utils.utility import read_data
from deep_speech2_baidu.data_utils.augmentor import AugmentationPipeline
from deep_speech2_baidu.data_utils.featurizer.speech_featurizer import SpeechFeaturizer
from deep_speech2_baidu.data_utils.segments import SpeechSegment
from deep_speech2_baidu.data_utils.normalizer import FeatureNormalizer


