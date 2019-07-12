"""Contains the text featurizer class."""

import os
import codecs
from typing import Tuple, List, Dict


class TextFeaturizer(object):
    """
    Text featurizer, for processing or extracting features from text.
    Currently, it only supports char-level tokenizing and conversion into a list of token indices.
    Note that the token indexing order follows the given vocabulary file.

    :param vocab_filepath: Filepath to load vocabulary for token indices conversion.
    """

    def __init__(self, vocab_filepath: str):
        self._vocab_dict, self._vocab_list = self._load_vocabulary_from_file(vocab_filepath)

    @staticmethod
    def _load_vocabulary_from_file(vocab_filepath: str) -> (Dict, List):
        
