"""Contains the text featurizer class."""

import codecs
from typing import Union, List, Dict


class TextFeaturizer(object):
    """
    Text featurizer, for processing or extracting features from text.
    Currently, it only supports char-level tokenizing and conversion into a list of token indices.
    Note that the token indexing order follows the given vocabulary file.

    :param vocab_filepath: File path to load vocabulary for token indices conversion.
    """

    def __init__(self, vocab_filepath: str):
        self._vocab_dict, self._vocab_list = self._load_vocabulary_from_file(vocab_filepath)

    def featurize(self, text: Union[str, List]) -> List[int]:
        """Convert the text to token indices"""
        if isinstance(text, str):
            text = list(text.strip())
        return [self._vocab_dict[token] for token in text]

    @property
    def vocab_list(self):
        return self._vocab_list

    @property
    def vocab_dict(self):
        return self._vocab_dict

    @property
    def vocab_size(self):
        return len(self._vocab_list)

    @staticmethod
    def _load_vocabulary_from_file(vocab_filepath: str) -> (Dict[str, int], List[str]):
        """Load vocabulary from file"""
        vocab_list = []
        with codecs.open(vocab_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    vocab_list.append(line)

        vocab_dict = {token: i for i, token in enumerate(vocab_list)}
        return vocab_dict, vocab_list




