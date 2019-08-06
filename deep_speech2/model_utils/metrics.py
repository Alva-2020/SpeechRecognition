"""
Calculate wer and cer
"""

import Levenshtein
import numpy as np
from typing import List, Union


class EditDistance:

    @staticmethod
    def char_error(decode: str, target: str) -> int:
        """Calculate the CE(char error) between decode and target """
        return Levenshtein.distance(decode, target)

    @staticmethod
    def word_error(decode: str, target: str, word_sep: str = " ") -> int:
        """ Calculate the WE(word error) between decode and target"""
        decode = decode.split(word_sep)
        target = target.split(word_sep)
        word2index = {w: i for i, w in set(decode + target)}
        decode = "".join([chr(word2index[w]) for w in decode])
        target = "".join([chr(word2index[w]) for w in target])
        return Levenshtein.distance(decode, target)

    @classmethod
    def char_error_from_list(cls, decode: List[int], target: List[int]) -> int:
        decode = "".join([chr(x) for x in decode])
        target = "".join([chr(x) for x in target])
        return cls.char_error(decode, target)

    @classmethod
    def char_error_from_matrix(cls, decode: Union[np.ndarray, List[List[int]]], target: Union[np.ndarray, List[List[int]]]) -> int:
        if len(decode) != len(target):
            raise ValueError("Unmatched length between decode %d and target %d" % (len(decode), len(target)))
        decode = ["".join([chr(x) for x in v]) for v in decode]
        target = ["".join([chr(x) for x in v]) for v in target]
        ces = [cls.char_error(x, y) for x, y in zip(decode, target)]




