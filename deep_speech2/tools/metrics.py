"""
Calculate wer and cer
"""

import Levenshtein
import numpy as np
from typing import List, Union, Iterable


class EditDistance:

    @staticmethod
    def _char_error(decode: str, target: str) -> int:
        """Calculate the CE(char error) between decode and target """
        return Levenshtein.distance(decode, target)

    @staticmethod
    def _word_error(decode: str, target: str, word_sep: str = " ") -> int:
        """ Calculate the WE(word error) between decode and target"""
        decode = decode.split(word_sep)
        target = target.split(word_sep)
        word2index = {w: i for i, w in set(decode + target)}
        decode = "".join([chr(word2index[w]) for w in decode])
        target = "".join([chr(word2index[w]) for w in target])
        return Levenshtein.distance(decode, target)

    @classmethod
    def char_error_rate(cls, decode: Iterable, target: Iterable) -> float:
        """Calculate the CER(char error rate) between decode and target"""
        if isinstance(decode, str) and isinstance(target, str):
            return cls._char_error(decode, target) / len(target)

        decode, target = list(decode), list(target)
        if len(decode) != len(target):
            raise ValueError("Unmatched length between decode: %d and target: %d" % (len(decode), len(target)))

        if isinstance(decode[0], int) and isinstance(target[0], int):
            decode = "".join([chr(x) for x in decode])
            target = "".join([chr(x) for x in target])
            return cls._char_error(decode, target) / len(target)

        if isinstance(decode[0], Iterable) and isinstance(target[0], Iterable):
            decode = ["".join([chr(x) for x in v]) for v in decode]
            target = ["".join([chr(x) for x in v]) for v in target]
            ces = [cls._char_error(x, y) / len(y) for x, y in zip(decode, target) if y != ""]
            return sum(ces) / len(ces)
