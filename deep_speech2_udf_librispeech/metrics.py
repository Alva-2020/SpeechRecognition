"""
Calculate wer and cer
"""

import Levenshtein
from typing import List, Tuple


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
    def cer(cls, decode: List[List[int]], target: List[List[int]]) -> float:
        if len(decode) != len(target):
            raise ValueError(f"Unmatched length between decode: `{len(decode)}` and target: `{len(target)}`.")
        decode = ["".join([chr(x) for x in v]) for v in decode]
        target = ["".join([chr(x) for x in v]) for v in target]
        ces = [cls._char_error(x, y) / len(y) for x, y in zip(decode, target) if y != ""]
        return sum(ces) / len(ces)

    @classmethod
    def wer(cls, decode: List[List[int]], target: List[List[int]]) -> float:
        if len(decode) != len(target):
            raise ValueError(f"Unmatched length between decode: `{len(decode)}` and target: `{len(target)}`.")
        decode = ["".join([chr(x) for x in v]) for v in decode]
        target = ["".join([chr(x) for x in v]) for v in target]
        wes = [cls._word_error(x, y) / len(y.split()) for x, y in zip(decode, target) if y != ""]
        return sum(wes) / len(wes)

    @classmethod
    def report(cls, decode: List[List[int]], target: List[List[int]]) -> Tuple[float, float]:
        if len(decode) != len(target):
            raise ValueError(f"Unmatched length between decode: `{len(decode)}` and target: `{len(target)}`.")

        n = len(decode)
        wer, cer = 0, 0
        for u, v in zip(decode, target):
            u = "".join([chr(x) for x in u])
            v = "".join([chr(x) for x in v])
            cer += (cls._char_error(u, v) / len(v))
            wer += (cls._word_error(u, v) / len(v.split()))
        return cer, wer

