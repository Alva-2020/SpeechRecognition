"""
Calculate wer and cer
"""

import Levenshtein


def word_error(decode: str, target: str, word_sep: str=" ") -> int:
    """ Calculate the WE(word error) between decode and target"""
    decode = decode.split(word_sep)
    target = target.split(word_sep)
    word2index = {w: i for i, w in set(decode + target)}
    decode = "".join([chr(word2index[w]) for w in decode])
    target = "".join([chr(word2index[w]) for w in target])
    return Levenshtein.distance(decode, target)


def char_error(decode: str, target: str) -> int:
    """Calculate the CE(char error) between decode and target """
    return Levenshtein.distance(decode, target)


