"""Deep speech decoder."""


import itertools
import Levenshtein
import numpy as np
from typing import Iterable


class DeepSpeechDecoder(object):
    """Greedy decoder implementation for Deep Speech model."""
    def __init__(self, labels: Iterable[str], blank_index: int = 28):
        self.vocab = {i: c for i, c in enumerate(labels)}
        self.blank_index = blank_index

    @staticmethod
    def wer(decode: str, target: str) -> float:
        """Computes the Word Error Rate (WER).

        WER is defined as the edit distance between the two provided sentences after tokenizing to words.
        :param decode: string of the decoded output.
        :param target: a string for the ground truth label.

        :return: A float number for the WER of the current decode-target pair.
        """
        words = set(decode.split() + target.split())
        word2char = {word: i for i, word in enumerate(words)}

        new_decode = "".join([chr(word2char[w]) for w in decode.split()])
        new_target = "".join([chr(word2char[w]) for w in target.split()])

        return Levenshtein.distance(new_decode, new_target) / len(target.split())

    @staticmethod
    def cer(decode: str, target: str) -> float:
        """Computes the Character Error Rate (CER).

        CER is defined as the edit distance between the two given strings.
        :param decode: a string of the decoded output.
        :param target: a string for the ground truth label.
        :return: A float number denoting the CER for the current sentence pair.
        """
        return Levenshtein.distance(decode, target) / len(target)

    def decode(self, logits: np.ndarray) -> str:
        # Choose the class with maximum probability.
        best = list(np.argmax(logits, axis=1))
        # Merge repeated chars.
        # Remove the blank index in the decoded sequence.
        # Index to char
        merge = [self.vocab[k] for k, _ in itertools.groupby(best) if k != self.blank_index]
        return "".join(merge)
