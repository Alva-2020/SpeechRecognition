"""External Scorer for Beam Search Decoder."""

import os
import kenlm
import numpy as np


class Scorer(object):
    """
    External scorer to evaluate a prefix or whole sentence in beam search decoding,
    including the score from n-gram language model and word count.

    :param alpha: Parameter associated with language model. Don't use language model when alpha = 0.
    :param beta: Parameter associated with word count. Don't use word count when beta = 0.
    :model_path: Path to load language model.
    """
    def __init__(self, alpha: float, beta: float, model_path: str):
        self.alpha = alpha
        self.beta = beta
        if not os.path.isfile(model_path):
            raise IOError("Invalid language model path: %s" % model_path)
        self.lm = kenlm.Model(model_path)

    def _language_model_log_score(self, sentence: str):
        res = 0.0
        for log_prob, ngram_length, oov in self.lm.full_scores(sentence, eos=None):
            res = log_prob
        return res

    def reset_params(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def _word_count(sentence: str):
        return len(sentence.strip().split(" "))

    def __call__(self, sentence: str, log=False):
        lm_log_score = self._language_model_log_score(sentence)
        word_cnt = self._word_count(sentence)
        if log:
            return self.alpha * lm_log_score + self.beta * np.log(word_cnt)
        lm_score = np.power(10, lm_log_score)
        return np.power(lm_score, self.alpha) + np.power(word_cnt, self.beta)
