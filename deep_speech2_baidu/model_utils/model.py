"""Contains DeepSpeech2 model."""

import sys
import os
import logging
import tensorflow as tf
from deep_speech2_baidu.decoders.scorer_deprecated import Scorer
from deep_speech2_baidu.decoders.decoders_deprecated import ctc_greedy_decoder, ctc_beam_search_decoder_batch
from typing import Union, Optional, Callable, List, Dict


logging.basicConfig(format="[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s")


class DeepSpeech2(object):

    def __init__(self, vocab_size: int, num_conv_layers: int, num_rnn_layers: int, num_rnn_cell: int, user_gru: bool,
                 pretrained_model_path: Optional[str], share_rnn_weights: bool):
        """
        DeepSpeech2Model class.

        :param vocab_size: Decoding vocabulary size.
        :param num_conv_layers: Number of stacking convolution layers.
        :param num_rnn_layers: Number of stacking RNN layers.
        :param num_rnn_cell: RNN layer size (number of RNN cells).
        :param pretrained_model_path: Pretrained model path. If None, will train from scratch.
        :param share_rnn_weights: Whether to share input-hidden weights between forward and backward directional RNNs.
                                  Notice that for GRU, weight sharing is not supported.
        """
        self._num_conv_layers = num_conv_layers
        self.inferer = None
        self.loss_inferer = None
        self.ext_scorer: Optional[Scorer] = None
        self.logger = logging.getLogger("")
        self.logger.setLevel(level=logging.INFO)
        pass

    def init_ext_scorer(self, alpha: float, beta: float, model_path: Optional[str]) -> None:
        if model_path:
            self.logger.info("Begin to initialize the external scorer for decoding!")
            self.ext_scorer = Scorer(alpha, beta, model_path)
            self.logger.info("End initializing scorer!")
        else:
            self.logger.info("No language model provided, decoding by pure beam search without scorer.")

    def decode_batch_beam_search(self, probs_split, beam_size: int,
                                 vocab_list: Union[List[str], Dict[int, str]],
                                 cutoff_prob: float=1.0, cutoff_top_n: int=40,
                                 num_processes: int=8, **kwargs) -> List[str]:
        """
        Decode by beam search for a batch of probs matrix input.

        :param probs_split: List of 2-D probability matrix
        :param beam_size: Width for Beam search.
        :param cutoff_prob: Cutoff probability in pruning, default 1.0, no pruning.
        :param cutoff_top_n: Cutoff number in pruning, only top cutoff_top_n characters with highest probs in
                             vocabulary will be used in beam search, default 40.
        :param vocab_list: List of tokens in the vocabulary, for decoding.
        :param num_processes: Number of processes (CPU) for decoder.
        :return: List of transcription texts.
        """
        if self.ext_scorer is not None:
            if "alpha" in kwargs and "beta" in kwargs:
                self.ext_scorer.reset_params(kwargs["alpha"], kwargs["beta"])
        num_processes = min(num_processes, len(probs_split))
        beam_search_results = ctc_beam_search_decoder_batch(
            probs_split=probs_split,
            vocabulary=vocab_list,
            beam_size=beam_size,
            num_processes=num_processes,
            ext_scoring_func=self.ext_scorer,
            cutoff_prob=cutoff_prob,
            cutoff_top_n=cutoff_top_n
        )

        return [result[0][1] for result in beam_search_results]

    def _adapt_feeding_dict(self, feed_dict: Dict):
        """
        Adapt feeding dict according to network struct.

        To remove impacts from padding part, we add scale_sub_region layer and sub_seq layer.
        For sub_seq layer, 'sequence_offset' and 'sequence_length' fields are appended.
        For each scale_sub_region layer 'convN_index_range' field is appended.
        :param feed_dict: Feeding is a map of field name and tuple index of the data that reader returns.
        :return: Adapted feeding dict.
        """
        adapted_feed_dict = feed_dict.copy()
        adapted_feed_dict["sequence_offset"] = len(adapted_feed_dict)
        adapted_feed_dict["sequecen_length"] = len(adapted_feed_dict)
        for i in range(self._num_conv_layers):
            adapted_feed_dict["conv%d_index_range" % i] = len(adapted_feed_dict)
        return adapted_feed_dict






    def train(self, train_batch_reader):
        pass

    @staticmethod
    def decode_batch_greedy(probs_split: List, vocab_list: Union[List[str], Dict[int, str]]) -> List[str]:
        """
        Decode by best path for a batch of probs matrix input.
        :param probs_split: List of 2-D probability matrix.
        :param vocab_list: List of tokens in the vocabulary, for decoding.
        :return: List of transcription texts.
        """
        return [ctc_greedy_decoder(probs_seq, vocab_list) for probs_seq in probs_split]






