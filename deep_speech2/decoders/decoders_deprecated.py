""" Contains various CTC decoders. """

import numpy as np
import multiprocessing
from itertools import groupby
from math import log
from typing import Dict, Callable, Optional, List, Union

# global func
ext_nproc_scorer: Optional[Callable] = None


def ctc_greedy_decoder(probs_seq, blank_index: int, vocabulary=None) -> Union[str, List[int]]:
    """
    CTC Greedy (best path) decoder.
    Path consisting of the most probable tokens are further post-processed to remove consecutive repetitions and all blanks.

    :param probs_seq: 2-D list of probabilities over the vocabulary for each character.
                      Each element is a list of float probabilities for one character.
    :param blank_index: The index indicating the blank.
    :param vocabulary: Vocabulary.
    :return: Decoding result string.
    """
    # dimension verification
    try:
        probs_seq = np.asarray(probs_seq)
    except Exception as e:
        raise ValueError("probs_seq is invalid, expected 2d array-like list.")

    total_len = probs_seq.shape[1]
    if total_len != len(vocabulary) + 1:
        raise ValueError("probs_seq dimension mismatched with vocabulary")

    # argmax to get the best index for each time step
    max_index_list = list(np.array(probs_seq).argmax(axis=1))
    # remove consecutive duplicate indexes
    index_list = [index_group[0] for index_group in groupby(max_index_list)]
    if vocabulary:
        # remove blank indexes
        # convert index list to string
        return "".join([vocabulary[index] for index in index_list if index != blank_index])
    return index_list


def ctc_beam_search_decoder(probs_seq, beam_size: int, vocabulary: Union[List[str], Dict[int, str]],
                            cutoff_prob: float=1.0, cutoff_top_n: int=40,
                            ext_scoring_func: Optional[Callable]=None, nproc: bool=False) -> List:
    """
    CTC Beam search decoder.

    It utilizes beam search to approximately select top best decoding labels and returning results in the descending order.
    The implementation is based on Prefix Beam Search(https://arxiv.org/abs/1408.2873), and the unclear part is redesigned.
    Two important modifications:
        1) in the iterative computation of probabilities, the assignment operation is changed to
           accumulation for one prefix may comes from different paths;
        2) the if condition "if l^+ not in A_prev then" after probabilities' computation is deprecated for it is
           hard to understand and seems unnecessary.

    :param probs_seq: 2-D list of probability distributions over each time step, with each element being a list of normalized
                      probabilities over vocabulary and blank.
    :param beam_size: Width for beam search.
    :param vocabulary: Vocabulary Dict.
    :param cutoff_prob: Cutoff probability in pruning,  default 1.0, no pruning.
    :param cutoff_top_n: Cutoff length in pruning, default 40.
    :param ext_scoring_func: External scoring function for partially decoded sentence, e.g. word count or language model.
    :param nproc: Whether the decoder used in multiprocesses.
    :return: List of tuples of log probability and sentence as decoding results, in descending order of the probability.
    """
    # dimension verification
    try:
        probs_seq = np.asarray(probs_seq)
    except Exception as e:
        raise ValueError("probs_seq is invalid, expected 2d array-like list.")

    total_len = probs_seq.shape[1]
    if total_len != len(vocabulary) + 1:
        raise ValueError("probs_seq dimension mismatched with vocabulary")

    # blank_id assign
    blank_id = len(vocabulary)

    # If the decoder called in the multiprocesses, then use the global scorer
    # instantiated in ctc_beam_search_decoder_batch().
    if nproc is True:
        global ext_nproc_scorer
        ext_scoring_func = ext_nproc_scorer

    # [initialize]
    # prefix_set_prev: the set containing selected prefixes
    # probs_b_prev: prefixes' probability ending with blank in previous step
    # probs_nb_prev: prefixes' probability ending with non-blank in previous step
    prefix_set_prev = {'\t': 1.0}
    probs_b_prev, probs_nb_prev = {'\t': 1.0}, {'\t': 0.0}

    # extend prefix in loop
    for time_step in range(len(probs_seq)):
        # prefix_set_next: the set containing candidate prefixes
        # probs_b_cur: prefixes' probability ending with blank in current step
        # probs_nb_cur: prefixes' probability ending with non-blank in current step
        prefix_set_next, probs_b_cur, probs_nb_cur = {}, {}, {}

        prob_idx = list(enumerate(probs_seq[time_step]))
        cutoff_len = len(prob_idx)
        # If pruning is enabled
        if cutoff_prob < 1.0 or cutoff_top_n < cutoff_len:
            prob_idx = sorted(prob_idx, key=lambda asd: asd[1], reverse=True)
            cutoff_len, cum_prob = 0, 0.0
            for i in range(len(prob_idx)):
                cum_prob += prob_idx[i][1]
                cutoff_len += 1
                if cum_prob >= cutoff_prob:
                    break
            cutoff_len = min(cutoff_len, cutoff_top_n)
            prob_idx = prob_idx[0:cutoff_len]

        for l in prefix_set_prev:
            if l not in prefix_set_next:
                probs_b_cur[l], probs_nb_cur[l] = 0.0, 0.0

            # extend prefix by travering prob_idx
            for c, prob_c in prob_idx:

                if c == blank_id:
                    probs_b_cur[l] += prob_c * (probs_b_prev[l] + probs_nb_prev[l])
                else:
                    last_char = l[-1]
                    new_char = vocabulary[c]
                    l_plus = l + new_char
                    if l_plus not in prefix_set_next:
                        probs_b_cur[l_plus], probs_nb_cur[l_plus] = 0.0, 0.0

                    if new_char == last_char:
                        probs_nb_cur[l_plus] += prob_c * probs_b_prev[l]
                        probs_nb_cur[l] += prob_c * probs_nb_prev[l]
                    elif new_char == ' ':
                        if (ext_scoring_func is None) or (len(l) == 1):
                            score = 1.0
                        else:
                            prefix = l[1:]
                            score = ext_scoring_func(prefix)
                        probs_nb_cur[l_plus] += score * prob_c * (probs_b_prev[l] + probs_nb_prev[l])
                    else:
                        probs_nb_cur[l_plus] += prob_c * (probs_b_prev[l] + probs_nb_prev[l])
                    # add l_plus into prefix_set_next
                    prefix_set_next[l_plus] = probs_nb_cur[
                                                  l_plus] + probs_b_cur[l_plus]
            # add l into prefix_set_next
            prefix_set_next[l] = probs_b_cur[l] + probs_nb_cur[l]
        # update probs
        probs_b_prev, probs_nb_prev = probs_b_cur, probs_nb_cur

        # store top beam_size prefixes
        prefix_set_prev = sorted(prefix_set_next.items(), key=lambda asd: asd[1], reverse=True)
        if beam_size < len(prefix_set_prev):
            prefix_set_prev = prefix_set_prev[:beam_size]
        prefix_set_prev = dict(prefix_set_prev)

    beam_result = []
    for seq, prob in prefix_set_prev.items():
        if prob > 0.0 and len(seq) > 1:
            result = seq[1:]
            # score last word by external scorer
            if (ext_scoring_func is not None) and (result[-1] != ' '):
                prob = prob * ext_scoring_func(result)
            log_prob = log(prob)
            beam_result.append((log_prob, result))
        else:
            beam_result.append((float('-inf'), ''))

    # output top beam_size decoding results
    beam_result = sorted(beam_result, key=lambda asd: asd[0], reverse=True)
    return beam_result


def ctc_beam_search_decoder_batch(probs_split, beam_size: int, vocabulary: Union[List[str], Dict[int, str]],
                                  num_processes: int, cutoff_prob: float=1.0, cutoff_top_n: int=40,
                                  ext_scoring_func: Optional[Callable]=None) -> List:
    """
    CTC beam search decoder using multiple processes.

    :param probs_split: 3D list with each element as an instance of 2-D list of probabilities used by ctc_beam_search_decoder().
    :param beam_size: Width for beam search.
    :param vocabulary: Vocabulary list.
    :param num_processes: Number of parallel processes.
    :param cutoff_prob: Cutoff probability in pruning, default 1.0, no pruning.
    :param cutoff_top_n: Cutoff length in pruning, default 40.
    :param num_processes: Number of parallel processes.
    :param ext_scoring_func: External scoring function for partially decoded sentence, e.g. word count or language model
    :return: List of tuples of log probability and sentence as decoding results, in descending order of the probability.
    """
    # use global variable to pass the externnal scorer to beam search decoder
    global ext_nproc_scorer
    ext_nproc_scorer = ext_scoring_func

    pool = multiprocessing.Pool(processes=num_processes)
    results = []

    for probs_seq in probs_split:
        kwds = {
            "probs_seq": probs_seq,
            "beam_size": beam_size,
            "vocabulary": vocabulary,
            "cutoff_prob": cutoff_prob,
            "cutoff_top_n": cutoff_top_n,
            "ext_scoring_func": None,
            "nproc": True
        }
        results.append(pool.apply_async(ctc_beam_search_decoder, kwds=kwds))

    pool.close()
    pool.join()

    beam_search_results = [result.get() for result in results]
    return beam_search_results
