
import numpy as np
import tensorflow as tf
from typing import Dict


def decode_ctc(num_result: np.ndarray, id2word: Dict[int, str]):
    result = num_result[:, :, :]
    in_len = [result.shape[1]]
    r, _ = tf.keras.backend.ctc_decode(result, input_length=in_len, greedy=True, beam_width=10, top_paths=1)
    r1 = tf.keras.backend.get_value(r[0])
    encoded = r1[0]
    text = [id2word[x] for x in encoded]
    return r1, text


def get_session(graph=None):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 程序最多只能占用指定gpu80%的显存
    config.gpu_options.allow_growth = True  # 程序按需申请内存
    sess = tf.Session(config=config, graph=graph)
    return sess
