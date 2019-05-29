
import numpy as np
import tensorflow.python.keras.backend as K
from typing import Dict


def decode_ctc(num_result: np.ndarray, id2word: Dict[int, str]):
    result = num_result[:, :, :]
    in_len = [result.shape[1]]
    r, _ = K.ctc_decode(result, input_length=in_len, greedy=True, beam_width=10, top_paths=1)
    r1 = K.get_value(r[0])
    encoded = r1[0]
    text = [id2word[x] for x in encoded]
    return r1, text
