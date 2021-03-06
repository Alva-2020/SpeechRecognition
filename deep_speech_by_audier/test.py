
import os
import sys
import pandas as pd
import platform
system = platform.system().lower()
sys.path.append("F:/Code projects/Python/SpeechRecognition" if system == "windows"
                else "/data/zhaochengming/projects/SpeechRecognition")

from tensorflow.python.keras import backend as K
from deep_speech_by_audier.model.speech import AcousticModel
from deep_speech_by_audier.model.utils import get_session, decode_ctc
from deep_speech_by_audier.constant import make_pny_vocab, DATA_SOURCE_DIR, AM_LOG_DIR, AM_MODEL_DIR
from deep_speech_by_audier.input_data import DataGenerator
from evan_utils.u_distance import _levenshtein
from tqdm import tqdm
from typing import Dict

tqdm.pandas(tqdm)  # can use tqdm_gui
K.clear_session()
PNY2ID: Dict[str, int] = make_pny_vocab()
ID2PNY: Dict[int, str] = {id_: pny for pny, id_ in PNY2ID.items()}
VOCAB_SIZE = len(PNY2ID)  # 1585
N_FEATURES = 200
FEATURE_TYPE = "fbank"
MODEL_TYPE = "DFCNN"
BATCH_SIZE = 1
DATA_SOURCE = os.path.join(DATA_SOURCE_DIR, "labeled_data.txt")
TEST_BATCH = DataGenerator(
    data_source=DATA_SOURCE, pinyin_sep="-", data_type="test", feed_model="speech", model_type=MODEL_TYPE,
    feature_type=FEATURE_TYPE, n_features=N_FEATURES, shuffle=False, batch_size=BATCH_SIZE, data_length=-1,
    am_vocab=PNY2ID
)


if __name__ == "__main__":
    model = AcousticModel(vocab_size=VOCAB_SIZE, n_features=N_FEATURES,
                          inference_model_type=MODEL_TYPE, learning_rate=0., is_training=False)
    print("Load acoustic model...")
    K.set_session(get_session())
    model.inference_model.load_weights(AM_MODEL_DIR)
    result_file = os.path.join(DATA_SOURCE_DIR, "test_result.txt")
    if os.path.exists(result_file):  # 如果存在文件就先删除
        os.remove(result_file)

    lines = []
    total_wer = 0.0
    for i in tqdm(range(len(TEST_BATCH)), total=len(TEST_BATCH)):
        inputs, _ = TEST_BATCH[i]  # inputs [BATCH_SIZE, N_FEATURES]
        src = TEST_BATCH.data[i, 0]
        x = inputs["the_inputs"]
        y_true = [ID2PNY[x] for x in inputs["the_labels"][0]]
        y_pred = model.inference_model.predict(x, batch_size=BATCH_SIZE, steps=1, use_multiprocessing=True)
        _, y_pred = decode_ctc(y_pred, ID2PNY)
        diff = _levenshtein(y_true, y_pred)
        wer = diff / len(y_true)
        total_wer += wer
        lines.append([src, y_true, y_pred, wer])

    print("Test AVG wer: %.4f" % (total_wer / len(TEST_BATCH)))
    with open(result_file, "w", encoding="utf-8") as f:
        for src, y_true, y_pred, wer in lines:
            y_true = " ".join(y_true)
            y_pred = " ".join(y_pred)
            f.write("\t".join([src, y_true, y_pred, wer]) + "\n")

