
import os
import sys
import numpy as np
import platform
system = platform.system().lower()
sys.path.append("F:/Code projects/Python/SpeechRecognition" if system == "windows"
                else "/data/zhaochengming/projects/SpeechRecognition")
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras import backend as K
from deep_speech_by_audier.constant import make_vocab, DATA_SOURCE_DIR, AM_LOG_DIR, AM_MODEL_DIR
from deep_speech_by_audier.input_data import DataGenerator
from deep_speech_by_audier.model.speech import AcousticModel
from deep_speech_by_audier.model.utils import get_session
from _utils.tf.util import get_board_log_path
from typing import Dict, Generator
import tensorflow as tf


K.clear_session()
PNY2ID: Dict[str, int] = make_vocab()
ID2PNY: Dict[int, str] = {id_: pny for pny, id_ in PNY2ID.items()}
SHUFFLE = True
FEATURE_TYPE = "fbank"
N_FEATURES = 200
BATCH_SIZE = 4
VOCAB_SIZE = len(PNY2ID)  # 1585
LEARNING_RATE = 8e-4
N_EPOCH = 10
MODEL_TYPE = "DFCNN"
DATA_SOURCE = os.path.join(DATA_SOURCE_DIR, "labeled_data.txt")


TRAIN_DATA = DataGenerator(
    data_source=DATA_SOURCE, pinyin_sep="-", data_type="train",
    is_shuffle=SHUFFLE, data_length=None, vocab=PNY2ID
)

DEV_DATA = DataGenerator(
    data_source=DATA_SOURCE, pinyin_sep="-", data_type="dev",
    is_shuffle=SHUFFLE, data_length=10, vocab=PNY2ID
)

BATCH_NUM = len(TRAIN_DATA.data) // BATCH_SIZE


def batch_wrapper(batch: Generator, model_type: str):
    model_type = model_type.upper()
    for inputs in batch:
        input_data_list, label_data_list, input_length, label_length = inputs
        if model_type == "DFCNN":
            input_length //= 8

        outputs = np.zeros(shape=(len(input_length), ))  # 一个空的输出用来占位满足keras.Model的fit_generator 输入API
        inputs = {
            "the_inputs": input_data_list,
            "the_labels": label_data_list,
            "the_input_length": input_length,
            "the_label_length": label_length
        }
        yield (inputs, outputs)


if __name__ == "__main__":
    print("AM_LOG_PATH: %s" % AM_LOG_DIR)

    batch = TRAIN_DATA.get_am_batch(feature_type=FEATURE_TYPE, n_features=N_FEATURES, batch_size=BATCH_SIZE)
    dev_batch = DEV_DATA.get_am_batch(feature_type=FEATURE_TYPE, n_features=N_FEATURES, batch_size=BATCH_SIZE)
    batch = batch_wrapper(batch, MODEL_TYPE)
    dev_batch = batch_wrapper(batch, MODEL_TYPE)

    graph = tf.Graph()
    tf.reset_default_graph()
    with graph.as_default():
        model = AcousticModel(vocab_size=VOCAB_SIZE, n_features=N_FEATURES,
                              inference_model_type=MODEL_TYPE, learning_rate=LEARNING_RATE, is_training=True)
        model.model.summary()
        print(model.model._feed_input_names)
        K.set_session(get_session(graph=graph))

        if os.path.exists(AM_MODEL_DIR):
            print("Load acoustic model...")
            model.model.load_weights(AM_MODEL_DIR)

        ckpt = "model_{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(
            filepath=os.path.join(AM_LOG_DIR, ckpt),
            monitor="val_loss", save_weights_only=False,
            verbose=1, save_best_only=True
        )

        model_name = "acoustic_model_{}".format(MODEL_TYPE)
        tensorboard = TensorBoard(log_dir=get_board_log_path(model_name), batch_size=BATCH_SIZE)

        model.model.fit_generator(
            batch, epochs=N_EPOCH, steps_per_epoch=BATCH_NUM, verbose=1,
            callbacks=[checkpoint, tensorboard], validation_data=dev_batch, validation_steps=200,
            use_multiprocessing=False, workers=1
        )
        model.model.save_weights(AM_MODEL_DIR)

