
import os
import sys
import platform
system = platform.system().lower()
sys.path.append("F:/Code projects/Python/SpeechRecognition" if system == "windows"
                else "/data/zhaochengming/projects/SpeechRecognition")
from tensorflow.python.keras.callbacks import ModelCheckpoint
from deep_speech_by_audier.constant import make_vocab, DATA_SOURCE_DIR, AM_LOG_DIR, AM_MODEL_DIR
from deep_speech_by_audier.input_data import DataGenerator
from deep_speech_by_audier.model.speech import AcousticModel
from typing import Dict

PNY2ID: Dict[str, int] = make_vocab()
ID2PNY: Dict[int, str] = {id_: pny for pny, id_ in PNY2ID.items()}
SHUFFLE = True
FEATURE_TYPE = "fbank"
N_FEATURES = 200
BATCH_SIZE = 4
VOCAB_SIZE = len(PNY2ID)  # 1585
LEARNING_RATE = 8e-4
N_EPOCH = 10
DATA_SOURCE = os.path.join(DATA_SOURCE_DIR, "labeled_data.txt")

TRAIN_DATA = DataGenerator(
    data_source=DATA_SOURCE_DIR, pinyin_sep="-", data_type="train",
    is_shuffle=SHUFFLE, data_length=None, vocab=PNY2ID
)

DEV_DATA = DataGenerator(
    data_source=DATA_SOURCE_DIR, pinyin_sep="-", data_type="dev",
    is_shuffle=SHUFFLE, data_length=10, vocab=PNY2ID
)

batch_num = len(TRAIN_DATA.data) // BATCH_SIZE
ckpt = "model_{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(
    filepath=os.path.join(AM_LOG_DIR, ckpt),
    monitor="val_loss", save_weights_only=False,
    verbose=1, save_best_only=True
)

model = AcousticModel(vocab_size=VOCAB_SIZE, n_features=N_FEATURES,
                      inference_model="DFCNN", learning_rate=LEARNING_RATE, is_training=True)


if __name__ == "__main__":
    batch = TRAIN_DATA.get_am_batch(feature_type=FEATURE_TYPE, n_features=N_FEATURES, batch_size=BATCH_SIZE)
    dev_batch = DEV_DATA.get_am_batch(feature_type=FEATURE_TYPE, n_features=N_FEATURES, batch_size=BATCH_SIZE)
    model.model.summary()

    if os.path.exists(AM_MODEL_DIR):
        print("Load acoustic model...")
        model.model.load_weights(AM_MODEL_DIR)

    model.model.fit_generator(
        batch, epochs=N_EPOCH, steps_per_epoch=batch_num, verbose=1,
        callbacks=[checkpoint], validation_data=dev_batch, validation_steps=200,
        use_multiprocessing=False, workers=1
    )
    model.model.save_weights(AM_MODEL_DIR)

