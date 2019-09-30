
import os
import argparse
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras import backend as K
from asrt_keras.data.dataset import AudioConfig, DatasetConfig, DataGenerator
from asrt_keras.model.model import AcousticModel
from _utils.utensorflow import get_board_log_path, get_session_config
from typing import List


MODEL_FILE = "am_model.h5"


def get_dataset_config(args: argparse.Namespace):
    audio_config = AudioConfig(
        sample_rate=args.sample_rate, window_ms=args.window_ms, stride_ms=args.stride_ms, normalize=args.is_normalize)
    return DatasetConfig(
        audio_config=audio_config, data_path=args.data_file, vocab_file_path=args.vocab_file,
        sortagrad=args.sortagrad, batch_size=args.batch_size)


def get_callbacks(args: argparse.Namespace) -> List[tf.keras.callbacks.Callback]:
    checkpoint_file = os.path.join(args.model_dir, "model_{epoch:02d}-{val_loss:.2f}.hdf5")
    checkpoints = ModelCheckpoint(
        filepath=checkpoint_file, monitor="val_loss", save_weights_only=False, verbose=1, save_best_only=True)
    tensorboard = TensorBoard(log_dir=get_board_log_path("DFCNN"), batch_size=args.batch_size)
    return [checkpoints, tensorboard]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, help="The path where labeled data placed.")
    parser.add_argument("--model_dir", type=str, help="The path where model saved.")
    parser.add_argument("--vocab_file", type=str, help="The path where vocabulary file placed.")
    parser.add_argument("--sortagrad", type=bool, default=True, help="Whether to sort input audio by length.")
    parser.add_argument("--sample_rate", type=int, default=16000, help="The sample rate for audio.")
    parser.add_argument("--window_ms", type=int, default=20, help="The frame length for spectrogram.")
    parser.add_argument("--stride_ms", type=int, default=10, help="The frame step for spectrogram.")
    parser.add_argument("--is_normalize", type=bool, default=True, help="whether normalize the audio feature.")
    parser.add_argument("--seed", type=int, default=1, help="The random seed.")
    parser.add_argument("--batch_size", type=int, default=4, help="The data feed batch size.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="The learning rate.")
    parser.add_argument("--epochs", type=int, default=100, help="The training epochs.")
    parser.add_argument("--gpu_num", type=int, default=1, help="The num of gpu to use.")
    args = parser.parse_args()

    tf.reset_default_graph()
    K.clear_session()
    K.set_session(tf.Session(config=get_session_config()))

    model_file = os.path.join(args.model_dir, MODEL_FILE)
    callbacks = get_callbacks(args)

    dataset_config = get_dataset_config(args)
    train = DataGenerator(partition="train", config=dataset_config, seed=args.seed)
    dev = DataGenerator(partition="dev", config=dataset_config, seed=args.seed)

    model = AcousticModel(
        vocab_size=train.n_labels, n_features=train.n_features, gpu_num=args.gpu_num,
        learning_rate=args.learning_rate, is_training=True)
    model.summary()

    if os.path.exists(model_file):
        print("Load acoustic model...")
        model.load_model(model_file)
    model.train(train, dev, epochs=args.epochs, callbacks=callbacks)
    model.save_model(model_file)
