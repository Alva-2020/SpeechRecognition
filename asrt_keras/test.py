
import os
import argparse
import tensorflow as tf
from tensorflow.python.keras import backend as K
from asrt_keras.data.dataset import DataGenerator
from asrt_keras.model.model import AcousticModel
from asrt_keras.train import MODEL_FILE, get_dataset_config
from _utils.utensorflow import get_session_config


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
    parser.add_argument("--batch_size", type=int, default=8, help="The data feed batch size.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="The learning rate.")
    parser.add_argument("--gpu_num", type=int, default=1, help="The num of gpu to use.")
    args = parser.parse_args()

    tf.reset_default_graph()
    K.clear_session()
    K.set_session(tf.Session(config=get_session_config()))

    dataset_config = get_dataset_config(args)
    test = DataGenerator(partition="test", config=dataset_config, seed=args.seed)

    model = AcousticModel(
        vocab_size=test.n_labels, n_features=test.n_features, gpu_num=args.gpu_num,
        learning_rate=args.learning_rate, is_training=False)

    model_file = os.path.join(args.model_dir, MODEL_FILE)
    if not os.path.exists(model_file):
        raise IOError(f"file not found in `{model_file}`.")
    model.inference_model.load_model(model_file)

    macro_avg_cer, micro_avg_cer = model.test(test, 100)
    print(f"Macro Avg. CER: {macro_avg_cer}, Micro Avg. CER: {micro_avg_cer}.")
