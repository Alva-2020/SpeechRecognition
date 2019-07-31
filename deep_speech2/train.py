
"""Trainer for DeepSpeech2 model."""

import os
import argparse
import configparser
import tensorflow as tf
from deep_speech2.model_utils.model import Model
from deep_speech2.data_utils.data import DataGenerator
from typing import Optional, Dict


params_tree = {
     "data": ["data_file", "vocab_file", "vocab_type", "mean_std_file"],
     "feature": ["sample_rate", "window_ms", "stride_ms", "max_freq", "specgram_type", "use_dB_normalization"],
     "model": ["rnn_hidden_size", "rnn_hidden_layers", "rnn_type", "fc_use_bias"],
     "train": ["batch_size", "random_seed", "learning_rate", "model_path"]
}


def load_params(args: argparse.Namespace) -> Dict:
    config = configparser.ConfigParser()
    if args.param_file:
        if os.path.isfile(args.param_file):
            config.read(args.param_file)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--param_file", type=str, help="The config file containing the whole parameters.")
    parser.add_argument("--data_file", type=str, help="The path to the labeled data file.")
    parser.add_argument("--vocab_file", type=str, help="The path to the vocabulary file.")
    parser.add_argument("--vocab_type", type=str, choices=["pny", "han", "eng"], default="pny", help="The vocab type.")
    parser.add_argument("--mean_std_file", type=str, help="The mean std file to normalize features.")
    parser.add_argument("--sample_rate", type=int, default=16000, help="The sample rate for audio processing.")
    parser.add_argument("--window_ms", type=int, default=20, help="The window length for processing audio features.")
    parser.add_argument("--stride_ms", type=int, default=10, help="The stride length for processing audio features.")
    parser.add_argument("--max_freq", type=float, help="The max freq to limit the audio features.")
    parser.add_argument("--specgram_type", type=str, default="linear", choices=["linear", "mfcc"], help="The feature type to generate")
    parser.add_argument("--use_dB_normalization", type=bool, default=True, help="Whether to normalize the audio to -20 dB before extracting the features.")
    parser.add_argument("--rnn_hidden_size", type=int, default=800, help="The hidden size of RNNs.")
    parser.add_argument("--rnn_hidden_layers", type=int, default=5, help="The num of RNN layers.")
    parser.add_argument("--rnn_type", type=str, default="gru", help="Type of RNN cell.")
    parser.add_argument("--fc_use_bias", type=bool, default=False, help="Whether use bias in the last fc layer.")
    parser.add_argument("--random_seed", type=int, default=0, help="The random seed to generate data.")
    parser.add_argument("--batch_size", type=int, default=4, help="The batch size of data fed in.")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="The learning rate of network.")
    args = parser.parse_args()

    train_data = DataGenerator(
        data_file=args.data_dir, data_type="train", batch_size=args.batch_size, vocab_file=args.vocab_file,
        vocab_type=args.vocab_type, mean_std_file=args.mean_std_file, stride_ms=args.stride_ms,
        window_ms=args.window_ms, max_freq=args.max_freq, sample_rate=args.sample_rate, specgram_type=args.specgram_type,
        use_dB_normalization=args.use_dB_normalization, random_seed=args.random_seed, keep_transcription_text=False
    )

    model_fn = Model(
        num_classes=num_classes, rnn_hidden_layers=args.rnn_hidden_layers, rnn_type=args.rnn_type,
        is_bidirectional=args.is_bidirectional, rnn_hidden_size=args.rnn_hidden_size, fc_use_bias=args.fc_use_bias)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args.model_dir)
    estimator.train(input_fn=)