
"""Trainer for DeepSpeech2 model."""

import os
import sys
import argparse
import time
import re
import tensorflow as tf
from deep_speech2_udf_librispeech.model_utils.model import Model
from deep_speech2_udf_librispeech.data import dataset
from deep_speech2_udf_librispeech.metrics import EditDistance
from _utils.confighandler import ConfigHandler
from _utils.utensorflow.utils import get_ckpt_global_step
from _utils.utensorflow.record import generate_feature_desc
from typing import List, Dict, Any
from tqdm import tqdm


BLANK_INDEX = 28
# Default vocabulary file
_VOCAB_FILE = os.path.join(os.path.dirname(__file__), "data/vocabulary.txt")
# Evaluation metrics
_WER_KEY = "WER"
_CER_KEY = "CER"
DATASET_NAME = "LibriSpeech"


def get_args(arg_parser: argparse.ArgumentParser,
             input_keys: List[str] = sys.argv[1:]) -> Dict[str, Any]:
    """
    A process wrapper for parsed args
    :param arg_parser: An `argparse.ArgumentParser` from argparse module.
    :param input_keys: A list of chars input from command line, often `sys.argv[1:]`
    :return: a simple dict.
    """
    args = arg_parser.parse_args()

    # get the arg keys from command line.
    argv = [re.sub("^--|=.*$", "", arg) for arg in input_keys if arg.startswith("--")]

    # input args from command line.
    given_args = {k: v for k, v in vars(args).items() if k in argv} if argv else {}

    if args.param_file:
        if os.path.exists(args.param_file):
            # if the config file exists, read config from file
            config = ConfigHandler.from_xml(args.param_file)
            # and updated with given args input from command line.
            if given_args:
                config.update(given_args, ignore_keys=["param_file"])
        else:
            # if the config file doesn't exist, read config from parser
            config = ConfigHandler.from_argparser(arg_parser, ignore_keys=["param_file"])
            # and save to the xml file.
            config.write(args.param_file)
    else:
        # if not given a config file, then read config from command line
        config = ConfigHandler.from_argparser(arg_parser, ignore_keys=["param_file"])

    return config.get_args(as_namespace=False)


def get_data_params(args: Dict[str, Any]) -> Dict[str, Any]:
    audio_config = dataset.AudioConfig(
        sample_rate=args["sample_rate"], window_ms=args["window_ms"],
        stride_ms=args["stride_ms"], normalize=args["is_normalize"])
    dataset_config = dataset.DatasetConfig(
        audio_config=audio_config, data_path=args["data_file"],
        vocab_file_path=args["vocab_file"], sortagrad=args["sortagrad"])
    return dict(dataset_config=dataset_config, batch_size=args["batch_size"], seed=args["random_seed"])


def get_model_params(args: Dict[str, Any]) -> Dict[str, Any]:
    return dict(
        rnn_hidden_layers=args["rnn_hidden_layers"], rnn_type=args["rnn_type"], is_bidirectional=args["is_bidirectional"],
        rnn_hidden_size=args["rnn_hidden_size"], fc_use_bias=args["fc_use_bias"], learning_rate=args["learning_rate"],
        gpu_num=args["gpu_num"])


def build_session(graph):
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
    return tf.Session(graph=graph, config=config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--param_file", type=str, help="The config file containing the whole parameters.")
    parser.add_argument("--data_file", type=str, help="The path where labeled data placed.")
    parser.add_argument("--model_dir", type=str, help="The path where model saved.")
    parser.add_argument("--vocab_file", type=str, default=_VOCAB_FILE, help="The path where vocabulary file placed.")
    parser.add_argument("--sortagrad", type=bool, default=True, help="Whether to sort input audio by length.")
    parser.add_argument("--sample_rate", type=int, default=16000, help="The sample rate for audio.")
    parser.add_argument("--window_ms", type=int, default=20, help="The frame length for spectrogram.")
    parser.add_argument("--stride_ms", type=int, default=10, help="The frame step for spectrogram.")
    parser.add_argument("--is_normalize", type=bool, default=True, help="whether normalize the audio feature.")
    parser.add_argument("--rnn_hidden_size", type=int, default=800, help="The hidden size of RNNs.")
    parser.add_argument("--rnn_hidden_layers", type=int, default=5, help="The num of layers of RNNs.")
    parser.add_argument("--use_bias", type=bool, default=True, help="Whether use bias at the last fc layer.")
    parser.add_argument("--is_bidirectional", type=bool, default=True, help="Whether rnn unit is bidirectional.")
    parser.add_argument("--rnn_type", type=str, default="gru", help="The rnn cell type.")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="The learning rate.")
    parser.add_argument("--seed", type=int, default=1, help="The random seed.")
    parser.add_argument("--batch_size", type=int, default=128, help="The data feed batch size.")
    parser.add_argument("--random_seed", type=int, default=0, help="The random seed to generate data.")
    parser.add_argument("--epochs", type=int, default=100, help="The training epochs.")
    parser.add_argument("--gpu_num", type=int, default=2, help="The num of gpu for training.")
    args = get_args(parser)

    if args["batch_size"] % args["gpu_num"] != 0:
        raise ValueError("The `batch_size` must be `gpu_num` * N")

    print("********** The total config **********")
    for key, value in args.items():
        print("{key}: {value}\n".format(key=key, value=value))
    print("\n\n")

    data_params = get_data_params(args)
    model_params = get_model_params(args)
    model_dir = args["model_dir"]

    # Init data
    train_data = dataset.DeepSpeechDataset(partition="train", label_padding_value=BLANK_INDEX, **data_params)
    eval_data = dataset.DeepSpeechDataset(partition="dev", label_padding_value=BLANK_INDEX, **data_params)

    feature_descriptions = generate_feature_desc(os.path.join(model_dir, "train_data.xml"))
    model = Model(num_classes=train_data.num_classes, n_features=train_data.n_features, **model_params)
    sess = build_session(model.graph)

    train_writer = tf.summary.FileWriter(logdir=os.path.join(model_dir, "train"), graph=sess.graph)
    eval_writer = tf.summary.FileWriter(logdir=os.path.join(model_dir, "test"))
    ckpt_state = tf.train.get_checkpoint_state(model_dir)

    if ckpt_state:
        model.restore(sess, ckpt_state.model_checkpoint_path)
        old_epoch = get_ckpt_global_step(ckpt_state)
    else:
        model.init(sess)
        old_epoch = 0

    epochs = args["epochs"]
    batch_size = args["batch_size"]
    print("Total Epochs: {}. Training batch size: {}".format(epochs, batch_size))

    old_eval_loss = float("inf")
    train_step = 0
    for epoch in range(epochs):

        # 1. Train Stage
        # 1.1. init
        train_loss = 0.
        tic = time.time()
        iteration = 0
        for i in tqdm(range(len(train_data)), desc="Epoch {}/{} Train Stage".format(epoch + 1, epochs)):
            train_feed = train_data[i]
            loss, train_summary = model.train(sess, train_feed)
            train_loss += loss
            train_step += 1
            train_writer.add_summary(train_summary, train_step)
            iteration += 1

        # 1.2 print train log
        toc = time.time()
        train_loss /= iteration
        print("{} stage cost: {:.4f}, time using: {:.2f}".format("Train", train_loss, toc - tic))

        # 2. Eval Stage
        # 2.1. int
        eval_loss, wer, cer = 0., 0., 0.
        tic = time.time()
        eval_step = train_step
        iteration = 0
        for i in tqdm(range(len(eval_data)), desc="Epoch {}/{} Eval Stage".format(epoch + 1, epochs)):
            eval_feed = eval_data[i]
            loss, eval_summary, eval_results, eval_labels = model.eval(sess, eval_feed)
            cer_, wer_ = EditDistance.report(eval_results, eval_labels)
            cer += cer_
            wer += wer_
            eval_loss += loss
            eval_step += 1
            eval_writer.add_summary(eval_summary, eval_step)
            iteration += 1

        # 2.2. print eval log
        toc = time.time()
        eval_loss /= iteration
        cer /= iteration
        print("{} stage cost: {:.4f}, cer: {:.2f}, time using: {:.2f}".format("Eval", eval_loss, cer, toc - tic))
        if eval_loss < old_eval_loss:
            model.saver.save(
                sess, save_path=os.path.join(model_dir, "model_{:4f}.ckpt".format(eval_loss)), global_step=train_step)
            old_eval_loss = eval_loss
        else:
            print("Eval loss not improved, before {:4f}, current {:4f}".format(old_eval_loss, eval_loss))

        train_data.batch_wise_shuffle()

    train_writer.close()
    eval_writer.close()

