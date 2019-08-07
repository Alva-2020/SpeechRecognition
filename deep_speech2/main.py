
"""Trainer for DeepSpeech2 model."""

import os
import sys
import argparse
import time
import dill as pickle
import tensorflow as tf
from deep_speech2.model_utils.model import Model
from deep_speech2.data_utils.data import DataGenerator
from deep_speech2.tools.metrics import EditDistance
from _utils.confighandler import ConfigHandler
from _utils.tensorflow import get_ckpt_global_step
from typing import List, Dict, Any
from tqdm import tqdm


def get_args(arg_parser: argparse.ArgumentParser,
             input_keys: List[str]=sys.argv[1:]) -> Dict[str, Any]:
    """
    A process wrapper for parsed args
    :param arg_parser: An `argparse.ArgumentParser` from argparse module.
    :param input_keys: A list of chars input from command line, often `sys.argv[1:]`
    :return: a simple dict.
    """
    args = arg_parser.parse_args()
    argv = [arg.lstrip("--") for arg in input_keys if arg.startswith("--")]  # get the arg keys from command line.
    given_args = {k: v for k, v in vars(args).items() if k in argv} if argv else {}  # input args from command line.

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
    return dict(
        data_file=args["data_file"], batch_size=args["batch_size"], vocab_file=args["vocab_file"],
        vocab_type=args["vocab_type"], mean_std_file=args["mean_std_file"], stride_ms=args["stride_ms"],
        window_ms=args["window_ms"], max_freq=args["max_freq"], sample_rate=args["sample_rate"],
        specgram_type=args["specgram_type"], use_dB_normalization=args["use_dB_normalization"],
        random_seed=args["random_seed"])


def get_model_params(args: Dict[str, Any]) -> Dict[str, Any]:
    return dict(
        rnn_hidden_layers=args["rnn_hidden_layers"], rnn_type=args["rnn_type"], is_bidirectional=args["is_bidirectional"],
        rnn_hidden_size=args["rnn_hidden_size"], fc_use_bias=args["fc_use_bias"], learning_rate=args["learning_rate"])


def build_session(graph):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 程序最多只能占用指定gpu80%的显存
    config.gpu_options.allow_growth = True  # 程序按需申请内存
    sess = tf.Session(graph=graph, config=config)
    return sess


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
    parser.add_argument("--is_bidirectional", type=bool, default=True, help="If rnn unit is bidirectional")
    parser.add_argument("--fc_use_bias", type=bool, default=False, help="Whether use bias in the last fc layer.")
    parser.add_argument("--model_dir", type=str, help="Where to save the model checkpoints.")
    parser.add_argument("--random_seed", type=int, default=0, help="The random seed to generate data.")
    parser.add_argument("--epochs", type=int, default=100, help="The training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="The batch size of data fed in.")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="The learning rate of network.")
    args = get_args(parser)

    print("********** The total config **********")
    for key, value in args.items():
        print("{key}: {value}\n".format(key=key, value=value))
    print("\n\n")

    input_params = get_data_params(args)
    model_params = get_model_params(args)
    model_dir = args["model_dir"]
    train_data_path = os.path.join(model_dir, "train_data.pickle")
    eval_data_path = os.path.join(model_dir, "eval_data.pickle")

    if os.path.exists(eval_data_path):
        print("Loading Dev data from files %s..." % eval_data_path)
        with open(eval_data_path, "rb") as f:
            eval_data = pickle.load(f)
        print("Loading Dev data successfully.")
    else:
        eval_data = DataGenerator(data_type="dev", keep_transcription_text=False, **input_params)
        print("Saving Dev data into files %s..." % eval_data_path)
        with open(eval_data_path, "wb") as f:
            pickle.dump(eval_data, f, pickle.HIGHEST_PROTOCOL)
        print("Saving Dev data successfully.")

    if os.path.exists(train_data_path):
        print("Loading Train data from files %s..." % train_data_path)
        with open(eval_data_path)
        train_data = pickle.load(train_data_path)
        print("Loading Train data successfully.")
    else:
        train_data = DataGenerator(data_type="train", keep_transcription_text=False, **input_params)
        print("Saving Train data into files %s..." % train_data_path)
        with open(train_data_path, "wb") as f:
            pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
        print("Saving Train data successfully.")

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
    old_eval_loss = float("inf")
    train_step = 0
    for epoch in range(epochs):

        # 1. Train Stage
        # 1.1. init
        train_loss = 0.
        tic = time.time()
        for i in tqdm(range(train_data.n_batches), desc="Epoch {}/{} Train Stage".format(epoch + 1, epochs)):

            loss, train_summary = model.train(
                sess, features=train_data[i].features, labels=train_data[i].labels,
                input_length=train_data[i].input_length, label_length=train_data[i].label_length)

            train_loss += loss
            train_step += 1
            train_writer.add_summary(train_summary, train_step)

        # 1.2 print train log
        toc = time.time()
        train_loss /= train_data.n_batches
        print("{} stage cost: {:.4f}, time using: {:.2f}".format("Train", train_loss, toc - tic))

        # 2. Eval Stage
        # 2.1. int
        eval_loss, cer = 0., 0.
        tic = time.time()
        eval_step = train_step
        for i in tqdm(range(eval_data.n_batches), desc="Epoch {}/{} Eval Stage".format(epoch + 1, epochs)):

            loss, eval_summary, eval_results = model.eval(
                sess, features=eval_data[i].features, labels=eval_data[i].labels,
                input_length=eval_data[i].input_length, label_length=eval_data[i].label_length)

            cer += EditDistance.char_error_rate(eval_results, eval_data[i].labels)
            eval_loss += loss
            eval_step += 1
            eval_writer.add_summary(eval_summary, eval_step)

        # 2.2. print eval log
        toc = time.time()
        eval_loss /= eval_data.n_batches
        cer /= eval_data.n_batches
        print("{} stage cost: {:.4f}, cer: {:.2f}, time using: {:.2f}".format("Eval", eval_loss, cer, toc - tic))
        if eval_loss < old_eval_loss:
            model.saver.save(
                sess, save_path=os.path.join(model_dir, "model_{:4f}.ckpt".format(eval_loss)), global_step=train_step)
            old_eval_loss = eval_loss
        else:
            print("Eval loss not improved, before {:4f}, current {:4f}".format(old_eval_loss, eval_loss))

        train_data.shuffle()

    train_writer.close()
    eval_writer.close()

