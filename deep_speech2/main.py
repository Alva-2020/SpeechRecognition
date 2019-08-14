
"""Trainer for DeepSpeech2 model."""

import os
import sys
import argparse
import time
import math
import tensorflow as tf
from deep_speech2.model_utils.model import Model
from deep_speech2.data_utils.data import DataGenerator
from deep_speech2.tools.metrics import EditDistance
from _utils.confighandler import ConfigHandler
from _utils.tensorflow.utils import get_ckpt_global_step
from _utils.tensorflow.record import generate_feature_desc
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
        data_file=args["data_file"], vocab_file=args["vocab_file"], vocab_type=args["vocab_type"],
        mean_std_file=args["mean_std_file"], stride_ms=args["stride_ms"], window_ms=args["window_ms"],
        max_freq=args["max_freq"], sample_rate=args["sample_rate"], specgram_type=args["specgram_type"],
        use_dB_normalization=args["use_dB_normalization"], random_seed=args["random_seed"])


def get_model_params(args: Dict[str, Any]) -> Dict[str, Any]:
    return dict(
        rnn_hidden_layers=args["rnn_hidden_layers"], rnn_type=args["rnn_type"], is_bidirectional=args["is_bidirectional"],
        rnn_hidden_size=args["rnn_hidden_size"], fc_use_bias=args["fc_use_bias"], learning_rate=args["learning_rate"])


def build_session(graph):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    return tf.Session(graph=graph, config=config)


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

    # Save data
    train_data = DataGenerator(partition="train", keep_transcription_text=False, **input_params)
    eval_data = DataGenerator(partition="dev", keep_transcription_text=False, **input_params)

    data_path_mapping = {
        "train": [os.path.join(model_dir, "train_data.record"), train_data],
        "eval": [os.path.join(model_dir, "eval_data.record"), eval_data],
    }

    for partition, (file, data) in data_path_mapping.items():
        if not os.path.exists(file):
            print("Record of {} not exists, start rebuilding.".format(partition))
            data.write_to_record(file)
        else:
            print("Record of {} already exists, will loading...".format(partition))
            print("Total num of samples for {partition}: {size}".format(partition=partition, size=len(data)))

    feature_descriptions = generate_feature_desc(os.path.join(model_dir, "train_data.xml"))
    model = Model(num_classes=train_data.num_classes, n_features=train_data.n_features,
                  feature_descriptions=feature_descriptions, **model_params)
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
        n_train_batches = int(math.ceil(len(train_data) / batch_size))
        model.stage_init(sess, [data_path_mapping["train"][0]], batch_size)
        iteration = 0
        for i in tqdm(range(n_train_batches), desc="Epoch {}/{} Train Stage".format(epoch + 1, epochs)):
            try:
                loss, train_summary = model.train(sess)
                train_loss += loss
                train_step += 1
                train_writer.add_summary(train_summary, train_step)
            except tf.errors.OutOfRangeError:
                break
            iteration += 1

        # 1.2 print train log
        toc = time.time()
        train_loss /= iteration
        print("{} stage cost: {:.4f}, time using: {:.2f}".format("Train", train_loss, toc - tic))

        # 2. Eval Stage
        # 2.1. int
        eval_loss, cer = 0., 0.
        tic = time.time()
        n_eval_batches = int(math.ceil(len(eval_data) / batch_size))
        eval_step = train_step
        model.stage_init(sess, [data_path_mapping["eval"][0]], batch_size)
        iteration = 0
        for i in tqdm(range(n_eval_batches), desc="Epoch {}/{} Eval Stage".format(epoch + 1, epochs)):
            try:
                loss, eval_summary, eval_results, eval_labels = model.eval(sess)
                cer += EditDistance.char_error_rate(eval_results, eval_labels)
                eval_loss += loss
                eval_step += 1
                eval_writer.add_summary(eval_summary, eval_step)
            except tf.errors.OutOfRangeError:
                break
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

    train_writer.close()
    eval_writer.close()

