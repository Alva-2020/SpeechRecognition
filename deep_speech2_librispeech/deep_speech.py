"""Main entry to train and evaluate DeepSpeech model."""

import os
import tensorflow as tf
import tensorflow.estimator as es
import argparse
import deep_speech2_librispeech.data.dataset as dataset
import deep_speech2_librispeech.model.network as deep_speech_model
import deep_speech2_librispeech.decoder as decoder
import _utils.tensorflow.misc.distribution_utils as distribution_utils
from _utils.tensorflow.utils import get_session_config
from _utils.tensorflow.logs import logger, hooks_helper
from _utils.tensorflow.app import parser_to_flags
from typing import List, Dict


# Default vocabulary file
_VOCAB_FILE = os.path.join(os.path.dirname(__file__), "data/vocabulary.txt")
# Evaluation metrics
_WER_KEY = "WER"
_CER_KEY = "CER"
DATASET_NAME = "LibriSpeech"


def compute_length_after_conv(max_time_steps, ctc_time_steps, input_length):
    """Computes the time_steps/ctc_input_length after convolution."""
    ctc_input_length = tf.cast(tf.multiply(input_length, ctc_time_steps), dtype=tf.float32)
    max_time_steps = tf.cast(max_time_steps, dtype=tf.float32)
    return tf.cast(tf.floordiv(ctc_input_length, max_time_steps), dtype=tf.int32)


def ctc_loss(label_length, ctc_input_length, labels, probs):
    """Computes the ctc loss for the current batch of predictions."""
    label_length = tf.cast(tf.squeeze(label_length), dtype=tf.int32)
    ctc_input_length = tf.cast(tf.squeeze(ctc_input_length), dtype=tf.int32)
    sparse_labels = tf.cast(tf.keras.backend.ctc_label_dense_to_sparse(labels, label_length), dtype=tf.int32)
    y_pred = tf.log(tf.transpose(probs, perm=[1, 0, 2]) + tf.keras.backend.epsilon())
    return tf.expand_dims(tf.nn.ctc_loss(labels=sparse_labels, inputs=y_pred, sequence_length=ctc_input_length), axis=1)


def evaluate_model(estimator: es.Estimator, speech_labels: List[str], entries, input_fn_eval) -> Dict[str, float]:
    """
    Evaluate the model performance using WER anc CER as metrics.

    WER: Word Error Rate
    CER: Character Error Rate

    :param estimator: estimator to evaluate.
    :param speech_labels: a list of string specifying all the character in the vocabulary.
    :param entries: a list of data entries (audio_file, file_size, transcript) for the given dataset.
    :param input_fn_eval: data input function for evaluation.

    :return Evaluation result containing 'wer' and 'cer' as two metrics.
    """
    # Get predictions
    predictions = estimator.predict(input_fn=input_fn_eval)

    # Get probabilities of each predicted class
    probs = [pred["probabilities"] for pred in predictions]
    num_of_examples = len(probs)
    targets = [entry[1] for entry in entries]  # The ground truth transcript

    total_wer, total_cer = 0., 0.
    greedy_decoder = decoder.DeepSpeechDecoder(speech_labels, blank_index=28)
    for prob, target in zip(probs, targets):
        decode = greedy_decoder.decode(prob)
        total_cer += greedy_decoder.cer(decode, target)
        total_wer += greedy_decoder.wer(decode, target)

    total_cer /= num_of_examples
    total_wer /= num_of_examples
    global_step = estimator.get_variable_value(tf.GraphKeys.GLOBAL_STEP)
    eval_results = {
        _WER_KEY: total_wer,
        _CER_KEY: total_cer,
        tf.GraphKeys.GLOBAL_STEP: global_step
    }
    return eval_results


def model_fn(features: Dict, labels, mode, params: Dict):
    """Define model function for deep speech model.

    :param features: a dictionary of input_data features.
                     It includes the data `input_length`, `label_length` and the `spectrogram features`.
    :param labels: a list of labels for the input data.
    :param mode: current estimator mode; should be one of `tf.estimator.ModeKeys.TRAIN`, `EVALUATE`, `PREDICT`.
    :param params: a dict of hyper parameters to be passed to model_fn.

    :return: EstimatorSpec parameterized according to the input params and the current mode.
    """
    global FLAGS
    num_classes = params["num_classes"]
    input_length = features["input_length"]
    label_length = features["label_length"]
    features = features["features"]

    # Create model
    model = deep_speech_model.DeepSpeech2(
        num_rnn_layers=FLAGS.rnn_hidden_layers, rnn_type=FLAGS.rnn_type, is_bidirectional=FLAGS.is_bidirectional,
        rnn_hidden_size=FLAGS.rnn_hidden_size, num_classes=num_classes, use_bias=FLAGS.use_bias)

    # predict mode
    if mode == es.ModeKeys.PREDICT:
        logits = model(features, training=False)
        predictions = {
            "logits": logits,
            "classes": tf.argmax(logits, axis=2),
            "probabilities": tf.nn.softmax(logits)
        }

        return es.EstimatorSpec(mode=mode, predictions=predictions)

    # train / eval mode
    logits = model(features, training=True)
    probs = tf.nn.softmax(logits)
    ctc_input_length = compute_length_after_conv(tf.shape(features)[1], tf.shape(probs)[1], input_length)
    loss = tf.reduce_mean(
        ctc_loss(label_length=label_length, ctc_input_length=ctc_input_length, labels=labels, probs=probs))
    opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    global_step = tf.train.get_or_create_global_step()
    minimize_op = opt.minimize(loss, global_step=global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group(minimize_op, update_ops)

    return es.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


def generate_dataset(data_dir: str, partition: str) -> dataset.DeepSpeechDataset:
    """Generate a speech dataset."""
    global FLAGS
    audio_conf = dataset.AudioConfig(
        sample_rate=FLAGS.sample_rate, window_ms=FLAGS.window_ms, stride_ms=FLAGS.stride_ms, normalize=True)
    data_conf = dataset.DatasetConfig(
        audio_config=audio_conf, data_path=data_dir, vocab_file_path=FLAGS.vocab_file, sortagrad=FLAGS.sortagrad)

    speech_dataset = dataset.DeepSpeechDataset(dataset_config=data_conf, partition=partition, seed=FLAGS.seed)
    return speech_dataset


def run_deep_speech():
    """Run deep speech training and eval loop."""
    global FLAGS
    tf.set_random_seed(FLAGS.seed)
    # Data preprocessing
    tf.logging.info("Data Processing...")
    train_speech_dataset = generate_dataset(FLAGS.data_dir, partition="train")
    eval_speech_dataset = generate_dataset(FLAGS.data_dir, partition="dev")

    # Number of label classes. Label string is "[a-z]' -"
    num_classes = len(train_speech_dataset.speech_labels)

    # not available in 1.4
    distribution_strategy = distribution_utils.get_distribution_strategy(num_gpus=FLAGS.num_gpus)
    run_config = es.RunConfig(train_distribute=distribution_strategy, session_config=get_session_config())

    estimator = es.Estimator(
        model_fn=model_fn, model_dir=FLAGS.model_dir, config=run_config, params={"num_classes": num_classes})

    run_params = {
        "batch_size": FLAGS.batch_size,
        "train_epochs": FLAGS.train_epochs,
        "rnn_hidden_size": FLAGS.rnn_hidden_size,
        "rnn_hidden_layers": FLAGS.rnn_hidden_layers,
        "rnn_type": FLAGS.rnn_type,
        "is_bidirectional": FLAGS.is_bidirectional,
        "use_bias": FLAGS.use_bias
    }

    benchmark_logger = logger.get_benchmark_logger()
    benchmark_logger.log_run_info(
        model_name="deep_speech", dataset_name="LibriSpeech", run_params=run_params, test_id=FLAGS.benchmark_test_id)

    train_hooks = hooks_helper.get_train_hooks(FLAGS.hooks, model_dir=FLAGS.model_dir, batch_size=FLAGS.batch_size)
    per_replica_batch_size = distribution_utils.per_replica_batch_size(FLAGS.batch_size, FLAGS.num_gpus)

    def input_fn_train():
        return train_speech_dataset.input_fn(batch_size=per_replica_batch_size)

    def input_fn_eval():
        return eval_speech_dataset.input_fn(batch_size=per_replica_batch_size)

    # total_training_cycle = FLAGS.train_epochs // FLAGS.epochs_between_evals
    total_training_cycle = FLAGS.train_epochs

    for cycle_index in range(total_training_cycle):
        tf.logging.info(f"Starting train cycle: {cycle_index + 1} / {total_training_cycle}")

        # Perform batch_wise dataset shuffling
        train_speech_dataset.batch_wise_shuffle(FLAGS.batch_size)

        # Train
        estimator.train(input_fn=input_fn_train, hooks=train_hooks)

        # Evaluation
        tf.logging.info("Starting to evaluate...")
        eval_results = evaluate_model(estimator, speech_labels=eval_speech_dataset.speech_labels,
                                      entries=eval_speech_dataset.entries, input_fn_eval=input_fn_eval)

        # Log the WER and CER results.
        benchmark_logger.log_evaluation_result(eval_results)
        tf.logging.info(
            f"Iteration {cycle_index + 1}: WER = {eval_results[_WER_KEY]:.2f}, CER = {eval_results[_CER_KEY]:.2f}")


def main(_):
    with logger.benchmark_context(FLAGS):
        run_deep_speech()


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="The path where labeled data placed.")
    parser.add_argument("--model_dir", type=str, help="The path where model saved.")
    parser.add_argument("--sortagrad", type=bool, default=True, help="Whether to sort input audio by length.")
    parser.add_argument("--sample_rate", type=int, default=16000, help="The sample rate for audio.")
    parser.add_argument("--window_ms", type=int, default=20, help="The frame length for spectrogram.")
    parser.add_argument("--stride_ms", type=int, default=10, help="The frame step for spectrogram.")
    parser.add_argument("--vocab_file", type=str, default=_VOCAB_FILE, help="The path where vocabulary file placed.")
    parser.add_argument("--rnn_hidden_size", type=int, default=800, help="The hidden size of RNNs.")
    parser.add_argument("--rnn_hidden_layers", type=int, default=5, help="The num of layers of RNNs.")
    parser.add_argument("--use_bias", type=bool, default=True, help="Whether use bias at the last fc layer.")
    parser.add_argument("--is_is_bidirectional", type=bool, default=True, help="Whether rnn unit is bidirectional.")
    parser.add_argument("--rnn_type", type=str, default="gru", help="The rnn cell type.")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="The learning rate.")
    parser.add_argument("--seed", type=int, default=1, help="The random seed.")
    parser.add_argument("--batch_size", type=int, default=128, help="The data feed batch size.")
    parser.add_argument("--train_epochs", type=int, default=10, help="The num of train epochs.")
    parser.add_argument("--num_gpus", type=int, default=2, help="The num of gpus to use.")
    parser.add_argument("--hooks", type=str, default="", help="The train hooks.")
    FLAGS = parser_to_flags(parser)

    tf.app.run(main)














