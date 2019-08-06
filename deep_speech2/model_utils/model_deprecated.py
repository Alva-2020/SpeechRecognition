"""Contains DeepSpeech2 model."""

import tensorflow as tf
from deep_speech2.model_utils.network_deprecated import DeepSpeech2
from typing import Union, Optional, Callable, List, Dict


class Model(object):
    """
    A model wrapper for tf.estimator.Estimator
    """
    def __init__(self, num_classes: int, rnn_hidden_layers: int, rnn_type: str,
                 is_bidirectional: bool, rnn_hidden_size: int, fc_use_bias: bool,
                 learning_rate: float):
        self.model = DeepSpeech2(
            num_rnn_layers=rnn_hidden_layers, rnn_type=rnn_type, is_bidirectional=is_bidirectional,
            rnn_hidden_size=rnn_hidden_size, num_classes=num_classes, fc_use_bias=fc_use_bias)
        self.lr = learning_rate

    def __call__(self, features: Dict, labels: List, mode: str) -> tf.estimator.EstimatorSpec:
        """
        Define model function for deep speech model.
        Note: This is a `model_fn` implementation. Its parameters set should not be changed.

        :param features: a dictionary of input_data features.
                         It includes the data `input_length`, `label_length` and the spectrogram `features`.
        :param labels: a list of labels for the input data.
        :param mode: current estimator mode; should be one of `tf.estimator.ModeKeys.TRAIN`, `EVALUATE`, `PREDICT`.
                     which is 'train', 'eval', 'infer' respectively.
        :return: EstimatorSpec.
        """
        input_length = features["input_length"]
        label_length = features["label_length"]
        features = features["features"]
        if mode == tf.estimator.ModeKeys.PREDICT:
            logits = self.model(features, training=False)
            predictions = {
                "classes": tf.argmax(logits, axis=2),
                "probabilities": tf.nn.softmax(logits),
                "logits": logits
            }
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # mode = `train` or `eval`
        # compute ctc loss
        logits = self.model(features, training=True)
        probs = tf.nn.softmax(logits)
        ctc_input_length = self.compute_length_after_conv(
            max_time_steps=tf.shape(features)[1], ctc_time_steps=tf.shape(probs)[1], input_length=input_length)

        loss = tf.reduce_mean(
            self.ctc_loss(label_length=label_length, ctc_input_length=ctc_input_length, labels=labels, logits=logits))

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss)

        # train_op for `train` mode.
        # train_op won't be used under 'eval' mode
        global_step = tf.train.get_or_create_global_step()
        minimize_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss, global_step=global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    @staticmethod
    def compute_length_after_conv(max_time_steps, ctc_time_steps, input_length) -> tf.Tensor:
        """
        Computes the time_steps/ctc_input_length after convolution.

        Suppose that the original feature contains two parts:
        1) Real spectrogram signals, spanning input_length steps.
        2) Padded part with all 0s.
        The total length of those two parts is denoted as max_time_steps, which is the padded length of the current batch.
        After convolution layers, the time steps of a spectrogram feature will be decreased.
        As we know the percentage of its original length within the entire length, we can compute the time steps
        for the signal after convolution as follows (using ctc_input_length to denote):
          `ctc_input_length` = (`input_length` / `max_time_steps`) * `output_length_of_conv`.
        This length is then fed into ctc loss function to compute loss.

        :param max_time_steps: max_time_steps for the batch, after padding.
        :param ctc_time_steps: number of timesteps after convolution.
        :param input_length: actual length of the original spectrogram, without padding.
        :return: the ctc_input_length after convolution layer.
        """
        return tf.to_int32(tf.floordiv(
            tf.to_float(tf.multiply(input_length, ctc_time_steps)), tf.to_float(max_time_steps)))

    @staticmethod
    def ctc_loss(label_length: tf.Tensor, ctc_input_length: tf.Tensor, labels, logits: tf.Tensor):
        """Computes the ctc loss for the current batch of predictions."""
        label_length = tf.to_int32(tf.squeeze(label_length))
        ctc_input_length = tf.to_int32(tf.squeeze(ctc_input_length))
        sparse_labels = tf.to_int32(tf.keras.backend.ctc_label_dense_to_sparse(labels, label_length))
        y_pred = tf.log(tf.transpose(logits, perm=[1, 0, 2]) + tf.keras.backend.epsilon())

        return tf.expand_dims(
            tf.nn.ctc_loss(labels=sparse_labels, inputs=y_pred, sequence_length=ctc_input_length),
            axis=1)

