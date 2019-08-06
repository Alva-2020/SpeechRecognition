"""Contains DeepSpeech2 model."""

import tensorflow as tf
from deep_speech2.model_utils.network import DeepSpeech2
from deep_speech2.model_utils.metrics import word_error, char_error
from typing import Union, Optional, Callable, List, Dict


class Model(object):

    def __init__(self, num_classes: int, n_features: int, rnn_hidden_layers: int, rnn_type: str,
                 is_bidirectional: bool, rnn_hidden_size: int, fc_use_bias: bool, learning_rate: float):

        self.acoustic_model = DeepSpeech2(
            num_rnn_layers=rnn_hidden_layers, rnn_type=rnn_type, is_bidirectional=is_bidirectional,
            rnn_hidden_size=rnn_hidden_size, num_classes=num_classes, fc_use_bias=fc_use_bias)

        tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope("Input"):
                self.features = tf.placeholder(dtype=tf.float32, shape=[None, None, n_features, 1], name="features")
                self.input_length = tf.placeholder(dtype=tf.int32, shape=[None, 1], name="input_length")
                self.label_length = tf.placeholder(dtype=tf.int32, shape=[None, 1], name="label_length")
                self.labels = tf.placeholder(dtype=tf.int32, shape=[None, None], name="labels")
                self.is_train = tf.placeholder(dtype=tf.bool, shape=[1], name="train_phase")

            with tf.name_scope("DeepSpeech2"):
                logits = self.acoustic_model(self.features, self.is_train)  # shape: [batch_size, max_time, num_classes]

            with tf.name_scope("Loss"):
                ctc_input_length = self.compute_length_after_conv(
                    max_time_steps=tf.shape(self.features)[1],
                    ctc_time_steps=tf.shape(logits)[1],
                    input_length=self.input_length)

                ctc_input_length = tf.squeeze(ctc_input_length)
                sparse_labels = tf.to_int32(tf.keras.backend.ctc_label_dense_to_sparse(self.labels, self.label_length))
                pred = tf.log(logits + tf.keras.backend.epsilon())

                batch_ctc_loss = tf.nn.ctc_loss(
                    labels=sparse_labels, inputs=pred, sequence_length=ctc_input_length, time_major=False)
                self.loss = tf.reduce_mean(batch_ctc_loss)

            tf.summary.scalar(name="ctc_loss", tensor=self.loss)

            # todo: Use external decode instead.
            with tf.name_scope("decode"):
                # decode: single element list    decode[0]: sparse tensor
                decode, _ = tf.nn.ctc_greedy_decoder(
                    inputs=tf.transpose(logits, perm=[1, 0, 2]), sequence_length=ctc_input_length, merge_repeated=True)
                self.decoded = tf.sparse_tensor_to_dense(decode[0])
                # self.ler = tf.reduce_mean(tf.edit_distance(self.result, self.labels))
                # tf.summary.scalar(name="ler", tensor=self.ler)

            self.classes = tf.argmax(logits, axis=2)
            self.probs = tf.nn.softmax(logits)
            global_step = tf.train.get_or_create_global_step()
            minimize_op = tf.train.AdamOptimizer(learning_rate=learning_rate)\
                .minimize(self.loss, global_step=global_step)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.train_op = tf.group(minimize_op, update_ops)

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver(max_to_keep=5)
            self.merge_summary = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))

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
