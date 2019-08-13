

"""
Contains DeepSpeech2 model.
Based on model without placeholders which are replaced by `dataset` api
"""

import tensorflow as tf
import _utils.numpy as unp
import _utils.tensorflow as utf
from deep_speech2.model_utils.network import DeepSpeech2
from typing import Union, List, Dict, Optional


class Model(object):

    def __init__(self, num_classes: int, n_features: int, rnn_hidden_layers: int, rnn_type: str,
                 is_bidirectional: bool, rnn_hidden_size: int, fc_use_bias: bool, learning_rate: float,
                 feature_descriptions: Optional[Dict[str, Union[tf.FixedLenFeature, tf.VarLenFeature]]]=None):

        self.acoustic_model = DeepSpeech2(
            num_rnn_layers=rnn_hidden_layers, rnn_type=rnn_type, is_bidirectional=is_bidirectional,
            rnn_hidden_size=rnn_hidden_size, num_classes=num_classes, fc_use_bias=fc_use_bias)
        self.num_classes = num_classes
        self.n_features = n_features
        self.lr = learning_rate
        self._graph = self._build_graph()
        self.data_reader: Optional[utf.record.RecordReader] =\
            utf.record.RecordReader(feature_descriptions) if feature_descriptions else None


    @property
    def graph(self):
        return self._graph

    @property
    def graph_init(self):
        return self._graph_init

    @property
    def saver(self):
        return self._saver

    def init(self, sess: tf.Session):
        sess.run(self.graph_init)

    def restore(self, sess: tf.Session, ckpt_path: str):
        print("Loading model checkpoints from {}  ...".format(ckpt_path))
        self.saver.restore(sess, ckpt_path)
        print("Loading model successfully!")

    def set_data_reader(self, config_file: str):
        self.data_reader = utf.record.RecordReader.from_config(config_file)

    def _build_data(self, input_files: tf.Tensor, batch_size: tf.Tensor) -> tf.data.Iterator:
        data = self.data_reader.read(input_files)
        data = data.shuffle(buffer_size=100)
        data = data.padded_batch(batch_size=batch_size,
                                 padded_shapes={"features": [None, None, 1], "labels": [None]},
                                 padding_values={"features": 0., "labels": self.num_classes - 1})
        iterator = data.make_initializable_iterator()
        return iterator

    def _build_graph(self):
        graph = tf.Graph()
        tf.reset_default_graph()
        with self._graph.as_default():
            with tf.name_scope("Input"):
                self.input_files = tf.placeholder(dtype=tf.string, shape=[None], name="files_path")
                self.batch_size = tf.placeholder(dtype=tf.int64, shape=[1], name="batch_size")  # must be int64
                self.is_train = tf.placeholder(dtype=tf.bool, shape=[1], name="train_phase")

            with tf.name_scope("Read"):
                self.data_iterator = self._build_data(self.input_files, self.batch_size)
                self.data_init = self.data_iterator.initializer
                next_iter = self.data_iterator.get_next()
                self.features, self.input_length, self.label_length, self.labels =\
                    [next_iter[key] for key in ["features", "input_length", "label_length", "labels"]]

                # check input data tensor's attribute
                utf.tensor.validate_tensor(self.features, dtype=tf.float32, shape=[None, None, self.n_features, 1])
                utf.tensor.validate_tensor(self.input_length, dtype=tf.int32, shape=[None, 1])
                utf.tensor.validate_tensor(self.label_length, dtype=tf.int32, shape=[None, 1])
                utf.tensor.validate_tensor(self.labels, dtype=tf.int32, shape=[None, None])

            with tf.name_scope("DeepSpeech2"):
                logits = self.acoustic_model(self.features, self.is_train)  # shape: [batch_size, max_time, num_classes]
                ctc_input_length = self.compute_length_after_conv(
                    max_time_steps=tf.shape(self.features)[1],
                    ctc_time_steps=tf.shape(logits)[1],
                    input_length=self.input_length)
                ctc_input_length = tf.squeeze(ctc_input_length)

            with tf.name_scope("decode"):
                # decode: single element list    decode[0]: sparse tensor
                decode, _ = tf.nn.ctc_greedy_decoder(
                    inputs=tf.transpose(logits, perm=[1, 0, 2]), sequence_length=ctc_input_length,
                    merge_repeated=True)
                self.decoded =\
                    tf.sparse_tensor_to_dense(decode[0], default_value=-1)  # -1 indicates the end of result
                # self.ler = tf.reduce_mean(tf.edit_distance(self.result, self.labels))
                # tf.summary.scalar(name="ler", tensor=self.ler)

            with tf.name_scope("Loss"):
                sparse_labels = tf.to_int32(tf.keras.backend.ctc_label_dense_to_sparse(self.labels, self.label_length))
                pred = tf.log(logits + tf.keras.backend.epsilon())

                batch_ctc_loss = tf.nn.ctc_loss(
                    labels=sparse_labels, inputs=pred, sequence_length=ctc_input_length, time_major=False)
                self.loss = tf.reduce_mean(batch_ctc_loss)

            tf.summary.scalar(name="ctc_loss", tensor=self.loss)

            self.classes = tf.argmax(logits, axis=2)
            self.probs = tf.nn.softmax(logits)
            global_step = tf.train.get_or_create_global_step()
            minimize_op = tf.train.AdamOptimizer(learning_rate=self.lr)\
                .minimize(self.loss, global_step=global_step)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.train_op = tf.group(minimize_op, update_ops)

            self._graph_init = tf.global_variables_initializer()
            self._saver = tf.train.Saver(max_to_keep=5)
            self.merge_summary = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))
        return graph

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

    def stage_init(self, sess: tf.Session, input_files: List[str], batch_size: int):
        sess.run(
            self.data_init,
            feed_dict={self.input_files: input_files, self.batch_size: batch_size})

    def train(self, sess: tf.Session):
        loss, _, summary = sess.run([self.loss, self.train_op, self.merge_summary], feed_dict={self.is_train: True})
        return loss, summary

    def eval(self, sess: tf.Session):
        loss, summary, results, labels, label_length =\
            sess.run([self.loss, self.merge_summary, self.decoded, self.labels, self.label_length],
                     feed_dict={self.is_train: False})
        results = [unp.trim(v, -1, "b").tolist() for v in results]  # drop -1 in tails, method from `_utils`
        labels = [label[:length] for label, length in zip(labels, label_length)]
        return loss, summary, results, labels

    def predict(self, sess: tf.Session):
        results = sess.run(self.decoded, feed_dict={self.is_train: False})
        results = [unp.trim(v, -1, "b").tolist() for v in results]  # drop -1 in tails, method from `_utils`
        return results

