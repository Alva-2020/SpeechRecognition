

"""
Contains DeepSpeech2 model.
Based on model without placeholders which are replaced by `dataset` api
"""

import re
import numpy as np
import tensorflow as tf
import _utils.numpy as unp
import _utils.tensorflow as utf
from deep_speech2.model_utils.network import DeepSpeech2
from typing import Union, List, Dict, Tuple


class Model(object):
    TOWER_NAME = "DeepSpeech2"

    def __init__(self, num_classes: int, n_features: int, rnn_hidden_layers: int, rnn_type: str,
                 is_bidirectional: bool, rnn_hidden_size: int, fc_use_bias: bool, learning_rate: float,
                 feature_descriptions: Dict[str, Union[tf.FixedLenFeature, tf.VarLenFeature]], num_gpu: int=1):

        self.acoustic_model = DeepSpeech2(
            num_rnn_layers=rnn_hidden_layers, rnn_type=rnn_type, is_bidirectional=is_bidirectional,
            rnn_hidden_size=rnn_hidden_size, num_classes=num_classes, fc_use_bias=fc_use_bias)
        self.num_classes = num_classes
        self.n_features = n_features
        self.lr = learning_rate
        self.num_gpu = num_gpu
        self.data_reader: utf.record.RecordReader = utf.record.RecordReader(feature_descriptions)
        self.graph = self._build_graph()

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

    def _build_data(self, input_files: tf.Tensor, batch_size: tf.Tensor) -> tf.data.Iterator:
        print(self.data_reader.feature_description)
        data = self.data_reader.read(input_files)
        data = data.shuffle(buffer_size=100)
        data = data.padded_batch(
            batch_size=batch_size,
            padded_shapes={
                "features": [None, self.n_features, 1],
                "labels": [None],
                "true_length": [1],
                "label_length": [1]},
            padding_values={
                "features": np.float32(0),
                "labels": np.int64(self.num_classes - 1),  # padded with blank index
                "true_length": np.int64(0),
                "label_length": np.int64(0)})
        iterator = data.make_initializable_iterator()
        return iterator

    def _build_graph(self):
        graph = tf.Graph()
        tf.reset_default_graph()
        with graph.as_default(), tf.device("/CPU:0"):
            with tf.name_scope("Input"):
                self.input_files = tf.placeholder(dtype=tf.string, shape=[None], name="files_path")
                self.batch_size = tf.placeholder(dtype=tf.int64, shape=None, name="batch_size")  # must be int64

            with tf.name_scope("Read"):
                self.data_iterator = self._build_data(self.input_files, self.batch_size)
                self.data_init = self.data_iterator.initializer

            with tf.name_scope("train"):
                global_step = tf.get_variable("global_step", shape=[], initializer=tf.constant_initializer(0), trainable=False)
                opt = tf.train.AdamOptimizer(learning_rate=self.lr)

            # Train on multi-gpu
            tower_grads = []
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(self.num_gpu):
                    with tf.device("/GPU:%d" % i):
                        with tf.name_scope("{tower_name}_{id}".format(tower_name=self.TOWER_NAME, id=i)) as scope:
                            input_iter = self.data_iterator.get_next()
                            loss = self.tower_loss(scope, input_iter, is_train=True, loss_key="train_loss")
                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()
                            grads = opt.compute_gradients(loss)
                            tower_grads.append(grads)

            self.train_loss = tf.add_n(tf.get_collection("train_loss"))

            with tf.name_scope("decode"):
                eval_input_iter = self.data_iterator.get_next()
                self.eval_features, self.eval_input_length, self.eval_label_length, self.eval_labels =\
                    self.read_input(eval_input_iter)
                self.eval_loss = self.acoustic_model.ctc_loss(
                    features=self.eval_features, input_length=self.eval_input_length,
                    label_length=self.eval_label_length, labels=self.eval_labels, is_train=False, loss_key="eval_loss")

                self.decoded = self.acoustic_model.decode(
                    features=self.eval_features, input_length=self.eval_input_length)

            grads = self.average_gradients(tower_grads)
            self.train_op = opt.apply_gradients(grads, global_step=global_step)
            self._graph_init = tf.global_variables_initializer()
            self._saver = tf.train.Saver(max_to_keep=5)
            self.merge_summary = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))
        return graph

    def read_input(self, input_iter):
        features, input_length, label_length, labels = \
            [input_iter[key] for key in ["features", "true_length", "label_length", "labels"]]

        features = tf.cast(features, dtype=tf.float32)
        input_length = tf.cast(input_length, dtype=tf.int32)
        label_length = tf.cast(label_length, dtype=tf.int32)
        labels = tf.cast(labels, dtype=tf.int32)

        # check input data tensor's attribute
        utf.tensor.validate_tensor(features, dtype=tf.float32, shape=[None, None, self.n_features, 1])
        utf.tensor.validate_tensor(input_length, dtype=tf.int32, shape=[None, 1])
        utf.tensor.validate_tensor(label_length, dtype=tf.int32, shape=[None, 1])
        utf.tensor.validate_tensor(labels, dtype=tf.int32, shape=[None, None])

        return features, input_length, label_length, labels

    def tower_loss(self, scope: str, input_iter, is_train, loss_key: str="train_loss"):
        """
        Calculate the total loss on a single tower running the CIFAR model.

        :param scope: unique prefix string identifying the tower, e.g. '{tower}_0'
        :param input_iter: The input tensor mapping, e.g. {"features": ..., "true_length": ..., "label_length": ..., "labels": ...}
        :param is_train: The train phase.
        :param loss_key: The collection key to control where value being added.

        :return: Tensor of shape [] containing the total loss for a batch of data
        """
        features, input_length, label_length, labels = self.read_input(input_iter)

        # Build the portion of the Graph calculating the losses.
        # Note that we will assemble the total_loss using a custom function below.
        _ = self.acoustic_model.ctc_loss(
            features, input_length, label_length, labels, is_train=is_train, loss_key=loss_key)

        # Assemble all of the losses for the current tower only.
        losses = tf.get_collection(loss_key, scope)

        # Calculate the total loss for the current tower.
        total_loss = tf.add_n(losses)

        # Attach a scalar summary to all individual losses and the total loss;
        # do the same for the averaged version of the losses.
        for l in losses + [total_loss]:
            # Remove '{tower}_[0-9]/' from the name in case this is a multi-GPU training
            # session. This helps the clarity of presentation on tensorboard.
            loss_name = re.sub('%s_' % self.TOWER_NAME, '', l.op.name)
            tf.summary.scalar(loss_name, l)
        return total_loss

    @staticmethod
    def average_gradients(tower_grads: List[List[Tuple]]):
        """
        Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        :param tower_grads: List
        :return: List of pairs of (gradient, variable) where the gradient has been averaged across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            # ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)

            grad = tf.concat(grads, axis=0)
            grad = tf.reduce_mean(grad, axis=0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to the Variable.
            v = grad_and_vars[0][1]
            average_grads.append((grad, v))
        return average_grads

    def stage_init(self, sess: tf.Session, input_files: List[str], batch_size: int):
        sess.run(
            self.data_init,
            feed_dict={self.input_files: input_files, self.batch_size: batch_size})

    def train(self, sess: tf.Session):
        loss, _, summary = sess.run([self.train_loss, self.train_op, self.merge_summary])
        return loss, summary

    def eval(self, sess: tf.Session):
        loss, summary, results, labels, label_length =\
            sess.run([self.eval_loss, self.merge_summary, self.decoded, self.eval_labels, self.eval_label_length])
        results = [unp.trim(v, -1, "b").tolist() for v in results]  # drop -1 in tails, method from `_utils`
        labels = [label[:length].tolist() for label, length in zip(labels, label_length.reshape(-1,))]
        return loss, summary, results, labels


