

"""
Contains DeepSpeech2 model.
Based on model without placeholders which are replaced by `dataset` api
"""

import re
import numpy as np
import tensorflow as tf
import _utils.unumpy as unp
import _utils.utensorflow as utf
from deep_speech2.model_utils.network import DeepSpeech2
from typing import Union, List, Dict, Tuple


class Model(object):
    TOWER_NAME = "DeepSpeech2"

    def __init__(self, num_classes: int, n_features: int, rnn_hidden_layers: int, rnn_type: str,
                 is_bidirectional: bool, rnn_hidden_size: int, fc_use_bias: bool, learning_rate: float,
                 feature_descriptions: Dict[str, Union[tf.FixedLenFeature, tf.VarLenFeature]], gpu_num: int = 1):

        self.acoustic_model = DeepSpeech2(
            num_rnn_layers=rnn_hidden_layers, rnn_type=rnn_type, is_bidirectional=is_bidirectional,
            rnn_hidden_size=rnn_hidden_size, num_classes=num_classes, fc_use_bias=fc_use_bias)
        self.num_classes = num_classes
        self.n_features = n_features
        self.lr = learning_rate
        self.gpu_num = gpu_num
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
        data = data.prefetch(buffer_size=1000)
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
                self.is_train = tf.placeholder(dtype=tf.bool, shape=None, name="train_phase")

            with tf.name_scope("Read"):
                self.data_iterator = self._build_data(self.input_files, self.batch_size)
                self.data_init = self.data_iterator.initializer
                self.features, self.input_length, self.label_length, self.labels =\
                    self.read_input(self.data_iterator.get_next(), split_num=self.gpu_num)

            with tf.name_scope("train"):
                global_step = tf.get_variable("global_step", shape=[], initializer=tf.constant_initializer(0), trainable=False)
                opt = tf.train.AdamOptimizer(learning_rate=self.lr)

            # Train on multi-gpu
            tower_grads = []
            tower_decoded = []
            with tf.variable_scope("Inference", reuse=tf.AUTO_REUSE):
                for i in range(self.gpu_num):
                    with tf.device("/GPU:%d" % i):
                        with tf.name_scope("{tower_name}_{id}".format(tower_name=self.TOWER_NAME, id=i)) as scope:
                            features, input_length, label_length, labels =\
                                [x[i] for x in [self.features, self.input_length, self.label_length, self.labels]]
                            loss = self.tower_loss(
                                scope, features, input_length, label_length, labels,
                                is_train=self.is_train, loss_key="loss")
                            grads = opt.compute_gradients(loss)
                            tower_grads.append(grads)

                            with tf.name_scope("decode"):
                                decoded = self.acoustic_model.decode(features=features, input_length=input_length)
                                tower_decoded.append(decoded)

            self.loss = tf.add_n(tf.get_collection("loss"))
            self.decoded = tower_decoded

            grads = self.average_gradients(tower_grads)
            self.train_op = opt.apply_gradients(grads, global_step=global_step)
            self._graph_init = tf.global_variables_initializer()
            self._saver = tf.train.Saver(max_to_keep=5)
            self.merge_summary = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))
        return graph

    def read_input(self, input_iter, split_num: int):
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

        # slice index in case of `batch_size` can't evenly split by `split_num`
        index = tf.shape(features)[0] - tf.shape(features)[0] % split_num
        features, input_length, label_length, labels =\
            [tf.split(x[: index], split_num, axis=0) for x in [features, input_length, label_length, labels]]

        return features, input_length, label_length, labels

    def tower_loss(self, scope: str, features, input_length, label_length, labels, is_train, loss_key: str="train_loss"):
        """
        Calculate the total loss on a single tower running the CIFAR model.

        :param scope: unique prefix string identifying the tower, e.g. '{tower}_0'
        :param features: Input `features`.
        :param input_length: Input `input_length`
        :param label_length: Input `label_length`
        :param labels: Input `labels`
        :param is_train: The train phase.
        :param loss_key: The collection key to control where value being added.

        :return: Tensor of shape [] containing the total loss for a batch of data
        """

        # Build the portion of the Graph calculating the losses.
        # Note that we will assemble the total_loss using a custom function below.
        loss = self.acoustic_model.ctc_loss(features, input_length, label_length, labels, is_train=is_train)
        tf.add_to_collection(loss_key, loss)

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
        loss, _, summary = sess.run([self.loss, self.train_op, self.merge_summary], feed_dict={self.is_train: True})
        return loss, summary

    def eval(self, sess: tf.Session):
        # results is a list
        loss, summary, results, labels, label_length =\
            sess.run([self.loss, self.merge_summary, self.decoded,
                      tf.concat(self.labels, axis=0), tf.concat(self.label_length, axis=0)],
                     feed_dict={self.is_train: False})
        # drop -1 in tails, method from `_utils`
        results = [unp.trim(v, -1, "b").tolist() for part in results for v in part]
        labels = [label[:length].tolist() for label, length in zip(labels, label_length.reshape(-1,))]
        return loss, summary, results, labels

    def predict(self, sess: tf.Session):
        # results is a list
        results = sess.run(self.decoded, feed_dict={self.is_train: False})
        # drop -1 in tails, method from `_utils`
        results = [unp.trim(v, -1, "b").tolist() for part in results for v in part]
        return results


