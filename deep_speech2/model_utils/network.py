"""Contains DeepSpeech2 Base layers and networks.
Based on https://github.com/tensorflow/models/blob/master/research/deep_speech/deep_speech_model.py
"""

import tensorflow as tf
from typing import Union, List, Tuple


# Supported rnn cells.
SUPPORTED_RNNS = {
    "lstm": tf.nn.rnn_cell.BasicLSTMCell,
    "rnn": tf.nn.rnn_cell.RNNCell,
    "gru": tf.nn.rnn_cell.GRUCell,
}

# Parameters for batch normalization.
_BATCH_NORM_EPSILON = 1e-5
_BATCH_NORM_DECAY = 0.997

# Filters of convolution layer
_CONV_FILTERS = 32


def batch_norm(inputs: tf.Tensor, training: Union[bool, tf.Tensor]) -> tf.Tensor:
    """
    Batch normalization layer.
    Note that the momentum to use will affect validation accuracy over time.
    Batch norm has different behaviors during training/evaluation. With a large
    momentum, the model takes longer to get a near-accurate estimation of the
    moving mean/variance over the entire training dataset, which means we need
    more iterations to see good evaluation results. If the training data is evenly
    distributed over the feature space, we can also try setting a smaller momentum
    (such as 0.1) to get good evaluation result sooner.

    :param inputs: input data for batch norm layer.
    :param training: a boolean or tensor to indicate if it is in training stage.
    :return: tensor output from batch norm layer.
    """
    return tf.layers.batch_normalization(
        inputs=inputs, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, fused=True, training=training
    )


def _conv_bn_layer(inputs: tf.Tensor, padding: Union[Tuple, List], filters: int,
                   kernel_size: Tuple, strides: Tuple, layer_id: int, training: Union[bool, tf.Tensor]) -> tf.Tensor:
    """
    Defines 2D convolutional + batch normalization layer.

    :param inputs: input data for convolution layer.
    :param padding: padding to be applied before convolution layer.
    :param filters: an integer, number of output filters in the convolution.
    :param kernel_size: a tuple specifying the height and width of the 2D convolution window.
    :param strides: a tuple specifying the stride length of the convolution.
    :param layer_id: an integer specifying the layer index.
    :param training: a boolean or tensor to indicate which stage we are in (training/eval).
    :return: tensor output from the current layer.
    """
    inputs = tf.pad(inputs, [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]])
    y = tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding="valid",
        use_bias=False, activation=tf.nn.relu6, name="cnn_{}".format(layer_id))
    return batch_norm(y, training=training)


def _rnn_layer(inputs: tf.Tensor, rnn_cell: tf.nn.rnn_cell.RNNCell, rnn_hidden_size: int, layer_id: int,
               is_batch_norm: bool, is_bidirectional: bool, training: Union[bool, tf.Tensor]) -> tf.Tensor:
    """
    Defines a batch normalization + rnn layer.

    :param inputs: input tensors for the current layer.
    :param rnn_cell: RNN cell instance to use.
    :param rnn_hidden_size: an integer for the dimensionality of the rnn output space.
    :param layer_id: an integer for the index of current layer.
    :param is_batch_norm: a boolean specifying whether to perform batch normalization on input states.
    :param is_bidirectional: a boolean specifying whether the rnn layer is bi-directional.
    :param training: a boolean to indicate which stage we are in (training/eval).
    :return: tensor output for the current layer.  `[batch_size, max_time, cell.output_size]`
    """
    if is_batch_norm:
        inputs = batch_norm(inputs, training)

    # Construct forward/backward RNN cells.
    fw_cell = rnn_cell(num_units=rnn_hidden_size, name="rnn_fw_{}".format(layer_id))
    if is_bidirectional:
        bw_cell = rnn_cell(num_units=rnn_hidden_size, name="rnn_bw_{}".format(layer_id))
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=fw_cell, cell_bw=bw_cell, inputs=inputs, dtype=tf.float32, swap_memory=True)
        rnn_outputs = tf.concat(outputs, axis=-1)
    else:
        rnn_outputs = tf.nn.dynamic_rnn(cell=fw_cell, inputs=inputs, dtype=tf.float32, swap_memory=True)
    return rnn_outputs


class DeepSpeech2(object):

    def __init__(self, num_rnn_layers: int, rnn_type: str, is_bidirectional: bool,
                 rnn_hidden_size: int, num_classes: int, fc_use_bias: bool):
        """
        Initialize DeepSpeech2 model.

        :param num_rnn_layers: an integer, the number of rnn layers. By default, it's 5.
        :param rnn_type: a string, one of the supported rnn cells: gru, rnn and lstm.
        :param is_bidirectional: a boolean to indicate if the rnn layer is bidirectional.
        :param rnn_hidden_size: an integer for the number of hidden states in each unit.
        :param num_classes: an integer, the number of output classes/labels.
        :param fc_use_bias: a boolean specifying whether to use bias in the last fc layer.
        """
        if rnn_type not in SUPPORTED_RNNS:
            raise ValueError("Invalid rnn type %s. Possible choices are %s." % (rnn_type, str(SUPPORTED_RNNS.keys())))
        self.num_rnn_layers = num_rnn_layers
        self.rnn_cell = SUPPORTED_RNNS[rnn_type.lower()]
        self.is_bidirectional = is_bidirectional
        self.rnn_hidden_size = rnn_hidden_size
        self.num_classes = num_classes
        self.fc_use_bias = fc_use_bias
        self.decode_pad_value = -1

    def inference(self, inputs: tf.Tensor, training: Union[bool, tf.Tensor]):
        # 1. Two CNN layers
        with tf.variable_scope("cnn", reuse=tf.AUTO_REUSE):
            inputs = _conv_bn_layer(
                inputs, padding=(20, 5), filters=_CONV_FILTERS, kernel_size=(41, 11),
                strides=(2, 2), layer_id=1, training=training)

            inputs = _conv_bn_layer(
                inputs, padding=(10, 5), filters=_CONV_FILTERS, kernel_size=(21, 11),
                strides=(2, 1), layer_id=2, training=training)

        with tf.variable_scope("reshape", reuse=tf.AUTO_REUSE):
            # output of conv_layer2 is of the shape [batch_size (N), times (T), features (F), channels (C)].
            batch_size = tf.shape(inputs)[0]
            feat_size = inputs.get_shape().as_list()[2]
            inputs = tf.reshape(inputs, shape=[batch_size, -1, feat_size * _CONV_FILTERS])

        with tf.variable_scope("rnn", reuse=tf.AUTO_REUSE):
            # 2. RNN layers:
            for layer_counter in range(self.num_rnn_layers):
                is_batch_norm = (layer_counter != 0)  # No batch normalization on the first layer.
                layer_id = layer_counter + 1
                inputs = _rnn_layer(
                    inputs=inputs, rnn_cell=self.rnn_cell, rnn_hidden_size=self.rnn_hidden_size, layer_id=layer_id,
                    is_batch_norm=is_batch_norm, is_bidirectional=self.is_bidirectional, training=training)

        with tf.variable_scope("fc", reuse=tf.AUTO_REUSE):
            # 3. FC Layer with batch norm
            inputs = batch_norm(inputs, training)
            # shape: [batch_size, max_time, num_classes]
            logits = tf.layers.dense(inputs, units=self.num_classes, use_bias=self.fc_use_bias)

        return logits

    @staticmethod
    def _compute_length_after_conv(max_time_steps, ctc_time_steps, input_length) -> tf.Tensor:
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
        return tf.cast(tf.floordiv(tf.cast(tf.multiply(input_length, ctc_time_steps), dtype=tf.float32),
                                   tf.cast(max_time_steps, dtype=tf.float32)),
                       dtype=tf.int32)

    @staticmethod
    def _ctc_loss(label_length, ctc_input_length, labels, logits) -> tf.Tensor:
        """Compute the ctc loss for current batch of predictions"""
        ctc_input_length = tf.cast(tf.squeeze(ctc_input_length), dtype=tf.int32)
        label_length = tf.cast(tf.squeeze(label_length), dtype=tf.int32)
        probs = tf.nn.softmax(logits)

        sparse_labels = tf.cast(
            tf.keras.backend.ctc_label_dense_to_sparse(labels, label_length), dtype=tf.int32)
        y_pred = tf.log(tf.transpose(probs, perm=[1, 0, 2]) + tf.keras.backend.epsilon())
        losses = tf.nn.ctc_loss(labels=sparse_labels, inputs=y_pred, sequence_length=ctc_input_length)
        return tf.reduce_mean(losses)

    def ctc_loss(self, features, input_length, label_length, labels, is_train: Union[bool, tf.Tensor]):
        """Compute the ctc loss for current batch of predictions by original input"""
        logits = self.inference(inputs=features, training=is_train)
        ctc_input_length = self._compute_length_after_conv(
            max_time_steps=tf.shape(features)[1],
            ctc_time_steps=tf.shape(logits)[1],
            input_length=input_length)
        loss = self._ctc_loss(label_length=label_length, ctc_input_length=ctc_input_length, labels=labels, logits=logits)
        return loss

    def decode(self, features: tf.Tensor, input_length: tf.Tensor) -> tf.Tensor:
        """Get the ctc decoded labels"""
        logits = self.inference(inputs=features, training=False)
        ctc_input_length = self._compute_length_after_conv(
            max_time_steps=tf.shape(features)[1],
            ctc_time_steps=tf.shape(logits)[1],
            input_length=input_length)
        ctc_input_length = tf.cast(tf.squeeze(ctc_input_length), dtype=tf.int32)
        decode_, _ = tf.nn.ctc_greedy_decoder(
            inputs=tf.transpose(logits, perm=[1, 0, 2]), sequence_length=ctc_input_length,
            merge_repeated=True)

        # -1 indicates the end of result
        return tf.sparse_tensor_to_dense(decode_[0], default_value=self.decode_pad_value)



