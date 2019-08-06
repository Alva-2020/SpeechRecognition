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


def batch_norm(inputs: tf.Tensor, training: bool) -> tf.Tensor:
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
    :param training: a boolean to indicate if it is in training stage.
    :return: tensor output from batch norm layer.
    """
    return tf.layers.batch_normalization(
        inputs=inputs, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, fused=True, training=training
    )


def _conv_bn_layer(inputs: tf.Tensor, padding: Union[Tuple, List], filters: int,
                   kernel_size: Tuple, strides: Tuple, layer_id: int, training: bool) -> tf.Tensor:
    """
    Defines 2D convolutional + batch normalization layer.

    :param inputs: input data for convolution layer.
    :param padding: padding to be applied before convolution layer.
    :param filters: an integer, number of output filters in the convolution.
    :param kernel_size: a tuple specifying the height and width of the 2D convolution window.
    :param strides: a tuple specifying the stride length of the convolution.
    :param layer_id: an integer specifying the layer index.
    :param training: a boolean to indicate which stage we are in (training/eval).
    :return: tensor output from the current layer.
    """
    inputs = tf.pad(inputs, [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]])
    y = tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding="valid",
        use_bias=False, activation=tf.nn.relu6, name="cnn_{}".format(layer_id))
    return batch_norm(y, training=training)


def _rnn_layer(inputs: tf.Tensor, rnn_cell: tf.nn.rnn_cell.RNNCell, rnn_hidden_size: int, layer_id: int,
               is_batch_norm: bool, is_bidirectional: bool, training: bool) -> tf.Tensor:
    """
    Defines a batch normalization + rnn layer.

    :param inputs: input tensors for the current layer.
    :param rnn_cell: RNN cell instance to use.
    :param rnn_hidden_size: an integer for the dimensionality of the rnn output space.
    :param layer_id: an integer for the index of current layer.
    :param is_batch_norm: a boolean specifying whether to perform batch normalization on input states.
    :param is_bidirectional: a boolean specifying whether the rnn layer is bi-directional.
    :param training: a boolean to indicate which stage we are in (training/eval).
    :return: tensor output for the current layer.
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
    """
    Initialize DeepSpeech2 model.

    :param num_rnn_layers: an integer, the number of rnn layers. By default, it's 5.
    :param rnn_type: a string, one of the supported rnn cells: gru, rnn and lstm.
    :param is_bidirectional: a boolean to indicate if the rnn layer is bidirectional.
    :param rnn_hidden_size: an integer for the number of hidden states in each unit.
    :param num_classes: an integer, the number of output classes/labels.
    :param fc_use_bias: a boolean specifying whether to use bias in the last fc layer.
    """
    def __init__(self, num_rnn_layers: int, rnn_type: str, is_bidirectional: bool,
                 rnn_hidden_size: int, num_classes: int, fc_use_bias: bool):
        if rnn_type not in SUPPORTED_RNNS:
            raise ValueError("Invalid rnn type %s. Possible choices are %s." % (rnn_type, str(SUPPORTED_RNNS.keys())))
        self.num_rnn_layers = num_rnn_layers
        self.rnn_cell = SUPPORTED_RNNS[rnn_type.lower()]
        self.is_bidirectional = is_bidirectional
        self.rnn_hidden_size = rnn_hidden_size
        self.num_classes = num_classes
        self.fc_use_bias = fc_use_bias

    def __call__(self, inputs: tf.Tensor, training: bool):
        # 1. Two CNN layers
        inputs = _conv_bn_layer(
            inputs, padding=(20, 5), filters=_CONV_FILTERS, kernel_size=(41, 11),
            strides=(2, 2), layer_id=1, training=training)

        inputs = _conv_bn_layer(
            inputs, padding=(10, 5), filters=_CONV_FILTERS, kernel_size=(21, 11),
            strides=(2, 1), layer_id=2, training=training)

        # output of conv_layer2 is of the shape [batch_size (N), times (T), features (F), channels (C)].
        batch_size = tf.shape(inputs)[0]
        feat_size = inputs.get_shape().as_list()[2]
        inputs = tf.reshape(inputs, shape=[batch_size, -1, feat_size * _CONV_FILTERS])

        # 2. RNN layers:
        for layer_counter in range(self.num_rnn_layers):
            is_batch_norm = (layer_counter != 0)  # No batch normalization on the first layer.
            layer_id = layer_counter + 1
            inputs = _rnn_layer(
                inputs=inputs, rnn_cell=self.rnn_cell, rnn_hidden_size=self.rnn_hidden_size, layer_id=layer_id,
                is_batch_norm=is_batch_norm, is_bidirectional=self.is_bidirectional, training=training)

        # 3. FC Layer with batch norm
        inputs = batch_norm(inputs, training)
        logits = tf.layers.dense(inputs, units=self.num_classes, use_bias=self.fc_use_bias)

        return logits
