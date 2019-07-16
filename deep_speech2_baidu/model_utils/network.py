"""Contains DeepSpeech2 Base layers and networks."""

import tensorflow as tf
from typing import Union, List, Tuple, Optional, Callable


def conv_bn_layer(inputs: tf.Tensor, kernel_size: int, num_channels_in: int, num_channels_out: int,
                  stride: Tuple[int, int], padding: str, bn_activation: Optional[Callable],
                  index_range_data):
    """
    Convolution layer with batch normalization.

    :param inputs: Input layer.
    :param kernel_size: The x dimension of a filter kernel. Or input a tuple for two image dimension.
    :param num_channels_in: Number of input channels.
    :param num_channels_out: Number of output channels.
    :param stride: convolution stride.
    :param padding: The x dimension of the padding. Or input a tuple for two image dimension.
    :param bn_activation: Activation type for `BatchNormalization`.
    :param index_range_data: Index range to indicate sub region.
    :return: Batch norm layer after convolution layer.
    """
    conv_layer = tf.layers.conv2d(
        inputs=input,
        kernel_size=kernel_size,
        filters=num_channels_out,
        strides=stride,
        padding=padding,
        activation=None,
        use_bias=False
    )

    batch_norm = bn_activation(tf.layers.batch_normalization(inputs=conv_layer, momentum=0.9, epsilon=1e-5))

    # todo with index_range_data
    return batch_norm


def bidirectional_simple_rnn_bn_layer(name: str, inputs: tf.Tensor, size: int,
                                      act: Optional[Callable], share_weights: bool):
    """
    Bi-directional simple rnn layer with sequence-wise batch normalization.
    The batch normalization is only performed on input-state weights.

    :param name: Name of the layer.
    :param inputs: Input layer.
    :param size: Number of RNN cells.
    :param act: Activation type.
    :param share_weights: Whether to share input-hidden weights between forward and backward directional RNNs.
    :return: Bidirectional simple rnn layer.
    """
    with tf.variable_scope(name):
        if share_weights:
            input_proj = tf.layers.dense(inputs=input, units=size, activation=None, use_bias=False)
            input_proj_bn = tf.layers.batch_normalization(inputs=input_proj)





