
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from typing import Optional, List

def normalize(inputs: tf.Tensor, epsilon:float=1e-8, scope: str="ln", reuse: Optional[bool]=None):
    """ Normalize tensor
    :param inputs: Tensor with 2 or more dims where first dim is `batch_size`
    :param epsilon: small floating number to prevent ZeroDivision
    :param scope: Optional scope name
    :param reuse: whether to reuse the weights of a previous layer by the same time
    :return: A tensor with same shape as inputs
    """
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()  # for example [batch, m, n]
        params_shape = inputs_shape[-1:]  # (n, )
        mean, variance = tf.nn.moments(inputs, axes=[-1], keep_dims=True)  # [batch, m, 1]

        beta = tf.Variable(tf.zeros(shape=params_shape, dtype=tf.float32))  # 选最后一个维度方便 broadcast  [n, ]
        gamma = tf.Variable(tf.ones(shape=params_shape, dtype=tf.float32))  # [n, ]
        normalized = (inputs - mean) / ((variance + epsilon) ** 0.5)
    return gamma * normalized + beta


def embedding(inputs: tf.Tensor, vocab_size: int, emb_size: int, zero_pad: bool=True,
              scale: bool=True, scope: str="embedding", reuse: Optional[bool]=None):
    """ Embeds a given tensor
    :param inputs: a tensor with `int32` or `int64` dtype
    :param vocab_size:  Vocabulary size e.g. the num of rows of `lookup_table`
    :param emb_size: Embedding size e.g. the num of columns of `lookup_table`
    :param zero_pad: if true, the first row (e.g. id = 0) is all constant 0
    :param scale: if true, outputs will be multiplied by (emb_size ** 0.5)
    :param scope: Optional scope name
    :param reuse: whether to reuse the weights of a previous layer by the same time
    :return: A tensor with one more rank than `inputs`, the last rank is emb_size
    """
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable("lookup_table", dtype=tf.float32, shape=[vocab_size, emb_size], initializer=xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat([tf.zeros(shape=[1, emb_size], dtype=tf.float32), lookup_table[1:, :]], axis=0)
        outputs = tf.nn.embedding_lookup(params=lookup_table, ids=inputs)
        if scale:
            outputs *= emb_size ** 0.5
    return outputs


def feed_forward(inputs: tf.Tensor, num_units: List[int]=[2048, 512],
                 scope: str="feed_forward", reuse: Optional[bool]=None):
    """ Point-wise feed forward net
    :param inputs: A 3d tensor with shape of [N, T, C]
    :param num_units: A list of two int
    :param scope: Optional scope name
    :param reuse: whether to reuse the weights of a previous layer by the same time
    :return: A 3d tensor with the same shape and dtype as inputs
    """
    with tf.variable_scope(scope, reuse=reuse):
        # [N, T, 2048]
        outputs = tf.layers.conv1d(inputs=inputs, filters=num_units[0], kernel_size=1, activation=tf.nn.relu, use_bias=True)
        # [N, T, 512]
        outputs = tf.layers.conv1d(inputs=outputs, filters=num_units[1], kernel_size=1, activation=None, use_bias=True)
        outputs += inputs  # Residual connection
    return normalize(outputs)


def label_smoothing(inputs: tf.Tensor, epsilon: float=0.1):
    """ Apply label smoothing
    :param inputs: A 3d tensor with shape [N, T, V]
    :param epsilon: smoothing rate
    :return: smoothed tensor with same shape as inputs
    """
    V = inputs.get_shape().as_list()[-1]  # number of last dim (channels)
    return (1 - epsilon) * inputs + epsilon / V


def multihead_attention(emb: tf.Tensor, queries: tf.Tensor, keys: tf.Tensor, num_units: Optional[int]=None,
                        num_heads: int=8, dropout_rate: float=0., is_training: bool=True, causality: bool=False,
                        scope: str="multihead_attention", reuse: Optional[bool]=None):
    """
    
    :param emb: A 3d tensor with shape [N, T_k, C_k], embeded tensor from a `lookup_table`
    :param queries: A 3d tensor with shape [N, T_q, C_q]
    :param keys: A 3d tensor with shape [N, T_k, C_k]
    :param num_units: Attention size -> C
    :param num_heads: Num of heads
    :param dropout_rate: dropout rate
    :param is_training: controler for dropout
    :param causality: If true, units that reference the future are masked
    :param scope: Optional scope name
    :param reuse: whether to reuse the weights of a previous layer by the same time
    :return: A 3d tensor with shape [N, T_q, C]
    """
    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]

        # Linear projections
        Q = tf.layers.dense(inputs=queries, units=num_units, activation=tf.nn.relu)  # [N, T_q, C]
        K = tf.layers.dense(inputs=keys, units=num_units, activation=tf.nn.relu)  # [N, T_k, C]
        V = tf.layers.dense(inputs=keys, units=num_units, activation=tf.nn.relu)  # [N, T_k, C]

        # Split and concat
        Q_ = tf.concat(tf.split(value=Q, num_or_size_splits=num_heads, axis=2), axis=0)  # [h * N, T_q, C / h]
        K_ = tf.concat(tf.split(value=K, num_or_size_splits=num_heads, axis=2), axis=0)  # [h * N, T_k, C / h]
        V_ = tf.concat(tf.split(value=V, num_or_size_splits=num_heads, axis=2), axis=0)  # [h * N, T_k, C / h]

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, perm=(0, 2, 1)))  # [h * N, T_q, T_k]

        # Scale
        outputs *= (num_units / num_heads) ** 0.5

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # [T_q, T_k]
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # 取下三角  [T_q, T_k]
            tril = tf.expand_dims(tril, axis=0)  # [1, T_q, T_k]
            key_masks = tf.tile(tril, multiples=[tf.shape(outputs)[0], 1, 1])  # [h * N, T_q, T_k]
        else:
            # Key masking
            key_masks = tf.sign(tf.abs(tf.reduce_sum(emb, axis=-1)))  # [N, T_k]
            key_masks = tf.expand_dims(tf.tile(key_masks, multiples=[num_heads, 1]), axis=1)  # [h * N, 1, T_k]
            key_masks = tf.tile(key_masks, multiples=[1, tf.shape(queries)[1], 1])  # [h * N, T_q, T_k]

        paddings = tf.ones_like(outputs) ** (-2 ** 32 + 1)  # [h * N, T_q, T_k]

        # Take elements based on condition, take x if true else y
        outputs = tf.where(condition=tf.equal(key_masks, 0), x=paddings, y=outputs)  # [h * N, T_q, T_k]

        # Activation
        outputs = tf.nn.softmax(outputs)  # [h * N, T_q, T_k]































class LanguageModel(object):

    def __init__(self, is_training: bool=True, hidden_units: int=512, learning_rate: float=3e-4):


