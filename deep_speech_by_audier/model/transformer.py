
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.training import HParams
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

        # Key masking
        # Causality = future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # [T_q, T_k]
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # 取下三角  [T_q, T_k]
            tril = tf.expand_dims(tril, axis=0)  # [1, T_q, T_k]
            key_masks = tf.tile(tril, multiples=[tf.shape(outputs)[0], 1, 1])  # [h * N, T_q, T_k]
        else:
            key_masks = tf.sign(tf.abs(tf.reduce_sum(emb, axis=-1)))  # [N, T_k]
            key_masks = tf.expand_dims(tf.tile(key_masks, multiples=[num_heads, 1]), axis=1)  # [h * N, 1, T_k]
            key_masks = tf.tile(key_masks, multiples=[1, tf.shape(queries)[1], 1])  # [h * N, T_q, T_k]

        paddings = tf.ones_like(outputs) ** (-2 ** 32 + 1)  # [h * N, T_q, T_k]

        # Take elements based on condition, take x if true else y
        outputs = tf.where(condition=tf.equal(key_masks, 0), x=paddings, y=outputs)  # [h * N, T_q, T_k]

        # Activation
        outputs = tf.nn.softmax(outputs)  # [h * N, T_q, T_k]

        # Query masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(emb, axis=-1)))  # [N, T_q]
        query_masks = tf.expand_dims(tf.tile(query_masks, multiples=[num_heads, 1]), axis=-1)  # [h * N, T_q, 1]
        query_masks = tf.tile(query_masks, multiples=[1, 1, tf.shape(keys)[1]])  # [h * N, T_q, T_k]
        outputs *= query_masks  # [h * N, T_q, T_k]

        # Dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)  # [h * N, T_q, T_k]

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # [h * N, T_q, C / h]

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_or_size_splits=num_heads, axis=0), axis=2)  # [N, T_q, C]

        # Residual connection
        outputs += queries

        # Normalize
        outputs = normalize(outputs)

    return outputs


def default_hparams():
    params = HParams(
        num_heads=8,
        num_blocks=6,
        input_vocab_size=50,
        label_vocab_size=50,
        max_length=100,
        hidden_units=512,
        dropout_rate=0.2,
        learning_rate=3e-4,
        is_training=True
    )
    return params


class LanguageModel(object):

    def __init__(self, num_heads: int=8, num_blocks: int=6, input_vocab_size: int=50,
                 label_vocab_size: int=50, max_length: int=100, hidden_units: int=512,
                 dropout_rate: float=0.2, learning_rate: float=3e-4, is_training: bool=True):
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.input_vocab_size = input_vocab_size
        self.label_vocab_size = label_vocab_size
        self.max_length = max_length
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.is_training = is_training
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = tf.Graph()
        tf.reset_default_graph()
        with graph.as_default():
            with tf.name_scope("input"):
                self.x = tf.placeholder(dtype=tf.int32, shape=[None, None], name="x")  # [batch, max_length]
                self.y = tf.placeholder(dtype=tf.int32, shape=[None, None], name="y")

            emb = embedding(inputs=self.x, vocab_size=self.input_vocab_size, emb_size=self.hidden_units,
                            zero_pad=True, scale=True, scope="embedding", reuse=None)
            with tf.name_scope("encoding"):
                x_ = tf.range(tf.shape(self.x)[1])  # [max_length, ]
                x_ = tf.expand_dims(x_, axis=0)  # [1, max_length]
                x_ = tf.tile(x_, multiples=[tf.shape(self.x)[0], 1])  # [batch, max_length]
                enc = embedding(inputs=x_, vocab_size=self.max_length, emb_size=self.hidden_units,
                                zero_pad=False, scale=False, scope="enc_pe", reuse=None)
                enc += emb
                enc = tf.layers.dropout(enc, rate=self.dropout_rate, training=self.is_training)

            # blocks
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    # Multihead Attention
                    enc = multihead_attention(emb=emb, queries=enc, keys=enc, num_units=self.hidden_units,
                                              num_heads=self.num_heads, dropout_rate=self.dropout_rate,
                                              is_training=self.is_training, causality=False)
            # Feed Forward
            self.outputs = feed_forward(inputs=enc, num_units=[4 * self.hidden_units, self.hidden_units])

            # Final Linear Projection
            with tf.name_scope("ACC cal"):
                logits = tf.layers.dense(inputs=self.outputs, units=self.label_vocab_size, activation=None)
                self.preds = tf.to_int32(tf.argmax(logits, axis=-1))
                self.istarget = tf.to_float(tf.not_equal(self.y, 0))  # 去掉padding部分的计算
                self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y)) * self.istarget) / tf.reduce_sum(self.istarget)
                tf.summary.scalar("acc", self.acc)

            if self.is_training:
                with tf.name_scope("loss"):
                    y_smoothed = label_smoothing(inputs=tf.one_hot(self.y, depth=self.label_vocab_size))
                    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits= logits, labels=y_smoothed)
                    self.mean_loss = tf.reduce_sum(loss * self.istarget) / (tf.reduce_sum(self.istarget))

                    self.global_step = tf.Variable(0, name="global_step", trainable=False)
                    opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-8)
                    self.train_op = opt.minimize(self.mean_loss, global_step=self.global_step)

                    tf.summary.scalar("mean_loss", self.mean_loss)
                    self.merged = tf.summary.merge_all()

        return graph







