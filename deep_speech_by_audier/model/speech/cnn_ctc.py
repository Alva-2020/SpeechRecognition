
import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, Input, MaxPooling2D, Reshape, Dense, Dropout, Lambda
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K


def conv2d(size):
    return Conv2D(filters=size, kernel_size=(3, 3), use_bias=True, activation="relu", padding="same", kernel_initializer="he_normal")

def norm(x):
    return BatchNormalization(axis=-1)(x)

def maxpool(x):
    return MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid")(x)

def dense(units, activation="relu"):
    return Dense(units=units, activation=activation, use_bias=True, kernel_initializer="he_normal")


def cnn_cell(size, x, pool=True):
    x = tf.keras.layers.





class Model(object):
    def __init__(self, vocab_size: int, n_features: int, learning_rate: float=8e-4, is_training: bool=True):
        self.vocab_size = vocab_size
        self.lr = learning_rate
        self.n_features = n_features
        self.is_training = is_training

    def _build_model(self):
        graph = tf.Graph()
        tf.reset_default_graph()
        with graph.as_default():
            with tf.name_scope("input"):
                self.inputs = tf.keras.Input(shape=(None, self.n_features, 1), name="inputs")

            with tf.name_scope("cnn"):
                h1 = tf.keras.layers.




