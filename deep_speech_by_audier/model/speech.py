
import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, Input, MaxPooling2D, Dense, Dropout, Lambda, Flatten, GRU, add
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K


class DFCNN(Model):
    def __init__(self, n_classes: int):
        super(DFCNN, self).__init__()  # tf 1.11 no need for `inputs` and `outputs`
        self.n_classes = n_classes

    def call(self, inputs, mask=None):
        h = self.cnn_cell(32, inputs, pool=True)
        h = self.cnn_cell(64, h, pool=True)
        h = self.cnn_cell(128, h, pool=True)
        h = self.cnn_cell(128, h, pool=False)
        h = self.cnn_cell(128, h, pool=False)
        h = Flatten()(h)
        h = Dropout(rate=0.2)(h)
        h = self.dense(256)(h)
        h = Dropout(rate=0.2)(h)
        return self.dense(self.n_classes, activation="softmax")(h)

    @staticmethod
    def conv2d(size):
        return Conv2D(filters=size, kernel_size=(3, 3), use_bias=True, activation="relu", padding="same",
                      kernel_initializer="he_normal")
    @staticmethod
    def norm(x):
        return BatchNormalization(axis=-1)(x)

    @staticmethod
    def maxpool(x):
        return MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid")(x)

    @staticmethod
    def dense(units, activation="relu"):
        return Dense(units=units, activation=activation, use_bias=True, kernel_initializer="he_normal")

    def cnn_cell(self, size, x, pool=True):
        x = self.norm(self.conv2d(size)(x))
        x = self.norm(self.conv2d(size)(x))
        if pool:
            x = self.maxpool(x)
        return x


class DFSMN(Model):
    pass


class BiGRU(Model):
    def __init__(self, n_classes: int):
        super(BiGRU, self).__init__()
        self.n_classes = n_classes

    def call(self, inputs, mask=None):
        h = Flatten()(inputs)
        h = self.dense(512, h)
        h = self.dense(512, h)
        h = self.bi_gru(512, h)
        h = self.bi_gru(512, h)
        h = self.bi_gru(512, h)
        h = self.dense(512, h)
        return self.dense(self.n_classes, h, activation="softmax")

    @staticmethod
    def bi_gru(units, x):
        x = Dropout(rate=0.2)(x)
        y1 = GRU(units=units, return_sequences=True, kernel_initializer="he_normal")(x)
        y2 = GRU(units=units, return_sequences=True, go_backwards=True, kernel_initializer="he_normal")(x)
        return add([y1, y2])

    @staticmethod
    def dense(units, x, activation="relu"):
        x = Dropout(rate=0.2)(x)
        return Dense(units=units, activation=activation, use_bias=True, kernel_initializer="he_normal")(x)


# class CTCModel(Model):
#     def __init__(self):
#         super(CTCModel, self).__init__()
#
#     def call(self, inputs, mask=None):
#         labels, y_pred, input_length, label_length = inputs
#         output = Lambda(function=self.ctc_lambda)(labels, y_pred, input_length, label_length)
#         return output
#
#     @staticmethod
#     def ctc_lambda(labels, y_pred, input_length, label_length):
#         y_pred = y_pred[:, :, :]
#         cost = K.ctc_batch_cost(y_true=labels, y_pred=y_pred, input_length=input_length, label_length=label_length)
#         return K.mean(cost)


class AcousticModel(object):
    def __init__(self, vocab_size: int, n_features: int, inference_model: str, learning_rate: float=8e-4, is_training: bool=True):
        self.vocab_size = vocab_size
        self.is_training = is_training
        self.n_features = n_features

        if inference_model.upper() == "DFCNN":
            self.inference_model = DFCNN(vocab_size)
        # elif inference_model.upper() == "DFSMN":
        #     self.inference_model = DFSMN(vocab_size)
        else:
            self.inference_model = BiGRU(vocab_size)
        self.lr = learning_rate
        self._build_model()
        if is_training:
            self._opt_init()

    def _build_model(self):
        with tf.name_scope("input"):
            self.inputs = Input(shape=[None, self.n_features, 1])
            self.labels = Input(shape=[None], dtype="int32")
            self.input_length = Input(shape=[1], dtype="int32")
            self.label_length = Input(shape=[1], dtype="int32")
        with tf.name_scope("inference"):
            self.y_pred = self.inference_model.call(self.inputs)
        with tf.name_scope("loss"):
            self.loss = self.ctc_loss(self.labels, self.y_pred, self.input_length, self.label_length)
            self.model = Model(inputs=[self.labels, self.y_pred, self.input_length, self.label_length], outputs=self.loss)

    def _opt_init(self):
        self.model.compile(
            optimizer=Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, decay=0.01, epsilon=1e-7),
            loss={"ctc": lambda inputs, outputs: outputs}
        )

    @staticmethod
    def ctc_loss(labels, y_pred, input_length, label_length):
        y_pred = y_pred[:, :, :]
        cost = K.ctc_batch_cost(y_true=labels, y_pred=y_pred, input_length=input_length, label_length=label_length)
        return K.mean(cost)
