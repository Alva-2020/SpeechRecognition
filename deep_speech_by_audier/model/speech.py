
from tensorflow import shape as tf_shape
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, Input, MaxPooling2D, Dense, Dropout, Reshape, Flatten, GRU, add, Lambda
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K


class DFCNN(object):
    def __init__(self, n_classes: int, n_features: int):
        self.n_classes = n_classes
        self.n_features = n_features
        inputs = Input(shape=[None, n_features, 1])
        self.model = Model(inputs=inputs, outputs=self._forward(inputs))

    def _forward(self, inputs):
        h = self.cnn_cell(32, inputs)  # conv[seq_len, n_features, 32] -> pool[seq_len // 2, n_features // 2, 32]
        h = self.cnn_cell(64, h)  # conv[seq_len // 2, n_features // 2, 64] -> pool[seq_len // 4, n_features // 4, 64]
        h = self.cnn_cell(128, h)  # conv[seq_len // 4, n_features // 4, 128] -> pool[seq_len // 8, n_features // 8, 128]
        h = self.cnn_cell(128, h, pool=False)  # conv[seq_len // 8, n_features // 8, 128]
        h = self.cnn_cell(128, h, pool=False)  # conv[seq_len // 8, n_features // 8, 128]
        h = Reshape((tf_shape(h)[1], self.n_features // 8 * 128))(h)
        h = Dropout(rate=0.2)(h)
        h = self.dense(256)(h)
        h = Dropout(rate=0.2)(h)
        return self.dense(self.n_classes, activation="softmax")(h)

    def __call__(self):
        return self.model

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


class DFSMN(object):
    pass


class BiGRU(object):
    def __init__(self, n_classes: int, n_features: int):
        self.n_classes = n_classes
        self.n_features = n_features
        inputs = Input(shape=[None, n_features, None])
        self.model = Model(inputs=inputs, outputs=self._forward(inputs))

    def _forward(self, inputs):
        h = Flatten()(inputs)
        h = self.dense(512, h)
        h = self.dense(512, h)
        h = self.bi_gru(512, h)
        h = self.bi_gru(512, h)
        h = self.bi_gru(512, h)
        h = self.dense(512, h)
        return self.dense(self.n_classes, h, activation="softmax")

    def __call__(self):
        return self.model

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
    def __init__(self, vocab_size: int, n_features: int, inference_model_type: str,
                 learning_rate: float=8e-4, is_training: bool=True):
        self.vocab_size = vocab_size
        self.is_training = is_training
        self.n_features = n_features
        self.model_type = inference_model_type.upper()
        self.lr = learning_rate
        self._build_model()
        if is_training:
            self._opt_init()

    def _build_model(self):
        self.inputs = Input(shape=[None, self.n_features, 1], dtype="float32", name="the_inputs")
        self.labels = Input(shape=[None], dtype="int32", name="the_labels")
        self.input_length = Input(shape=[1], dtype="int32", name="the_input_length")
        self.label_length = Input(shape=[1], dtype="int32", name="the_label_length")
        if self.model_type == "DFCNN":
            self.inference_model = DFCNN(self.vocab_size, self.n_features)()
        else:
            self.inference_model = BiGRU(self.vocab_size, self.n_features)()
        self.y_pred = self.inference_model(self.inputs)
        self.loss = Lambda(function=self.ctc_loss, name="ctc_loss")([self.labels, self.y_pred, self.input_length, self.label_length])  # function 只接受一个占位输入
        self.model = Model(inputs=[self.inputs, self.labels, self.input_length, self.label_length], outputs=self.loss)

    def _opt_init(self):
        self.model.compile(
            optimizer=Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, decay=0.01, epsilon=1e-7),
            # 这里的outputs就是loss，而不是基于inputs和outputs计算的损失
            # key的名称必须是层的名称，如上文中的ctc_loss
            loss={"ctc_loss": lambda inputs, outputs: outputs}
        )

    @staticmethod
    def ctc_loss(inputs):
        labels, y_pred, input_length, label_length = inputs
        y_pred = y_pred[:, :, :]
        cost = K.ctc_batch_cost(y_true=labels, y_pred=y_pred, input_length=input_length, label_length=label_length)
        return K.mean(cost)
