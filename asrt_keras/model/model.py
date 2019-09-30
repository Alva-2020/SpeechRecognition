
import numpy as np
from tensorflow.python.keras.layers import Dense, Dropout, Input, Reshape, Lambda, Activation, Conv2D, MaxPooling2D
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import History
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import Sequence, multi_gpu_model
from ..config import MODELKEYS
from tqdm import tqdm
from typing import Tuple
from _utils.udistance import EditDistance


class _Model:
    """An alter edition of DFCNN"""
    def __init__(self, n_classes: int, n_features: int):
        self.n_classes = n_classes
        self.n_features = n_features

    def __call__(self, inputs):
        h = self.cnn_cell(32, (0.05, 0.05), True, inputs)
        h = self.cnn_cell(64, (0.1, 0.1), True, h)
        h = self.cnn_cell(128, (0.15, 0.15), True, h)
        h = self.cnn_cell(128, (0.2, 0.2), False, h)
        h = self.cnn_cell(128, (0.2, 0), False, h)
        h = Reshape((-1, self.n_features // 8 * 128))(h)  # Flatten
        h = self.dropout(0.3)(h)
        h = self.dense(128)(h)
        h = self.dropout(0.3)(h)
        return self.dense(self.n_classes, activation="softmax")(h)

    @staticmethod
    def conv2d(size):
        return Conv2D(filters=size, kernel_size=(3, 3), use_bias=True, activation="relu",
                      padding="same", kernel_initializer="he_normal")

    @staticmethod
    def maxpool(x):
        return MaxPooling2D(pool_size=2, strides=None, padding="valid")(x)

    @staticmethod
    def dense(units, activation="relu"):
        return Dense(units=units, activation=activation, use_bias=True, kernel_initializer="he_normal")

    @staticmethod
    def dropout(rate: float):
        return Dropout(rate) if rate > 0. else Lambda(lambda x: x)

    def cnn_cell(self, filter_size: int, drop_rates: Tuple[float, float], pool: bool, x):
        x = self.conv2d(filter_size)(x)
        x = self.dropout(drop_rates[0])(x)
        x = self.conv2d(filter_size)(x)
        if pool:
            x = self.maxpool(x)
        x = self.dropout(drop_rates[1])(x)
        return x

    @staticmethod
    def get_decode_input_length(x):
        return K.shape(x)[1] // 8


class AcousticModel(object):
    def __init__(self, vocab_size: int, n_features: int, gpu_num: int = 1, learning_rate: float = 1e-3,
                 is_training: bool = True):
        self.n_classes = vocab_size
        self.is_training = is_training
        self.n_features = n_features
        self._inference = _Model(self.n_classes, self.n_features)
        self.inputs = Input(shape=[None, self.n_features, 1], dtype="float32", name=MODELKEYS.INPUT)
        self.y_pred = self._inference(self.inputs)

        self.gpu_num = gpu_num
        self.lr = learning_rate
        self.inference_model = Model(inputs=self.inputs, outputs=self.y_pred)
        if is_training:
            self._opt_init()

    # @property
    # def test_func(self):
    #     return K.function(inputs=self.inputs, outputs=self.y_pred)

    def _opt_init(self):
        self.labels = Input(shape=[None], dtype="int32", name=MODELKEYS.LABELS)
        self.input_length = Input(shape=[1], dtype="int32", name=MODELKEYS.INPUT_LENGTH)
        self.label_length = Input(shape=[1], dtype="int32", name=MODELKEYS.LABEL_LENGTH)

        self.ctc_input_length = Lambda(function=self.get_ctc_input_length, name="ctc_input_length")(self.input_length)
        self.loss = Lambda(function=self.ctc_loss, name="ctc_loss")\
            ([self.labels, self.y_pred, self.ctc_input_length, self.label_length])  # function 只接受一个占位输入
        self.ctc_model = Model(inputs=[self.inputs, self.labels, self.input_length, self.label_length], outputs=self.loss)
        if self.gpu_num > 1:
            self.train_model = multi_gpu_model(model=self.ctc_model, gpus=self.gpu_num)
        else:
            self.train_model = self.ctc_model

        self.train_model.compile(
            optimizer=Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, decay=0.0, epsilon=1e-7),
            # 这里的outputs就是loss，而不是基于inputs和outputs计算的损失
            # key的名称必须是层的名称，如上文中的ctc_loss
            loss={"ctc_loss": lambda inputs, outputs: outputs}
        )

    @staticmethod
    def get_ctc_input_length(x):
        return x // 8

    @staticmethod
    def ctc_loss(inputs):
        labels, y_pred, ctc_input_length, label_length = inputs
        y_pred = y_pred[:, :, :]
        cost = K.ctc_batch_cost(y_true=labels, y_pred=y_pred, input_length=ctc_input_length, label_length=label_length)
        return K.mean(cost)

    def predict(self, inputs: np.ndarray, input_length: int):
        """Predict on a single speech sample
        :param inputs: The audio samples, require of shape [?, n_features, 1]
        :param input_length: The time frames of samples before decode.
        :return: List of ids of labels.
        """
        if inputs.ndim != 3 or inputs.shape[-1] != 1 or inputs.shape[-2] != self.n_features:
            raise TypeError(f"Require inputs to be of shape [?, {self.n_features}, 1], got {inputs.shape}.")
        if not isinstance(input_length, int):
            raise TypeError(f"Require input length to be a int scalar, got {type(input_length)}.")

        inputs = [inputs]  # batch_size = 1
        input_length = [self.get_ctc_input_length(input_length)]
        base_pred = self.inference_model.predict(inputs)[:, :, :]
        r = K.ctc_decode(base_pred, input_length=input_length, greedy=True, beam_width=100, top_paths=1)
        encoded_ids = K.get_value(r[0][0])[0]
        return encoded_ids

    def train(self, train_data: Sequence, test_data: Sequence, epochs: int = 1, callbacks=None) -> History:
        """
        Train for a single epoch, with both `train` and `eval` step in this process
        :return: a History object
        """
        h = self.train_model.fit_generator(
            generator=train_data,
            validation_data=test_data,
            validation_steps=200,
            epochs=epochs,
            verbose=1,
            steps_per_epoch=len(train_data),
            use_multiprocessing=True,
            workers=4,
            callbacks=callbacks)
        return h

    def test(self, test_data: Sequence, test_num: int = -1) -> Tuple[float, float]:
        """Test on `test_data` for `test_num` batches, compute cer.
        :return: A tuple of cer in different calculation scope.
        `macro_avg_cer`, averaged cer by total testing samples, each sample is equally considered regardless of length.
        `micro_avg_cer`, averaged cer by total testing words.
        """
        if test_num <= 0 or test_num > len(test_data):
            test_num = len(test_data)

        n_tests, n_words, errors = 0, 0, 0
        total_cer = 0.
        for i in tqdm(range(test_num)):  # loop on a bunch of batches
            inputs, _ = test_data[i]  # select a single batch
            data, data_length, labels = inputs[MODELKEYS.INPUT], inputs[MODELKEYS.INPUT_LENGTH], inputs[MODELKEYS.LABELS]
            for x, length, label in zip(data, data_length, labels):  # loop on single batch
                length = np.asscalar(length)
                encoded_ids = self.predict(x, length)
                edit_distance = EditDistance.distance_with_tokens(encoded_ids, label)
                n_tests += 1  # Total num of test
                n_words += len(label)  # Total target words
                errors += edit_distance  # Total char errors
                total_cer += (edit_distance / len(label))  # Total sum of cer

        macro_avg_cer = total_cer / n_tests  # Avg cer by total samples.
        micro_avg_cer = errors / n_words  # Avg cer by total words.
        return macro_avg_cer, micro_avg_cer

    def save_model(self, file: str):
        """Save main model, ctc_model in this class"""
        self.ctc_model.save(file, overwrite=True)

    def load_model(self, file: str):
        """Load weights from model file"""
        self.ctc_model.load_weights(file)

    def summary(self):
        self.inference_model.summary()
