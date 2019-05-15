
import tensorflow as tf


class Model(object):
    def __init__(self, n_features: int, n_rnn_units: int, n_rnn_layers: int, n_classes: int,
                 bidirectional: bool=False, learning_rate: float=0.01, momentum: float=0.9):
        self.n_features = n_features
        self.n_rnn_units = n_rnn_units
        self.n_rnn_layers = n_rnn_layers
        self.n_classes = n_classes
        self.bidirectional = bidirectional
        self.lr = learning_rate
        self.momentum = momentum
        self.graph = self._build_graph()

    def _inference(self, inputs, seq_len, num_hidden, num_layers):
        rnn_init = tf.random_normal_initializer(stddev=0.1)
        if self.bidirectional:
            cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=self.n_rnn_units, initializer=rnn_init,
                                              state_is_tuple=True)
            cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=self.n_rnn_units, initializer=rnn_init,
                                              state_is_tuple=True)
            cells_fw = tf.nn.rnn_cell.MultiRNNCell(cells=[cell_fw] * self.n_rnn_layers)
            cells_bw = tf.nn.rnn_cell.MultiRNNCell(cells=[cell_bw] * self.n_rnn_layers)

            # fw_outputs, bw_outputs: [batch_size, max_step, n_rnn_units]
            (fw_outputs, bw_outputs), (_, _) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cells_fw, cell_bw=cells_bw,
                inputs=self.inputs, sequence_length=self.seq_len,
                time_major=False, dtype=tf.float32)
            # [batch_size, max_step, 2 * n_rnn_units]
            outputs = tf.concat([fw_outputs, bw_outputs], axis=2)
            # [batch_size * max_step, 2 * n_rnn_units]
            outputs = tf.reshape(outputs, shape=[-1, 2 * self.n_rnn_units])

            # Another RNN API
            # cell_fw = tf.contrib.rnn.LSTMCell(
            #     num_hidden, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1), state_is_tuple=True
            # )
            #
            # cell_bw = tf.contrib.rnn.LSTMCell(
            #     num_hidden, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1), state_is_tuple=True
            # )
            #
            # cells_fw = [cell_fw] * num_layers
            # cells_bw = [cell_bw] * num_layers
            #
            # outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            #     cells_fw=cells_fw, cells_bw=cells_bw, inputs=inputs, dtype=tf.float32, sequence_length=seq_len
            # )
            # outputs = tf.reshape(outputs, [-1, num_hidden])
        else:
            cell = tf.nn.rnn_cell.LSTMCell(num_units=self.n_rnn_units, initializer=rnn_init,
                                           state_is_tuple=True)
            cells = [cell] * self.n_rnn_layers
            stack = tf.nn.rnn_cell.MultiRNNCell(cells=cells)
            # [batch_size, max_step, n_rnn_units]
            outputs, _ = tf.nn.dynamic_rnn(
                cell=stack, inputs=self.inputs, sequence_length=self.seq_len,
                time_major=False, dtype=tf.float32)
            # [batch_size * max_step, n_rnn_units]
            outputs = tf.reshape(outputs, shape=[-1, self.n_rnn_units])

        # [batch_size * max_step, n_classes]
        logits = tf.layers.dense(inputs=outputs, units=self.n_classes, activation=None)
        # [batch_size, max_step, n_classes]
        logits = tf.reshape(logits, shape=[tf.shape(self.inputs)[0], -1, self.n_classes])
        # [max_step, batch_size, n_classes]
        logits = tf.transpose(logits, [1, 0, 2])  # for ctc input requirement

        return logits

    def _build_graph(self):
        graph = tf.Graph()
        tf.reset_default_graph()
        with graph.as_default():
            with tf.name_scope("input"):
                # [batch_size, max_step, n_features] 满足RNN time_major=False的需求
                self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, None, self.n_features])
                self.labels = tf.sparse_placeholder(dtype=tf.int32)
                self.seq_len = tf.placeholder(dtype=tf.int32, shape=[None])

            with tf.name_scope("inference"):
                logits = self._inference(self.inputs, self.seq_len, self.n_rnn_units, self.n_rnn_layers)

            with tf.name_scope("loss"):
                ctc_loss = tf.nn.ctc_loss(labels=self.labels, inputs=logits, sequence_length=self.seq_len)  # [batch_size]
                self.loss = tf.reduce_mean(ctc_loss)
                self.train_op = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=self.momentum).minimize(self.loss)
            with tf.name_scope("decode"):
                # decode: single element list    decode[0]: sparse tensor
                decode, _ = tf.nn.ctc_greedy_decoder(inputs=logits, sequence_length=self.seq_len, merge_repeated=True)
                self.result = decode[0]
                self.ler = tf.reduce_mean(tf.edit_distance(tf.cast(self.result, dtype=tf.int32), self.labels))

            self.init = tf.global_variables_initializer()
        return graph
