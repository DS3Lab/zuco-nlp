import tensorflow as tf

seed_value = 1

class TextRNN(object):

    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size,
                 l2_reg_lambda, weights, rel_pos_embedding_size, rel_pos_cardinality, lstm_units,
                 pos_tags_embedding_size, pos_tags_cardinality, with_eye_tracking, et_features_size,
                 et_number_of_bins, et_embedding_dimension, with_eeg, eeg_features_size, use_normalization_layer):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_rel_pos_first = tf.placeholder(tf.int32, [None, sequence_length], name="input_rel_pos_first")
        self.input_rel_pos_second = tf.placeholder(tf.int32, [None, sequence_length], name="input_rel_pos_second")
        self.input_pos_tags = tf.placeholder(tf.int32, [None, sequence_length], name="input_pos_tags")
        if et_number_of_bins is False:
            self.input_et = tf.placeholder(tf.float32, [None, sequence_length, et_features_size], name="input_et")
        else:
            self.input_et = tf.placeholder(tf.int32, [None, sequence_length, et_features_size], name="input_et")
        self.input_eeg = tf.placeholder(tf.float32, [None, sequence_length, eeg_features_size], name="input_eeg")
        self.sequence_lengths = tf.placeholder(tf.int32, [None], name="sequence_lengths")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.input_dropout_keep_prob = tf.placeholder(tf.float32, name="input_dropout_keep_prob")
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        self.weights = tf.constant(weights, dtype=tf.float32)

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.01)

        # todo: test seed here
        tf.set_random_seed(seed_value)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            tf.set_random_seed(seed_value)
            feature_list = []

            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0, seed=seed_value),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            feature_list.append(self.embedded_chars)

            self.rel_pos_first_embedding_dict = tf.Variable(
                tf.random_uniform([rel_pos_cardinality[0], rel_pos_embedding_size], -1.0, 1.0, seed=seed_value),
                name="rel_pos_first_embedding_dict")
            self.embedded_rel_pos_first = tf.nn.embedding_lookup(self.rel_pos_first_embedding_dict, self.input_rel_pos_first)
            feature_list.append(self.embedded_rel_pos_first)

            self.rel_pos_second_embedding_dict = tf.Variable(
                tf.random_uniform([rel_pos_cardinality[1], rel_pos_embedding_size], -1.0, 1.0, seed=seed_value),
                name="rel_pos_second_embedding_dict")
            self.embedded_rel_pos_second = tf.nn.embedding_lookup(self.rel_pos_second_embedding_dict, self.input_rel_pos_second)
            feature_list.append(self.embedded_rel_pos_second)

            self.pos_tags_embedding_dict = tf.Variable(
                tf.random_uniform([pos_tags_cardinality, pos_tags_embedding_size], -1.0, 1.0, seed=seed_value),
                name="pos_tags_embedding_dict")
            self.embedded_pos_tags = tf.nn.embedding_lookup(self.pos_tags_embedding_dict, self.input_pos_tags)
            feature_list.append(self.embedded_pos_tags)

            if with_eye_tracking:
                if not et_number_of_bins:
                    feature_list.append(self.input_et)
                elif et_number_of_bins:
                    self.et_embedding_dicts = [None] * et_features_size
                    self.embedded_et = [None] * et_features_size
                    for i in range(et_features_size):
                        self.et_embedding_dicts[i] = tf.Variable(
                            tf.random_uniform([et_number_of_bins, et_embedding_dimension], -1.0, 1.0, seed=seed_value),
                            name="et_embedding_dict_" + str(i))
                        self.embedded_et[i] = tf.nn.embedding_lookup(self.et_embedding_dicts[i], self.input_et[:, :, i])
                    feature_list.extend(self.embedded_et)

            if with_eeg:
                feature_list.append(self.input_eeg)

            self.embedded_all = tf.concat(feature_list, 2)

        if use_normalization_layer:
            self.embedded_all = tf.contrib.layers.layer_norm(self.embedded_all)

        self.lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
        self.lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
        self.lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(self.lstm_fw_cell, self.input_dropout_keep_prob, seed=seed_value)
        self.lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(self.lstm_bw_cell, self.input_dropout_keep_prob, seed=seed_value)
        self.value, self.output_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.lstm_fw_cell, cell_bw=self.lstm_bw_cell, inputs=self.embedded_all,
                                                        sequence_length=self.sequence_lengths, dtype=tf.float32)

        logits = tf.layers.dense(tf.concat([self.output_state[0][0], self.output_state[0][1], self.output_state[1][0], self.output_state[1][1]], 1),
                                 units=num_classes)

        self.scores = logits
        self.probabilities = tf.nn.softmax(self.scores, 1, name="probabilities")
        self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda*l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")