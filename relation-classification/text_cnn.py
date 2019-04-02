import tensorflow as tf

seed_value = 1

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters,
                 l2_reg_lambda, weights, rel_pos_embedding_size, rel_pos_cardinality, pos_tags_embedding_size, pos_tags_cardinality,
                 with_eye_tracking, et_features_size, et_number_of_bins, et_embedding_dimension, with_eeg, eeg_features_size):

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
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.input_dropout_keep_prob = tf.placeholder(tf.float32, name="input_dropout_keep_prob")
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        self.weights = tf.constant(weights, dtype=tf.float32)

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.01)

        tf.set_random_seed(seed_value)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            tf.set_random_seed(seed_value)
            feature_list = []

            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0, seed=seed_value), name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            feature_list.append(self.embedded_chars_expanded)

            self.rel_pos_first_embedding_dict = tf.Variable(
                tf.random_uniform([rel_pos_cardinality[0], rel_pos_embedding_size], -1.0, 1.0, seed=seed_value),
                name="rel_pos_first_embedding_dict")
            self.embedded_rel_pos_first = tf.nn.embedding_lookup(self.rel_pos_first_embedding_dict, self.input_rel_pos_first)
            self.embedded_rel_pos_first_expanded = tf.expand_dims(self.embedded_rel_pos_first, -1)
            feature_list.append(self.embedded_rel_pos_first_expanded)

            self.rel_pos_second_embedding_dict = tf.Variable(
                tf.random_uniform([rel_pos_cardinality[1], rel_pos_embedding_size], -1.0, 1.0, seed=seed_value),
                name="rel_pos_second_embedding_dict")
            self.embedded_rel_pos_second = tf.nn.embedding_lookup(self.rel_pos_second_embedding_dict, self.input_rel_pos_second)
            self.embedded_rel_pos_second_expanded = tf.expand_dims(self.embedded_rel_pos_second, -1)
            feature_list.append(self.embedded_rel_pos_second_expanded)

            self.pos_tags_embedding_dict = tf.Variable(
                tf.random_uniform([pos_tags_cardinality, pos_tags_embedding_size], -1.0, 1.0, seed=1),
                name="pos_tags_embedding_dict")
            self.embedded_pos_tags = tf.nn.embedding_lookup(self.pos_tags_embedding_dict, self.input_pos_tags)
            self.embedded_pos_tags_expanded = tf.expand_dims(self.embedded_pos_tags, -1)
            feature_list.append(self.embedded_pos_tags_expanded)

            if with_eye_tracking:
                if not et_number_of_bins:
                    self.embedded_et_expanded = tf.expand_dims(self.input_et, -1)
                    feature_list.append(self.embedded_et_expanded)
                elif et_number_of_bins:
                    self.et_embedding_dicts = [None] * et_features_size
                    self.embedded_et = [None] * et_features_size
                    self.embedded_et_expanded = [None] * et_features_size
                    for i in range(et_features_size):
                        self.et_embedding_dicts[i] = tf.Variable(
                            tf.random_uniform([et_number_of_bins, et_embedding_dimension], -1.0, 1.0, seed=seed_value),
                            name="et_embedding_dict_" + str(i))
                        self.embedded_et[i] = tf.nn.embedding_lookup(self.et_embedding_dicts[i], self.input_et[:, :, i])
                        self.embedded_et_expanded[i] = tf.expand_dims(self.embedded_et[i], -1)
                    feature_list.extend(self.embedded_et_expanded)

            if with_eeg:
                self.embedded_eeg_expanded = tf.expand_dims(self.input_eeg, -1)
                feature_list.append(self.embedded_eeg_expanded)

            self.embedded_all = tf.concat(feature_list, 2)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                if with_eye_tracking and not with_eeg:
                    if not et_number_of_bins:
                        filter_shape = [filter_size, embedding_size + 2 * rel_pos_embedding_size +
                                        pos_tags_embedding_size + et_features_size, 1, num_filters]
                    else:
                        filter_shape = [filter_size, embedding_size + 2 * rel_pos_embedding_size +
                                        pos_tags_embedding_size + et_embedding_dimension * et_features_size, 1, num_filters]
                elif with_eeg and not with_eye_tracking:
                    filter_shape = [filter_size, embedding_size + 2 * rel_pos_embedding_size +
                                    pos_tags_embedding_size + eeg_features_size, 1, num_filters]

                # note: this only works for raw normalized ET features
                elif with_eye_tracking and with_eeg:
                    filter_shape = [filter_size, embedding_size + 2 * rel_pos_embedding_size +
                                    pos_tags_embedding_size + et_features_size + eeg_features_size, 1, num_filters]

                else:
                    filter_shape = [filter_size, embedding_size+2*rel_pos_embedding_size + pos_tags_embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1, seed=seed_value), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_all,
                    #self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        self.h_drop = tf.nn.dropout(self.h_pool_flat, self.input_dropout_keep_prob, seed=seed_value, name="dropout")

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer(seed=seed_value))
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b",)
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.probabilities = tf.nn.softmax(self.scores, 1, name="probabilities")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
