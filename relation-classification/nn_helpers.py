def train_step_rnn(x_batch, rel_pos_first_batch, rel_pos_second_batch, pos_tags_batch, sequence_lengths_batch, et_batch, eeg_batch, y_batch,
                   learning_rate, global_step, train_op, dropout_keep_prob, sess, rnn):
    # A single training step

    feed_dict = {
        rnn.input_x: x_batch,
        rnn.input_rel_pos_first: rel_pos_first_batch,
        rnn.input_rel_pos_second: rel_pos_second_batch,
        rnn.input_pos_tags: pos_tags_batch,
        rnn.sequence_lengths: sequence_lengths_batch,
        rnn.input_et: et_batch,
        rnn.input_eeg: eeg_batch,
        rnn.input_y: y_batch,
        rnn.input_dropout_keep_prob: dropout_keep_prob,
        rnn.learning_rate: learning_rate
    }
    _, step, loss = sess.run([train_op, global_step, rnn.loss], feed_dict)


def dev_step_rnn(x_batch, rel_pos_first_batch, rel_pos_second_batch, pos_tags_batch, sequence_lengths_batch, et_batch, eeg_batch, y_batch,
                 global_step, dropout_keep_prob, sess, rnn):
    # Evaluates model on a dev set

    feed_dict = {
        rnn.input_x: x_batch,
        rnn.input_rel_pos_first: rel_pos_first_batch,
        rnn.input_rel_pos_second: rel_pos_second_batch,
        rnn.input_pos_tags: pos_tags_batch,
        rnn.sequence_lengths: sequence_lengths_batch,
        rnn.input_et: et_batch,
        rnn.input_eeg: eeg_batch,
        rnn.input_y: y_batch,
        rnn.input_dropout_keep_prob: dropout_keep_prob
    }
    step, loss, probs = sess.run([global_step, rnn.loss, rnn.probabilities], feed_dict)
    return step, loss, probs


def train_step_cnn(x_batch, rel_pos_first_batch, rel_pos_second_batch, pos_tags_batch, et_batch, eeg_batch, y_batch, learning_rate,
                   global_step, train_op, dropout_keep_prob, sess, cnn):
    # A single training step

    feed_dict = {
        cnn.input_x: x_batch,
        cnn.input_rel_pos_first: rel_pos_first_batch,
        cnn.input_rel_pos_second: rel_pos_second_batch,
        cnn.input_pos_tags: pos_tags_batch,
        cnn.input_et: et_batch,
        cnn.input_eeg: eeg_batch,
        cnn.input_y: y_batch,
        cnn.input_dropout_keep_prob: dropout_keep_prob,
        cnn.learning_rate: learning_rate
    }

    _, step, loss = sess.run([train_op, global_step, cnn.loss], feed_dict)


def dev_step_cnn(x_batch, rel_pos_first_batch, rel_pos_second_batch, pos_tags_batch, et_batch, eeg_batch, y_batch, global_step, dropout_keep_prob, sess, cnn):
    # Evaluates model on a dev set

    feed_dict = {
        cnn.input_x: x_batch,
        cnn.input_rel_pos_first: rel_pos_first_batch,
        cnn.input_rel_pos_second: rel_pos_second_batch,
        cnn.input_pos_tags: pos_tags_batch,
        cnn.input_et: et_batch,
        cnn.input_eeg: eeg_batch,
        cnn.input_y: y_batch,
        cnn.input_dropout_keep_prob: dropout_keep_prob
    }
    step, loss, probs = sess.run([global_step, cnn.loss, cnn.probabilities], feed_dict)
    return step, loss, probs
