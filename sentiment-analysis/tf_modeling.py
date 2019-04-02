import tensorflow as tf
import data_science_helpers as dsh
import data_loading_helpers as dlh
from constants import constants
import numpy as np
import datetime
import time
import pandas as pd
import os
import os.path
from configuration import new_configs

from matplotlib import pyplot as plt

N_ET_FEATURES = 5

def tf_average_recall_and_precision(predicted, actual, n_classes):
    avg_recall = 0
    avg_precision = 0
    with tf.name_scope("Precision_and_Recall"):
        for i in range(n_classes):
            mask_recall = tf.cast(tf.equal(predicted, i), tf.float32)
            mask_precision = tf.cast(tf.equal(actual, i), tf.float32)
            recall_i = tf.reduce_mean(tf.cast(tf.equal(predicted, actual), tf.float32) * mask_recall)/tf.reduce_mean(mask_recall)
            precision_i = tf.reduce_mean(tf.cast(tf.equal(predicted, actual), tf.float32) * mask_precision)/tf.reduce_mean(mask_precision)
            avg_recall += recall_i/n_classes
            avg_precision += precision_i / n_classes
    return avg_recall, avg_precision

def add_tf_classification_measures(model):
    """
    Adds classification measures and summaries to the model, added measures are:
        Accuracy
        Recall, Precision and F1 Score for Negatives
        Recall, Precision and F1 Score for Positives

    :param model:   (obj)       Tensorflow Model to which the measures must be added

    :return:
        summaries:  (tf.obj)    Tensorflow summary of the added measures
    """
    summaries = []
    actual_values = tf.argmax(model.data_placeholders['TARGETS'], 1, name="true_values")

    target_hits = tf.cast(tf.equal(model.predictions, actual_values), tf.float32)
    model.accuracy = tf.reduce_mean(target_hits)

    accuracy_summary = tf.summary.scalar("exact_current_accuracy", model.accuracy)
    summaries.append(accuracy_summary)

    if model.input_config["BINARY_CLASSIFICATION"]:
        neg_value = 0
        pos_value = 1
    else:
        neg_value = 0
        pos_value = 2

    with tf.name_scope("Classification_Metrics"):
        mask_recall_neg = tf.cast(tf.equal(model.predictions, neg_value), tf.float32)
        model.recall_neg = tf.reduce_mean(target_hits * mask_recall_neg) / tf.reduce_mean(mask_recall_neg)

        mask_precision_neg = tf.cast(tf.equal(actual_values, neg_value), tf.float32)
        model.precision_neg = tf.reduce_mean(target_hits * mask_precision_neg) / tf.reduce_mean(mask_precision_neg)

        model.f1_neg = 2 * (model.recall_neg*model.precision_neg) / (model.recall_neg+model.precision_neg)

        mask_recall_pos = tf.cast(tf.equal(model.predictions, pos_value), tf.float32)
        model.recall_pos = tf.reduce_mean(target_hits * mask_recall_pos) / tf.reduce_mean(mask_recall_pos)

        mask_precision_pos = tf.cast(tf.equal(actual_values, pos_value), tf.float32)
        model.precision_pos = tf.reduce_mean(target_hits * mask_precision_pos) / tf.reduce_mean(mask_precision_pos)

        model.f1_pos = 2 * (model.recall_pos * model.precision_pos) / (model.recall_pos + model.precision_pos)

        recall_neg_summary = tf.summary.scalar("recall_neg", model.recall_neg)
        precision_neg_summary = tf.summary.scalar("precision_neg", model.precision_neg)
        f1_neg_summary = tf.summary.scalar("F1_Score_neg", model.f1_neg)
        recall_pos_summary = tf.summary.scalar("recall_pos", model.recall_pos)
        precision_pos_summary = tf.summary.scalar("precision_pos", model.precision_pos)
        f1_pos_summary = tf.summary.scalar("F1_Score_pos", model.f1_pos)

    all_summaries = tf.summary.merge([recall_neg_summary, precision_neg_summary, recall_pos_summary, precision_pos_summary,
                                      f1_neg_summary, f1_pos_summary, accuracy_summary])

    return all_summaries


class AugmentedRNN:
    def __init__(self, input_config = {}, vocab_size = None, max_sentence_length = 41):

        self.input_config = new_configs.complete_config(input_config)

        self.data_placeholders = {}
        self.sentence_features = []

        # TODO: Fix the approach with which the dimensions are defined
        self.we_dim = 300
        self.et_features_size = 5
        self.eeg_features_size = 105

        self.vocab_size = vocab_size
        self.max_sentence_length = max_sentence_length

        self.lstm_units = self.input_config['LSTM_UNITS']

        self.num_classes = 2 if self.input_config['BINARY_CLASSIFICATION'] else 3
        # Initialize results and results files
        # self.results = dict(accuracy=[])
        # self.result_file = open(self.RESULTS_FILE_PATH, 'w', encoding="utf-8")

        self.CreateGraph()

    def _print(self, message):
        # Controls message printing depending on verbosity
        if self.input_config['VERBOSE']:
            print(message)

    def _output_fold_results(self, count_fold, results, result_file):
        """
        Handles printing to a file the performance statistics for each cross-validation fold

        :param count_fold:  (int)   Number of folds for cross-validation
        :param results:     (list)  Sequence of results for each fold to be printed on file
        :param result_file: (file)  Opened file where to print the results

        :return:
            None
        """
        result_string = "Fold {}:".format(count_fold)
        for metric in results.keys():
            result_string = result_string + "\n {}: {}".format(metric, results[metric])
        print(result_string, file=result_file)
        self._print(result_string)

    def _produce_result_output(self, results, result_file):
        # Print final results (averaged over all folds) to file and console
        result_string = "\nAVERAGE (ALL FOLDS):\nAcc: {}".format(np.average(results["accuracy"]))
        print(result_string, file=result_file)
        self._print(result_string)

    def _create_word_embedding_variables(self):
        self.data_placeholders['WORD_IDXS'] = tf.placeholder(tf.int32, [None, self.max_sentence_length], name="input_x")
        self.W = tf.Variable(tf.random_uniform([self.vocab_size, self.we_dim], -1.0, 1.0), name="W")
        self.embedded_chars = tf.nn.embedding_lookup(self.W, self.data_placeholders['WORD_IDXS'])
        self.sentence_features.append(self.embedded_chars)

    def _create_et_variables(self):
        self.data_placeholders['ET'] = tf.placeholder(tf.float32, [None, self.max_sentence_length, self.et_features_size], name="input_et")
        self.sentence_features.append(self.data_placeholders['ET'])

    def _create_eeg_variables(self):
        if self.input_config['EEG_TO_PIC']:
            self.data_placeholders['EEG'] = tf.placeholder(tf.float32,[None, self.max_sentence_length, self.eeg_features_size, 1],
                                                           name="input_eeg")

        else:
            self.data_placeholders['EEG'] = tf.placeholder(tf.float32,[None, self.max_sentence_length, self.eeg_features_size],
                                                           name="input_eeg")
            self.sentence_features.append(self.data_placeholders['EEG'])

    def CreateGraph(self):
        # Placeholders for input, output and dropout
        self.data_placeholders['SEQUENCE_LENGTHS'] = tf.placeholder(tf.int32, [None], name="sequence_lengths")
        self.data_placeholders['TARGETS'] = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")

        self.DROPOUT_KEEP_PROB = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.L2_REG_LAMBDA = tf.placeholder(tf.float32)
        self.L1_REG_LAMBDA = tf.placeholder(tf.float32)
        self.LEARNING_RATE = tf.placeholder(tf.float32)

        # TODO Right now there is no l2_loss! Implement it if needed (first check via tensorboard)
        # Embedding layer
        with tf.device(self.input_config['TF_DEVICE']), tf.name_scope("embedding"):

            # Build the input
            if self.input_config['WORD_EMBEDDINGS']:
                self._create_word_embedding_variables()
            if self.input_config['EYE_TRACKING']:
                self._create_et_variables()
            if self.input_config['EEG_SIGNAL']:
                self._create_eeg_variables()

            self.sentence_features = tf.concat(self.sentence_features, 2)


        if self.input_config['USE_NORMALIZATION_LAYER']:
            self.sentence_features = tf.contrib.layers.layer_norm(self.sentence_features)
        print(self.sentence_features)


        self.lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_units)
        self.lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_units)
        self.lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(self.lstm_fw_cell, self.DROPOUT_KEEP_PROB, seed = 1234)
        self.lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(self.lstm_bw_cell, self.DROPOUT_KEEP_PROB, seed = 2345)
        (self.output_fw, self.output_bw), self.output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.lstm_fw_cell, cell_bw=self.lstm_bw_cell,
                                                                        inputs=self.sentence_features,
                                                                        sequence_length=self.data_placeholders['SEQUENCE_LENGTHS'],
                                                                        dtype=tf.float32)

        # Time Series Prediction part
        self.lstm_outputs = tf.concat([self.output_fw, self.output_bw], 2)


        # PAY A LOT OF ATTENTION HERE!!
        if self.input_config["ATTENTION_EMBEDDING"]:
            H_reshape = tf.reshape(self.lstm_outputs, [-1, 2 * self.lstm_units])
            with tf.name_scope("self-attention"):
                initializer = tf.contrib.layers.xavier_initializer()

                d_a_size = 40
                r_size = 2

                self.W_s1 = tf.get_variable("W_s1", shape=[2 * self.lstm_units, d_a_size], initializer=initializer)
                _H_s1 = tf.nn.tanh(tf.matmul(H_reshape, self.W_s1))
                self.W_s2 = tf.get_variable("W_s2", shape=[d_a_size, r_size], initializer=initializer)
                _H_s2 = tf.matmul(_H_s1, self.W_s2)
                _H_s2_reshape = tf.transpose(tf.reshape(_H_s2, [-1, self.max_sentence_length, r_size]), [0, 2, 1])
                self.attention_weights = tf.nn.softmax(_H_s2_reshape, name="attention")


            with tf.name_scope("sentence-embedding"):
                self.embedding = tf.matmul(self.attention_weights, self.lstm_outputs)
                self.embedding_dim = self.embedding.get_shape()[1] * self.embedding.get_shape()[2]
                self.embedding = tf.reshape(self.embedding, [-1, self.embedding_dim])
        # ----TILL HERE----

        else:
            self.embedding = tf.concat([self.output_states[0][0], self.output_states[0][1],
                                        self.output_states[1][0], self.output_states[1][1]], 1) # If this is active attention is not used
            self.embedding_dim = self.embedding.get_shape()[1]

        if self.input_config['HIDDEN_LAYER_UNITS']>0:
            self.hidden = tf.layers.dense(self.embedding, units=self.input_config['HIDDEN_LAYER_UNITS'])
            self.scores = tf.layers.dense(self.hidden, units=self.num_classes)

        else:
            self.scores = tf.layers.dense(self.embedding, units=self.num_classes)

        self.probabilities = tf.nn.softmax(self.scores, 1, name="probabilities")
        self.predictions = tf.argmax(self.scores, 1, name="predictions")

        self.actual_values = tf.argmax(self.data_placeholders['TARGETS'], 1, name="true_values")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.data_placeholders['TARGETS'])
            vars = tf.trainable_variables()
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in vars])
            l1_loss = tf.add_n([tf.norm(v, ord = 1) for v in vars])
            self.classification_loss = tf.reduce_mean(losses) + self.L2_REG_LAMBDA*l2_loss  + self.L1_REG_LAMBDA*l1_loss
            self.loss = 0.0 # Needed only to uniform among all models

        loss_summary = tf.summary.scalar("loss", self.classification_loss)

        classification_summaries = add_tf_classification_measures(self)

        self.summary = tf.summary.merge([classification_summaries, loss_summary])

    def reset_graph(self):
        tf.reset_default_graph()

# TODO: recreate training so that there is only one function that puts together the losses and can shift focus from one to the other.

def prepare_session(input_config):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=input_config['PER_PROCESS_GPU_FRACTION'])
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
    sess = tf.Session(config=session_conf)
    return sess

def train_tf(model, data_obj, train_idxs, val_idxs, sess, input_config, spec_rate = 1.0, only_new_weights = False,
             n_batch_eval = 1, tensorboard_dir = None, initialize = True):
    """
    Train a tensorflow model object, characterized by a loss and a input layer

    :param model:           (obj)       An object containing a tensorflow model with model.loss and model.layers["input"]
    :param train_data:      (df)        Data on which to train the model
    :param val_data:        (df)        Data on which to validate the model
    :param sess:            (tf sess)   Tensorflow session in which to create the graph and do the training

    :param input_config:    (dict)      Configuration of model parameters -- see configs.py for default example
    :param spec_rate:       (float)      % of importance to give to classification
    :param only_new_weights:(bool)

    :param n_batch_eval:    (int)       Number of batches of training between 2 evaluations on the test set
    :param tensorboard_dir: (str)       Directory where to put tensorboard summary logs
    :param initialize:      (bool)      True if weights are to be initialized, False if they are to be kept
    :param is_time_series:  (bool)      True if the model is on time series data

    :return:
        None
    """
    # Define Training procedure
    learning_rate = input_config['INITIAL_LR']
    batch_size = input_config['BATCH_SIZE']
    n_epochs = input_config['NUM_EPOCHS']


    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(model.LEARNING_RATE)

    loss = model.classification_loss * spec_rate + model.loss * (1 - spec_rate)

    if only_new_weights:
        vars_to_optimize = [variable_name for layer_name in model.classification_layer_names[1:] for variable_name in
                            model.dense_variables[layer_name].trainable_variables]
    else:
        vars_to_optimize = tf.trainable_variables(scope=None)

    grads_and_vars = optimizer.compute_gradients(loss, var_list=vars_to_optimize)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    tf.set_random_seed(111)
    if initialize:
        # Initialize variables
        if only_new_weights:
            init_new_vars_op = tf.variables_initializer(vars_to_optimize + optimizer.variables() + [global_step])
            sess.run(init_new_vars_op)
        else:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            if input_config['WORD_EMBEDDINGS']:
                    sess.run(model.W.assign(data_obj.initial_word_embeddings))

    if tensorboard_dir is not None:
        train_writer = tf.summary.FileWriter(tensorboard_dir + "train")
        test_writer = tf.summary.FileWriter(tensorboard_dir + "test")
        train_writer.add_graph(sess.graph)

    # Generate batches
    batches_rnn = dsh.batch_iter(np.array(train_idxs), batch_size, n_epochs)
    #fig = plt.figure()
    #accuracies = []
    while True:
        batch_rnn = next(batches_rnn, None)
        if batch_rnn is not None:
            feed_dict = {model.data_placeholders[variable] : data_obj.placeholder_fillers[variable][batch_rnn] for variable in model.data_placeholders.keys()}

            feed_dict[model.DROPOUT_KEEP_PROB] = input_config['DROPOUT_KEEP_PROB']
            feed_dict[model.L2_REG_LAMBDA] = input_config['L2_REG_LAMBDA']
            feed_dict[model.L1_REG_LAMBDA] = input_config['L1_REG_LAMBDA']
            feed_dict[model.LEARNING_RATE] = learning_rate

            accuracy_train, train_loss, train_summary, _ = sess.run([model.accuracy, loss, model.summary, train_op], feed_dict = feed_dict)
            if input_config["VERBOSE"]:
                print("Training Accuracy: {}, Training Loss:{}".format(accuracy_train, train_loss))
        else:
            break

        current_step = tf.train.global_step(sess, global_step)
        if tensorboard_dir:
            train_writer.add_summary(train_summary, global_step=current_step)

        if (current_step % n_batch_eval == 0):
            learning_rate /= 2**(1.0/input_config['HALVE_LR_EVERY_PASSES'])
            feed_dict = {model.data_placeholders[variable] : data_obj.placeholder_fillers[variable][val_idxs] for variable in model.data_placeholders.keys()}

            feed_dict[model.DROPOUT_KEEP_PROB] = 1.0
            feed_dict[model.L2_REG_LAMBDA] = 0.0
            feed_dict[model.L1_REG_LAMBDA] = 0.0
            feed_dict[model.LEARNING_RATE] = 0.0

            test_summary, accuracy_val, total_loss = sess.run([model.summary, model.accuracy, loss], feed_dict = feed_dict)
            if tensorboard_dir:
                test_writer.add_summary(test_summary, global_step=current_step)
            print("\nTest Accuracy: {}, Test Loss:{}\n".format(accuracy_val, total_loss))
            #accuracies.append(accuracy_val)
            #plot_loss(accuracies, fig)

    if tensorboard_dir:
        train_writer.close()
        test_writer.close()

def test_tf(model, data_obj, sess, input_config, spec_rate = 1.0):
    feed_dict = {model.data_placeholders[variable]: data_obj.placeholder_fillers[variable] for variable in
                 model.data_placeholders.keys()}

    feed_dict[model.DROPOUT_KEEP_PROB] = 1.0
    feed_dict[model.L2_REG_LAMBDA] = 0.0
    feed_dict[model.L1_REG_LAMBDA] = 0.0
    feed_dict[model.LEARNING_RATE] = 0.0

    loss = model.classification_loss * spec_rate + model.loss * (1 - spec_rate)
    actual_values, accuracy_val, predictions, total_loss = sess.run([model.actual_values, model.accuracy, model.predictions, loss], feed_dict=feed_dict)
    print("Total loss is : " + str(total_loss))
    print("Accuracy is : " + str(accuracy_val))
    return predictions, actual_values



def cross_validate_config_accuracy(model, data_obj, input_config, create_table=True,
                                   specialized_embeddings = True, save_all_predictions = True, tensorboard_save = True):                          #TODO: Add     DATA_MASKS
    # model_start = "model(pictures_shape={}, ConvLayersConfigs={}, DenseLayersConfigs={})".format(pic_shape, conv_dicts_paper, dense_dicts_paper)
    # NOT READY! NEEDS TO KEEP TRACK OF RESULTS, ALSO MAKE SURE TO RESTART GRAPH WHENEVER YOU WANT TO DO A NEW FOLD
    filedir = input_config["RESULTS_FILE_PATH"]
    n_folds = input_config["NUMBER_OF_CV_SPLITS"]
    avg_accuracy = 0
    results_list = []
    predictions_dfs_list = []
    count_fold = 0
    sentence_idxs = np.array(list(set(data_obj.sentence_numbers)))
    print(sentence_idxs)
    for sentences_train_idxs, sentences_dev_idxs in dsh.get_train_and_dev_indices(sentence_idxs, n_folds):
        train_sentences = sentence_idxs[sentences_train_idxs]
        dev_sentences = sentence_idxs[sentences_dev_idxs]
        train_indices = data_obj.extract_data_idxs_from_sentence_numbers(train_sentences)
        dev_indices = data_obj.extract_data_idxs_from_sentence_numbers(dev_sentences)
        count_fold += 1
        print("Initializing new session")
        sess = prepare_session(input_config)
        only_new_weights = specialized_embeddings == False
        tensorboard_dir = filedir + "Tensorboard_results/" + input_config["Config_name"] + "/" + str(input_config["SUBJECTS"]) + "/Fold" + str(count_fold) + "/"
        tensorboard_dir = tensorboard_dir if tensorboard_save else None
        train_tf(model, data_obj, train_idxs=train_indices, val_idxs=dev_indices, sess=sess, input_config = input_config,
                 spec_rate = 1.0, only_new_weights = only_new_weights, n_batch_eval = input_config["EVALUATE_EVERY"],
                 tensorboard_dir = tensorboard_dir, initialize = True)


        if create_table:
            results_dict = dict(input_config)
            results_dict["Fold"] = count_fold
            results_dict["Time"] = time.time()
            feed_dict = {model.data_placeholders[variable] : data_obj.placeholder_fillers[variable][dev_indices] for variable in model.data_placeholders.keys()}

            feed_dict[model.DROPOUT_KEEP_PROB] = 1.0
            feed_dict[model.L2_REG_LAMBDA] = 0.0
            feed_dict[model.LEARNING_RATE] = 0.0

            results_dict["Accuracy"] = sess.run(model.accuracy, feed_dict=feed_dict)
            results_dict["Precision_neg"],results_dict["Recall_neg"],results_dict["F1_neg"] = sess.run(
                [model.precision_neg, model.recall_neg, model.f1_neg], feed_dict=feed_dict)

            results_dict["Precision_pos"], results_dict["Recall_pos"], results_dict["F1_pos"] = sess.run(
                [model.precision_pos, model.recall_pos, model.f1_pos], feed_dict=feed_dict)

            avg_accuracy += results_dict["Accuracy"]/n_folds
            results_list.append(results_dict)

        if save_all_predictions:
            feed_dict = {model.data_placeholders[variable]: data_obj.placeholder_fillers[variable][dev_indices] for
                         variable in model.data_placeholders.keys()}

            feed_dict[model.DROPOUT_KEEP_PROB] = 1.0
            feed_dict[model.L2_REG_LAMBDA] = 0.0
            feed_dict[model.LEARNING_RATE] = 0.0

            sentence_numbers = data_obj.sentence_numbers[dev_indices]

            if input_config["ATTENTION_EMBEDDING"] == True:
                predictions, targets, attention_vals = sess.run([model.predictions, model.actual_values, model.attention_weights], feed_dict=feed_dict)
                attention_vals = (np.array(attention_vals) * 1000).astype(np.int16).tolist()
                fold_predictions_df = pd.DataFrame({'Predicted': predictions, 'Target': targets, "Sentence_n": sentence_numbers,
                                                    "Attention_weights":attention_vals})
            else:
                predictions, targets = sess.run([model.predictions, model.actual_values], feed_dict=feed_dict)
                fold_predictions_df = pd.DataFrame({'Predicted': predictions, 'Target': targets, "Sentence_n": sentence_numbers})

            fold_predictions_df["Fold"] = count_fold
            fold_predictions_df["Time"] = time.time()
            for config_element in input_config.keys():
                fold_predictions_df[config_element] = str(input_config[config_element])

            predictions_dfs_list.append(fold_predictions_df)

    print("Classification CV is complete, yields accuracy: " + str(avg_accuracy))

    if save_all_predictions:
        print("\n    Saving all predictions... \n")
        predictions_df = pd.DataFrame(pd.concat(predictions_dfs_list))
        predictions_path = filedir + "Predictions/" + input_config["Config_name"] + "_pred.csv"
        if os.path.isfile(predictions_path):
            old_predictions = pd.read_csv(predictions_path, index_col=0)
            all_results_df = pd.DataFrame(pd.concat([old_predictions, predictions_df]))
            all_results_df.to_csv(predictions_path)
        else:
            predictions_df.to_csv(predictions_path)
    results_df = pd.DataFrame.from_records(results_list)
    return results_df


