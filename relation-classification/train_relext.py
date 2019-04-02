#! /usr/bin/env python
import datetime
import random
import time
import sys
import os
import numpy as np
import tensorflow as tf
import data_helpers
import data_science_helpers
import output
from nn_helpers import train_step_rnn, dev_step_rnn, train_step_cnn, dev_step_cnn
from text_cnn import TextCNN
from text_rnn import TextRNN
from feature_engineering import eye_tracking_features
from feature_engineering import eeg_features_word_level

seed_value = 1
os.environ['PYTHONHASHSEED'] = str(seed_value)

random.seed(seed_value)
startTime = datetime.datetime.now()

tf.set_random_seed(seed_value)


def initialize_parameters():
    # ========================= Parameters =========================
    # Model Hyperparameters
    tf.flags.DEFINE_boolean("enable_word_embeddings", True, "Enable/disable the word embedding (default: True)")
    tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
    tf.flags.DEFINE_string("filter_sizes", "2,3,4,5,6,7", "Comma-separated filter sizes (default: '2,3,4,5,6')")
    tf.flags.DEFINE_integer("num_filters", 192, "Number of filters per filter size (default: 192)")
    tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.6)")
    tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate (default: 0.001)")
    tf.flags.DEFINE_float("l2_reg_lambda", 0.5, "L2 regularization lambda (default: 0.5)")
    tf.flags.DEFINE_integer("lstm_units", 600, "Dimensionality of the LSTM (default: 600)")
    tf.flags.DEFINE_integer("ensemble_size", 20, "Size of the ensemble (default: 20)")
    tf.flags.DEFINE_integer("rel_pos_embedding_size", 20, "Dimensionality of relative positions embedding (default: 20)")
    tf.flags.DEFINE_integer("pos_tags_embedding_size", 30, "Dimensionality of POS tags embedding (default: 30)")
    tf.flags.DEFINE_integer("et_embedding_dimension", 20, "Dimensionality of binned eye-tracking feature embeddings (default: 20)")

    # Training parameters
    tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs_rnn", 200, "Number of training epochs for the RNN (default: 200)")
    tf.flags.DEFINE_integer("num_epochs_cnn", 10, "Number of training epochs for the CNN (default: 10)")
    tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)") # in this case: dev set = test set

    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
    tf.flags.DEFINE_float("decay_coefficient", 2.5, "Decay coefficient (default: 2.5)")


# Get current configuration as command line argument
cfg = data_helpers.read_config_file(sys.argv[1])
dataset_name = cfg["datasets"]["default"]
task = dataset_name.split("_")[1].upper()
evaluation_dataset_name = cfg["train_test_split"]["evaluation"] if cfg["train_test_split"][
                                                                       "type"] == "fixed_split" else dataset_name

# Initialize output files
time_for_filenames = time.localtime()
output_files = output.initialize_result_file(evaluation_dataset_name, time_for_filenames, cfg["features"]["subjects"])
output_files.update(output.initialize_submission_file(evaluation_dataset_name, time_for_filenames, cfg["features"]["subjects"]))

initialize_parameters()
FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
print("\nParameters:")
for attr, value in tf.app.flags.FLAGS.flag_values_dict().items():
    print("{}={}".format(attr.upper(), value))
    print("{}={}".format(attr.upper(), value), file=output_files['result'])
print("")

# ========================= Data Preparation =========================


# Load data
print("Loading data...")
dataset, x, x_text, y, vocab_processor, _, max_doc_length = data_helpers.get_processed_dataset(cfg, dataset_name)
sequence_lengths = np.argmin(x, axis=1)
sentence_order = data_helpers.get_sentence_order(dataset)
print("Data loaded.")

print("Loading relative positions...")
relative_positions, _, relative_positions_cardinality = data_helpers.get_relative_positions(cfg, dataset_name)
print("Relative positions loaded.")

print("Loading POS tags...")
pos_tags, pos_tags_cardinality = data_helpers.get_pos_tags(cfg, dataset_name, dataset["filenames"])
print("POS tags loaded.")

# ========================= Gaze Features =========================

# Load ET data if necessary
if cfg["features"]["gaze"] is True:
    print("Loading eye-tracking data...")
    print("ET CONFIG: ", "binned: ", cfg["features"]["binned"], "\ndata from subjects: ", cfg["features"]["subjects"], "\n\n", file=output_files['result'])
    et = eye_tracking_features.get_eye_tracking_features(matlab_files=cfg["datasets"]["matlab_files"], subject_names=cfg["features"]["subjects"],
                                                         number_of_bins=cfg["features"]["binned"],
                                                         sentence_numbers=None, task=task, max_sequence_length=max_doc_length)[sentence_order]
    print("ET data loaded.")
else:
    print("NO eye-tracking features used.")
    print("NO eye-tracking features", file=output_files['result'])
    # Phony values
    et = np.zeros((x.shape[0], x.shape[1], 2))

# ========================= EEG Features =========================

# Load EEG data if necessary
if cfg["features"]["eeg"] is True:
    print("Loading EEG data...")
    print("EEG CONFIG: ", cfg["features"]["eeg_config"], "\ndata from subjects: ", cfg["features"]["subjects"], file=output_files['result'])
    eeg = eeg_features_word_level.get_eeg_features(matlab_files=cfg["datasets"]["matlab_files"], eeg_subject_from=cfg["features"]["subjects"],
                                                         eeg_config=cfg["features"]["eeg_config"],
                                                         sentence_numbers=None, pca_dimension=-1, task=task, max_sequence_length=max_doc_length)[sentence_order]

    print("EEG data loaded.")
else:
    print("NO EEG features used.")
    print("NO EEG features", file=output_files['result'])
    # Phony values
    eeg = np.zeros((x.shape[0], x.shape[1], 2))

# ========================= Preprocessing =========================

# Randomly shuffle data
np.random.seed(seed_value)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
relative_positions_shuffled = [r[shuffle_indices] for r in relative_positions]
sequence_lengths_shuffled = sequence_lengths[shuffle_indices]
pos_tags_shuffled = pos_tags[shuffle_indices]
et_shuffled = et[shuffle_indices]
eeg_shuffled = eeg[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Read entities for each sentence
entity_ids = data_helpers.get_entity_ids(cfg, dataset_name, dataset["filenames"])
entities_shuffled = np.array(entity_ids)[shuffle_indices]

# Initialize results
count_fold = 0
results = dict(accuracy=[], precision=[], recall=[], f1=[])
final_predictions_strings = []
target_strings = []
sentences = []
entities = []
probabilities_of_predicted = []
final_reverse_labels = []

# Load word embeddings
print("Loading word embeddings...")
init = None
if FLAGS.enable_word_embeddings and cfg['word_embeddings']['default'] is not None:
    embedding_name = cfg['word_embeddings']['default']
    embedding_dimension = cfg['word_embeddings'][embedding_name]['dimension']
    initW = data_helpers.get_word_embeddings(vocab_processor, embedding_name, embedding_dimension, cfg)
else:
    embedding_dimension = FLAGS.embedding_dim
print("Word embeddings loaded")

# Train each ensemble on each fold separately
for train_indices, dev_indices in data_science_helpers.get_train_and_dev_indices(cfg, y_shuffled, dataset["filenames"], dataset_name, evaluation_dataset_name, shuffle_indices):

    final_probabilities_by_network = np.zeros((len(dev_indices), y.shape[1], FLAGS.ensemble_size))
    final_probabilities_entities_only_by_network = np.zeros((len(dev_indices), y.shape[1], FLAGS.ensemble_size))

    count_fold += 1
    print("#####FOLD " + str(count_fold) + " #####")

    x_dev, y_dev = x_shuffled[dev_indices], y_shuffled[dev_indices]
    y_dev_decoded = y_dev.argmax(1)  # Decode one-hot encoded labels
    relative_positions_dev = [r[dev_indices] for r in relative_positions_shuffled]
    pos_tags_dev, sequence_lengths_dev = pos_tags_shuffled[dev_indices], sequence_lengths_shuffled[dev_indices]
    et_dev, eeg_dev = et_shuffled[dev_indices], eeg_shuffled[dev_indices]

    x_train, pos_tags_train, sequence_lengths_train, et_train, eeg_train, y_train, relative_positions_train = data_science_helpers.prepare_training_data(
        x_shuffled, relative_positions_shuffled, pos_tags_shuffled, sequence_lengths_shuffled, et_shuffled, eeg_shuffled, y_shuffled,
        train_indices)

    class_weights_for_cross_entropy = np.ones(len(cfg["datasets"][dataset_name]["categories"]))

    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    for number_of_network in range(0, FLAGS.ensemble_size):
        print("====== NETWORK " + str(number_of_network) + " ========")
        tf.set_random_seed(6789 + number_of_network)

        # ========================= Training =========================
        with tf.Graph().as_default():
            # Random seed for TensorFlow
            tf.set_random_seed(seed_value + number_of_network)

            session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
            sess1 = tf.Session(config=session_conf)
            sess2 = tf.Session(config=session_conf)

            with sess1.as_default() and sess2.as_default():

                tf.set_random_seed(seed_value + number_of_network)
                rnn = TextRNN(sequence_length=x_train.shape[1], num_classes=y_train.shape[1],
                              vocab_size=len(vocab_processor.vocabulary_), embedding_size=embedding_dimension,
                              l2_reg_lambda=FLAGS.l2_reg_lambda, weights=class_weights_for_cross_entropy,
                              rel_pos_embedding_size=FLAGS.rel_pos_embedding_size,
                              rel_pos_cardinality=relative_positions_cardinality,
                              lstm_units=FLAGS.lstm_units, pos_tags_embedding_size=FLAGS.pos_tags_embedding_size,
                              pos_tags_cardinality=pos_tags_cardinality,
                              with_eye_tracking=cfg["features"]["gaze"], et_features_size=et.shape[2],
                              et_number_of_bins=cfg["features"]["binned"],
                              et_embedding_dimension=FLAGS.et_embedding_dimension,
                              with_eeg=cfg["features"]["eeg"], eeg_features_size=eeg.shape[2], use_normalization_layer=True)

                cnn = TextCNN(sequence_length=x_train.shape[1], num_classes=y_train.shape[1],
                              vocab_size=len(vocab_processor.vocabulary_), embedding_size=embedding_dimension,
                              filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))), num_filters=FLAGS.num_filters,
                              l2_reg_lambda=FLAGS.l2_reg_lambda, weights=class_weights_for_cross_entropy,
                              rel_pos_embedding_size=FLAGS.rel_pos_embedding_size,
                              rel_pos_cardinality=relative_positions_cardinality,
                              pos_tags_embedding_size=FLAGS.pos_tags_embedding_size,
                              pos_tags_cardinality=pos_tags_cardinality,
                              with_eye_tracking=cfg["features"]["gaze"], et_features_size=et.shape[2],
                              et_number_of_bins=cfg["features"]["binned"],
                              et_embedding_dimension=FLAGS.et_embedding_dimension,
                              with_eeg=cfg["features"]["eeg"], eeg_features_size=eeg.shape[2])

                # Define Training procedure
                global_steps = [tf.Variable(0, name="global_step_1", trainable=False),
                                tf.Variable(0, name="global_step_2", trainable=False)]
                optimizers = [tf.train.AdamOptimizer(rnn.learning_rate), tf.train.AdamOptimizer(cnn.learning_rate)]
                grads_and_vars = [optimizers[0].compute_gradients(rnn.loss), optimizers[1].compute_gradients(cnn.loss)]
                train_ops = [optimizers[0].apply_gradients(grads_and_vars[0], global_step=global_steps[0]),
                             optimizers[1].apply_gradients(grads_and_vars[1], global_step=global_steps[1])]

                # Initialize all variables
                sess1.run(tf.global_variables_initializer())
                sess2.run(tf.global_variables_initializer())
                if initW is not None:
                    # Overwrite initialized random value with the embeddings
                    sess1.run(rnn.W.assign(initW))
                    sess2.run(cnn.W.assign(initW))

                # Generate batches
                batches_rnn = data_helpers.batch_iter(
                    list(zip(x_train, relative_positions_train[0], relative_positions_train[1], pos_tags_train,
                             sequence_lengths_train, et_train, eeg_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs_rnn, shuffle=True)

                batches_cnn = data_helpers.batch_iter(
                    list(zip(x_train, relative_positions_train[0], relative_positions_train[1], pos_tags_train,
                             sequence_lengths_train, et_train, eeg_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs_cnn, shuffle=True)

                # Training loop. For each batch...
                while True:

                    batch_rnn = next(batches_rnn, None)
                    if batch_rnn is not None:
                        x_batch, rel_pos_first_batch, rel_pos_second_batch, pos_tags_batch, sequence_lengths_batch, et_batch, eeg_batch, y_batch = zip(
                            *batch_rnn)
                        train_step_rnn(x_batch, rel_pos_first_batch, rel_pos_second_batch, pos_tags_batch,
                                       sequence_lengths_batch, et_batch, eeg_batch, y_batch,
                                       FLAGS.learning_rate, global_steps[0], train_ops[0], FLAGS.dropout_keep_prob, sess1, rnn)

                    batch_cnn = next(batches_cnn, None)
                    if batch_cnn is not None:
                        x_batch, rel_pos_first_batch, rel_pos_second_batch, pos_tags_batch, sequence_lengths_batch, et_batch, eeg_batch, y_batch = zip(
                            *batch_cnn)
                        train_step_cnn(x_batch, rel_pos_first_batch, rel_pos_second_batch, pos_tags_batch, et_batch, eeg_batch, y_batch,
                                       FLAGS.learning_rate, global_steps[1], train_ops[1], FLAGS.dropout_keep_prob, sess2, cnn)

                    if batch_rnn is None and batch_cnn is None:
                        break


                    # count RNN and CNN steps
                    current_step_rnn = tf.train.global_step(sess1, global_steps[0])
                    current_step_cnn = tf.train.global_step(sess2, global_steps[1])
                    current_step = max(current_step_cnn, current_step_rnn)

                    if current_step % FLAGS.evaluate_every == 0:
                        print("\nEvaluation:")
                        step, loss_rnn, probs_rnn = dev_step_rnn(x_dev, relative_positions_dev[0], relative_positions_dev[1],
                                                                 pos_tags_dev, sequence_lengths_dev, et_dev, eeg_dev, y_dev, global_steps[0],
                                                                 1.0, sess1, rnn)

                        _, loss_cnn, probs_cnn = dev_step_cnn(x_dev, relative_positions_dev[0],
                                                              relative_positions_dev[1], pos_tags_dev, et_dev, eeg_dev, y_dev,
                                                              global_steps[1], 1.0, sess2, cnn)

                        # Only for development purposes: Test different methods of weighting CNN and RNN predictions
                        weights = sequence_lengths_dev / (max(sequence_lengths_dev) - min(sequence_lengths_dev))
                        pred_y = [np.argmax(probs_rnn, axis=1), np.argmax(probs_cnn, axis=1),
                                  np.argmax(probs_rnn + probs_cnn, axis=1),
                                  np.argmax((weights * probs_rnn.T + (1 - weights) * probs_cnn.T).T, axis=1),
                                  np.argmax(((1 - weights) * probs_rnn.T + weights * probs_cnn.T).T, axis=1)]
                        names_pred_strategy = ["RNN", "CNN", "Combined", "Weighted", "Weighted Inv."]

                        res_dev = [output.get_fold_results_statistics(y_dev_decoded, p, cfg["datasets"][dataset_name]["categories"], verbose=False) for p in pred_y]

                        print("Loss RNN: {}   Loss CNN: {}".format(loss_rnn, loss_cnn))
                        for n, rd in zip(names_pred_strategy, res_dev):
                            print(
                                "{}: step {}, acc: {}, pr: {}, rec: {}, f1: {}".format(n, current_step, rd['accuracy'],
                                                                                       rd['precision'], rd['recall'],
                                                                                       rd['f1']))

                # Get predictions and probabilities for this network
                _, lossRNN, final_probabilities_rnn = dev_step_rnn(x_dev, relative_positions_dev[0], relative_positions_dev[1],
                                                                   pos_tags_dev, sequence_lengths_dev, et_dev, eeg_dev, y_dev, global_steps[0],
                                                                   1.0, sess1, rnn)

                _, lossCNN, final_probabilities_cnn = dev_step_cnn(x_dev, relative_positions_dev[0],
                                                                   relative_positions_dev[1], pos_tags_dev, et_dev, eeg_dev, y_dev,
                                                                   global_steps[1], 1.0, sess2, cnn)

                # Get average of RNN and CNN final probabilities
                final_probabilities_by_network[:, :, number_of_network] = (
                                                                                  final_probabilities_rnn + final_probabilities_cnn) / 2.0

    final_probabilities_ensemble = np.average(final_probabilities_by_network, axis=2)
    final_class_predictions_ensemble = np.argmax(final_probabilities_ensemble, axis=1)

    # Print accuracy if y_test is defined
    if y_dev is not None:

        # CALCULATE RESULTS
        fold_results = output.get_fold_results_statistics(y_dev_decoded, final_class_predictions_ensemble, cfg["datasets"][dataset_name]["categories"])

        # SAVE PREDICTIONS FOR THE CURRENT FOLD
        print(dataset['target_names'])
        final_reverse_labels.extend(["REVERSE" if "-REV" in dataset['target_names'][y] else "NOT-REVERSE" for y in
                                     final_class_predictions_ensemble])
        final_predictions_strings.extend(
            [dataset['target_names'][y].replace("-REV", "") for y in final_class_predictions_ensemble])

        # target_strings.extend([datasets['target_names'][y] for y in y_dev_decoded])
        # sentences.extend(np.array(datasets['data'])[shuffle_indices][dev_indices].tolist())
        entities.extend(entities_shuffled[dev_indices].tolist())
        # probabilities_of_predicted.extend(["{0:.2f}".format(max(probs)) for probs in final_probabilities_ensemble])

        classification_report, confusion_matrix = output.get_classification_report(y_dev_decoded,
                                                                            final_class_predictions_ensemble, cfg["datasets"][dataset_name]["categories"])
        output.output_fold_results(count_fold, fold_results, classification_report, output_files["result"],
                                   confusion_matrix)

        for key in results:
            results[key].append(fold_results[key])

# ========================= Output =========================

output.produce_result_output(output_files, results)
output.produce_submission_output(output_files, final_predictions_strings, entities)

for key in output_files:
    output_files[key].close()

print("\nexecution time:")
print(datetime.datetime.now() - startTime)
