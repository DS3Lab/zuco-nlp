import sys
import os

set_wd_to = "./"

os.chdir(set_wd_to)
sys.path.append(set_wd_to)
import importlib
import data_loading_helpers as dlh
from configuration import new_configs
from sklearn.model_selection import ParameterGrid
from constants import constants
import tf_modeling as tfm
import pandas as pd
import summary_funcs as sf
from SST_experiment import stts_utils as sttsu
import numpy as np
import time


def train_and_test_config(configs, config_name = "UNKNOWN", save_data = True):
    for config in configs:
        complete_config = new_configs.complete_config(config)
        db = sttsu.stts_data_box(complete_config, {"train":"stts_train_sentence_level.csv",
                                                   "dev":"stts_dev_sentence_level.csv",
                                                   "test":"stts_test_sentence_level.csv"})


        db.shuffle_data(121)
        #db.oversample_underreepresented_classes(seed = 123)

        train_idxs = np.where(db.data_name == "train")[0]

        val_idxs = np.where(db.data_name == "dev")[0]
        test_idxs = np.where(db.data_name == "test")[0]



        print("SENTENCES in train are : {}".format(db.sentence_numbers[train_idxs]))
        print("TARGETS in train shape is : {}".format(db.placeholder_fillers["TARGETS"][train_idxs].shape))
        print("TARGETS in train ratios is : {}".format(db.placeholder_fillers["TARGETS"][train_idxs].mean(0)))

        print("SENTENCES in val are : {}".format(db.sentence_numbers[val_idxs]))
        print("TARGETS in val shape is : {}".format(db.placeholder_fillers["TARGETS"][val_idxs].shape))
        print("TARGETS in val ratios is : {}".format(db.placeholder_fillers["TARGETS"][val_idxs].mean(0)))
        model = tfm.AugmentedRNN(input_config=complete_config, vocab_size=len(db.vocab_processor.vocabulary_),
                                 max_sentence_length=db.vocab_processor.max_document_length)

        sess = tfm.prepare_session(complete_config)
        tfm.train_tf(model, db, train_idxs, val_idxs, sess, complete_config, spec_rate = 1.0, only_new_weights = False,
                                n_batch_eval = complete_config['EVALUATE_EVERY'], tensorboard_dir = None, initialize = True)

        db.sample_data_from_idxs(test_idxs)
        print(db.data_name)
        predictions, actuals = tfm.test_tf(model, db, sess, complete_config, spec_rate=1.0)

        print("From Preds VS Targets Accuracy is : {}".format(np.mean(np.equal(np.array(predictions), np.array(actuals)))))

        cv_results_dict = dict(config)
        # TODO: Change with precision_negatives, precision_positives, f1_positives, recall_negatives, recall_positives, f1_negatives, accuracy REMEMBER TO CHANGE SUMMARIES AS WELL
        predictions_df = pd.DataFrame({"Predicted" : predictions, "Target" : actuals})
        for par in cv_results_dict:
            predictions_df[par] = cv_results_dict[par]
        predictions_df["Time"] = time.time()

        data_name = new_configs.default_config['RESULTS_FILE_PATH'] + "Predictions/" + cv_results_dict["Config_name"] + "_pred.csv"
        if os.path.isfile(data_name):
            old_pred = pd.read_csv(data_name, index_col=0)
            columns_to_keep = [col for col in old_pred.columns if "Unnamed" not in col]
            old_pred = old_pred[columns_to_keep]
            complete_pred = pd.DataFrame(pd.concat([old_pred, predictions_df]))
            complete_pred.to_csv(data_name)
        else:
            predictions_df.to_csv(data_name)

        model.reset_graph()


def add_hyperpars_to_optimize(config):
    config['LSTM_UNITS'] = [150, 300]
    config['HIDDEN_LAYER_UNITS'] = [50]
    config['USE_NORMALIZATION_LAYER'] = [True]
    config['ATTENTION_EMBEDDING'] = [True]
    config['L2_REG_LAMBDA'] = [0.00005]
    config['L1_REG_LAMBDA'] = [0.00002, 0.00001]
    config['DROPOUT_KEEP_PROB'] = [.5]
    config['NUM_EPOCHS'] = [10]
    config["INITIAL_LR"] = [0.001]
    config["HALVE_LR_EVERY_PASSES"] = [6]
    config["BATCH_SIZE"] = [64, 128, 256]
    return config


#config = {}
#config["Config_name"] = "All"
#config["WORD_EMBEDDINGS"] = False
#config["EYE_TRACKING"] = False
#config["EEG_SIGNAL"] = True
#config["CONCATENATE_OR_STACK"] = 'STACK'
#config["SUBJECTS"] = constants.SUBJECT_NAMES
#config2 = dict(config)
#config["WORD_EMBEDDINGS"] = True
#config3 = dict(config2)
#config["EYE_TRACKING"] = True
#config["L2"] = True

#configs = [config]#, config2, config3]
#cross_validate_configs(configs, "All", save_data = True)


def extract_configs(args):
    config = {}
    config["Config_name"] = ["Config_stts2_final"]

    if '-b' in args:
        config["BINARY_CLASSIFICATION"] = [True]
        config["Config_name"][0] = config["Config_name"][0] + '_binary'

    if '-we' in args:
        config["Config_name"][0] = config["Config_name"][0] + '_WE'
        config["WORD_EMBEDDINGS"] = [True]
    else:
        config["WORD_EMBEDDINGS"] = [False]

    if '-et' in args:
        config["Config_name"][0] = config["Config_name"][0] + '_ET'
        config["EYE_TRACKING"] = [True]

    if '-eeg' in args:
        config["Config_name"][0] = config["Config_name"][0] + '_EEG'
        config["EEG_SIGNAL"] = [True]

    return config


if __name__ == "__main__":
    par_lists = extract_configs(sys.argv)
    config_name = par_lists.get("Config_name", ["No_name"])[0]

    opt = "-opt" in sys.argv
    if opt:
        par_lists = add_hyperpars_to_optimize(par_lists)

    par_lists['VERBOSE'] = ["-v" in sys.argv]
    save_data = "-s" in sys.argv

    print("Running configuration {}.\n Having parameters {}.".format(config_name, par_lists))
    configs = list(ParameterGrid(par_lists))
    train_and_test_config(configs, config_name, save_data)
