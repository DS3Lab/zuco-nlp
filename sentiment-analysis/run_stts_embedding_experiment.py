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

# configs = list(ParameterGrid(new_configs.config_WE_ET_EEG)) #_ET_EEG

def cross_validate_configs(configs, config_name = "UNKNOWN", save_data = True):
    results_list = []
    for config in configs:
        complete_config = new_configs.complete_config(config)
        db = sttsu.stts_data_box(complete_config)
        seed = config.get("Random_Seed", 123)
        db.shuffle_data(seed)
        if complete_config["Oversample"]:
            db.oversample_underreepresented_classes(seed=seed)
        db.shuffle_data(seed)

        model = tfm.AugmentedRNN(input_config=complete_config, vocab_size=len(db.vocab_processor.vocabulary_),
                                 max_sentence_length=db.vocab_processor.max_document_length)

        results_file = tfm.cross_validate_config_accuracy(model, data_obj = db, input_config = complete_config,
                                                          create_table=True, specialized_embeddings = True,
                                                          tensorboard_save = False)

        cv_results_dict = dict(config)
        # TODO: Change with precision_negatives, precision_positives, f1_positives, recall_negatives, recall_positives, f1_negatives, accuracy REMEMBER TO CHANGE SUMMARIES AS WELL
        stats = ['Accuracy', 'Precision_neg', 'Recall_neg', 'F1_neg', 'Precision_pos', 'Recall_pos', 'F1_pos']
        for stat in stats:
            cv_results_dict[stat] = results_file[stat].mean()

        results_list.append(cv_results_dict)
        model.reset_graph()

    if save_data:
        summary = pd.DataFrame.from_records(results_list)
        summary_path = new_configs.default_config['RESULTS_FILE_PATH'] + "Summaries/"
        summary_name = "Summary_" + config_name + ".csv"
        if os.path.isfile(summary_path + summary_name):
            old_summary = pd.read_csv(summary_path + summary_name)
            complete_summary = pd.DataFrame(pd.concat([old_summary, summary]))
            complete_summary.to_csv(summary_path + summary_name)
        else:
            summary.to_csv(summary_path + summary_name)

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
    config["Config_name"] = ["Config_stts"]

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

    config["NUMBER_OF_CV_SPLITS"] = [10]
    return config

def optimal_parametrization(config, args):
    config['HIDDEN_LAYER_UNITS'] = [50]
    config['USE_NORMALIZATION_LAYER'] = [True]
    config['ATTENTION_EMBEDDING'] = [True]
    if "-b" in args:
        config['LSTM_UNITS'] = [300]
        config['INITIAL_LR'] = [0.001]
        config['HALVE_LR_EVERY_PASSES'] = [3]

    else:
        config['LSTM_UNITS'] = [150]
        config['INITIAL_LR'] = [0.001]
        config['HALVE_LR_EVERY_PASSES'] = [9]

    config['L2_REG_LAMBDA'] = [0.000]
    config['DROPOUT_KEEP_PROB'] = [.5]
    config['BATCH_SIZE'] = [32]
    config['NUM_EPOCHS'] = [10]
    config["Oversample"] = [True]
    return config

if __name__ == "__main__":
    par_lists = extract_configs(sys.argv)
    config_name = par_lists.get("Config_name", ["No_name"])[0]
    if "-opt" in sys.argv:
        par_lists = optimal_parametrization(par_lists, sys.argv)
    par_lists['VERBOSE'] = ["-v" in sys.argv]
    save_data = "-s" in sys.argv

    print("Running configuration {}.\n Having parameters {}.".format(config_name, par_lists))
    configs = list(ParameterGrid(par_lists))
    cross_validate_configs(configs, config_name, save_data)


