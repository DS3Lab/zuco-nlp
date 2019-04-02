from sklearn.datasets import load_files
import math
import collections
from scipy import io
#from constants import constants
#import nltk
from nltk.corpus import stopwords
import h5py
import numpy as np
import pickle as pkl
import data_loading_helpers as dlh
from constants import constants
import pandas as pd
import re
import time
stopwords.words('english')
stopwords = stopwords.words('english')

def do_print(string, file):
    print(string)
    print(string, file = file)

def is_real_word(word):
    return re.search('[a-zA-Z0-9]', word)

def open_subject_sentence_data(subject):
    filepath = "./Data_to_preprocess/results" + subject + "_SR_ica_new.mat"
    f = h5py.File(filepath)
    return f

def load_matlab_string(matlab_extracted_object):
    extracted_string = u''.join(chr(c) for c in matlab_extracted_object)
    return extracted_string

#bad_channels_data = pd.read_csv("../eeg-sentiment-binary/eeg-quality/badChannelsEEG.csv")
def get_bad_channels(idx, subject, task = "sentiment-binary"):
    # TODO: Discuss and fix bad_channels loading
    session = int(idx/50) + 1
    if task == "sentiment-binary":
        session_file_name1 = "SR" + str(session)
        session_file_name2 = "SNR" + str(session)
    else:
        raise Exception("only sentiment-binary task available so far")
    file_filter = (bad_channels_data['file'] == session_file_name1) | (bad_channels_data['file'] == session_file_name2)
    subject_filter = bad_channels_data['sj'] == subject
    bad_channels = bad_channels_data.loc[file_filter & subject_filter]["bad_channels"]
    bad_channels = bad_channels.values[0].split(" ") if bad_channels.values else None
    return bad_channels

def extract_word_order_from_fixations(fixations_order_per_word):
    if not fixations_order_per_word:
        return []
    fxs_list = [list(fixs) if len(fixs.shape)>1 else [] for fixs in fixations_order_per_word]
    n_tot_fixations = len(sum(fxs_list, []))
    words_fixated_in_order = []
    for fixation_n in range(n_tot_fixations):
        mins_per_word_idx = np.array([min(i) if len(i)>0 else np.nan for i in fxs_list])
        next_word_fixated = int(np.nanargmin(mins_per_word_idx)) # Seems to work like this
        fxs_list[next_word_fixated].remove(min(fxs_list[next_word_fixated]))
        words_fixated_in_order.append(next_word_fixated)
    return words_fixated_in_order


def extract_word_level_data(data_container, word_objects, eeg_float_resolution = np.float16):
    available_objects = list(word_objects)
    contentData = word_objects['content']
    fixations_order_per_word = []
    if "rawEEG" in available_objects:
        rawData = word_objects['rawEEG']
        icaData = word_objects['IC_act_automagic']
        etData = word_objects['rawET']
        # TODO: Double check that this works
        ffdData = word_objects['FFD']
        gdData = word_objects['GD']
        gptData = word_objects['GPT']
        trtData = word_objects['TRT']
        nFixData = word_objects['nFixations']
        fixPositions = word_objects["fixPositions"]
        assert len(contentData) == len(etData) == len(icaData) == len(rawData), "different amounts of different data!!"

        zipped_data = zip(rawData, icaData, etData, contentData, ffdData, gdData, gptData, trtData, nFixData, fixPositions)
        word_level_data = {}
        word_idx = 0
        for raw_eegs_obj, ica_eegs_obj, ets_obj, word_obj, ffd, gd, gpt, trt, nFix, fixPos in zipped_data:
            word_string = load_matlab_string(data_container[word_obj[0]])
            if is_real_word(word_string):
                data_dict = {}
                data_dict["RAW_EEG"] = extract_all_fixations(data_container, raw_eegs_obj[0], eeg_float_resolution)
                data_dict["ICA_EEG"] = extract_all_fixations(data_container, ica_eegs_obj[0], eeg_float_resolution)
                data_dict["RAW_ET"] = extract_all_fixations(data_container, ets_obj[0], np.float32)
                # TODO: Fill the following in!!! <---- IMPORTANT TO DO ASAP
                data_dict["FFD"] = data_container[ffd[0]].value[0, 0] if len(data_container[ffd[0]].value.shape) == 2 else None
                data_dict["GD"] = data_container[gd[0]].value[0, 0] if len(data_container[gd[0]].value.shape) == 2 else None
                data_dict["GPT"] = data_container[gpt[0]].value[0, 0] if len(data_container[gpt[0]].value.shape) == 2 else None
                data_dict["TRT"] = data_container[trt[0]].value[0, 0] if len(data_container[trt[0]].value.shape) == 2 else None
                data_dict["nFix"] = data_container[nFix[0]].value[0, 0] if len(data_container[nFix[0]].value.shape) == 2 else None
                fixations_order_per_word.append(fixPos)

                data_dict["word_idx"] = word_idx
                # TODO: data_dict["word2vec_idx"] = Looked up after through the actual word.
                data_dict["content"] = word_string
                word_idx += 1
                word_level_data[word_idx] = data_dict
            else:
                print(word_string + " is not a real word.")
    else:
        # If there are no word-level data it will be word embeddings alone
        word_level_data = {}
        word_idx = 0
        for word_obj in contentData:
            word_string = load_matlab_string(data_container[word_obj[0]])
            if is_real_word(word_string):
                data_dict = {}
                #TODO: Make sure it was a good call to convert the below from {} to None
                data_dict["RAW_EEG"] = {}
                data_dict["ICA_EEG"] = {}
                data_dict["RAW_ET"] = {}
                data_dict["FFD"] = None
                data_dict["GD"] = None
                data_dict["GPT"] = None
                data_dict["TRT"] = None
                data_dict["nFix"] = None
                data_dict["word_idx"] = word_idx
                data_dict["content"] = word_string
                word_level_data[word_idx] = data_dict
                word_idx += 1
            else:
                print(word_string + " is not a real word.")
        sentence = " ".join([load_matlab_string(data_container[word_obj[0]]) for word_obj in word_objects['content']])
        print("Only available objects for the sentence '{}' are {}.".format(sentence, available_objects))
    word_level_data["word_reading_order"] = extract_word_order_from_fixations(fixations_order_per_word)
    return word_level_data


def extract_all_fixations(data_container, word_data_object, float_resolution = np.float16):
    word_data = data_container[word_data_object]
    fixations_data = {}
    if len(word_data.shape) > 1:
        for fixation_idx in range(word_data.shape[0]):
            fixations_data[fixation_idx] = np.array(data_container[word_data[fixation_idx][0]]).astype(float_resolution)
    return fixations_data


def extract_sentence_level_data(subject, eeg_float_resolution=np.float16):
    # TODO: consider adding smoothing for signals!!
    f = open_subject_sentence_data(subject)
    sentence_data = f['sentenceData']
    rawData = sentence_data['rawData']
    icaData = sentence_data['IC_act_automagic']
    contentData = sentence_data['content']
    wordData = sentence_data['word']
    dataset, x, x_text, y, _ = dlh.get_processed_dataset(dataset_path="data/sentences", binary=False, verbose=True,
                                                         labels_from=None)
    sentence_order = dlh.get_sentence_order(dataset)
    sentence_level_data = {}
    for idx in range(len(rawData)): # raw data is an used but they all should be the same in length (400 for ternary, about 2/3 of that for binary)
        data_dict = {}
        obj_reference_raw = rawData[idx][0]
        data_dict["RAW_EEG"] = np.array(f[obj_reference_raw]).astype(eeg_float_resolution)
        obj_reference_ica = icaData[idx][0]
        data_dict["ICA_EEG"] = np.array(f[obj_reference_ica]).astype(eeg_float_resolution)
        obj_reference_content = contentData[idx][0]
        data_dict["content"] = load_matlab_string(f[obj_reference_content])
        # do_print(data_dict["content"], report_file)
        data_dict["sentence_number"] = idx
        label_idx = np.where(np.array(sentence_order) == idx)[0][0]
        data_dict["label"] = np.array(y[label_idx])
        data_dict["word_embedding_idxs"] = np.array(x[label_idx, :])
        data_dict["label_content"] = dataset['data'][label_idx]
        label_n = np.where(data_dict["label"] == 1)[0][0]
        data_dict["label_name"] = dataset['target_names'][label_n]
        bad_channels = get_bad_channels(idx, subject)
        data_dict["bad_channels"] = bad_channels.split(" ") if type(bad_channels) == str else None
        data_dict["word_level_data"] = extract_word_level_data(f, f[wordData[idx][0]], eeg_float_resolution=eeg_float_resolution)
        sentence_level_data[idx] = data_dict
    return sentence_level_data

def create_all_subjects_data(filename, eeg_float_resolution=np.float16):
    all_subjects_dict = {}
    for subject in constants.SUBJECT_NAMES:
        print(subject)
        all_sentences_info = extract_sentence_level_data(subject, eeg_float_resolution=eeg_float_resolution)
        all_subjects_dict[subject] = all_sentences_info
        subject_file = filename + "_" + subject + ".pickle"
        print("Data saved in file " + subject_file)
        with open(subject_file, "wb") as f:
            pkl.dump(all_sentences_info, f)
    return all_subjects_dict
