import re
import numpy as np
from sklearn.datasets import load_files
import os.path
import shutil
from configuration import new_configs
import word_embedding_helpers as weh
from constants import constants
import tflearn
import pickle as pkl
import tf_modeling as tfm
from constants import constants
import itertools
import pandas as pd
import time

#import topo_lib as tl

N_ELECTRODES = 105
N_ET_VARIABLES = 4

def reshape_sentence_features(sentence_features):
    """
    Pads the end of each sentence shorter than the max length with "0" which has no place in Vocab

    :param sentence_features:   (list)  List of arrays, each representing one sentence composed of integers

    :return:
        sentences_array:    (array) Array with ncol = max signal length, nrow = n sentences
    """
    max_length = np.max([sf.shape[0] for sf in sentence_features])
    padded_sentences = [np.pad(sf, ((0, max_length-sf.shape[0]),(0,0)), 'constant') for sf in sentence_features]
    sentences_array = np.stack(padded_sentences, axis=0)
    return sentences_array

def iterFlatten(root):
    if isinstance(root, (list, tuple)):
        for element in root:
            for e in iterFlatten(element):
                yield e
    else:
        yield root

def compress_eegs(eeg_list, single_vectors = False):
    if single_vectors:
        trt_array = np.concatenate([fixation for fixation in eeg_list if not is_nan_or_none(fixation)], 0)
    else:
        trt_array = np.concatenate(eeg_list, 0)
    return trt_array.mean(0)


def all_arrays(mixed_list):
    list_of_elements = iterFlatten(mixed_list)
    list_of_arrays = [element for element in list_of_elements if element is not None]
    return list_of_arrays


def is_nan_or_none(value):
    if value is None:
        return True
    else:
        return np.isnan(value).all()

def extract_normalization_values(list_of_2Darrays):
    all_sentences_consecutive = np.concatenate(list_of_2Darrays, 0)
    #print(all_sentences_consecutive.shape)
    normalization_values = {}
    zeros = np.zeros(all_sentences_consecutive.shape)
    condition = np.abs(all_sentences_consecutive) < np.inf
    no_extreme_vals = np.where(condition, all_sentences_consecutive, zeros)
    normalization_values['mean'] = np.nanmean(no_extreme_vals, 0)
    #print(normalization_values['mean'].shape)
    normalization_values['stdev'] = np.nanstd(no_extreme_vals, 0)
    normalization_values['stdev'] = np.where(normalization_values['stdev'] != 0, normalization_values['stdev'], 0.001)
    return normalization_values

def normalize_fixations(sentence_word_level_eegs, normalization_values):
    normalized_fixations_list = [normalize_list_of_eegs(fixations, normalization_values) if fixations is not None else None for fixations in sentence_word_level_eegs]
    return normalized_fixations_list

def normalize_list_of_eegs(eegs, normalization_values):
    eegs = [normalize_array(eeg, normalization_values["mean"], normalization_values["stdev"]) for eeg in eegs]
    return eegs

def normalize_array(array, mean, std):
    centered_array = array - mean
    outlier_boundary = 3 * std
    zeros = np.zeros(centered_array.shape)
    centered_array = np.where(np.abs(centered_array) < 3 * std, centered_array, zeros)
    normalized_array = np.divide(centered_array, outlier_boundary)
    #normalized_array = np.where(normalized_array < -1, normalized_array, np.array(-1))
    # centered_array[centered_array < - outlier_boundary] = - outlier_boundary[centered_array < - outlier_boundary]
    return normalized_array


def clean_str(string):
    string = string.replace(".", "")
    string = string.replace(",", "")
    string = string.replace("--", "")
    string = string.replace("`", "")
    string = string.replace("''", "")
    string = string.replace("' ", " ")
    string = string.replace("*", "")
    string = string.replace("\\", "")
    string = string.replace(";", "")
    string = string.replace("- ", " ")
    string = string.replace("/", "-")
    string = string.replace("!", "")
    string = string.replace("?", "")
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()

def load_data_labels(dataset):
    # Split by words
    x_text = dataset['data']
    x_text = [clean_str(sent) for sent in x_text]

    # Generate labels
    labels = []
    for i in range(len(x_text)):
        label = [0 for j in dataset['target_names']]
        label[dataset['target'][i]] = 1
        labels.append(label)
    y = np.array(labels)
    return [x_text, y]

def get_processed_dataset(dataset_path, binary, verbose, labels_from):
    """
    Load and process raw dataframes to a usable format for training

    :param dataset_path:    (str)   Path to the dataset to process
    :param binary:          (bool)  Output sentiment-binary format
    :param verbose:         (bool)  Verbose output
    :param labels_from:     (str)   Name of the subject from whom to take the labels

    :return:
        dataset:            (~dic)  Dictionary-like object containing the data from dataset_path
        x:                  (array) Array of integers, ncol = max phrase length, nrow = number of sentences, each value being the vocab index of the c-th word in the r-th sentence
        x_text:             (list)  List of all sentences
        y:                  (list)  List of lists, each sublist is the OHE vector of the response variable
        vocab_processor:    (tfobj) Converts word idx to word and vice versa
    """

    categories = ["NEGATIVE", "POSITIVE"] if binary else None
    subfolder_name = labels_from or "all"

    dataset = load_files(container_path=dataset_path + "/" + subfolder_name, categories=categories,
                          load_content=True, shuffle=False, encoding='utf-8')

    x_text, y = load_data_labels(dataset)

    # Build vocabulary
    max_sentence_length = max([len(x.split(" ")) for x in x_text])

    # TODO: UNDERSTAND WHAT VOCAB DOES EXACTLY AND WHY IS IT NEEDED!
    vocab_processor = tflearn.data_utils.VocabularyProcessor(max_sentence_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    if(verbose):
        print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
        print("Max. Sentence Length: {:d}".format(max_sentence_length))

    return dataset, x, x_text, y, vocab_processor

def unzip_data_files(zip_path, folder_path, verbose=True):
    """
    Unzip data files

    :param zip_path:        (str)   Path where to find the zipped file to unzip
    :param folder_path:     (str)   Path where to put the unzipped file
    :param verbose:         (bool)  Verbose output

    :return:
    (None)
    """
    if os.path.isdir(folder_path):
        if(verbose):
            print("Deleting " + folder_path + "...")
        #Removes the folder if it already exists
        shutil.rmtree(folder_path)
        if (verbose):
            print(folder_path + " deleted")

    if (verbose):
        print("Unzipping data files...")
    #Unpacks the zipped in zip_path file to a folder in folder_path
    shutil.unpack_archive(zip_path, folder_path)
    if (verbose):
        print("Data files unzipped")

def get_sentence_order(dataset):
    # Find numbers between '/' and '.txt' in filenames, convert them to integers and return a list
    return [int(x) for x in [re.findall('/([0-9]+).txt$', filename)[0] for filename in dataset["filenames"]]]



class last_data_box:
    """
    Class to contain all data loaded for the language model augmented via EEG and ET
    """
    def __init__(self, input_config = new_configs.default_config, seed = 100):
        np.random.seed(seed)
        self.input_config = new_configs.complete_config(input_config)

        if type(self.input_config['SUBJECTS']) != list:
            self.input_config['SUBJECTS'] = [self.input_config['SUBJECTS']]

        self._load_vp_data()


        self.word_idxs = []
        self.et = []
        self.eeg = []
        self.sentence_lengths = []
        self.targets = []
        self.sentence_numbers = []

        self.placeholder_fillers = {}


        for subject in self.input_config['SUBJECTS']:
            print(subject)
            self._load_subject(subject)

        if self.input_config["WORD_EMBEDDINGS"] == True:
            self.initial_word_embeddings = weh.get_word_embeddings(vocab_processor=self.vocab_processor,
                                                                   filename=self.input_config['WORD_EMBEDDINGS_PATH'],
                                                                   binary_classification=self.input_config['BINARY_CLASSIFICATION'],
                                                                   labels_from_subject=None,
                                                                   save_filename=self.input_config['Config_name'])

    def _load_vp_data(self):
        # Load main sentences data
        # unzip_data_files(zip_path=self.DATASET_ZIPPED_FILE_PATH, folder_path=self.DATASETS_PATH, verbose=self.VERBOSE)
        dataset, x, x_text, y, self.vocab_processor = get_processed_dataset(dataset_path=self.input_config['DATASETS_PATH'],
                                                                            binary=self.input_config['BINARY_CLASSIFICATION'],
                                                                            verbose=self.input_config['VERBOSE'],
                                                                            labels_from=None)
        #return dataset, x, x_text, y

    def _load_subject(self, subject):
        subject_data_path = self.input_config['ALL_PREPROCESSED_DATA_PATH'] + 'Sentence_data_' + subject + '.pickle'
        with open(r'{}'.format(subject_data_path), "rb") as subject_file:
            subject_data = pkl.load(subject_file)

        if self.input_config['SENTENCE_LEVEL'] == True:
            self._load_sentence_level_subject(subject_data)
        elif self.input_config['SENTENCE_LEVEL'] == False:
            self._load_word_level_subject(subject_data)
        else:
            raise Exception(str(self.input_config['SENTENCE_LEVEL']) + "is not a valid value for SENTENCE_LEVEL parameter.")

    def _load_sentence_level_subject(self, subject_data):
        raise Exception("Not implemented yet!")

    def _load_word_level_subject(self, subject_data):
        self._load_word_level_subject_reading_order(subject_data)

    def _load_word_level_subject_written_order(self, subject_data):
        raise Exception("Not implemented yet!")


    def _load_word_level_subject_reading_order(self, subject_data):
        # TODO: Make sure if it's reading order it's only one subject


        if self.input_config["WORD_EMBEDDINGS"] == True:
            self.placeholder_fillers["WORD_IDXS"] = self._extract_word_level_idxs_reading_order(subject_data)

        if self.input_config["EYE_TRACKING"] == True:
            self.placeholder_fillers["ET"] = self._extract_word_level_ets_reading_order(subject_data)

        if self.input_config["EEG_SIGNAL"] == True:
            self.placeholder_fillers["EEG"] = self._extract_word_level_eegs_reading_order(subject_data)

    def _extract_word_level_idxs_reading_order(self, subject_data):
        read_sentences = []
        for sentence in subject_data.keys():
            word_level_data = subject_data[sentence]
            reading_order = word_level_data["word_reading_order"]
            word_idxs = self._extract_word_level_idxs(word_level_data)
            reading_order_idxs = word_idxs[reading_order]
            read_sentences.append(reading_order_idxs)
        read_sentences = constant_pad_list_of_arrays(read_sentences, constant = np.nan)
        return read_sentences

    def _extract_word_level_ets_reading_order(self, subject_data):
        eye_tracking = []
        for sentence in subject_data.keys():
            word_level_data = subject_data[sentence]
            reading_order = word_level_data["word_reading_order"]
            single_sentence_et = self._extract_word_level_ets_features(word_level_data)
            single_sentence_et = single_sentence_et[reading_order]
            eye_tracking.append(single_sentence_et)
        eye_tracking = constant_pad_list_of_arrays(eye_tracking, constant = np.nan)
        return eye_tracking

    def _extract_word_level_eegs_reading_order(self, subject_data):
        eeg_signal = []
        for sentence in subject_data.keys():
            word_level_data = subject_data[sentence]
            single_sentence_eeg = self._extract_fixations(word_level_data)
            eeg_signal.append(single_sentence_eeg)
        eeg_signal = constant_pad_list_of_arrays(eeg_signal, constant = np.nan)
        return eeg_signal

    def _extract_word_level_idxs(self, word_level_data):
        word_embedding_idxs = [list(self.vocab_processor.fit_transform([clean_str(word_level_data[word_idx]['content'])]))[0]
                               for word_idx in word_level_data.keys() if type(word_idx) == int]
        word_idxs_array = np.array(word_embedding_idxs)[:, 0]  # Everything else is zeros
        # print(word_idxs_array.shape)
        return word_idxs_array

    def _extract_word_level_ets_features(self, word_level_data):
        word_ets = []
        for word_idx in word_level_data.keys():
            if type(word_idx) == int:
                FFD = word_level_data[word_idx]["FFD"] or np.nan
                GD = word_level_data[word_idx]["GD"] or np.nan
                GPT = word_level_data[word_idx]["GPT"] or np.nan
                TRT = word_level_data[word_idx]["TRT"] or np.nan
                nFix = word_level_data[word_idx]["nFix"] or np.nan
                fixation_ets = np.array([FFD, GD, GPT, TRT, nFix])
                word_ets.append(fixation_ets)
        word_ets = np.array(word_ets)
        return word_ets

    def _extract_fixations(self, word_level_data, eeg_format):
        self.eeg_features_size = constants.FEATURE_DIMENSIONS[eeg_format]
        word_eegs = []
        dummy_eeg = np.empty((self.eeg_features_size,1,))
        dummy_eeg[:] = np.nan
        fixation_word_idxs = word_level_data["word_reading_order"]
        for word_idx in fixation_word_idxs:
            eeg = word_level_data[word_idx].pop(0)
            if not is_nan_or_none(eeg):
                word_eegs.append(eeg)
            else:
                word_eegs.append(dummy_eeg)
        print(word_eegs)
        print([eeg_mat.shape for eeg_mat in word_eegs])
        eegs_matrix = constant_pad_list_of_arrays(word_eegs, constant = np.nan)
        print(eegs_matrix)
        print(eegs_matrix.shape)
        return eegs_matrix

    def keep_sentence_selection(self, selection_of_sentence_numbers):
        # Can be used for downsample or oversample too
        data_idxs_to_keep = np.where(np.isin(self.sentence_numbers, selection_of_sentence_numbers))[0]
        self.sample_data_from_idxs(data_idxs_to_keep)

    def sample_data_from_idxs(self, idxs):
        self.sentence_numbers = self.sentence_numbers[idxs]

        for data_name in self.placeholder_fillers.keys():
            self.placeholder_fillers[data_name] = self.placeholder_fillers[data_name][idxs]


def constant_pad_list_of_arrays(arrays_list, constant = np.nan):
    assert all([len(array.shape) == len(arrays_list[0].shape) for array in arrays_list]), "Arrays must all have the same number of dimensions!"
    array_dimensions = np.array([array.shape for array in arrays_list])
    array_max_dims = array_dimensions.max(0)
    new_arrays = [pad_array(array, array_max_dims, constant = constant) for array in arrays_list]
    final_array = np.array(new_arrays)
    return final_array

def pad_array(array, objective_dims, constant = np.nan):
    padding_dimensions = tuple([(0, objective_dims[i] - array.shape[i]) for i in range(len(objective_dims))])
    new_array = np.pad(array, padding_dimensions, 'constant', constant_values=constant)
    return new_array















class new_data_box:
    """
    Class to contain all data loaded for the language model augmented via EEG and ET
    """
    def __init__(self, input_config = new_configs.default_config, seed = 100):
        np.random.seed(seed)
        self.input_config = new_configs.complete_config(input_config)

        if type(self.input_config['SUBJECTS']) != list:
            self.input_config['SUBJECTS'] = [self.input_config['SUBJECTS']]

        self._load_vp_data()


        self.word_idxs = []
        self.et = []
        self.eeg = []
        self.sentence_lengths = []
        self.targets = []
        self.sentence_numbers = []

        self.placeholder_fillers = {}


        for subject in self.input_config['SUBJECTS']:
            print(subject)
            only_we = self.input_config["WORD_EMBEDDINGS"] and not (self.input_config["EYE_TRACKING"] or self.input_config["EEG_SIGNAL"])
            if only_we:
                subject = None
            # assert (subject is None) != eeg_or_et, 'You cannot have eeg nor et for None subj'
            if subject is None:
                subject = 'ZPH'  # Just a random one to pick the sentence data from!
            self._load_subject(subject)

            self.sentence_lengths.append(self.sentence_level_data['sentence_length'])
            self.targets.append(self.sentence_level_data['targets'])
            self.sentence_numbers.append(np.array(self.sentence_level_data['sentence_numbers']))

            if self.input_config["WORD_EMBEDDINGS"]:
                self.word_idxs.append(self.sentence_level_data['word_idxs'])
            if self.input_config["EYE_TRACKING"]:
                print("\nET shape for subject {} is {}.\n".format(subject, self.sentence_level_data['eye_tracking'].shape))
                self.et.append(self.sentence_level_data['eye_tracking'])
            if self.input_config["EEG_SIGNAL"]:
                print("\nEEG shape for subject {} is {}.\n".format(subject, self.sentence_level_data['eeg_signals'].shape))
                self.eeg.append(self.sentence_level_data['eeg_signals'])

        if self.input_config["JOIN_SUBJECTS_METHOD"] == 'STACK':
            print("Stacking...")
            self.stack_data()
        elif self.input_config["JOIN_SUBJECTS_METHOD"] == 'CONCATENATE':
            print("Concatenating...")
            self.concatenate_data()
        elif self.input_config["JOIN_SUBJECTS_METHOD"] == 'AVERAGE':
            print("Averaging...")
            self.average_data()

        if self.input_config["BINARY_CLASSIFICATION"]:
            self._binarize(binarization_method=self.input_config['BINARY_FORMAT'])

        #if self.input_config['EEG_TO_PIC']:
        #    self.eeg_to_topoplots()

        if self.input_config["WORD_EMBEDDINGS"] == True:
            self.initial_word_embeddings = weh.get_word_embeddings(vocab_processor=self.vocab_processor,
                                                                   filename=self.input_config['WORD_EMBEDDINGS_PATH'],
                                                                   binary_classification=self.input_config['BINARY_CLASSIFICATION'],
                                                                   labels_from_subject=None,
                                                                   save_filename=self.input_config['Config_name'])


    def stack_data(self):
        # This only works because ATM all word-level eeg and et data are squeezed, make sure to then add the padding
        self.placeholder_fillers['SEQUENCE_LENGTHS'] = np.array(list(iterFlatten(self.sentence_lengths)))
        self.sentence_lengths = None
        self.placeholder_fillers['TARGETS'] = np.concatenate(self.targets, 0)
        self.targets = None
        self.sentence_numbers = np.concatenate(self.sentence_numbers, 0)

        if self.input_config["WORD_EMBEDDINGS"]:
            self.placeholder_fillers['WORD_IDXS'] = np.concatenate(self.word_idxs, 0)
            self.word_idxs = None
        if self.input_config["EYE_TRACKING"]:
            self.placeholder_fillers['ET'] = np.concatenate(self.et, 0)
            self.et = None
        if self.input_config["EEG_SIGNAL"]:
            self.placeholder_fillers['EEG'] = np.concatenate(self.eeg, 0)
            self.eeg = None

    def concatenate_data(self):
        # This only works because ATM all word-level eeg and et data are squeezed, make sure to then add the padding
        assert all(self.sentence_numbers[0].tolist() == rest.tolist() for rest in self.sentence_numbers), 'Concatenation failed due to different order of sentences'
        assert self.input_config['SENTENCE_LEVEL'] == False, "Concatenation is not suitable for Sentence Level"

        self.placeholder_fillers['SEQUENCE_LENGTHS'] = np.array(self.sentence_lengths).max(axis = 0)
        self.sentence_lengths = None
        self.placeholder_fillers['TARGETS'] = np.array(self.targets[0])
        self.targets = None
        self.sentence_numbers = np.array(self.sentence_numbers[0])

        if self.input_config["WORD_EMBEDDINGS"]:
            self.placeholder_fillers['WORD_IDXS'] = np.array(self.word_idxs[0]) # The same for all ppl
            self.word_idxs = None
        if self.input_config["EYE_TRACKING"]:
            self.placeholder_fillers['ET'] = np.concatenate(self.et, 2)
            self.et = None
        if self.input_config["EEG_SIGNAL"]:
            print(self.eeg)
            self.placeholder_fillers['EEG'] = np.concatenate(self.eeg, 2)
            self.eeg = None

    def average_data(self):
        # This only works because ATM all word-level eeg and et data are squeezed, make sure to then add the padding
        assert all(self.sentence_numbers[0].tolist() == rest.tolist() for rest in self.sentence_numbers), 'Averaging failed due to different order of sentences'
        assert self.input_config['SENTENCE_LEVEL'] == False, "Averaging is not suitable for Sentence Level"

        self.placeholder_fillers['SEQUENCE_LENGTHS'] = np.array(self.sentence_lengths).max(axis = 0)
        self.sentence_lengths = None
        self.placeholder_fillers['TARGETS'] = np.array(self.targets[0])
        self.targets = None
        self.sentence_numbers = np.array(self.sentence_numbers[0])

        if self.input_config["WORD_EMBEDDINGS"]:
            self.placeholder_fillers['WORD_IDXS'] = np.array(self.word_idxs[0])
            self.word_idxs = None
        if self.input_config["EYE_TRACKING"]:
            self.placeholder_fillers['ET'] = np.array(self.et).mean(axis = 0)
            self.et = None
        if self.input_config["EEG_SIGNAL"]:
            self.placeholder_fillers['EEG'] = np.array(self.eeg).mean(axis = 0)
            self.eeg = None

    def shuffle_data(self, seed = 123):
        np.random.seed(seed)
        n_sentences = len(self.placeholder_fillers['SEQUENCE_LENGTHS'])
        shuffled_idxs = np.random.permutation(np.arange(n_sentences))

        self.placeholder_fillers['SEQUENCE_LENGTHS'] = self.placeholder_fillers['SEQUENCE_LENGTHS'][shuffled_idxs]
        self.placeholder_fillers['TARGETS'] = self.placeholder_fillers['TARGETS'][shuffled_idxs]
        self.sentence_numbers = self.sentence_numbers[shuffled_idxs]

        if self.input_config["WORD_EMBEDDINGS"]:
            self.placeholder_fillers['WORD_IDXS'] = self.placeholder_fillers['WORD_IDXS'][shuffled_idxs]
        if self.input_config["EYE_TRACKING"]:
            self.placeholder_fillers['ET'] = self.placeholder_fillers['ET'][shuffled_idxs]
        if self.input_config["EEG_SIGNAL"]:
            self.placeholder_fillers['EEG'] = self.placeholder_fillers['EEG'][shuffled_idxs]

    def _binarize(self, binarization_method = 'POS_VS_NEG'):
        neutral_targets = self.placeholder_fillers['TARGETS'][:, 1]
        if binarization_method == 'POS_VS_NEG':
            non_neutral_idxs = np.where(neutral_targets == 0)[0]
            self.placeholder_fillers['TARGETS'] = self.placeholder_fillers['TARGETS'][:,[0,2]]
            self.sentence_numbers = self.sentence_numbers[non_neutral_idxs]
            for data_name in self.placeholder_fillers.keys():
                print(self.placeholder_fillers[data_name])
                self.placeholder_fillers[data_name] = self.placeholder_fillers[data_name][non_neutral_idxs]

        elif binarization_method == 'POS_VS_NONPOS':
            self.placeholder_fillers['TARGETS'] = self.placeholder_fillers['TARGETS'][:, [0,2]]
            self.placeholder_fillers['TARGETS'][:, 0] = self.placeholder_fillers['TARGETS'][:, 0] + neutral_targets

        elif binarization_method == 'NEG_VS_NONNEG':
            self.placeholder_fillers['TARGETS'] = self.placeholder_fillers['TARGETS'][:, [0,2]]
            self.placeholder_fillers['TARGETS'][:,1] = self.placeholder_fillers['TARGETS'][:,1] + neutral_targets
        else:
            raise ValueError('binarization_method can only assume the following values: POS_VS_NEG or POS_VS_NONPOS or NEG_VS_NONNEG')

    def extract_data_idxs_from_sentence_numbers(self, sentence_numbers):
        data_idxs = np.where(np.in1d(self.sentence_numbers, sentence_numbers))[0]
        return data_idxs

    def _load_vp_data(self):
        # Load main sentences data
        # unzip_data_files(zip_path=self.DATASET_ZIPPED_FILE_PATH, folder_path=self.DATASETS_PATH, verbose=self.VERBOSE)
        dataset, x, x_text, y, self.vocab_processor = get_processed_dataset(dataset_path=self.input_config['DATASETS_PATH'],
                                                                            binary=self.input_config['BINARY_CLASSIFICATION'],
                                                                            verbose=self.input_config['VERBOSE'],
                                                                            labels_from=None)
        #return dataset, x, x_text, y

    def _load_subject(self, subject):
        subject_data_path = self.input_config['ALL_PREPROCESSED_DATA_PATH'] + 'Sentence_data_' + subject + '.pickle'
        with open(r'{}'.format(subject_data_path), "rb") as subject_file:
            subject_data = pkl.load(subject_file)

        if self.input_config['SENTENCE_LEVEL'] == True:
            self._load_sentence_level_subject(subject_data)
        elif self.input_config['SENTENCE_LEVEL'] == False:
            self._load_word_level_subject(subject_data)
        else:
            raise Exception(str(self.input_config['SENTENCE_LEVEL']) + "is not a valid value for SENTENCE_LEVEL parameter.")

    def _load_sentence_level_subject(self, subject_data):
        self.sentence_level_data = {}
        eeg_format = self.input_config["EEG_SIGNAL_FORMAT"]
        self.sentence_level_data['eeg_signals'] = []
        self.sentence_level_data['sentence_length'] = []
        self.sentence_level_data['targets'] = []
        self.sentence_level_data['sentence_numbers'] = []

        for sentence in subject_data.keys():
            if not is_nan_or_none(subject_data[sentence][eeg_format]):
                self.sentence_level_data['eeg_signals'].append(subject_data[sentence][eeg_format])
                self.sentence_level_data['sentence_length'].append(subject_data[sentence][eeg_format].shape[0])
                self.sentence_level_data['targets'].append(subject_data[sentence]['label'])
                self.sentence_level_data['sentence_numbers'].append(sentence)

        if self.input_config['NORMALIZE_EEG']:
            normalization_values = extract_normalization_values(self.sentence_level_data['eeg_signals'])
            self.sentence_level_data['eeg_signals'] = normalize_list_of_eegs(self.sentence_level_data['eeg_signals'], normalization_values)

        self.sentence_level_data['eeg_signals'] = reshape_sentence_features(self.sentence_level_data['eeg_signals'])


    def _load_word_level_subject(self, subject_data):
        compress_eeg = True# if self.input_config["EEG_SIGNAL_FORMAT"] in ['RAW_EEG', 'ICA_EEG', 'ERR_EEG', 'EMB_EEG'] else False
        compress_et = True# if self.input_config["EYE_TRACKING_FORMAT"] in ['RAW_ET'] else False
        self.sentence_level_data = {}
        self.sentence_level_data['eeg_signals'] = []
        self.sentence_level_data['sentence_length'] = []
        self.sentence_level_data['eye_tracking'] = []
        self.sentence_level_data['word_idxs'] = []
        self.sentence_level_data['targets'] = []
        self.sentence_level_data['sentence_numbers'] = []

        for sentence in subject_data.keys():
            print(sentence)
            word_level_data = subject_data[sentence]["word_level_data"]
            if self.input_config["WORD_EMBEDDINGS"] == True:
                self.sentence_level_data['word_idxs'].append(self._extract_word_level_idxs(word_level_data))
                # TODO: check how this works

            if self.input_config["EYE_TRACKING"] == True:
                self.sentence_level_data['eye_tracking'].append(self._extract_word_level_ets(word_level_data))
            if self.input_config["EEG_SIGNAL"] == True:
                eeg = self._extract_word_level_eegs(word_level_data)
                self.sentence_level_data['eeg_signals'].append(eeg)

            self.sentence_level_data['sentence_length'].append(len([i for i in word_level_data.keys() if type(i) == int]))
            self.sentence_level_data['targets'].append(subject_data[sentence]['label'])
            self.sentence_level_data['sentence_numbers'].append(sentence)



        # Normalize over sentences
        if self.input_config["EYE_TRACKING"] == True and self.input_config["NORMALIZE_ET"] == True:
            arrays_list = all_arrays(self.sentence_level_data['eye_tracking'])
            norm_dict = extract_normalization_values(arrays_list)
            self.sentence_level_data['eye_tracking'] = [normalize_fixations(ets, norm_dict) for ets in
                                                       self.sentence_level_data['eye_tracking']]

        if self.input_config["EEG_SIGNAL"] == True and self.input_config["NORMALIZE_EEG"] == True:
            arrays_list = all_arrays(self.sentence_level_data['eeg_signals'])
            norm_dict = extract_normalization_values(arrays_list)
            self.eeg_means = norm_dict['mean']
            self.eeg_stdevs = norm_dict['stdev']
            self.sentence_level_data['eeg_signals'] = [normalize_fixations(eegs, norm_dict) for eegs in self.sentence_level_data['eeg_signals']]

        # Squeeze to one vector for word
        # TODO: check that padding is present, pad every sentence, then check for sentence level eeg padding at intra subject and intersubject level
        if self.input_config["EYE_TRACKING"] == True and compress_et == True:
            self.sentence_level_data['eye_tracking'] = self.compress_eye_tracking_by_word(self.sentence_level_data['eye_tracking'])

        if self.input_config["EEG_SIGNAL"] == True and compress_eeg == True:
            self.sentence_level_data['eeg_signals'] = self.compress_eeg_signals_by_word(self.sentence_level_data['eeg_signals'])


        if self.input_config["WORD_EMBEDDINGS"] == True:
            self.sentence_level_data['word_idxs'] = [np.pad(sf, (0, self.vocab_processor.max_document_length-sf.shape[0]), 'constant') for sf in self.sentence_level_data['word_idxs']]
        if self.input_config["EYE_TRACKING"] == True:
            self.sentence_level_data['eye_tracking'] = reshape_sentence_features(self.sentence_level_data['eye_tracking'])
        if self.input_config["EEG_SIGNAL"] == True:
            self.sentence_level_data['eeg_signals'] = reshape_sentence_features(self.sentence_level_data['eeg_signals'])


    def _extract_word_level_idxs(self, word_level_data):
        word_embedding_idxs = [list(self.vocab_processor.fit_transform([clean_str(word_level_data[word_idx]['content'])]))[0]
                               for word_idx in word_level_data.keys() if type(word_idx) == int]
        word_idxs_array = np.array(word_embedding_idxs)[:, 0]  # Everything else is zeros
        # print(word_idxs_array.shape)
        return word_idxs_array

    def _extract_word_level_ets(self, word_level_data):
        et_format = self.input_config['EYE_TRACKING_FORMAT']
        # TODO: de-hardcode fixation_data variable
        fixation_data = False
        word_ets = []
        for word_idx in word_level_data.keys():
            if type(word_idx) == int:
                fixation_ets = []
                if fixation_data:
                    for fixation in word_level_data[word_idx][et_format].keys():
                        et = word_level_data[word_idx][et_format][fixation]
                        if not is_nan_or_none(et):
                        # Added due to particular (rare) fixations = [4, 0]
                        # TODO: Extract nFixations aside
                            if len(et.shape) == 2:
                                fixation_ets.append(np.array(et))
                            else:
                                print(et)
                        else:
                            pass
                else:
                    FFD = word_level_data[word_idx]["FFD"] or 0
                    GD = word_level_data[word_idx]["GD"] or 0
                    GPT = word_level_data[word_idx]["GPT"] or 0
                    TRT = word_level_data[word_idx]["TRT"] or 0
                    nFix = word_level_data[word_idx]["nFix"] or 0
                    fixation_ets = np.array([FFD, GD, GPT, TRT, nFix])
                if len(fixation_ets) > 0:
                    word_ets.append(fixation_ets)
                else:
                    word_ets.append(None)
        return word_ets

    def _extract_word_level_eegs(self, word_level_data):
        eeg_format = self.input_config['EEG_SIGNAL_FORMAT']
        word_eegs = []
        for word_idx in word_level_data.keys():
            if type(word_idx) == int:
                fixation_eegs = []
                for fixation in word_level_data[word_idx][eeg_format].keys():
                    eeg = word_level_data[word_idx][eeg_format][fixation]
                    if not is_nan_or_none(eeg):
                        fixation_eegs.append(np.array(eeg))
                        if hasattr(self, 'eeg_features_size'):
                            if self.eeg_features_size != eeg.shape[1]:
                                raise ValueError("All eeg fixations must have the same amount of features!")
                            else:
                                pass
                        else:
                            self.eeg_features_size = eeg.shape[1]
                    else:
                        pass

                if len(fixation_eegs) > 0:
                    word_eegs.append(fixation_eegs)
                else:
                    word_eegs.append(None)
        return word_eegs


    def compress_eeg_signals_by_word(self, eegs):
        # TODO: change this so that it works also for ICA data
        n_var_eeg = self.eeg_features_size
        single_vectors = False

        dummy_eeg = np.zeros(n_var_eeg)
        eegs = [np.array([compress_eegs(fixations, single_vectors = single_vectors) # Fix this so that "single vectors" is not needed
                          if fixations is not None else dummy_eeg for fixations in word_level_eegs])
                for word_level_eegs in eegs]
        return eegs


    def compress_eye_tracking_by_word(self, ets):
        #TODO: add another element with "fixations" isolated from eeg and
        n_var_et = N_ET_VARIABLES
        dummy_et = np.zeros(n_var_et + 1)
        #ets = [np.array([np.concatenate([np.array([len(fixations)]), compress_eegs(fixations)])
        #                 if fixations is not None else dummy_et for fixations in word_level_ets])
        #        for word_level_ets in ets]
        ets = [np.array([fixations if fixations is not None else dummy_et for fixations in word_level_ets])
               for word_level_ets in ets]
        return ets

    def join_to_data_array(self, data_list):
        data_list = [np.array(list_of_word_vectors) for list_of_word_vectors in data_list]
        data_array = reshape_sentence_features(data_list)
        return data_array

    def eeg_to_topoplots(self, topoplot_dim = (40, 40), exp_decay = 10.0):
        #TODO: Generalize topoplot to all formats
        if self.input_config["EEG_SIGNAL"] != True:
            raise Exception("Cannot create EEG topoplots without EEG data!")
        elif self.input_config["EEG_SIGNAL_FORMAT"] != "RAW_EEG":
            raise ValueError("Topoplot creation currently only available for RAW_EEG format.")
        elif self.input_config["JOIN_SUBJECTS_METHOD"] == 'CONCATENATE':
            raise ValueError("Topoplot creation not available with concatenated data.")

        self.placeholder_fillers['EEG'] = tl.convert_ts_array_to_topo_array(self.placeholder_fillers['EEG'],
                                                                            nrow = topoplot_dim[0], ncol = topoplot_dim[1],
                                                                            exp_decay = exp_decay)

    def eeg_to_freq_pictures(self, freq_pic_dim = (40, 40)):
        #TODO: Generalize frequency pictures to all formats
        if self.input_config["EEG_SIGNAL"] != True:
            raise Exception("Cannot create EEG topoplots without EEG data!")
        elif self.input_config["EEG_SIGNAL_FORMAT"] != "RAW_EEG":
            raise ValueError("Topoplot creation currently only available for RAW_EEG format.")
        elif self.input_config["SENTENCE_LEVEL"] != True:
            raise Exception("Topoplot creation currently only available for SENTENCE_LEVEL eeg.")

    def keep_sentence_selection(self, selection_of_sentence_numbers):
        # Can be used for downsample or oversample too
        data_idxs_to_keep = np.where(np.isin(self.sentence_numbers, selection_of_sentence_numbers))[0]
        self.sample_data_from_idxs(data_idxs_to_keep)

    def sample_data_from_idxs(self, idxs):
        self.placeholder_fillers['SEQUENCE_LENGTHS'] = self.placeholder_fillers['SEQUENCE_LENGTHS'][idxs]
        self.placeholder_fillers['TARGETS'] = self.placeholder_fillers['TARGETS'][idxs]
        self.sentence_numbers = self.sentence_numbers[idxs]

        if self.input_config["WORD_EMBEDDINGS"]:
            self.placeholder_fillers['WORD_IDXS'] = self.placeholder_fillers['WORD_IDXS'][idxs]
        if self.input_config["EYE_TRACKING"]:
            self.placeholder_fillers['ET'] = self.placeholder_fillers['ET'][idxs]
        if self.input_config["EEG_SIGNAL"]:
            self.placeholder_fillers['EEG'] = self.placeholder_fillers['EEG'][idxs]

    def oversample_underreepresented_classes(self, per_class_amount = None):
        per_class_totals = np.sum(self.placeholder_fillers['TARGETS'], 0)

        if per_class_amount is not None:
            assert per_class_amount>=max(per_class_totals), "Requested per_class_amount must be larger than current amounts to oversample"
        else:
            per_class_amount = max(per_class_totals)
        idxs_to_select = np.arange(self.placeholder_fillers['TARGETS'].shape[0]).tolist()
        for class_i in range(per_class_totals.shape[0]):
            if per_class_totals[class_i] < per_class_amount:
                idxs_to_select = idxs_to_select + self._oversample_class(class_i, per_class_amount - per_class_totals[class_i])
        self.sample_data_from_idxs(idxs = idxs_to_select)
        return idxs_to_select

    def _oversample_class(self, class_to_oversample, oversample_amount):
        class_i_n = self.placeholder_fillers["TARGETS"][:, class_to_oversample]
        class_i_idxs = np.where(class_i_n == 1)[0]
        n_class_i_idxs = len(class_i_idxs)
        random_sample = np.random.randint(0, n_class_i_idxs, oversample_amount).tolist()
        idxs_to_repeat = [class_i_idxs[i] for i in random_sample]
        return idxs_to_repeat



# TODO: Add loading in fixation order!
def load_all_word_eegs(subject, eeg_format = 'RAW_EEG'):
    subject_data_path = 'Sentence_level_data/Sentence_data_' + subject + '.pickle'
    with open(r'{}'.format(subject_data_path), 'rb') as subject_file:
        subject_data = pkl.load(subject_file)
    fixations_per_word_per_sentence = [[[subject_data[sentence]['word_level_data'][word][eeg_format][fixation]
                                         if not is_nan_or_none(subject_data[sentence]['word_level_data'][word][eeg_format][fixation]) else None
                                         for fixation in subject_data[sentence]['word_level_data'][word][eeg_format].keys()]
                                        if not subject_data[sentence]['word_level_data'][word][eeg_format] is None else None
                                        for word in subject_data[sentence]['word_level_data'].keys()]
                                       for sentence in subject_data.keys()]

    fixations_eegs_and_none = iterFlatten(fixations_per_word_per_sentence)
    fixations_eegs = [fixation for fixation in fixations_eegs_and_none if fixation is not None]
    return fixations_eegs

def create_subject_embeddings(subject):
    eeg_data = load_all_word_eegs(subject)
    sentence_lengths = [array.shape[0] for array in eeg_data]
    eeg_data = reshape_sentence_features(eeg_data)
    return eeg_data, sentence_lengths

class word_eegs_container:
    #ADD ALL OTHER INFOS FOR VISUALIZING
    def __init__(self, input_config):
        self.placeholder_fillers = {}

        eeg_data = load_all_word_eegs(input_config['SUBJECTS'], eeg_format=input_config['EEG_SIGNAL_FORMAT'])
        self.placeholder_fillers['SEQUENCE_LENGTHS'] = np.array([array.shape[0] for array in eeg_data])
        self.placeholder_fillers['EEG'] = reshape_sentence_features(eeg_data).astype(np.float64)

        if input_config['NORMALIZE_EEG'] == True:
            self.normalize_eeg()

        self.shuffle_data()

    def normalize_eeg(self):
        min_stdev = 0.001
        self.eeg_means = self.placeholder_fillers['EEG'].mean(axis = (0, 1))
        self.eeg_features_size = self.eeg_means.shape[0]
        self.eeg_stdevs = self.placeholder_fillers['EEG'].std(axis=(0, 1))
        self.eeg_stdevs = np.where(self.eeg_stdevs > 0.001, self.eeg_stdevs, min_stdev)
        self.placeholder_fillers['EEG'] = (self.placeholder_fillers['EEG'] - self.eeg_means)/self.eeg_stdevs

    def shuffle_data(self, seed = 123):
        np.random.seed(seed)
        n_sentences = len(self.placeholder_fillers['SEQUENCE_LENGTHS'])
        shuffled_idxs = np.random.permutation(np.arange(n_sentences))

        self.placeholder_fillers['SEQUENCE_LENGTHS'] = self.placeholder_fillers['SEQUENCE_LENGTHS'][shuffled_idxs]
        self.placeholder_fillers['EEG'] = self.placeholder_fillers['EEG'][shuffled_idxs]



def update_subject_ts_emb_and_errors(model, sess, input_config, dat_obj):
    if input_config['SENTENCE_LEVEL'] == True:
        subject_data = update_subject_ts_emb_and_errors_sentence_level(model, sess, input_config, dat_obj)
    else:
        subject_data = update_subject_ts_emb_and_errors_word_level(model, sess, input_config, dat_obj)
    return subject_data


def update_subject_ts_emb_and_errors_word_level(model, sess, input_config, dat_obj):
    subject = input_config["SUBJECTS"]
    eeg_format = input_config["EEG_SIGNAL_FORMAT"]
    subject_data_path = 'Sentence_level_data/Sentence_data_' + subject + '.pickle'
    with open(r'{}'.format(subject_data_path), 'rb') as subject_file:
        subject_data = pkl.load(subject_file)
    for sentence in subject_data.keys():
        print(sentence)
        word_level_data = subject_data[sentence]["word_level_data"]
        for word in word_level_data.keys():
            single_word_data = word_level_data[word][eeg_format]
            if single_word_data is not None:
                errs = {}
                embs = {}
                for fixation in single_word_data.keys():
                    eeg = single_word_data[fixation]

                    if not is_nan_or_none(eeg):
                        seq_len = eeg.shape[0]
                        max_len = model.max_length
                        if input_config['NORMALIZE_EEG']:
                            eeg = (eeg - dat_obj.means) / dat_obj.stdevs
                        padded_eeg = np.pad(eeg, ((0, max_len - seq_len), (0, 0)), 'constant')
                        errors, embedding = tfm.extract_ts_embedding_and_errors(model, sess, input_config, eeg = padded_eeg, seq_length = seq_len)
                        embs[fixation] = embedding
                        errs[fixation] = np.array(errors[range(seq_len)])
                    else:
                        embs[fixation] = eeg
                        errs[fixation] = eeg
                word_level_data[word]["ERR_EEG"] = errs
                word_level_data[word]["EMB_EEG"] = embs
            else:
                word_level_data[word]['ERR_EEG'] = None
                word_level_data[word]['EMB_EEG'] = None
        subject_data[sentence]["word_level_data"] = word_level_data
    with open(r'{}'.format(subject_data_path), 'wb') as subject_file:
        pkl.dump(subject_data, subject_file)
    return subject_data

def update_subject_ts_emb_and_errors_sentence_level(model, sess, input_config, dat_obj):
    subject = input_config["SUBJECTS"]
    eeg_format = input_config["EEG_SIGNAL_FORMAT"]
    subject_data_path = 'Sentence_level_data/Sentence_data_' + subject + '.pickle'
    with open(r'{}'.format(subject_data_path), 'rb') as subject_file:
        subject_data = pkl.load(subject_file)
    for sentence in subject_data.keys():
        eeg = subject_data[sentence][eeg_format]
        if not is_nan_or_none(subject_data[sentence][eeg_format]):
            errs = {}
            embs = {}
            if input_config['NORMALIZE_EEG']:
                eeg = (eeg - dat_obj.eeg_means) / dat_obj.eeg_stdevs
                seq_len = eeg.shape[0]
                max_len = model.max_length
                padded_eeg = np.pad(eeg, ((0, max_len - seq_len), (0, 0)), 'constant')
                errors, embedding = tfm.extract_ts_embedding_and_errors(model, sess, input_config, eeg = padded_eeg, seq_length = seq_len)
                subject_data[sentence]["ERR_EEG"] = errs
                subject_data[sentence]["EMB_EEG"] = embs
            else:
                subject_data[sentence]['ERR_EEG'] = None
                subject_data[sentence]['EMB_EEG'] = None
    with open(r'{}'.format(subject_data_path), 'wb') as subject_file:
        pkl.dump(subject_data, subject_file)
    return subject_data



















