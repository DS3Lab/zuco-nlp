import numpy as np
import data_loading_helpers as dlh
import json
import pandas as pd
import word_embedding_helpers as weh
import tflearn
import re


def reshape_sentence_features(sentence_features, max_len = None):
    """
    Pads the end of each sentence shorter than the max length with "0" which has no place in Vocab

    :param sentence_features:   (list)  List of arrays, each representing one sentence composed of integers

    :return:
        sentences_array:    (array) Array with ncol = max signal length, nrow = n sentences
    """
    max_length = max_len if max_len is not None else np.max([sf.shape[0] for sf in sentence_features])
    padded_sentences = [np.pad(sf, ((0, max_length - sf.shape[0]), (0, 0)), 'constant') for sf in sentence_features]
    sentences_array = np.stack(padded_sentences, axis=0)
    return sentences_array


def is_real_word(word):
    return re.search('[a-zA-Z0-9]', word)

class united_data_box:
    def __init__(self, data_boxes_list):
        self.input_config = data_boxes_list[0].input_config
        self.placeholder_fillers = {}
        for data_name in data_boxes_list[0].placeholder_fillers.keys():
            data_list = [data_box.placeholder_fillers[data_name] for data_box in data_boxes_list]
            self.placeholder_fillers["data_name"] = np.concatenate(data_list, 0)

        self.sentence_numbers = []
        for data_box in data_boxes_list:
            self.sentence_numbers.append(data_box.sentence_numbers + len(self.sentence_numbers))



def strip_punctuation(word):
    word = word.strip(".")
    word = word.strip(",")
    word = word.strip("?")
    word = word.strip("!")
    word = word.strip(")")
    word = word.strip("(")
    word = word.lower()

    return word


def get_norm_vals(array):
    norm_vals = {}
    norm_vals["means"] = array.mean(axis = 0)
    norm_vals["stdevs"] = array.std(axis=0)
    norm_vals["stdevs"] = np.where(array.std(axis=0) > 0.0001, array.std(axis=0), 0.0001)
    return norm_vals

def normalize_vec(vec, norm_vals):
    return (vec - norm_vals["means"])/norm_vals["stdevs"]

def normalize_embeddings(embeddings_dict):
    all_embs = np.array([embeddings_dict[word] for word in embeddings_dict.keys()])
    norm_vals = get_norm_vals(all_embs)
    embeddings_dict = {word : normalize_vec(embeddings_dict[word], norm_vals) for word in embeddings_dict.keys()}
    return embeddings_dict

class stts_data_box(object):
    def __init__(self, input_config, sentences_data_dir = {"all" : "stts_all_sentence_level.csv"}, vocab = None, max_document_length = None, seed = 100):
        np.random.seed(seed)

        self.input_config = input_config

        sentences_data = []
        for data_name in sentences_data_dir.keys():
            sentences_data_i = pd.read_csv("SST_data/" + sentences_data_dir[data_name], header=None)
            sentences_data_i["NAME"] = data_name
            sentences_data.append(sentences_data_i)
        sentences_data = pd.concat(sentences_data)

        self.data_name = np.array(sentences_data["NAME"])
        # TODO: This gives error, fix it checking how to add nothing
        self.sentences = [[strip_punctuation(word) for word in sentence.split(" ") if is_real_word(word)]
                          for sentence in sentences_data.iloc[:, 0]]
        rebuilt_sentences = [" ".join(sentence) for sentence in self.sentences]
        self.targets = sentences_data.iloc[:, 1]

        max_sentence_length = max_document_length if max_document_length is not None else max([len(sentence) for sentence in self.sentences])
        print("Max sentence length is: {}".format(max_sentence_length))

        if self.input_config["WORD_EMBEDDINGS"] == True:
            self.vocab_processor = tflearn.data_utils.VocabularyProcessor(max_sentence_length, min_frequency=3, vocabulary = vocab)

        if self.input_config["EYE_TRACKING"] == True:
            with open("embedding_data/type_dict_gaze.json", "r") as f:
                self.et_dictionary = json.load(f)
            if self.input_config["NORMALIZE_ET"] == True:
                self.et_dictionary = normalize_embeddings(self.et_dictionary)

        if self.input_config["EEG_SIGNAL"] == True:
            with open("embedding_data/type_dict_eeg.json", "r") as f:
                self.eeg_dictionary = json.load(f)
            if self.input_config["NORMALIZE_EEG"] == True:
                self.eeg_dictionary = normalize_embeddings(self.eeg_dictionary)

        self.targets_map = {"POSITIVE": 2, "NEUTRAL": 1, "NEGATIVE": 0}
        self.sentence_numbers = []
        self.placeholder_fillers = {}
        self.placeholder_fillers["SEQUENCE_LENGTHS"] = []
        we_embeddings = list(self.vocab_processor.fit_transform(rebuilt_sentences))

        et_embeddings = []
        eeg_embeddings = []
        count_i = 0
        for sentence in self.sentences:
            self.sentence_numbers.append(count_i)
            count_i += 1
            self.placeholder_fillers["SEQUENCE_LENGTHS"].append(len(sentence))
            et_list = []
            eeg_list = []
            for word in sentence:
                if self.input_config["EYE_TRACKING"] == True:
                    et_list.append(self.load_word_et(word))
                if self.input_config["EEG_SIGNAL"] == True:
                    eeg_list.append(self.load_word_eeg(word))

            if self.input_config["EYE_TRACKING"] == True:
                et_embeddings.append(np.array(et_list))
            if self.input_config["EEG_SIGNAL"] == True:
                eeg_embeddings.append(np.array(eeg_list))

        self.create_unified_data(we_embeddings, et_embeddings, eeg_embeddings, max_len = max_sentence_length)

        self._load_initial_we()

        if input_config["BINARY_CLASSIFICATION"]:
            self._drop_neutrals()

        self.shuffle_data()

    def create_unified_data(self, we_embeddings, et_embeddings, eeg_embeddings, max_len = None):
        if self.input_config["WORD_EMBEDDINGS"] == True:
            self.placeholder_fillers["WORD_IDXS"] = np.array(we_embeddings)
            print("we_embeddings have therefore shape: {}".format(self.placeholder_fillers["WORD_IDXS"].shape))
        if self.input_config["EYE_TRACKING"] == True:
            self.placeholder_fillers["ET"] = reshape_sentence_features(et_embeddings, max_len = max_len)
        if self.input_config["EEG_SIGNAL"] == True:
            self.placeholder_fillers["EEG"] = reshape_sentence_features(eeg_embeddings, max_len = max_len)

        neg_sent = np.array(list(map(lambda x: x == "NEGATIVE", self.targets)))
        neg_sent = np.reshape(neg_sent, [neg_sent.shape[0], 1])
        neut_sent = np.array(list(map(lambda x: x == "NEUTRAL", self.targets)))
        neut_sent = np.reshape(neut_sent, [neut_sent.shape[0], 1])
        pos_sent = np.array(list(map(lambda x: x == "POSITIVE", self.targets)))
        pos_sent = np.reshape(pos_sent, [pos_sent.shape[0], 1])
        self.placeholder_fillers["TARGETS"] = np.concatenate([neg_sent, neut_sent, pos_sent], 1).astype(np.int16)

        self.placeholder_fillers["SEQUENCE_LENGTHS"] = np.array(self.placeholder_fillers["SEQUENCE_LENGTHS"])

        self.sentence_numbers = np.array(self.sentence_numbers)

    def _drop_neutrals(self):
        neutral_targets = self.placeholder_fillers['TARGETS'][:, 1]
        non_neutral_idxs = np.where(neutral_targets == 0)[0]
        self.placeholder_fillers['TARGETS'] = self.placeholder_fillers['TARGETS'][:, [0, 2]]
        self.sample_data_from_idxs(non_neutral_idxs)

    def shuffle_data(self, seed=123):
        np.random.seed(seed)
        n_sentences = len(self.placeholder_fillers['SEQUENCE_LENGTHS'])
        shuffled_idxs = np.random.permutation(np.arange(n_sentences))
        self.sample_data_from_idxs(shuffled_idxs)

    def oversample_underreepresented_classes(self, per_class_amount = None, seed = 123):
        np.random.seed(seed)
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

    def _oversample_class(self, class_to_oversample, oversample_amount):
        class_i_n = self.placeholder_fillers["TARGETS"][:, class_to_oversample]
        class_i_idxs = np.where(class_i_n == 1)[0]
        n_class_i_idxs = len(class_i_idxs)
        random_sample = np.random.randint(0, n_class_i_idxs, oversample_amount).tolist()
        idxs_to_repeat = [class_i_idxs[i] for i in random_sample]
        return idxs_to_repeat

    def sample_data_from_idxs(self, idxs):
        self.sentence_numbers = self.sentence_numbers[idxs]
        self.data_name = self.data_name[idxs]

        for data_name in self.placeholder_fillers.keys():
            self.placeholder_fillers[data_name] = self.placeholder_fillers[data_name][idxs]

    def _load_initial_we(self):
        # TODO: check how this works
        self.initial_word_embeddings = weh.get_word_embeddings(vocab_processor=self.vocab_processor,
                                                               filename=self.input_config['WORD_EMBEDDINGS_PATH'],
                                                               binary_classification=self.input_config['BINARY_CLASSIFICATION'],
                                                               labels_from_subject=None,
                                                               save_filename=self.input_config['Config_name'])

    def load_word_et(self, word, dummy_et=np.zeros(5)):
        et_emb = self.et_dictionary.get(word, dummy_et)
        return et_emb

    def load_word_eeg(self, word, dummy_eeg=np.zeros(105)):
        eeg_emb = self.eeg_dictionary.get(word, dummy_eeg)
        return eeg_emb

    def extract_data_idxs_from_sentence_numbers(self, sentence_numbers):
        data_idxs = np.where(np.in1d(self.sentence_numbers, sentence_numbers))[0]
        return data_idxs
