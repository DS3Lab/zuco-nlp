import numpy as np
from scipy import io

from feature_engineering import pca
from feature_engineering.feature_helpers import reshape_sentence_features, is_real_word

N_ELECTRODES = 105
N_FREQUENCY_BANDS = 8
MAX_OUTLIER_TOLERANCE = 3


def count_words(word_array):
    return sum(1 for i in range(len(word_array)) if is_real_word(word_array[i]))

def get_fixation_features(fixation):
    avg = np.average(fixation, axis=1)
    avg[np.isnan(avg)] = 0
    return avg

def get_word_features(word):

    # No fixation
    if not hasattr(word, 'rawEEG') or isinstance(word.rawEEG, float):
        #print("Nan")
        return None

    #No fixation
    if not hasattr(word, 'rawEEG') or word.rawEEG.size == 0:
        #print("No fixations")
        return None

    #Single fixation
    if word.rawEEG.ndim == 2:
        #print("Single fixation")
        return get_fixation_features(word.rawEEG)

    #Multiple fixations
    if word.rawEEG.ndim == 1:
        #print("Multiple fixations")
        for f in word.rawEEG:
            if np.all(np.isnan(f)):
                #print("all nan")
                return None
        return np.average([get_fixation_features(fixation) for fixation in word.rawEEG if not np.all(np.isnan(fixation))], axis=0)
        #return np.average([get_fixation_features(fixation) for fixation in word.rawEEG], axis=0)

    print("Failed!")
    return None

def get_power_spectrum_word_features(word, eeg_config):
    if eeg_config == "POWER_SPECTRUM_FEATURES_TRT":
        word_features = np.concatenate([word.TRT_t1, word.TRT_t2, word.TRT_a1, word.TRT_a2, word.TRT_b1, word.TRT_b2, word.TRT_g1, word.TRT_g2])
    elif eeg_config == "POWER_SPECTRUM_FEATURES_FF":
        word_features = np.concatenate([word.FFD_t1, word.FFD_t2, word.FFD_a1, word.FFD_a2, word.FFD_b1, word.FFD_b2, word.FFD_g1, word.FFD_g2])

    word_features[np.isnan(word_features)] = 0
    return word_features

def get_sentence_features(sentence):
    sentence_features = [get_word_features(word) for word in sentence.word if is_real_word(word)]
    return np.array([np.zeros(N_ELECTRODES) if sf is None else sf for sf in sentence_features])

def get_normalized_sentence_features(sentence, normalization_values):
    #Get and normalize word features for each sentence
    mu, sigma = normalization_values['mu'], normalization_values['sigma']
    sentence_features = [get_word_features(word) for word in sentence.word if is_real_word(word)]
    #Replace non-fixated by averages
    sentence_features = np.array([mu if sf is None else sf for sf in sentence_features])
    #Center, remove outliers and rescale
    centered_sentence_features = sentence_features - normalization_values['mu']
    # TODO: assigning mu here appears to be a mistake, perhaps it shall be modified to 0? Will do, consider re-evaluating in future.
    #filtered_sentence_features = np.where(np.abs(centered_sentence_features) < MAX_OUTLIER_TOLERANCE * sigma, centered_sentence_features, mu)
    filtered_sentence_features = np.where(np.abs(centered_sentence_features) < MAX_OUTLIER_TOLERANCE * sigma, centered_sentence_features, 0)
    return filtered_sentence_features/(MAX_OUTLIER_TOLERANCE*sigma)

def get_power_spectrum_sentence_features(sentence, eeg_config):
    sentence_features = [get_power_spectrum_word_features(word, eeg_config) for word in sentence.word if is_real_word(word)]
    sentence_features_with_value = [sf for sf in sentence_features if sf.size>0]
    if len(sentence_features_with_value) > 0:
        default_value = np.mean([sentence_features_with_value], axis=1)
    else:
        default_value = np.zeros(N_ELECTRODES*N_FREQUENCY_BANDS)
    return np.vstack([default_value if sf.size==0 else sf for sf in sentence_features])

def get_normalization_values(data):
    #Get average and std for each electrode
    sentence_features = []
    count = 0
    for sentence in data:
        #print("Sentence position: " + str(count))
        sentence_features.append([get_word_features(word) for word in sentence.word if is_real_word(word)])
        #sentence_features.append([get_word_features(word) for word in sentence.word if is_real_word(word) and hasattr(word, 'rawEEG') and word.rawEEG.size != 0])
        count += 1

    mu, sigma = [], []

    for n_electrode in np.arange(N_ELECTRODES):
        all_voltages_electrode = []
        for sentence in sentence_features:
            all_voltages_electrode.extend([word[n_electrode] for word in sentence if word is not None])
        mu.append(np.average(all_voltages_electrode))
        sigma.append(np.max([np.std(all_voltages_electrode), 0.0001]))

    return np.array(mu), np.array(sigma)

def get_eeg_features_single_subject(file_name, eeg_config, sentence_numbers, pca_dimension, max_sequence_length):
    """
    Extract requested subject word-level EEG features from file at path file_name for all sentences

    :param file_name:           (str)
    :param eeg_config:          (int)   idx of the required configuration, see eeg_configuration.py
    :param sentence_numbers:    (list)  list of integers pointing at the sentences to extract
    :param pca_dimension:       (int)   Number of PC to be extracted via PCA, if -1 no PCA is performed

    :return:
        reshaped_features:  (array) np.array of EEGs of shape max_EEG_length*n_words padded by zeros
    """
    print(file_name)
    data = io.loadmat(file_name, squeeze_me=True, struct_as_record=False)['sentenceData']
    if sentence_numbers is not None:
        data = data[sentence_numbers]

    if eeg_config == "RAW_NORMALIZED_FEATURES":
        print("extracting RAW NORMALIZED FEATURES")
        normalization_values = {}
        normalization_values['mu'], normalization_values['sigma'] = get_normalization_values(data)
        sentence_features = [get_normalized_sentence_features(sentence, normalization_values) for sentence in data]
    elif eeg_config == "RAW_FEATURES":
        print("extracting RAW FEATURES")
        sentence_features = [get_sentence_features(sentence) for sentence in data]
    elif eeg_config == "POWER_SPECTRUM_FEATURES_TRT" or eeg_config == "POWER_SPECTRUM_FEATURES_FFD":
        print("extracting POWER SPECTRUM FEATURES")
        sentence_features = [get_power_spectrum_sentence_features(sentence, eeg_config) for sentence in data]

    if pca_dimension>0:
        sentence_features = pca.do_pca(sentence_features, pca_dimension)
    # Zero padding for shorter sentences
    reshaped_features = reshape_sentence_features(sentence_features, max_sequence_length)
    return reshaped_features


def get_eeg_features(eeg_subject_from, eeg_config, sentence_numbers, pca_dimension, task, max_sequence_length, matlab_files):

    features_by_subject = []
    for subject in eeg_subject_from:
        print("Reading EEG data for subject: " + subject)
        file_name = matlab_files + "results" + subject + "_" + task + ".mat"
        features_by_subject.append(get_eeg_features_single_subject(file_name, eeg_config, sentence_numbers, pca_dimension, max_sequence_length))

    return np.average(features_by_subject, axis=0)


