import numpy as np
from scipy import io
from feature_engineering.feature_helpers import reshape_sentence_features, is_real_word

# Eye-tracking features: First Fixation Duration (FFD), Gaze Duration (GD), Go Past Time (GPT), Total Reading Time (TRT), Number of Fixations (nFixations)
N_ET_FEATURES = 5


def get_word_features(word):
    if type(word.FFD) == int:
        return np.array([word.nFixations, word.FFD, word.GD, word.GPT, word.TRT])

    return np.zeros(N_ET_FEATURES)


def get_normalization_values(data):
    # Get average and std for each electrode
    sentence_features = []
    for sentence in data:
        sentence_features.append([get_word_features(word) for word in sentence.word if is_real_word(word)])

    mu, sigma = [], []

    for n_feature in np.arange(N_ET_FEATURES):
        all_values_this_feature = []
        for sentence in sentence_features:
            all_values_this_feature.extend([word[n_feature] for word in sentence])
        mu.append(np.average(all_values_this_feature))
        sigma.append(np.max([np.std(all_values_this_feature), 0.0001]))

    return np.array(mu), np.array(sigma)


def get_normalized_sentence_features(sentence, normalization_values):
    # Get and normalize word features for each sentence
    mu, sigma = normalization_values['mu'], normalization_values['sigma']
    sentence_features = [get_word_features(word) for word in sentence.word if is_real_word(word)]
    sentence_features = np.array(sentence_features)
    # Center and rescale
    centered_sentence_features = sentence_features - normalization_values['mu']

    return centered_sentence_features/sigma


def get_binned_sentence_features(data, number_of_bins):
    sentence_features = []
    for sentence in data:
        sentence_features.append([get_word_features(word) for word in sentence.word if is_real_word(word)])

    bins = []

    for n_feature in np.arange(N_ET_FEATURES):
        all_values_this_feature = []
        for sentence in sentence_features:
            all_values_this_feature.extend([word[n_feature] for word in sentence])
        bin = np.percentile(all_values_this_feature, q=np.linspace(100/number_of_bins, 100, number_of_bins))
        bin[number_of_bins-1] += 1
        bins.append(bin)

    binned_sentence_features = []
    for sentence in sentence_features:
        sentence_2d_array = np.array(sentence)
        for n_feature in np.arange(N_ET_FEATURES):
            sentence_2d_array[:,n_feature] = np.digitize(sentence_2d_array[:, n_feature], bins[n_feature])
        binned_sentence_features.append(sentence_2d_array.astype(int))

    return binned_sentence_features


def get_et_features_single_subject(file_name, sentence_numbers, number_of_bins, max_sequence_length):
    data = io.loadmat(file_name, squeeze_me=True, struct_as_record=False)['sentenceData']
    if sentence_numbers is not None:
        data = data[sentence_numbers]

    if not number_of_bins:
        print("extracting RAW NORMALIZED FEATURES")
        # note: this is not RAW_ET! it is just the features, but not binned!
        normalization_values = {}
        normalization_values['mu'], normalization_values['sigma'] = get_normalization_values(data)
        sentence_features = [get_normalized_sentence_features(sentence, normalization_values) for sentence in data]
    else:
        print("extracting BINNED FEATURES: ", number_of_bins, " bins")
        sentence_features = get_binned_sentence_features(data, number_of_bins)

    return reshape_sentence_features(sentence_features, max_sequence_length)


def get_eye_tracking_features(number_of_bins, sentence_numbers, subject_names, task, max_sequence_length, matlab_files):

    features_by_subject = []
    for subject in subject_names:
        print("Reading ET data for subject: " + subject)
        file_name = matlab_files + "results" + subject + "_" + task + ".mat"
        features_by_subject.append(get_et_features_single_subject(file_name, sentence_numbers, number_of_bins, max_sequence_length))

    if number_of_bins:
        return np.median(features_by_subject, axis=0)
    else:
        return np.mean(features_by_subject, axis=0)