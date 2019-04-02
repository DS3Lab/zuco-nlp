import re
import numpy as np


def reshape_sentence_features(sentence_features, max_seq_length_text):
    # Pad features with zeros given the different sentence lengths
    max_length = np.max([sf.shape[0] for sf in sentence_features])
    if max_seq_length_text > max_length:
        max_length = max_seq_length_text
    padded_sentences = [np.pad(sf, ((0, max_length-sf.shape[0]),(0,0)), 'constant') for sf in sentence_features]
    return np.stack(padded_sentences, axis=0)


def is_real_word(word):
    return re.search('[a-zA-Z0-9]', word.content)