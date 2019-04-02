import numpy as np

SUBJECT_NAMES = ['ZPH', 'ZMG', 'ZAB', 'ZJN', 'ZKH', 'ZGW', 'ZJS', 'ZKB', 'ZDM', 'ZJM', 'ZKW']

FEATURE_DIMENSIONS = {"RAW_ET":4,
                      "RAW_EEG":105,
                      "ICA_EEG":61
                      }

ANSWERED_SENTENCES_INDICES = np.array([3, 11, 12, 16, 21, 25, 28, 38, 49, 55, 68, 78, 88,
                                       103, 122, 124, 128, 135, 137, 152, 156, 161, 171, 176, 183, 184,
                                       187, 190, 193, 201, 215, 227, 262, 265, 266, 274, 287, 318, 363,
                                       374, 375, 383, 391, 394, 395, 397])

EEG_SENTIMENT_ELECTRODES = [2,3,4,5,7,8,9]



stopwords = {"an", "for", "do", "its", "of","off", "is", "s", "am", "or", "as",
             "from", "him", "each", "the", "are", "we", "these", "his", "me",
             "were", "her", "this", "our", "their", "up", "to", "ours", "she",
             "at", "them", "and", "in", "on", "that", "so", "did", "now", "he",
             "you", "has", "i", "t", "my", "a", "by", "it"}