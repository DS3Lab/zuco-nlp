"""
Constants shared across files.
"""
import re

# special tokens and number regex
UNK = '_UNK'  # unk/OOV word/char
WORD_START = '<w>'  # word star
WORD_END = '</w>'  # word end
NUM = 'NUM'  # number normalization string
NUMBERREGEX = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")

# tasks
POS = 'pos'  # part-of-speech tagging
CHUNK = 'chunk'  # chunking
SRL = 'srl'  # semantic role labeling
# --- tasks for ZuCo data
# NLP tasks
NER = 'ner'  # named entity recognition
SENTI2 = 'sentiment2'  # sentiment analysis (phrase level)
SENTI3 = 'sentiment3'  # sentiment analysis (phrase level)
REX = 'relext'  # relation extraction

# Eye-tracking tasks
MFD = 'mfd'  # mean fixation duration
FIXP = 'fixp'  # fixation probability
NFIX = 'nfix'  # number of fixations
FFD = 'ffd'  # first fixation duration
TRT = 'trt'  # total rading time

# EEG tasks
EEG_T = 'eeg_t'  # theta
EEG_A = 'eeg_a'  # alpha
EEG_B = 'eeg_b'  # beta
EEG_G = 'eeg_g'  # gamma

# EEG+ET
ALL = 'all'

# ------
FREQ = 'freq'

TASK_NAMES = [POS, SENTI2, SENTI3, NER, REX, EEG_T, EEG_A, EEG_B, EEG_G, NFIX, FIXP, TRT, FFD, ALL, MFD, FREQ]

# domains
#DOMAINS = ['bc', 'bn', 'mz', 'nw', 'wb', 'tc', 'pt', 'mr', 'zab', 'freq', 'zph', 'zjn', 'zjm', 'zkw', 'avg', 'freq8', 'zdm', 'zkh', 'zkb', 'zdn', 'zjs', 'zgw', 'zmg', 'concat', 'concat_et', 'concat_eeg', 'concat_all', 'avg_et', 'avg_all', 'avg_eeg']
DOMAINS = ['baseline', 'avg']

# model files
MODEL_FILE = 'sluice_net.model'
PARAMS_FILE = 'sluice_net_params.pkl'

# optimizers
SGD = 'sgd'
ADAM = 'adam'

# type of layer connections
STITCH = 'stitch'
CONCAT = 'concat'
SKIP = 'skip'
NONE = 'none'

# cross-stitch and layer-stitch initialization schemes
BALANCED = 'balanced'
IMBALANCED = 'imbalanced'
