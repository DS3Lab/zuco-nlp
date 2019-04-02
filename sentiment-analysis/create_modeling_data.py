import sys
import os

set_wd_to = "./"

os.chdir(set_wd_to)
sys.path.append(set_wd_to)
from preprocessing import data_creation_utils as dcu
import importlib
import numpy as np
import sys

if __name__ == '__main__':
    if "-s" in sys.argv:
        sys.stdout = open("../eeg-sentiment-binary/Results_files/preprocessing_report.txt",'wt')

    eeg_format = np.float16 if "-low_def" in sys.argv else np.float64
    res_dict = dcu.create_all_subjects_data("Sentence_level_data/Sentence_data", eeg_format)
    if "-s" in sys.argv:
        sys.stdout.close()
        sys.stdout = sys.__stdout__




