import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
import matplotlib as mpl
from matplotlib import pyplot as plt
import pylab
import seaborn as sns
from constants import constants
from sklearn import metrics
import json
import re

def strip_punctuation(word):
    word = word.strip(".")
    word = word.strip(",")
    word = word.strip("?")
    word = word.strip("!")
    word = word.strip(")")
    word = word.strip("(")
    word = word.lower()

    return word

def is_real_word(word):
    return re.search('[a-zA-Z0-9]', word)


def extract_changing_features(df):
    changing_columns = []
    for column_name in df.columns:
        if len(set(df[column_name]))>1:
            changing_columns.append(column_name)
        else:
            pass
    return changing_columns

def groupby_prec_rec_f1(df, group_by_cols, pred_name, target_name):
    targets = list(set(df[target_name]))
    for target in targets:
        df["TP_" + str(target)] = (df[pred_name] == target) & (df[target_name] == target)
        df["FP_" + str(target)] = (df[pred_name] == target) & (df[target_name] != target)
        df["FN_" + str(target)] = (df[pred_name] != target) & (df[target_name] == target)
    df["Accuracy"] = df[pred_name] == df[target_name]
    confusion_matrix_columns = [col for col in df.columns if (("TP" in col) | ("FP" in col) | ("FN" in col))]
    summary_df = df[confusion_matrix_columns + ["Accuracy"] + group_by_cols].groupby(group_by_cols).mean().reset_index()
    for target in targets:
        summary_df["Precision_" + str(target)] = summary_df["TP_" + str(target)] / (summary_df["TP_" + str(target)] + summary_df["FP_" + str(target)])
        summary_df["Recall_" + str(target)] = summary_df["TP_" + str(target)] / (summary_df["TP_" + str(target)] + summary_df["FN_" + str(target)])
        summary_df["F1_" + str(target)] = 2 * (1 / (( 1 / summary_df["Precision_" + str(target)]) + (1 / summary_df["Recall_" + str(target)])))
    stats = ["Precision", "Recall", "F1"]
    all_stat_columns = []
    for stat in stats:
        stat_columns = [col for col in summary_df.columns if stat in col]
        summary_df[stat] = summary_df[stat_columns].fillna(0).mean(axis = 1)
        all_stat_columns = all_stat_columns + stat_columns
    return summary_df.loc[:, group_by_cols + stats + ["Accuracy"] + all_stat_columns]




