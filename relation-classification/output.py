import time
from itertools import groupby
import numpy as np
from sklearn import metrics


def initialize_output_files(evaluation_dataset_name, startTimeForFilenames, subjects):
    result_file = initialize_result_file(evaluation_dataset_name, startTimeForFilenames)['result']
    predictions_file = open("predictions_" + time.strftime("%d.%m.%y-%H.%M", startTimeForFilenames) + "_" + "_".join(subjects), 'w',
                            encoding="utf-8")
    print(evaluation_dataset_name + "\n", file=predictions_file)
    wrong_predictions_detail_file = open(
        "wrongPredictionsDetail_" + time.strftime("%d.%m.%y-%H.%M", startTimeForFilenames) + "_" + "_".join(subjects),
        'w', encoding="utf-8")
    print(evaluation_dataset_name + "\n", file=wrong_predictions_detail_file)
    submission_file = initialize_submission_file(evaluation_dataset_name, startTimeForFilenames)['submission']

    return dict(result=result_file, predictions=predictions_file,
                wrong_predictions_detail=wrong_predictions_detail_file, submission=submission_file)


def initialize_submission_file(evaluation_dataset_name, startTimeForFilenames, subjects):
    submission_file = open("submission_" + time.strftime("%d.%m.%y-%H.%M", startTimeForFilenames)+ "_" + "_".join(subjects), 'w',
                           encoding="utf-8")
    print(evaluation_dataset_name, file=submission_file)
    return dict(submission=submission_file)


def initialize_result_file(evaluation_dataset_name, startTimeForFilenames, subjects):
    result_file = open("results_" + time.strftime("%d.%m.%y-%H.%M", startTimeForFilenames) + "_" + "_".join(subjects), 'w', encoding="utf-8")
    print("DATASET=" + evaluation_dataset_name + "\n", file=result_file)
    return dict(result=result_file)


def get_fold_results_statistics(true_y, predictions_y, target_names, verbose=True):
    fold_results = {}

    fold_results["accuracy"] = metrics.accuracy_score(true_y, predictions_y)

    for metric in ("precision", "recall", "f1"):
        fold_results[metric] = []

    labels = np.arange(len(target_names))

    for label in labels:
        fold_results["precision"].append(metrics.precision_score(true_y, predictions_y, average='micro', labels=[label]))
        fold_results["recall"].append(metrics.recall_score(true_y, predictions_y, average='micro', labels=[label]))
        fold_results["f1"].append(metrics.f1_score(true_y, predictions_y, average='micro', labels=[label]))

    for metric in ("precision", "recall", "f1"):
        fold_results[metric] = np.average(fold_results[metric])

    if verbose:
        print("FOLD RESULTS:")
        for metric in ("precision", "recall", "f1"):
            print("{}: {}".format(metric, fold_results[metric]))

    return fold_results


def output_fold_results(count_fold, results, class_report, result_file, confusion_matrix=None):
    #####OUTPUT FOLD RESULTS
    print("\n\nfold " + str(count_fold) + ":", results["accuracy"], results["precision"], results["recall"], results["f1"],
          file=result_file)
    print(
        "\nEvaluation per class for fold {} (please do NOT consider the averages for analysis):".format(
            count_fold), file=result_file)
    print(class_report, file=result_file)

    if (confusion_matrix is not None):
        print("\nConfusion matrix for fold {}:".format(count_fold), file=result_file)
        print(confusion_matrix, file=result_file)


def produce_full_output(predictions_strings, target_strings, probabilities_of_predicted, sentences, entities, files,
                        results):
    # OFFICIAL SUBMISSION FORMAT OUTPUT
    produce_submission_output(files, predictions_strings, entities)

    all_output = list(zip(predictions_strings, probabilities_of_predicted, target_strings, sentences))

    # DEBUG OUTPUT
    for predicted, prob, real, sentence in all_output:
        print("{} ({}) --> {}:\n{}".format(predicted, prob, real, sentence), file=files["predictions"])

    # WRONG PREDICTIONS DETAIL FOR DEBUG
    all_output_sorted = sorted(all_output, key=lambda tup: (tup[0], tup[2]))
    for predicted_class, lines in groupby(all_output_sorted, lambda tup: tup[0]):
        print("\nPredicted Class: {}\n".format(predicted_class), file=files["wrong_predictions_detail"])
        for line in lines:
            if predicted_class != line[2]:
                print("{} --> {} ({})".format(line[3], line[2], line[1]), file=files["wrong_predictions_detail"])

    print("\n===========================================\n", file=files["wrong_predictions_detail"])

    all_output_sorted = sorted(all_output, key=lambda tup: (tup[2], tup[0]))
    for right_class, lines in groupby(all_output_sorted, lambda tup: tup[2]):
        print("\nCorrect Class: {}\n".format(right_class), file=files["wrong_predictions_detail"])
        for line in lines:
            if right_class != line[0]:
                print("{} -/-> {} ({})".format(line[3], line[0], line[1]), file=files["wrong_predictions_detail"])

    produce_result_output(files, results)

    for key in files:
        files[key].close()


def produce_result_output(files, results):
    # Print final results (averaged over all folds) to file and console
    print("\n --- Final results, averaged over all networks and folds:", file=files["result"])
    print("\naverage", np.average(results["accuracy"]), np.average(results["precision"]), np.average(results["recall"]),
          np.average(results["f1"]))
    print("\naverage", np.average(results["accuracy"]), np.average(results["precision"]), np.average(results["recall"]),
          np.average(results["f1"]), file=files["result"])


def produce_submission_output(files, predictions_strings, entities):
    # Print final results (averaged over all folds) to file and console
    for fps, e in zip(predictions_strings, entities):
        if (fps == "NONE"):
            continue
        reverse_string = "" if (e[2] == "NOT-REVERSE") else ",{}".format(e[2])
        print("{}({},{}{})".format(fps, e[0], e[1], reverse_string), file=files["submission"])


def get_classification_report(true_y, predictions_y, target_names):

    labels = np.arange(len(target_names))

    return metrics.classification_report(true_y, predictions_y, labels=labels, target_names=target_names), \
           metrics.confusion_matrix(true_y, predictions_y)