from sklearn.model_selection import KFold
import numpy as np
from sklearn import metrics

def get_train_and_dev_indices(y_shuffled, cv_splits):
    k_fold = KFold(n_splits=cv_splits, random_state=10)
    return k_fold.split(y_shuffled)

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    #for x in data:
        #print(x)
    data = np.array(data)
    data_size = len(data)
    print("data size:")
    print(data_size)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    np.random.seed(1901)
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]



def get_fold_results_statistics(true_y, predictions_y):
    """
    Takes in true values and predicted values and gives back a series of useful metrics

    :param true_y:          (list)  Actual values of Y
    :param predictions_y:   (list)  Predicted values of Y

    :return:
        fold_results:   (dic)   dictionary containing all metrics
    """
    fold_results = {}
    fold_results["accuracy"] =  metrics.accuracy_score(true_y, predictions_y)
    for category in set(true_y):
        predictions_category = list(map(lambda x: x == category, predictions_y))
        true_category = list(map(lambda x: x == category, true_y))
        fold_results[str(category) + "_f1score"] = metrics.f1_score(true_category, predictions_category)
        fold_results[str(category) + "_precision"] = metrics.precision_score(true_category, predictions_category)
        fold_results[str(category) + "_recall"] = metrics.recall_score(true_category, predictions_category)
    #prec_and_rec = micro_avg_prf1(true_category, predictions_category)
    #fold_results["micro_avg_precision"] = prec_and_rec[0]
    #fold_results["micro_avg_recall"] = prec_and_rec[1]

    return fold_results

def micro_avg_prf1(actual, predicted):
    categories = set(actual).union(set(predicted))
    p_num = 0
    p_den = len(actual)
    r_num = 0
    r_den = len(actual)
    for category in categories:
        TP_c = sum([category == true and category == pred for true,pred in zip(actual, predicted)])
        FP_c = sum([category != true and category == pred for true,pred in zip(actual, predicted)])
        TN_c = sum([category != true and category != pred for true,pred in zip(actual, predicted)])
        FN_c = sum([category == true and category != pred for true,pred in zip(actual, predicted)])
        count_true = actual.count(category)
        count_pred = predicted.count(category)
        prec = TP_c/(TP_c + FP_c)
        rec = TP_c/(TP_c + FN_c)
        p_num += prec * count_true
        r_num += rec * count_pred


    micro_precision = p_num / p_den
    micro_recall = r_num / r_den
    return [micro_precision, micro_recall]

