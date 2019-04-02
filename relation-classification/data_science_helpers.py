import re
import numpy as np
from sklearn.model_selection import KFold

NONE_CLASS = 3; COMPARE_CLASS = 0
np.random.seed(47820)


def get_train_and_dev_indices(cfg, y_shuffled, filenames, dataset_name, evaluation_dataset_name,shuffle_indices):
    split_type = cfg["train_test_split"]["type"]
    print(split_type)

    if (split_type == "kfold"):
        # Use SKLearn to do the splitting
        k_fold = KFold(n_splits=cfg["train_test_split"]["kfold"]["number_of_folds"], random_state=10, shuffle=False)
        return k_fold.split(y_shuffled)
    elif (split_type == "fixed_split"):
        # Look for the right indices from the filenames, which are assumed to be equal to the numbers of the samples
        start_test_index = cfg["train_test_split"]["fixed_split"][dataset_name][evaluation_dataset_name][
            "start_test_split_index"]
        end_test_index = cfg["train_test_split"]["fixed_split"][dataset_name][evaluation_dataset_name][
            "end_test_split_index"]

        p = re.compile('/([0-9]+).txt$')
        numbers = np.array([int(p.search(filename).group(1)) for filename in filenames])
        test_mask = ((numbers >= start_test_index) & (numbers <= end_test_index))[shuffle_indices]
        test_indices = np.where(test_mask)[0]
        train_indices = np.where(np.logical_not(test_mask))[0]
        return [[train_indices, test_indices]]
    else:
        raise Exception("Train-test split type incorrectly configured. Check config!")


def get_class_weights_for_cross_entropy(y):
    class_counts = np.unique(np.argmax(y, 1), return_counts=True)[1]
    class_probs = class_counts / np.sum(class_counts)
    class_weights_for_cross_entropy = 1 / class_counts
    class_weights_for_cross_entropy = class_weights_for_cross_entropy / np.dot(class_weights_for_cross_entropy,
                                                                               class_probs)
    return class_weights_for_cross_entropy


def prepare_training_data(x_shuffled, relative_positions_shuffled, pos_tags_shuffled, sequence_lengths_shuffled, et_shuffled,
                          eeg_shuffled, y_shuffled, train_indices):
    x_train, relative_positions_train, pos_tags_train, sequence_lengths_train, et_train, eeg_train, y_train = x_shuffled[train_indices], [
        r[train_indices] for r in relative_positions_shuffled], pos_tags_shuffled[train_indices], \
                                                                                         sequence_lengths_shuffled[
                                                                                             train_indices], et_shuffled[train_indices], eeg_shuffled[train_indices], y_shuffled[
                                                                                             train_indices]
    indices = np.arange(len(train_indices))

    return x_train[indices], pos_tags_train[indices], sequence_lengths_train[indices], et_train[indices], \
           eeg_train[indices], y_train[indices], [r[indices] for r in relative_positions_train]
