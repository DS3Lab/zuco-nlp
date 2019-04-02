from sklearn.model_selection import KFold
import numpy as np


def split_folds(filename, outdir, folds):
    """ splits sentences in the input file into folds for training, development and testing"""

    infile = open(filename, "r").read()
    features = infile.split('\n\n')
    print(len(features), " sentences in total.")

    X = np.array(features)

    # Shuffle sentences:
    np.random.seed(123)
    np.random.shuffle(X)

    kf = KFold(n_splits=folds, shuffle=False, random_state=None)
    kf.get_n_splits(X)

    fold = 0
    outfilename = filename.split("/")[-1]

    for train_index, test_index in kf.split(X):

        dev = len(train_index) - int(len(test_index)/2)  # 10% of the data for dev, 20% for test
        dev_index = train_index[dev:]
        new_train_index = train_index[:dev]  # 70% of the data

        # print("TRAIN:", train_index, "TEST:", test_index)
        print("TRAIN sentences: ", len(new_train_index), "DEV sentences: ", len(dev_index), "TEST sentences: ",
              len(test_index))
        X_train, X_dev, X_test = X[new_train_index], X[dev_index], X[test_index]

        train_file = open(outdir + "train-" + str(fold) + "-" + outfilename, "w", encoding="utf-8")
        for tok in X_train:
            print(tok, file=train_file, end="\n\n")
        dev_file = open(outdir + "dev-" + str(fold) + "-" + outfilename, "w", encoding="utf-8")
        for tok in X_dev:
            print(tok, file=dev_file, end="\n\n")
        test_file = open(outdir + "test-" + str(fold) + "-" + outfilename, "w", encoding="utf-8")
        for tok in X_test:
            print(tok, file=test_file, end="\n\n")

        fold += 1


def main():
    infile = "ner/zuco.ner.4eeg.5et.freq.avg.8.tsv"
    outdir = "folds/"

    # Set number of folds
    folds = 5

    print("Splitting folds for ", infile)
    split_folds(infile, outdir, folds)


if __name__ == '__main__':
    main()
