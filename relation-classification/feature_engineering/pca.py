import numpy as np
from sklearn.decomposition import PCA


def do_pca(sentence_features, n_dimensions):
    # Make everything into one big matrix (words x features_per_word)
    sentence_lengths = [sf.shape[0] for sf in sentence_features]
    X = np.concatenate(sentence_features, axis = 0)

    # Transform with PCA
    pca = PCA(n_components=n_dimensions)
    X_transformed = pca.fit_transform(X)

    # Put everything back into shape
    reduced_dim_sentence_features = np.split(X_transformed, np.cumsum(sentence_lengths)[:-1])
    return reduced_dim_sentence_features
