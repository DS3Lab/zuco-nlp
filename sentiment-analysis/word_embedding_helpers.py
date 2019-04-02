import numpy as np
import pickle
import os.path

USED_WORD_EMBEDDINGS_FILE_PATH_TEMPLATE = "./embeddings/used_word_embeddings_{}{}{}.bin"

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)

def get_word_embeddings(vocab_processor, filename, binary_classification, labels_from_subject, save_filename = ""):
    """
    Function to extract the word embeddings for each word. Right now 300-dimensional embeddings are used.

    :param vocab_processor:         (tfobj) Converts word idx to word and vice versa
    :param filename:                (str)   Path of file containing the new embeddings
    :param binary_classification:   (bool)  True if response variable is binary
    :param labels_from_subject:     (str)   Name of the subject from whom to take the labels

    :return:
        used_embeddings:    (array) Pre-used and saved word2vec embeddings of dimension embedding_dimension*STRANGE_DIMENSION
            or
        embedding_vectors:  (array) New downloaded word2vec embeddings, found at the path in parameter filename, dimension vocabulary_length*embedding_dimension

    NOTE: If one wants to use new embeddings for person p, the path must be given and the old saved ones must be erased
    """
    # check if preprocessed file exists
    #classification_type_string = "binary" if binary_classification else "ternary"
    #used_word_embeddings_file_path = USED_WORD_EMBEDDINGS_FILE_PATH_TEMPLATE.format(classification_type_string, labels_from_subject or "", save_filename)
    #if os.path.exists(used_word_embeddings_file_path):
    #    return load_object(used_word_embeddings_file_path)

    # if pre-used file does not exist
    print("\n ABOUT TO LOAD EMBEDDINGS \n")
    embedding_vectors = load_embedding_vectors_word2vec(vocab_processor.vocabulary_, filename, binary=True)
    print("\n EMBEDDINGS LOADED \n")
    #save_object(obj=embedding_vectors, filename=used_word_embeddings_file_path)
    return embedding_vectors

def load_embedding_vectors_word2vec(vocabulary, filename, binary):
    """
    Loads word embeddings for the vocabulary from a downloaded general vocabulary (GoogleNews-vectors-negative300.bin)

    :param vocabulary:  (tfobj) Converts word idx to word and vice versa
    :param filename:    (str)   Path of file containing the new embeddings
    :param binary:      (bool)  Loading method related variable, unrelated to categorical response values

    :return:
        embedding_vectors:  (array) array of floats with word embeddings of shape embedding_dimension*vocab_dimension
    """
    # load embedding_vectors from the word2vec
    encoding = 'utf-8'
    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        # initial matrix with random uniform
        np.random.seed(189)
        embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
        if binary:
            binary_len = np.dtype('float32').itemsize * vector_size
            found_words = 0
            for line_no in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    if ch == b'':
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    if ch != b'\n':
                        word.append(ch)
                word = str(b''.join(word), encoding=encoding, errors='strict')
                idx = vocabulary.get(word)
                if idx != 0:
                    found_words += 1
                    embedding_vectors[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.seek(binary_len, 1)
            print("NUMBER OF WORDS WITH EMBEDDINGS: {}".format(found_words))
            print("NUMBER OF WORDS WITHOUT EMBEDDINGS: {}".format(len(vocabulary)-found_words))
        else:
            for line_no in range(vocab_size):
                line = f.readline()
                if line == b'':
                    raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                parts = str(line.rstrip(), encoding=encoding, errors='strict').split(" ")
                if len(parts) != vector_size + 1:
                    raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
                word, vector = parts[0], list(map('float32', parts[1:]))
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = vector
        f.close()
        return embedding_vectors
