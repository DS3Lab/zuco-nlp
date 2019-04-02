import json
import re
import os
import shutil
import yaml
import numpy as np
from sklearn.datasets import load_files
from tensorflow.contrib import learn


def get_sentence_order(dataset):
    return [int(filename.split("/")[3].split("_")[0]) for filename in dataset["filenames"]]


def read_config_file(filename):
    """Read current config file"""

    with open(filename, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    return cfg


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
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


def get_datasets_zuco(container_path=None, categories=None, load_content=True,
                       encoding='utf-8'):
    """
    Load text files with categories as subfolder names.
    Individual samples are assumed to be files stored a two levels folder structure.
    :param container_path: The path of the container
    :param categories: List of classes to choose, all classes are chosen by default (if empty or omitted)
    :param shuffle: shuffle the list or not
    :param random_state: seed integer to shuffle the dataset
    :return: data and labels of the dataset
    """

    datasets = load_files(container_path=container_path, categories=categories,
                          load_content=load_content, shuffle=False, encoding=encoding)
    return datasets


def load_data_labels(datasets):
    """
    Load data and labels
    :param datasets:
    :return:
    """

    # Split by words
    x_text = datasets['data']
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    labels = []
    for i in range(len(x_text)):
        label = [0 for j in datasets['target_names']]
        label[datasets['target'][i]] = 1
        labels.append(label)
    y = np.array(labels)
    return [x_text, y]


def load_embedding_vectors_word2vec(vocabulary, filename, binary):
    # load embedding_vectors from the word2vec
    encoding = 'utf-8'
    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        # initial matrix with random uniform
        np.random.seed(123)
        embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
        if binary:
            binary_len = np.dtype('float32').itemsize * vector_size
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
                    embedding_vectors[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.seek(binary_len, 1)
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


def load_embedding_vectors_glove(vocabulary, filename, vector_size):
    # load embedding_vectors from the glove
    # initial matrix with random uniform

    np.random.seed(123)
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    f = open(filename, encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        idx = vocabulary.get(word)
        if idx != 0:
            embedding_vectors[idx] = vector
    f.close()
    return embedding_vectors


def load_entities_by_file(file_path):
    with open(file_path, encoding="utf-8", mode='r') as entities_by_file_number_json_file:
        return json.load(entities_by_file_number_json_file)


def get_entity_ids(cfg, dataset_name, filenames):
    entities_by_file = load_entities_by_file(cfg["datasets"][dataset_name]["entities_file_path"])
    return [entities_by_file[x] for x in
                  [re.findall('/([0-9]+_[0-9]+).txt$', filename)[0] for filename in filenames]]


def get_word_embeddings(vocab_processor, embedding_name, embedding_dimension, cfg):
    vocabulary = vocab_processor.vocabulary_
    initW = None
    if embedding_name == 'word2vec':
        # load embedding vectors from the word2vec
        print("Load word2vec file {}".format(cfg['word_embeddings']['word2vec']['path']))
        initW = load_embedding_vectors_word2vec(vocabulary,
                                                             cfg['word_embeddings']['word2vec']['path'],
                                                             cfg['word_embeddings']['word2vec']['binary'])
        print("word2vec file has been loaded")
    elif embedding_name == 'glove':
        # load embedding vectors from the glove
        print("Load glove file {}".format(cfg['word_embeddings']['glove']['path']))
        initW = load_embedding_vectors_glove(vocabulary,
                                                          cfg['word_embeddings']['glove']['path'],
                                                          embedding_dimension)
        print("glove file has been loaded\n")
    elif embedding_name == 'w2v_nlp':
        # load embedding vectors from the glove
        print("Load word2vec retrained file {}".format(cfg['word_embeddings']['w2v_nlp']['path']))
        initW = load_embedding_vectors_glove(vocabulary,
                                                          cfg['word_embeddings']['w2v_nlp']['path'],
                                                          embedding_dimension)
        print("w2v_nlp file has been loaded\n")

        return initW


def get_processed_dataset(cfg, dataset_name):
    datasets = get_datasets_zuco(container_path=cfg["datasets"][dataset_name]["container_path"],
                                                 categories=cfg["datasets"][dataset_name]["categories"])
    x_text, y = load_data_labels(datasets)

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Max. Document Length: {:d}".format(max_document_length))

    x_text_split = [t.split(' ') for t in x_text]
    opening_tags = [[i for i, j in enumerate(t) if j == '<e>'] for t in x_text_split]
    closing_tags = [[i for i, j in enumerate(t) if j == '</e>'] for t in x_text_split]

    for tags in [opening_tags, closing_tags]:
        for i,o in enumerate(tags):
            if len(o) != 2:
                print(i)
                print(o)
                print(x_text_split[i])

    x_text_entities = []
    for t, o, c in zip(x_text_split, opening_tags, closing_tags):
        # Without tags
        # x_text_entities.append(t[(o[0]+1):c[0]]+t[(o[1]+1):c[1]])
        # With tags
        x_text_entities.append( " ".join( t[o[0]:(c[0]+1)] + t[o[1]:(c[1]+1)] ) )

    max_reduced_document_length = max([len(x.split(" ")) for x in x_text_entities])
    vocab_processor_entities = learn.preprocessing.VocabularyProcessor(max_reduced_document_length, vocabulary=vocab_processor.vocabulary_)
    x_entities = np.array(list(vocab_processor_entities.transform(x_text_entities)))

    return datasets, x, x_text, y, vocab_processor, x_entities, max_document_length


def get_relative_positions(cfg, dataset_name):
    position_names = ["first", "second"]
    relative_positions_all = []
    relative_positions_reversed_all = []
    lengths_all = []

    for i in (0, 1):
        file_path = cfg["datasets"][dataset_name]["relative_positions_" + position_names[i] + "_file_path"]
        relative_positions = open(file_path, encoding="utf-8", mode="r").read().splitlines()
        max_document_length = len(relative_positions[0].split(" "))
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

        relative_positions_transformed = np.array(list(vocab_processor.fit_transform(relative_positions)))

        unknown_symbol = list(vocab_processor.transform(["-999"]))[0][0]
        relative_positions_transformed_reversed = np.array([np.pad(np.flip(x[x != unknown_symbol], axis=0), pad_width=(0, max_document_length-len(x[x != unknown_symbol])),
                                                           mode='constant', constant_values=(0, unknown_symbol)) for x in relative_positions_transformed])

        relative_positions_all.append(relative_positions_transformed)
        relative_positions_reversed_all.append(relative_positions_transformed_reversed)
        lengths_all.append(len(vocab_processor.vocabulary_))

    return relative_positions_all, relative_positions_reversed_all, lengths_all


def get_pos_tags(cfg, dataset_name, filenames):
    file_path = cfg["datasets"][dataset_name]["pos_tags_file_path"]
    pos_tags = open(file_path, encoding="utf-8", mode="r").read().splitlines()
    max_document_length = len(pos_tags[0].split(" "))
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

    pos_tags = list(vocab_processor.fit_transform(pos_tags))
    # pos_tags_sorted = [pos_tags[int(x)] for x in [re.findall('/([0-9]+).txt$', filename)[0] for filename in filenames]]

    # return (np.array(pos_tags_sorted), len(vocab_processor.vocabulary_))
    return np.array(pos_tags), len(vocab_processor.vocabulary_)
