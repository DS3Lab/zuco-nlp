# Relation Classification

## Data

The data used can be found in `zuco_nr_cleanphrases/`.

## Relation classification with gaze and EEG

The code is based on the winning SemEval system for relation extraction and classification ([Rotsztejn et al., 2018](https://arxiv.org/pdf/1804.02042.pdf)).

## Type aggregation






## Preprocessing

### Word embeddings
Download the pre-trained Glove embeddings or train your own and put them in `embeddings/`.

### Part-of-speech tags
1. Make sure you have the Stanford CoreNLP installed correctly and that the server is started.
2. Use the script `annotate_pos_tags.py` to generate the PoS tags for you samples.
3. Move the new `pos_tags.txt` file into `preprocessing/`.

### Relative positions
Use the script `calculate_relative_positions.py` to generate the relative position files.

### Human features
See `feature_engineering/` for more details about augmenting the system with eye-tracking and/or EEG features.



## Training the system

### Configuration
Set your configurations in the the `config.yml` file.


### Prerequisites
Python 3, tensorflow, numpy

### Training a model
Run with Python 3:

`$ CUDA_VISIBLE_DEVICES=5 python train_relext.py config.yml`