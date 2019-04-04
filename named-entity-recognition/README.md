##Code

The code for these multi-task learning experiments in `ner-tagger/` was adapted from the [Named Entity Recognition Tool repository](https://github.com/glample/tagger).

The original code refers to the following paper:

Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer (2016). 
[Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360). 

Please also refer to their paper if reusing this code.

The original [Named Entity Recognition Tool repository](https://github.com/glample/tagger) includes usage details.
In this version adapted the reading of the input data to include eye-tracking and EEG features and the addition of eye-tracking/EEG embeddings to the neural architecture.
Additionaly, we changed the code to work with Python 3. 

##Data

The NER annotations can be found [here](https://github.com/DS3Lab/ner-at-first-sight).

Sample data including EEG and eye-tracking features can be found in `data/`. It contains the following features in each row:

1. word
2.-18. eye-tracking features
19.-26. EEG features
27. NER label

`data/type-lexicons/` contains a type_aggregated lexicon for ZuCo with eye-tracking and EEG feature values in the same order.

## Example usage 

The following command trains an NER model including eye-tracking _and_ EEG features:

`THEANO_FLAGS='dnn.conv.algo_bwd_filter=deterministic,dnn.conv.algo_bwd_data=deterministic' python3 train.py --train ${train-file} \ 
                                                                                                                 --dev ${dev-file} \
                                                                                                                 --test ${test-file} \
                                                                                                                 --tag_scheme iob \
                                                                                                                 --zeros 1 \
                                                                                                                 --with_eeg_gaze 1 \
                                                                                                                 --bins ${number-of-bins} \
                                                                                                                 --fold ${number-of-fold} \`

