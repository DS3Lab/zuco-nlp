## Code

The code for these multi-task learning experiments in `sluice-networks/` was adapted from the [Sluice Networks repository](https://github.com/sebastianruder/sluice-networks).

The original code refers to the following paper:

Sebastian Ruder, Joachim Bingel, Isabelle Augenstein, Anders Søgaard (2017). 
[Sluice networks: Learning what to share between loosely related tasks](https://arxiv.org/abs/1705.08142). 
arXiv preprint arXiv:1705.08142.

Please also refer to their paper if reusing this code.

In this version we only adapted the reading of the input data and the possible tasks and task combinations. The original [Sluice Networks repository](https://github.com/sebastianruder/sluice-networks) includes installation instructions.

## Data

The data for each NLP task is provided separately in `data/`. It contains the following features in each row:

1. document ID
2. sentence ID
3. word ID
4. word
5. part of speech (UNKNOWN for all words, since not used in our experiments) 
6. NLP task label \
    for NER: B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC \
    for relations: e.g. B-founder, I-founder, etc. \
    for sentiment-binary: B-NEUT, B-NOTNEUT, etc. \
    for sentiment-ternary: B-NEG, B-POS, B-NEUT, I-NEG, etc.

7. (EEG) Aggregated theta frequency band values (eeg_t)
8. (EEG) Aggregated alpha frequency band values (eeg_a)
9. (EEG) Aggregated beta frequency band values (eeg_b)
10. (EEG) Aggregated gamma frequency band values (eeg_g)
11. (eye tracking) mean fixation duration (mfd)
12. (eye tracking) fixation probability (fixp)
13. (eye tracking) number of fixations (nfix)
14. (eye tracking) first fixation duration (ffd)
15. (eye tracking) total reading time (trt)
16. word frequency (freq - extracted from the British National Corpus)

Use the scrips `cv_splits.py` to split the data in training, development and test file for each fold.

## Example usage 

The following command can be used to train and evaluate a model with NER as a main task and learning gaze features as auxilliary tasks:

`python3 run_sluice_net.py --dynet-devices GPU:${gpu} 
                           --dynet-autobatch 1 
                           --dynet-seed 123 
                           --dynet-mem 2048 
                           --epochs ${epochs} 
                           --task-names ner mfd fixp nfix ffd trt 
                           --h-layers 3  
                           --pred-layer 2 2 2 2 2 2 
                           --cross-stitch 
                           --layer-connect skip 
                           --num-subspaces 2 
                           --constrain-matrices 1 2 
                           --patience 3 
                           --train-dir ${train_dir} 
                           --dev-dir ${dev_dir} 
                           --test-dir ${test_dir}test 
                           --train avg --test avg  
                           --model-dir model/${model_name} 
                           --log-dir ${log_dir} 
                           --result-file ${result_file}`
