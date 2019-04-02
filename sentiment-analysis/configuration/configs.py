from constants import constants

# TODO: Change so that it does
default_config = {
    'Config_name': "UNNAMED_CONFIG",
            # Data paths
    'DATASETS_PATH': "data/sentences",
    'WORD_EMBEDDINGS_PATH': "embeddings/GoogleNews-vectors-negative300.bin",
    'ALL_PREPROCESSED_DATA_PATH': 'Sentence_level_data/',
            # Data of interest with respective formats
    'WORD_EMBEDDINGS': True,
    'EYE_TRACKING': False,
    'EYE_TRACKING_FORMAT': 'RAW_ET', # RAW (BINNED to be added, also add NUMBER_OF_BINS and EMBEDDING_DIM then)
    'NORMALIZE_ET': True,
    'EEG_SIGNAL': False,
    'EEG_SIGNAL_FORMAT': 'RAW_EEG', # RAW_EEG, ICA_EEG (FF, TRT to be added)
    'EEG_TO_PIC': False,
    'NORMALIZE_EEG': True,
    'SENTENCE_LEVEL': False,
    'SUBJECTS': None,
    'JOIN_SUBJECTS_METHOD': 'AVERAGE', # TODO: Used only if more than one subject is selected, check it raises ERROR, can assume values AVERAGE, STACK and CONCATENATE
            # Classification type
    'BINARY_CLASSIFICATION': False,
    'BINARY_FORMAT': 'POS_VS_NEG', # POS_VS_NEG, POS_VS_NONPOS or NEG_VS_NONNEG
            # Data preprocessing was moved
    # 'PCA_DIMENSION': -1,
    # 'USE_LDS_SMOOTHING': False,
# TODO: CHANGE THIS SO THAT IT CONSIDERS ALL MODELS
            # Model Hyperparameters config
    'LSTM_UNITS': 200,
    'HIDDEN_LAYER_UNITS': 50,
    'USE_NORMALIZATION_LAYER': False,
    'ATTENTION_EMBEDDING': False,
            # Learning rate hyperparameters
    'INITIAL_LR': .001,
    'HALVE_LR_EVERY_PASSES': 2,
            # Regularization
    'L2_REG_LAMBDA': 0.00001,
    'L1_REG_LAMBDA': 0.0000,
    'DROPOUT_KEEP_PROB': .5,
            # Learning phases hyperparameters
    'BATCH_SIZE': 16,
    'NUM_EPOCHS': 10,
            # Evaluation
    'EVALUATE_EVERY': 10,
    'NUMBER_OF_CV_SPLITS': 10,
            # Machine config
    'PER_PROCESS_GPU_FRACTION': .5,
    'TF_DEVICE': '/cpu:0', # maybe should be changed for the server? what does it do exactly?
            # Results config
    'RESULTS_FILE_PATH': "./Results_files/",
    'VERBOSE': False
}

def complete_config(config_dict):
    """
    Function to complete a configuration, the configuration must be in it's final format and not contain lists

    :param config:  (dic)   Dictionary of defined parameters

    :return:
        completed_config:   (dic)   Dictionary containing the default config updated via the defined parameters
    """
    completed_config = dict(default_config)
    completed_config.update(config_dict)

    word_level_features = completed_config['WORD_EMBEDDINGS'] or completed_config['EYE_TRACKING']
    assert word_level_features==False or completed_config['SENTENCE_LEVEL']==False, "Requested features are not available at Sentence Level."

    return completed_config
