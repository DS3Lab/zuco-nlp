## Data

In addition to the code in this repository, wou will need the following data: \
	- The Google 300 dimensional embeddings from https://github.com/mmihaltz/word2vec-GoogleNews-vectors. The embedding file must be placed in the folder `embeddings/`. \
	- The ZuCo data in their latest (January 2019) format. Those data must be placed in the `Data_to_preprocess/` folder. Please contact the first author it you require this exact version of the data.


## Code

1. You will then need an environment with the following packages:
	    
	h5py                2.7.1              
	matplotlib          2.2.3      
	mxnet               1.0.0.post2
	nltk                3.3        
	numpy               1.14.2     
	pandas              0.20.3     
	pip                 18.0       
	scikit-learn        0.19.0     
	scipy               0.19.1     
	seaborn             0.9.0      
	setuptools          39.0.1     
	statsmodels         0.9.0      
	tensorboard         1.7.0      
	tensorflow          1.7.0      
	tflearn             0.3.2      
	wheel               0.30.0     

2. After setting up the environment and all the data, you will need to preprocess them to create the training data. To do that run `python3 create_modeling_data.py` from this directory. 
Potential paramenters for this run are `-s` if you want to save a report of this preprocessing and `-low_def` if you want to save the newly preprocessed EEG signal (the most intensive component memory-wise) with low definition (np.float16).

3. Now you can run all experiments via these three scripts:
	- run_features_model.py for the experiment on ZuCo data only. \
		e.g. `python3 run_features_model.py -we -et -eeg -b -s -opt`
	- run_stts_embedding_experiment.py for the experiment on SST data with Cross-Validation. \
		e.g. `python3 run_stts_embedding_experiment.py -we -et -eeg -b -s -opt`
	- run_stts_embedding_experiment2.py for the experiment on SST data with the official train, dev and test division. \
		e.g. `python3 run_stts_embedding_experiment2.py -we -et -eeg -b -s -opt`

The parameters for those runs are: \
	`-v`	if you want a verbose output \
	`-b`	if you want the binary experiment \
	`-we`	if you want word embeddings to be used \
	`-et`	if you want eye-tracking to be used \
	`-eeg`	if you want eeg signal to be used \
	`-s`	if the data should be saved \
	`-opt`	if optimized parameters should be added \
	
It is also possible to modify the model's parameters from the file of the run (so any of the files written above) or from the default configuration present in configuration/new_configs.py (N.B.: the default configuration is overwritten by any choice made in the `run_*` scripts.).
