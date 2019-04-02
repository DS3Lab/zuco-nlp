## How to run the "Passive Human Supervision" experiments

1. You will need, together with this repo, additional data such as:
	- The google 300 dimensional embeddings from https://github.com/mmihaltz/word2vec-GoogleNews-vectors. The embedding file must be placed in the folder "embeddings".
	- The ZuCo data in their latest (January 2019) format. Those data must be placed in the "Data_to_preprocess" folder.

2. You will then need a virtual environment with the following packages:
	    
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

3. After setting up the environment and all the data, you will need to preprocess them to create the training data. To do that run "python3 create_modeling_data.py" from this folder. Potential paramenters for this run are "-s" if you want to save a report of this preprocessing and "-low_def" if you want to save the newly preprocessed EEG signal (the most intensive component memory-wise) with low definition (np.float16).

4. You are now ready to run all experiments via the 3 scripts:
	- run_features_model.py for the experiment on ZuCo data only.
		e.g. python3 run_features_model.py -we -et -eeg -b -s -opt
	- run_stts_embedding_experiment.py for the experiment on SST data with Cross-Validation.
		e.g. python3 run_stts_embedding_experiment.py -we -et -eeg -b -s -opt
	- run_stts_embedding_experiment2.py for the experiment on SST data with the official train, dev and test division.
		e.g. python3 run_stts_embedding_experiment2.py -we -et -eeg -b -s -opt

The parameters for those runs are:
	(-v)	if you want a verbose output
	(-b)	if you want the binary experiment
	(-we)	if you want word embeddings to be used
	(-et)	if you want eye-tracking to be used
	(-eeg)	if you want eeg signal to be used
	(-s)	if the data should be saved
	(-opt)	if optimized parameters should be added
It is also possible to modify the model's parameters from the file of the run (so any of the files written above) or from the default configuration present in configuration/new_configs.py (N.B.: the default configuration is overwritten by any choice made in the "run_*" scripts.).
