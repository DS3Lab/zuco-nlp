# Augmenting the system with human features

The details about the recorded eye-tracking and electroencephalography data can be found in [Hollenstein et al., 2018](https://www.nature.com/articles/sdata2018291).

The ZuCo corpus can be downloaded [here](https://osf.io/q3zws/).

Set the preferences for the cognitive features in `config.yml`.

## Eye-tracking features

We use the following 5 eye-tracking features: First Fixation Duration (FFD), Gaze Duration (GD), Go Past Time (GPT), Total Reading Time (TRT), Number of Fixations (nFixations).

They can be used as raw normalized values (binned: False in `config.yml`) or as binned values (binned: any integer in `config.yml`).

The eye-tracking embedding dimension is set to 20.

## EEG features

EEG features can be used as raw values (eeg_config: RAW_FEATURES in `config.yml`), raw normalized values (eeg_config: RAW_NORMALIZED_FEATURES in `config.yml`).

In addition only the power spectrum (frequency band) features during certain eye-tracking times can be used:
eeg_config: POWER_SPECTRUM_FEATURES_TRT or POWER_SPECTRUM_FEATURES_FFD in `config.yml` to use the frequency band features during total reading time (TRT) or first fixation duration (FFD), respectively.