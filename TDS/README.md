# TDS Baselines

This repository provides codes to reproduce TDS baselines in the paper. You should use them together with [wav2letter](https://github.com/facebookresearch/wav2letter).

## Data
- Two lists of supervised training data with 10 hours and 1 hour.
- Two sets of tokens for phonemes and characters.
- Two lexicons to map words to phonemes and characters.

## Experiments
### Model Architectures
- A TDS model with 20 million parameters is provided for training on the limited supervised data.
- A TDS model with 37 million parameters is provided for training on both supervised data and pseudo labels.

### Configurations
#### Acoustic model
Acoustic model training config files for each set-up.

#### Decoding
Optimal decoding parameters of each model. You can use wav2letter decoder to 
- Get optimal WER
- Generate pseudo-labels. 
