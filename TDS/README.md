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
Acoustic model training config files for each set-up. Note that the 20-millioin-parameter TDS models are trained on 8 GPUs each, while the 37-millioin-parameter ones are on 64 GPUs. See [tutorials](https://github.com/facebookresearch/wav2letter/blob/master/docs/train.md#distributed) about how to run distributed training. 

Sample command:
```sh
</path/to/your>/wav2letter/build/Train \
--flagsfile=</path/to/your>/libri-light/TDS/experiments/config/acoustic_model/10h+pseudo-label_letter_37M_TDS.cfg \
--enable_distributed=true
```

#### Decoding
Optimal decoding parameters of each model. You can use wav2letter decoder to 
- Get optimal WER
- Generate pseudo-labels. 

We use the official Librispeech 4-gram LM, which can be downloaded from [here](http://www.openslr.org/11/), in all the decoding experiments. 

Sample command:
```sh
</path/to/your>/wav2letter/build/Decode \
--flagsfile=</path/to/your>/libri-light/TDS/experiments/config/decoding/10h+pseudo-label_letter_37M_TDS.cfg \
--sclite=</path/to/your/output_folder>
```
