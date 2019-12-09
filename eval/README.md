# Eval

You will find here all relevant evaluation launched on the LibriLight-dataset.

## ABX

### Setup

To setup the ABX evaluation script you need to compile the cython code it relies on. Just do:
```console
cd ABX_src
python setup.py build_ext --inplace
```

Checkout that everything works properly with:
```console
cd ABX_src
nosetests -d
```

Download the Librilight `.item` files necessary to run the ABX evaluation here: [ABX_data.tgz](https://dl.fbaipublicfiles.com/librilight/data/ABX_data.tgz).

There are 4 `.item` files constructed from the Librispeech dev and test set: `dev-clean.item`, `dev-other.item`, `test-clean.item`, and `test-other.item`, containing the labels for the ABX evaluation.

### How to run the ABX evaluation ?


Dump your features in .pt (torch), .npz or .npy (numpy) format somewhere. Your features dataset should look like this:

```console
\data_dir
  file_name_0.extension
  file_name_1.extension
  ...
```

Each file should contain a 2D-vector of shape Sequence_size x Feature_dimension.

Then run:
```console
python eval_ABX.py $PATH_FEATURE_DIR  $PATH_TO_ABX_ITEMS/$DB_NAME.item --file_extension $EXTENSION --out $OUTPUT_DIR --feature_size $FEATURE_SIZE
```

Where `$DB_NAME` is one of the 4 evaluation datasets (`dev-clean`, `dev-other`, `test-clean`, `test-other) and 
`$FEATURE_SIZE` is the duration (in s) of one feature of the model (for a `10ms` frame rate, this would be `0.01`).


## Pre-computed checkpoints

Some pre-computed model trained with CPC are available for use ! In order to load a model just use CPC_loader.py, for example to retrieve the model trained on the 60k hours dataset:

```console
python CPC_loader.py 60k $PATH_OUTPUT_CHECKPOINT
```

You can directly evaluate the ABX score on this checkpoint by running:
```console
python eval_ABX.py $PATH_AUDIO_DIR  ABX_data/$DB_NAME.item --file_extension $EXTENSION --out $OUTPUT_DIR --path_checkpoint $PATH_OUTPUT_CHECKPOINT
```

Where $EXTENSION corresponds to an audio foramt (.wav, .flac ...)

## PER

### Setup

To setup the PER evaluation script you need to compile the cython code it relies on. Just do:
```console
cd PER_src
python setup.py build_ext --inplace
```
### How to run the PER evaluation ?

First you need to train a linear classifier on your features. Just run:
```console
python eval_PER.py train
```

Then you can run the PER computation, for example on librispeech100/test-clean:
```console
python eval_PER.py per $PATH_OUT/checkpoint.pt $PATH_TO_TEST_CLEAN $PATH_TO_TEST_CLEAN_PHONES --file_extension .flac
```


## WER

### Setup
* wav2letter python bindings: [(how-to)](https://github.com/facebookresearch/wav2letter/tree/master/bindings/python).
* KenLM-based Librispeech language model, can be found [here](http://www.openslr.org/11/) or downloaded [here](https://dl.fbaipublicfiles.com/librilight/data/4-gram.bin); it should be placed into `WER_data/`.
* lexicon, [download](https://dl.fbaipublicfiles.com/librilight/data/lexicon.txt.gz); it should be placed into `WER_data/`.
* jiwer, installable via `pip install jiwer`.

### How to run the WER evaluation?

Training a letter classifier on top of a pre-trained CPC model:
```console
python eval_WER.py --path_train=$PATH_FINETUNING --path_val=$PATH_TO_DEV_CLEAN --path_checkpoint=$PATH_OUT/checkpoint.pt --lr=1e-3  --n_epochs=50 --p_dropout=0.1 --output=$OUTPUT_DIR 

```
Evaluating it with wav2letter decoder:
```console
python eval_WER.py --path_checkpoint=$PATH_OUT/checkpoint.pt --lr=1e-3  --n_epochs=50 --p_dropout=0.1 --output=$OUTPUT_DIR --path_wer=$PATH_TO_TEST_CLEAN
```

You can also train and evaluate afterwards, in a single command:
```console
python eval_WER.py --path_train=$PATH_FINETUNING --path_val=$PATH_TO_DEV_CLEAN --path_checkpoint=$PATH_OUT/checkpoint.pt --lr=1e-3  --n_epochs=50 --p_dropout=0.1 --output=$OUTPUT_DIR --path_wer=$PATH_TO_TEST_CLEAN
```


