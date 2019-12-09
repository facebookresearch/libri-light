# Libri-light data preparation



Here we provide scripts to reconstruct the raw dataset from scratch, including data download, conversion into flac,
Voice Activity Detection (VAD) and Signal to Noise (SNR) computation,
meta-data construction, and dataset filtering and splitting.

## Download the data

### 1. Get and prepare the unlabelled data

*Download.* The unlabelled data is spit into 3 subsets of increasing lengths (small, medium, large). These splits are done to enable doing experiments on smaller amounts of data (also, downloading the large dataset can take about one day).

-  [small.tar  (577 hours, 35 GB)](https://dl.fbaipublicfiles.com/librilight/data/small.tar)   
-  [medium.tar (5193 hours, 321 GB)](https://dl.fbaipublicfiles.com/librilight/data/medium.tar)
-  [large.tar  (51934 hours, 3.05 TB)](https://dl.fbaipublicfiles.com/librilight/data/large.tar)

In addition, we also provide a 4th subset containing potentially duplicated books.

- [unlab_duplicate.tar.gz  (4500 hours, 274 GB)](https://dl.fbaipublicfiles.com/librilight/data/duplicate.tar)

The directory structure of the archives is the same as for librispeech:

    dataset_name/speakerID/book_name/

where `dataset_name` is `small`, `medium`, `large`, or `duplicate`, `speakerID` is the librivox speakerID (a number), and `book_name` the name of the original LibriVox audiobook file. Inside each directory, you should find a set of `.flac` and `.json` files. See below for the format  of the `.json` files.

By combining these subsets, you can construct the 3 splits described in the Libri-light paper:

- *unlab-58k*  : small+medium+large
- *unlab-5.8k* : small+medium
- *unlab-580*  : small


Once the dataset is downloaded, "untarred" and organised into a directory `UNLAB_DIR` you can check its statistics with the command:

```console
    python build_all_stats.py UNLAB_DIR OUTPUT_DIR
```
This will construct in `OUTPUT_DIR` two .png files (plus .json files in a .cache directory)

*Segment.* Each file may be rather long and may not fit into memory.  As a final step, we provide a script to segment the files into roughly 60sec sequences obtained by concatenating consecutive voice activity chunks.

```console
    python cut_by_vad.py --input_dir INPUT_DIR --output_dir OUTPUT_DIR
```

In `OUTPUT_DIR`, there will be the same structure as above, but each file_name directory will have a list of smaller files (`.flac`). You can change this step as fits your model. 


### 2. Get the limited-supervision train data

The limited supervision training sets are built on Librispeech. They consist in 10h, 1h, and 10 minute splits with orthographic transciptions and aligned phoneme transcriptions, which can be used to train small models or fine-tune pretrained ones. This can be downloaded here:

- [librispeech_finetuning.tgz  (10 hours, 0.6 GB)](https://dl.fbaipublicfiles.com/librilight/data/librispeech_finetuning.tgz)   

The directory structure is the following:

     1h/         # data of the 1h split (made up of 6 folds of 10 min)
         0/         # first 10 min fold
           clean/     # 2 speakers, clean
           other/     # 2 speakers, other
         ...      
         5/         # last 10 min fold
           clean/     # 2 speakers, clean
           other/     # 2 speakers, other
     9h/         # remaining data of the 10h split (10h=1h+9h)
         clean /    # 12 speakers, clean
         other/     # 12 speakers, other
     phones/     # phoneme alignment for all of the files


The 10h split is made by combining the data from the `9h/` and the `1h` directories. The 1h split is itself made of 6 folds of 10 min splits. The `phone/` directory contains the frame-wise phoneme transcription of the various splits (the IPA phone mappings are in `phone_mapping.json`). There is also the phoneme transcription of the librispeech dev and test sets.  


Alternatively, you can reconstruct this dataset by downloading by hand librispeech and running the scripts in `rebuild_limited_train/`.


### 3. Get the dev and test sets (for evaluation)

These are the standard librispeech dev and test sets. They can be gotten at the following address:

    wget http://www.openslr.org/resources/12/dev-clean.tar.gz
    wget http://www.openslr.org/resources/12/dev-other.tar.gz
    wget http://www.openslr.org/resources/12/test-clean.tar.gz
    wget http://www.openslr.org/resources/12/test-other.tar.gz



## Regenerate the entire data from scratch (or build it in another language !)

We provide scripts that would completely regenerate the raw data from the librivox repository.
To download the audio data from libri-vox.
```console
python download_librivox.py $OUTPUT_DOWNLOAD
```

You can actually try to download data in another language. To do so, use the --language option (for example --language French)

Then, you will need to unzip the data, just use:
```console
python unzip_and_convert.py unzip $OUTPUT_DOWNLOAD -o $OUTPUT_MP3
```

And convert them from .mp3 to flac:
```console
python unzip_and_convert.py unzip $OUTPUT_MP3 -o $OUTPUT_FLAC -f .flac
```

Finally, complete the metadata files:
```console
python complete_metadata.py --path_metadata $OUTPUT_DOWNLOAD --out_dir $OUTPUT_FLAC --ignore_cache
```


![pipeline](data_preparation_pipeline.svg)


## Format of Json files

We created one json file per LibriVox audio file. This is different from the LibriVoc distribution that had one json per book and each book could have several files.

     {
     "speaker" : "960"    # LibriVox speaker ID
     "book_meta": {       # a bunch of LibriVox metadata concerning the book relevant to that file
       "id":  "319"                  # LibriVox book ID
       "title": "History of Holland" # LibriVox book title
       ...                        
       "genre": [                    # LibriVox genre
        "*Non-fiction",
        "History"
         ],                          # from this point, this is our own-libri-light metadata:
       "Dramatic Readings": false,   # boolean for dramatic vs normal reading
       "meta_genre" : "Literature"   # meta-genre among, 7 possibilities
        },                           # ["Literature", "Science, Craft & Essay", "Undefined", "Religion", "Poetry", "Theater", "Ancien"]
     "snr": 5.391,                   # Signal to Noise Ratio computed on the basis of Voice Activity Detection
     "voice_activity": [             # onsets and offsets (in seconds) of each VAD segments
      [
       0.4,
       12.32
       ],
       ...
