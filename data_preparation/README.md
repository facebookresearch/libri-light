# Libri-light data preparation



Here we provide scripts to reconstruct the raw dataset from scratch, including data download, conversion into flac,
Voice Activity Detection (VAD) and Signal to Noise (SNR) computation,
meta-data construction, and dataset filtering and splitting.

## Download the data

### 1. The unlabelled data

The unlabelled data is spit into 3 subsets of increasing lengths (small, medium, large). These splits are done to enable doing experiments on smaller amounts of data (also, downloading the large dataset can take about one day). 

-  [unlab_small.tar  (  577 hours,  35 GB, 1135)](https://dl.fbaipublicfiles.com/librilight/data/unlab_small.tar)   
-  [unlab_medium.tar ( 5193 hours, 321 GB)](https://dl.fbaipublicfiles.com/librilight/data/unlab_medium.tar) 
-  [unlab_large.tar  (51934 hours, 3.05 TB)](https://dl.fbaipublicfiles.com/librilight/data/large.tar)
    
In addition, we also provide a 4th subset containing potentially duplicated books.

- [unlab_duplicate.tar.gz  ( 4500 hours,  274 GB)](https://dl.fbaipublicfiles.com/librilight/data/duplicate.tar)

The directory structure of the archives is the same as for librispeech: 

    dataset_name/speakerID/file_name/

where dataset_name is small, medium, large, or duplicate, speakerID is the librivox speakerID (a number), and file_name the name of the original LibriVox audio file. Inside each directory, you should find a .flac and a .json. See below for the structure of the .json file.

Once the dataset is downloaded and "untarred", into unlab_data/ you can check its statistics with the command

     python build_all_stats.py unlab_data/ unlab_data/ unlab_stats/
     
You'll find two png and a .json file

### 2. The limited supervision train data

The limited supervision rests on librispeech. If you do not already have it, you should first download librispeech:

       mkdir librispeech
       cd librispeech
       wget http://www.openslr.org/resources/12/train-clean-100.tar.gz # train-clean-100 of librispeech 
       wget http://www.openslr.org/resources/12/train-other-500.tar.gz # train-other-500 of librispeech


Then run the following script:

       make_limited_train/split train-clean-100 train-other-500

### 3. The dev and test sets 

These are the standard librispeech dev and test sets. They can be gotten at the following address:

      wget http://www.openslr.org/resources/12/dev-clean.tar.gz
      wget http://www.openslr.org/resources/12/dev-other.tar.gz
      wget http://www.openslr.org/resources/12/test-clean.tar.gz
      wget http://www.openslr.org/resources/12/test-other.tar.gz
      

## Cut the data into segments


We also provide scripts for preparing the raw dataset into useable segments for model training (segmentation).



## Regenerate the entire data from scratch

![pipeline](data_preparation_pipeline.svg)


## Json file

We created one json file per LibriVox audio file. This is different from the LibriVoc distribution that had one json per book and each book could have several files. 

     { 
     "speaker" : "960"    # LibriVox speaker ID
     "book_meta": {       # a bunch of LibriVox metadata concerning the book relevant to that file
       "id":  319                    # LibriVox book ID 
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
