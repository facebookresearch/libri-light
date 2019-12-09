# Libri-light data preparation



Here we provide scripts to reconstruct the raw dataset from scratch, including data download, conversion into flac,
Voice Activity Detection (VAD) and Signal to Noise (SNR) computation,
meta-data construction, and dataset filtering and splitting.

## Download the data

the data is spit into 3 subsets of increasing lengths (small, medium, large). These splits are done to enable doing experiments on smaller amounts of data (also, downloading the large dataset can take about one day). 

-  [unlab_small.tar  (  577 hours,  35 GB, 1135)](https://dl.fbaipublicfiles.com/librilight/data/unlab_small.tar)   
-  [unlab_medium.tar ( 5193 hours, 321 GB)](https://dl.fbaipublicfiles.com/librilight/data/unlab_medium.tar) 
-  [unlab_large.tar  (51934 hours, 3.05 TB)](https://dl.fbaipublicfiles.com/librilight/data/large.tar)
    
In addition, we also provide a 4th subset containing potentially duplicated books.

- [unlab_duplicate.tar.gz  ( 4500 hours,  274 GB)](https://dl.fbaipublicfiles.com/librilight/data/duplicate.tar)

to launch downloading in the background, you can type

     wget https://dl.fbaipublicfiles.com/librilight/data/small.tar &
     wget https://dl.fbaipublicfiles.com/librilight/data/medium.tar &
     wget https://dl.fbaipublicfiles.com/librilight/data/large.tar &
     

The structure of the repos are 

## Do the stats



## Cut the data into segments


We also provide scripts for preparing the raw dataset into useable segments for model training (segmentation).



## Regenerate the entire data from scratch

![pipeline](data_preparation_pipeline.svg)
