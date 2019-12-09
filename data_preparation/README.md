# Libri-light data preparation



Here we provide scripts to reconstruct the raw dataset from scratch, including data download, conversion into flac,
Voice Activity Detection (VAD) and Signal to Noise (SNR) computation,
meta-data construction, and dataset filtering and splitting.

## Download the data

the data is spit into 3 subsets of increasing lengths (small, medium, large). The subsets are included into one another, which means that if you download medium you don't need to download small, and if you download large, you don't needs to download medium. These splits are done to enable doing experiments on smaller amounts of data. 

   [ll_unlab_small.tar.gz  (  577 hours,  35 GB)](s3://dl.fbaipublicfiles.com/librilight/data/small.tar)
   
   [ll_unlab_medium.tar.gz ( 5193 hours, 321 GB)](s3://dl.fbaipublicfiles.com/librilight/data/medium.tar)
   
   [ll_unlab_large.tar.gz  (51934 hours, 3.05 TB)](s3://dl.fbaipublicfiles.com/librilight/data/large.tar)
    
In addition, we also provide a 4th subset containing potentially duplicated books.

   [ll_unlab_duplicate.tar.gz  ( 4500 hours,  274 GB)](s3://dl.fbaipublicfiles.com/librilight/data/duplicate.tar)


## Do the stats

## Cut the data into segments


We also provide scripts for preparing the raw dataset into useable segments for model training (segmentation).



## Regenerate the entire data from scratch

![pipeline](data_preparation_pipeline.svg)
