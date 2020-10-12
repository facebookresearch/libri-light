# Getting Librispeech subsample

This collection of scripts was used to subsample Librispeech data into 10h, 1h, and 10 min chunks. 
These chunks are then used for fine-tuning the models trained with Librilight.

The goal of the splitting process is to ensure that the chunks are approximately balanced w.r.t. 
to the gender of the speakers and noise levels.

Since the samples are nested, the workflow assumes that they are generated in turns, starting from the largest.

# 10 hours sample
```
python sample_10h.py --root_clean=<path to librispeech train-100-clean> --root_other=<path to train-500-other> --meta_path=<path to metadata>
```
the script will generate the subsample and output some statistics for it, but it will not be writen to a disc unless
you provide `--target_dir` option. In this case, it would be materialized on disk.
```
python sample_10h.py --root_clean=<path to librispeech train-100-clean> --root_other=<path to train-500-other> --meta_path=<path to metadata> --target_dir=10h
```

# 1 hour sample
Next step is selecting 1h sample from the 10h one, obtained above:
```
python select_1h.py --meta_path=<path to metadata> --root_10h=<path to 10h sample>
```
Again, to actually materialize
```
python select_1h.py --meta_path=<path to metadata> --root_10h=<path to 10h sample> --target_dir=./1h
```
As a result, the files would be moved form `root_10h` (making it effectively 9h).

# Splitting 1 hour in 6 x 10 minutes
Finally, we split 1h sample in 10 samples by 6 minutes:
```
python split_1h_in10min.py --root_1h=1h --target_dir=6x10min --meta_path=<path to metadata>
```

# Other
`get_stats.py` would output the stats for a particular directory and `clean_texts.py` would prune all texts that correspond to excluded files.

`build_dataset.sh` is a script for re-generating the Librispeech samples we release.
