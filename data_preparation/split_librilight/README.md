# Split and filter LibriLight data

## Preparation
First step is to find speakers that appear in Librispeech test and dev datasets,

```console
python extract_test_speakers.py --librispeech_meta=<path to librispeech meta files>
```

Next, taking the existing VAD files with per-frame silence probabilities, we find segments 
of speech that are not shorter than L ms and separated at least with L ms of silence. We use 
L = 6 (frames) * 80 (ms / frame).

```console
python prepare_vads.py --vad_root=<path to the directory with vads>
```

Further step is to build the audio-file metadata, which would contain both book meta-data and 
individual file's SNR/VAD records. To do that, we run

```console
python puts_json.py  --librivox_dir=<librivox metadata path> --librivox_processed=<downloaded and converted flac files> \
    --vad_preprocessed=<path to prepared vads> --snr_preprocessed=<path to snr records> --test_speakers=<path to file with hold-out speaker ids>
```

This command would (a) generate a file `processing_results.json` containing some diagnostic statistics, and
(b) place json files with meta-data alongside the audio files.

After that, we can decide on the data split. This command with produce three json files, each describing sets of 
selected (nested) sets of files, each having 10x less audio-time:

```console
python split.py --librivox_processed=<directory with metadata jsons and flac files> --sampling_steps=3 --divisor=10
```
The produced files would be named as `split_0.json` (largest), `split_1.json` (second largest), etc. They also
contain some rudimentary statistics of the selected data.

Finally, you can actually copy the selected files to a specified directory ("materialize") by running
```console
python materialize_split.py --src_dir<directory with metadata jsons and flac files> --dst_dir=<dst-small> --json=split_2.json
```
If you want to exclude other splits (e.g. make the `medium` split directory not contain files from the `small`), you can use `--minus` parameter:
```console
python materialize_split.py --src_dir<directory with metadata jsons and flac files> --dst_dir=<dst> --json=split_1.json --minus=split_2.json
```
