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