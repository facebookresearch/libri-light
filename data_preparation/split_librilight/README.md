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

