# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple
import argparse
import json
import pathlib
import multiprocessing


def get_args():
    parser = argparse.ArgumentParser(
        description="Transform per-frame VAD files into segments with voice activity")

    parser.add_argument('--vad_root', type=str, required=True)
    parser.add_argument('--time_step_ms', type=float, default=80)
    parser.add_argument('--p_threshold', type=float, default=0.999)
    parser.add_argument('--len_threshold_frames', type=int,
                        default=6)  # 6 frames ~ 0.5s
    parser.add_argument('--n_workers', type=int, default=32)

    parser.add_argument('--output', type=str, default='vads.json')

    args = parser.parse_args()
    return args


def parse_vad(fname):
    with open(fname, 'r') as f:
        probs = f.read()
        probs = [float(x) for x in probs.split()]
    return probs


def split_vad(silence_probs: List[float], p_silence_threshold: float, len_threshold: int) -> List[Tuple[int, int]]:
    """Given a sequence `p_probs` of silence probabilities p, this function
    returns intervals of speech activity, such that (a) those intervals are separated by
    at least `len_threshold` of silent frames (p > `p_silence_threshold`), 
    (b) are themselves longer than `len_threshold`.

    Arguments:
        silence_probs -- list of silence probabilities
        p_silence_threshold -- all frames with silence probability above this thresholds
            are considered as silence
        len_threshold -- minimal length of silence and non-silence segments

    Returns: list of tuples (start_speech_frame, first_silence_frame_after_start or end_of_sequence)
    """
    segments = []

    start = None
    i = 0
    n = len(silence_probs)

    while i < len(silence_probs) and silence_probs[i] > p_silence_threshold:
        i += 1
    # supported invariants: `start` points to the frame where speech starts, i >= start
    start = i

    while i < n:
        # scroll until first silence frame
        if silence_probs[i] < p_silence_threshold:
            i += 1
            continue

        # now i points to the first silence frame
        # look ahead: do we have at least len_threshold silence frames?
        all_silence = True
        for j in range(i + 1, min(i + len_threshold, n)):
            all_silence = all_silence and silence_probs[j] > p_silence_threshold
            if not all_silence:
                break

        if not all_silence:
            # no we don't: disregard the silence, go further
            # starting from the first non-silence frame
            i = j
        else:
            # we do have enough silence for a split
            if i - start > len_threshold:
                segments.append((start, i))

            while i < n and silence_probs[i] > p_silence_threshold:
                i += 1
            start = i
            i += 1

    if i - start > len_threshold and start < n:
        segments.append((start, i))

    return segments


def process(task):
    name, args = task
    vads = parse_vad(name)
    segments = split_vad(vads, args.p_threshold, args.len_threshold_frames)
    name = str(name.parent.name) + '/' + str(name.name)
    return (name, (segments, len(vads)))


if __name__ == '__main__':
    fname2segments = {}

    args = get_args()

    tasks = [(x, args) for x in pathlib.Path(args.vad_root).rglob("*.vad")]
    print(f'Found {len(tasks)} vad files')

    with multiprocessing.Pool(processes=args.n_workers) as pool:
        fname2segments = pool.map(process, tasks)
    fname2segments = dict(fname2segments)

    with open(args.output, 'w') as f:
        f.write(json.dumps(fname2segments, sort_keys=True))
