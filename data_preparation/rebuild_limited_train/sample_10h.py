# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from utils import get_histogram, get_speakers, traverse_tree, full_records, print_stats, materialize
import argparse
import pathlib
import random


def do_split_10h(records, speakers, max_seconds_per_speaker, min_seconds_per_speaker, total_seconds):
    """
    Greedily selecting speakers, provided we don't go over budget
    """
    scaler = 1.0 / 16000  # sampling rate
    speaker2time = get_histogram(records, lambda_key=lambda r: r.speaker.id,
                                 lambda_value=lambda r: r.length * scaler)

    speakers = set([r.speaker.id for r in records])
    speakers = sorted(speakers)
    random.shuffle(speakers)

    time_taken = 0.0
    speakers_taken = []

    for speaker in speakers:
        current_speaker_time = speaker2time[speaker]
        if min_seconds_per_speaker <= current_speaker_time <= max_seconds_per_speaker and current_speaker_time < total_seconds - time_taken:
            speakers_taken.append(speaker)
            time_taken += current_speaker_time

    speakers_taken = set(speakers_taken)

    records_filtered = [r for r in records if r.speaker.id in speakers_taken]
    return records_filtered


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_minutes_per_speaker', type=int, default=30)
    parser.add_argument('--min_minutes_per_speaker', type=int, default=20)
    parser.add_argument('--total_minutes', type=int, default=150)

    parser.add_argument('--target_dir', type=str)

    parser.add_argument('--seed', type=int, default=179)

    parser.add_argument('--root_clean', type=str)
    parser.add_argument('--root_other', type=str)
    parser.add_argument('--meta_path', type=str)

    args = parser.parse_args()

    if args.max_minutes_per_speaker <= 0:
        args.max_minutes_per_speaker = float('inf')
    return args


if __name__ == '__main__':
    args = get_args()
    random.seed(args.seed)

    speakers = get_speakers(pathlib.Path(args.meta_path) / 'SPEAKERS.TXT')
    print('Found speakers', len(speakers))

    for gender in ['m', 'f']:
        for root, tag in zip([args.root_clean, args.root_other], ['clean', 'other']):
            print(f'Selecting from {root}, gender {gender}, tag {tag}')

            fname2length = traverse_tree(root)
            records = full_records(speakers, fname2length)

            records = filter(lambda x: x.speaker.gender.lower()
                             == gender, records)
            records = list(records)

            records_filtered = do_split_10h(
                records, speakers, args.max_minutes_per_speaker * 60, args.min_minutes_per_speaker * 60, args.total_minutes * 60)
            print_stats(records_filtered)

            if args.target_dir:
                materialize(records_filtered, args.target_dir, tag=tag)
