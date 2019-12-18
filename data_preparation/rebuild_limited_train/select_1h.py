# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from utils import get_histogram, get_speakers, traverse_tree, full_records, print_stats, materialize
import argparse
import pathlib
import random


def do_split(records, seconds_per_speaker):
    speakers = list(set([r.speaker.id for r in records]))
    random.shuffle(speakers)
    records_filtered = []

    for speaker in speakers:
        time_taken = 0
        speaker_records = [r for r in records if r.speaker.id == speaker]

        for r in speaker_records:
            if time_taken > seconds_per_speaker * 16000:
                break
            time_taken += r.length

            records_filtered.append(r)

    return records_filtered


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_seconds_per_speaker', type=int, default=150)
    parser.add_argument('--target_dir', type=str)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--root_10h', type=str)
    parser.add_argument('--meta_path', type=str)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    random.seed(args.seed)

    speakers = get_speakers(pathlib.Path(args.meta_path) / 'SPEAKERS.TXT')

    print(f'Total {len(speakers)} speakers')
    fname2length = traverse_tree(args.root_10h)
    all_records = full_records(speakers, fname2length)
    print(f'Got {len(all_records)} records')

    for gender in ['f', 'm']:
        for t, tag in zip(['train-clean-100', 'train-other-500'], ['clean', 'other']):
            records = list(all_records)
            print(f'Selecting from {t}, gender {gender}, tag {tag}')

            records = filter(lambda x: x.speaker.gender.lower()
                             == gender and x.speaker.subset == t, records)
            records = list(records)

            records_filtered = do_split(records, args.max_seconds_per_speaker)
            print_stats(records_filtered)

            if args.target_dir:
                materialize(records_filtered, args.target_dir,
                            tag=tag, move=True)
