# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from utils import get_histogram, get_speakers, traverse_tree, full_records, print_stats, materialize
import pathlib
import argparse
import random


def do_split(records):
    speakers = set([r.speaker.id for r in records])

    for speaker in speakers:
        speaker_records = [r for r in records if r.speaker.id == speaker]
        yield speaker_records


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', type=str)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--root_1h', type=str)
    parser.add_argument('--meta_path', type=str)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    random.seed(args.seed)

    speakers = get_speakers(pathlib.Path(args.meta_path) / 'SPEAKERS.TXT')
    fname2length = traverse_tree(args.root_1h)
    all_records = full_records(speakers, fname2length)
    print(f'Got {len(all_records)} records')

    for gender in ['f', 'm']:
        for t, tag in zip(['train-clean-100', 'train-other-500'], ['clean', 'other']):
            records = list(all_records)
            print(f'Selecting from {t}, gender {gender}')

            records = filter(lambda x: x.speaker.gender.lower()
                             == gender and x.speaker.subset == t, records)
            records = list(records)
            print(f'{len(records)} utterances in the split')

            for i, split in enumerate(do_split(records)):
                print_stats(split)

                if args.target_dir:
                    materialize(split, args.target_dir + f'/{i}/', tag=tag)
