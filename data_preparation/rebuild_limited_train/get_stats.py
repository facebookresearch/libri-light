# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from utils import get_histogram, get_speakers, traverse_tree, full_records, print_stats, materialize
import pathlib
import argparse
import random


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str)
    parser.add_argument('--meta_path', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    speakers = get_speakers(pathlib.Path(args.meta_path) / 'SPEAKERS.TXT')

    fname2length = traverse_tree(args.root)
    records = full_records(speakers, fname2length, subset_name=None)
    print(f'Utterances: {len(records)}')

    time_by_gender = get_histogram(
        records, lambda_key=lambda r: r.speaker.gender, lambda_value=lambda r: r.length / 16000)
    print('Time by gender, seconds', time_by_gender)

    time_by_subset = get_histogram(
        records, lambda_key=lambda r: r.speaker.subset, lambda_value=lambda r: r.length / 16000)
    print('Time by subset, seconds', time_by_subset)

    speaker_freq = get_histogram(
        records, lambda_key=lambda r: r.speaker.id, lambda_value=lambda r: 1)
    print('Number of uniq speakers', len(speaker_freq))

    book_lengths = get_histogram(
        records, lambda_key=lambda r: r.book, lambda_value=lambda r: r.length)

    scaler = 1.0 / 16000
    max_length = max(book_lengths.values()) * scaler
    min_length = min(book_lengths.values()) * scaler
    mean_length = sum(book_lengths.values()) / len(book_lengths) * scaler

    print(
        f'Book length disrtibution, seconds,  min: {min_length}, mean: {mean_length}, max: {max_length}; n_books={len(book_lengths)}')
