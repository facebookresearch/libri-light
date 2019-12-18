# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from collections import defaultdict
import argparse
import json
import pathlib
import math


def get_stats(fnames, fnames2jsons):

    total_seconds = sum(fnames2jsons[fname]
                        ['file_length_sec'] for fname in fnames)

    files_per_genre = defaultdict(int)
    seconds_per_genre = defaultdict(int)
    snr_per_genre = defaultdict(float)

    mean_snr = 0.0

    unique_speakers = set()
    unique_books = set()

    for fname in fnames:
        data = fnames2jsons[fname]

        snr = data['snr'] if not math.isnan(data['snr']) else 0.0
        seconds = data['file_length_sec']

        mean_snr += snr * seconds

        if 'genre' not in data['book_meta'] or data['book_meta']['genre'] is None:
            file_genres = ['<none>']
        else:
            file_genres = data['book_meta']['genre']

        for genre in file_genres:
            files_per_genre[genre] += 1
            seconds_per_genre[genre] += seconds
            snr_per_genre[genre] += snr * seconds

        unique_speakers.add(data['speaker'])
        unique_books.add(data['book_meta']['id'])

    for g in snr_per_genre:
        snr_per_genre[g] /= seconds_per_genre[g]

    mean_snr /= total_seconds
    return seconds_per_genre, files_per_genre, snr_per_genre, total_seconds, unique_books, unique_speakers, mean_snr


def get_genre2time(fnames, fnames2jsons):
    seconds_per_genre = defaultdict(int)

    for fname in fnames:
        data = fnames2jsons[fname]
        if 'genre' not in data['book_meta'] or data['book_meta']['genre'] is None:
            file_genres = ['<none>']
        else:
            file_genres = data['book_meta']['genre']

        for genre in file_genres:
            seconds_per_genre[genre] += data['file_length_sec']

    return seconds_per_genre


def get_genre2files(fnames, fnames2jsons):
    genre_files = defaultdict(list)

    for fname in fnames:
        data = fnames2jsons[fname]
        if 'genre' not in data['book_meta'] or data['book_meta']['genre'] is None:
            file_genres = ['<none>']
        else:
            file_genres = data['book_meta']['genre']

        for genre in file_genres:
            genre_files[genre].append(fname)

    return genre_files


def get_fname2json(fnames):
    fname2json = {}
    for fname in fnames:
        with open(fname, 'r') as f:
            data = json.load(f)
        fname2json[fname] = data
    return fname2json


def subselect(fnames, files2jsons, divisor=10):
    overall_time = sum(
        fnames2jsons[fname]['file_length_sec'] for fname in fnames)
    print('Selecting from', overall_time / 60 / 60, 'hours')

    genre2time = get_genre2time(fnames, fnames2jsons)

    genre2budget = {}
    for genre, time in genre2time.items():
        genre2budget[genre] = time // divisor

    time_selected = 0
    selected_files = []

    for fname in fnames:
        if time_selected > overall_time // divisor:
            break

        data = fnames2jsons[fname]
        if 'genre' not in data['book_meta'] or data['book_meta']['genre'] is None:
            file_genres = ['<none>']
        else:
            file_genres = data['book_meta']['genre']
        length = data['file_length_sec']

        fits = True
        for file_genre in file_genres:
            fits = fits and (
                file_genre not in genre2budget or genre2budget[file_genre] > length)

        if fits:
            time_selected += length
            selected_files.append(fname)
            for file_genre in file_genres:
                if file_genre in genre2budget:
                    genre2budget[file_genre] -= length

    overall_time = sum(
        fnames2jsons[fname]['file_length_sec'] for fname in selected_files)
    print('Selected', overall_time / 60 / 60, 'hours')

    return selected_files


def take_n(x, n):
    for i, k in enumerate(x):
        yield k

        if i == n - 1:
            break


def get_args():
    parser = argparse.ArgumentParser(description='Reads a direcctory with flac/meta-data files and decides how to split them in '
        'three nested sets, roughly balancing genres')
    parser.add_argument('--librivox_processed', type=str)
    parser.add_argument('--sampling_steps', type=int, default=3)
    parser.add_argument('--size_divisor', type=int, default=10)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    fnames = list(
        take_n(pathlib.Path(args.librivox_processed).rglob('*.json'), n=1000 if args.debug else -1))
    fnames2jsons = get_fname2json(fnames)

    for sampling_step in range(args.sampling_steps):
        seconds_per_genre, files_per_genre, snr_per_genre, total_seconds, unique_books, unique_speakers, mean_snr = get_stats(
            fnames, fnames2jsons)

        print('Total seconds', total_seconds, ' = ',
              total_seconds / 60 / 60, ' hours')
        print('Unique speakers', len(unique_speakers), ' unique books',
              len(unique_books), ' files ', len(fnames))
        print('Time-weighted snr', mean_snr)

        with open(f'split_{sampling_step}.json', 'w') as f:
            dump = [(genre, {'seconds': seconds, 'hours': seconds / 60 / 60,
                             'files': files_per_genre[genre],
                             'mean_snr': snr_per_genre[genre]}) for (genre, seconds) in seconds_per_genre.items()]

            fnames_as_str = [str(f) for f in fnames]
            f.write(json.dumps({
                'distribution': dump,
                'files': fnames_as_str,
                'n_speakers': len(unique_speakers),
                'n_books': len(unique_books),
                'n_files': len(fnames),
                'time_weighted_snr': mean_snr},
                indent=1))

        fnames = subselect(
            fnames, fnames2jsons, divisor=args.size_divisor)
