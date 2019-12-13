from collections import defaultdict
import argparse
import json
import pathlib
import shutil
import os
import copy
import random
import math

def get_speech_len(file_meta):
    return file_meta['file_length_sec']
    #return file_meta['book_meta']['totaltimesecs']


def get_stats(fnames, fnames2jsons):

    total_seconds = sum(get_speech_len(fnames2jsons[fname]) for fname in fnames)

    files_per_genre = defaultdict(int)
    seconds_per_genre = defaultdict(int)
    snr_per_genre = defaultdict(float)

    mean_snr = 0.0

    unique_speakers = set()
    unique_books = set()

    for fname in fnames:
        data = fnames2jsons[fname]

        snr = data['snr'] if not math.isnan(data['snr']) else 0
        seconds = get_speech_len(data)

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
            seconds_per_genre[genre] += get_speech_len(data)

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


def nearly_stratified_sample(fnames, files2jsons):
    overall_time = sum(
        get_speech_len(fnames2jsons[fname]) for fname in fnames)
    print('Stratified sampling from', overall_time / 3600, 'hours')

    genre2time = get_genre2time(fnames, fnames2jsons)

    genre2budget = {}
    for genre, time in genre2time.items():
        genre2budget[genre] = time // 10

    time_selected = 0
    selected_files = []

    for fname in fnames:
        if time_selected > overall_time // 10:
            break

        data = fnames2jsons[fname]
        if 'genre' not in data['book_meta'] or data['book_meta']['genre'] is None:
            file_genres = ['<none>']
        else:
            file_genres = data['book_meta']['genre']
        length = get_speech_len(data)

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
        get_speech_len(fnames2jsons[fname]) for fname in selected_files)
    print('Stratified sampled', overall_time / 3600, 'hours')

    return selected_files


def take_n(x, n):
    for i, k in enumerate(x):
        yield k

        if i == n - 1:
            break


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--librivox_processed', type=str,
                        default='/checkpoint/kharitonov/librilight_converted_flac/')

    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--sampling_steps', type=int, default=3)
    parser.add_argument('--take_n', type=int, default=1000)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    random.seed(args.seed)

    fnames = list(
        take_n(pathlib.Path(args.librivox_processed).rglob('*.json'), args.take_n))
    fnames2jsons = get_fname2json(fnames)

    for sampling_step in range(args.sampling_steps):
        seconds_per_genre, files_per_genre, snr_per_genre, total_seconds, unique_books, unique_speakers, mean_snr = get_stats(
            fnames, fnames2jsons)

        print('Total seconds', total_seconds, ' = ',
              total_seconds / 60 / 60, ' hours')
        print('Unique speakers', len(unique_speakers), ' unique books',
              len(unique_books), ' files ', len(fnames))
        print('Time-weighted snr', mean_snr)

        with open(f'genres_{sampling_step}.txt', 'w') as f:
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

        fnames = nearly_stratified_sample(
            fnames, fnames2jsons)
