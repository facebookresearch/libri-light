# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Set

import argparse
import json
import pathlib
import copy
import dataclasses


@dataclasses.dataclass
class BookError:
    no_match_snr: Set[str] = dataclasses.field(default_factory=set)
    no_match_speaker: Set[str] = dataclasses.field(default_factory=set)
    test_speakers: Set[str] = dataclasses.field(default_factory=set)
    fuzzy_matched_speaker: Set[str] = dataclasses.field(default_factory=set)
    ok: int = 0

    def update(self, other):
        self.no_match_snr.update(other.no_match_snr)
        self.no_match_speaker.update(other.no_match_speaker)
        self.test_speakers.update(other.test_speakers)
        self.fuzzy_matched_speaker.update(other.fuzzy_matched_speaker)
        self.ok += other.ok

    def as_dict(self):
        file_stats = {
            'ok': self.ok,
            'no_match_speaker': len(self.no_match_speaker),
            'no_match_snr': len(self.no_match_snr),
            'test_speakers': len(self.test_speakers),
            'fuzzy_matched_speaker': len(self.fuzzy_matched_speaker),
        }

        files_excluded = {
            'no_match_speaker': list(self.no_match_speaker),
            'no_match_snr': list(self.no_match_snr),
            'test_speakers': list(self.test_speakers),
            'fuzzy_matched_speaker': list(self.fuzzy_matched_speaker),
        }

        results = dict(file_stats=file_stats, files_excluded=files_excluded)
        return results


def normalize(name):
    """
    We normalize directory names to fix some name misalignments in vad/snr data, 
    unzipped data, and the original meta-data jsons. Examples:
        - SNR has typical format of `.../1000things_1807_librivox_wav/1000things_00_fowler.wav`,
        when the correspoding meta-data is in `1000things_1807_librivox_64kb_mp3_metadata.json`
        - unzipped/converted flac is in `.../rock_me_to_sleep_1502/` while the correspoding meta-data
        is in `.../rock_me_to_sleep_1502.poem_librivox_64kb_mp3_metadata.json`

    The normalization is done by removing the suffixes removed/added by ffmpeg.
    """

    for suffix in ['.poem_', '_librivox', '_64kb', '_wav']:
        pos = name.find(suffix)
        if pos != -1:
            name = name[:pos]
            return name
    return name


def parse_downloaded_jsons(librivox_dir, duplicates=None):
    name2json = {}
    n_duplicates = 0

    fnames = pathlib.Path(librivox_dir).rglob(f"*_metadata.json")
    for meta_fname in fnames:
        if duplicates and meta_fname.name in duplicates:
            n_duplicates += 1
            continue

        with open(meta_fname,  'r') as f:
            meta_data = json.load(f)

        speaker_fname = str(meta_fname)[:-13] + 'speaker_data.json'
        with open(speaker_fname,  'r') as f:
            speaker_data = json.loads(f.read())

        root_name = str(meta_fname.name)
        root_name = normalize(root_name)

        name2json[root_name] = dict(meta=meta_data, speaker_data=speaker_data)
    return name2json, n_duplicates


def read_snr(fname):
    import csv
    name2snr = []

    lineno = 0
    errors = 0
    with open(fname, 'r') as f:
        r = csv.reader(f, delimiter='\t')
        for row in r:
            parent_name = pathlib.Path(row[0]).parent.name
            parent_name = str(parent_name)

            name = pathlib.Path(row[0]).stem
            name = str(name)
            snr = float(row[1])
            name2snr.append(((parent_name, name), snr))

    snr_table = {}

    for fname, snr in name2snr:
        dir_name = normalize(fname[0])
        fname = fname[1]

        if dir_name not in snr_table:
            snr_table[dir_name] = {}
        snr_table[dir_name][fname] = snr
    return snr_table


def process_dir(normalized_book_name, dir_name, name2meta, file_times, voice_activities, snr_table, test_speakers, extension='*.flac'):
    speaker2file = dict(zip(name2meta[normalized_book_name]['speaker_data']
                            ['names'], name2meta[normalized_book_name]['speaker_data']['readers']))

    assert normalized_book_name in name2meta
    assert normalized_book_name in snr_table, normalized_book_name
    assert normalized_book_name in voice_activities and normalized_book_name in file_times

    errors = BookError()

    for file_name in dir_name.glob(extension):
        fname = file_name.stem
        assert fname.endswith('_64kb')
        fname = fname[:-5]  # cut _64kb

        if fname not in snr_table[normalized_book_name]:
            errors.no_match_snr.add(fname)
            continue

        assert fname in voice_activities[normalized_book_name]
        assert fname in file_times[normalized_book_name]

        if fname in speaker2file:
            speakers = speaker2file[fname]
        else:
            match = [z for z in speaker2file.keys() if z.startswith(fname)]
            if len(match) != 1:
                errors.no_match_speaker.add(fname)
                continue
            else:
                errors.fuzzy_matched_speaker.add(fname)
            speakers = speaker2file[match[0]]

        if speakers is None:
            errors.no_match_speaker.add(fname)
            continue

        if len(speakers) != 1:
            errors.no_match_speaker.add(fname)
            continue

        speaker = speakers[0]
        if int(speaker) in test_speakers:
            errors.test_speakers.add(fname)
            continue

        errors.ok += 1

        target = file_name.parent / (file_name.stem + '.json')
        data = copy.deepcopy(name2meta[normalized_book_name])
        del data['speaker_data']
        data['speaker'] = speaker
        data['file_length_sec'] = file_times[normalized_book_name][fname]
        del data['meta']['totaltime']
        del data['meta']['trancription_status']
        meta = data['meta']
        del data['meta']
        data['book_meta'] = meta

        assert fname in snr_table[normalized_book_name], (
            fname, normalized_book_name)
        data['snr'] = round(snr_table[normalized_book_name][fname], 4)
        data['voice_activity'] = [(round(x[0], 4), round(x[1], 4))
                                  for x in voice_activities[normalized_book_name][fname]]

        with open(target, 'w') as fout:
            out = json.dumps(data, indent=1)
            fout.write(out)

    return errors


def get_voice_activities(vad_preprocessed, seconds_per_frame=80.0/1000.0):
    file_times = {}
    voice_activities = {}

    total_time = 0.0
    with open(vad_preprocessed, 'r') as f:
        segments = json.loads(f.read())

        for k, v in segments.items():
            dir_name, fname = k.split('/')
            assert fname.endswith('.vad')
            fname = fname[:-4]  # cut vad

            dir_name = normalize(dir_name)
            if dir_name not in file_times:
                file_times[dir_name] = {}
                voice_activities[dir_name] = {}

            file_times[dir_name][fname] = v[1] * seconds_per_frame
            voice_activities[dir_name][fname] = [
                (round(seconds_per_frame * x[0], 2), round(seconds_per_frame * x[1], 2)) for x in v[0]]
            total_time += v[1]

    total_time *= seconds_per_frame
    print('Total time in VAD files, hours', total_time / 60 / 60)
    return file_times, voice_activities


def get_duplicates(path):
    with open(path, 'r') as f:
        duplicates = json.loads(f.read())

    duplicates_to_remove = set()

    for duplicate_cluster in duplicates:
        for d in duplicate_cluster[1:]:
            duplicates_to_remove.add(d)
    return duplicates_to_remove


def get_args():
    parser = argparse.ArgumentParser(
        "Prepares jsons with meta information per single audiofile. "
        "The jsons are placed alongside with the audiofiles and have the same name")

    parser.add_argument('--librivox_dir', type=str, required=True)
    parser.add_argument('--vad_preprocessed', type=str, default='vads.json')
    parser.add_argument('--snr_preprocessed', type=str,
                        default='vad_based_snr_all.tsv')
    parser.add_argument('--librivox_processed', type=str, required=True)
    parser.add_argument('--test_speakers', type=str,
                        default='test_speakers.json')
    parser.add_argument('--title_duplicates', type=str,
                        default='title_duplicates.json')
    parser.add_argument('--millis_per_frame', type=float,
                        default=80.0)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    with open(args.test_speakers, 'r') as f:
        test_speakers = json.loads(f.read())['test_speakers']

    duplicates_to_remove = get_duplicates(args.title_duplicates)
    print(f'Removing {len(duplicates_to_remove)} duplicates')

    snr_table = read_snr(args.snr_preprocessed)
    voice_times, voice_activities = get_voice_activities(
        args.vad_preprocessed, seconds_per_frame=args.millis_per_frame/1000.0)
    name2json, n_duplicates = parse_downloaded_jsons(
        args.librivox_dir, duplicates_to_remove)

    unmatched_names = set()

    dir_paths = list(pathlib.Path(args.librivox_processed).glob('*'))

    for d in dir_paths:
        assert d.is_dir()

    aggregated_errors = BookError()

    for dir_path in dir_paths:
        book_name = str(dir_path.name)
        normalized_book_name = normalize(book_name)

        if normalized_book_name in name2json:
            errors = process_dir(normalized_book_name, dir_path, name2json,
                                 voice_times, voice_activities, snr_table, test_speakers)
            aggregated_errors.update(errors)
        else:
            unmatched_names.add(book_name)

    print('Done, flushing stats...')
    with open('processing_results.json', 'w') as f:
        results = aggregated_errors.as_dict()
        results['duplicate_books'] = n_duplicates
        f.write(json.dumps(results, indent=1))
