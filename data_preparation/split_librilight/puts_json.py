from typing import Set

import argparse
import json
import pathlib
import copy
import dataclasses
import multiprocessing

from common import canoninize_name, parse_downloaded_jsons


@dataclasses.dataclass
class BookError:
    no_match_snr: Set[str] = dataclasses.field(default_factory=set)
    no_match_vad: Set[str] = dataclasses.field(default_factory=set)
    no_match_speaker: Set[str] = dataclasses.field(default_factory=set)
    multiple_speakers: Set[str] = dataclasses.field(default_factory=set)
    name_collision: Set[str] = dataclasses.field(default_factory=set)
    test_speakers: Set[str] = dataclasses.field(default_factory=set)
    fuzzy_matched_speaker: Set[str] = dataclasses.field(default_factory=set)
    ok: int = 0

    def update(self, other):
        self.no_match_snr.update(other.no_match_snr)
        self.no_match_vad.update(other.no_match_vad)
        self.no_match_speaker.update(other.no_match_speaker)
        self.multiple_speakers.update(other.multiple_speakers)
        self.name_collision.update(other.name_collision)
        self.test_speakers.update(other.test_speakers)
        self.fuzzy_matched_speaker.update(other.fuzzy_matched_speaker)
        self.ok += other.ok

    def __str__(self):
        return \
            f"ok: {self.ok}, no_match_snr: {len(self.no_match_snr)}, no_match_vad: {len(self.no_match_vad)}, " + \
            f"no_match_speaker: {len(self.no_match_speaker)}, multiple_speakers: {len(self.multiple_speakers)}, " + \
            f"fuzzy_matched_speaker: {len(self.fuzzy_matched_speaker)}"


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--librivox_dir', type=str,
                        default='/checkpoint/kharitonov/LibriVox_updated_metadata/')
    parser.add_argument('--vad_preprocessed', type=str, default='vads.json')
    parser.add_argument('--snr_preprocessed', type=str,
                        default='/checkpoint/kharitonov/unsupervised/vad_based_snr_all.tsv')
    parser.add_argument('--librivox_processed', type=str,
                        default='/checkpoint/kharitonov/librilight_converted_flac/')
    parser.add_argument('--test_speakers', type=str,
                        default='test_speakers.json')
    parser.add_argument('--title_duplicates', type=str,
                        default='title_duplicates.json')

    args = parser.parse_args()
    return args


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
        dir_name = canoninize_name(fname[0])
        fname = fname[1]

        if dir_name not in snr_table:
            snr_table[dir_name] = {}
        snr_table[dir_name][fname] = snr
    return snr_table


def process_dir(normalized_book_name, dir_name, name2meta, voice_times, voice_activities, snr_table, test_speakers, extension='*.flac'):
    speaker2file = dict(zip(name2meta[normalized_book_name]['speaker_data']
                            ['names'], name2meta[normalized_book_name]['speaker_data']['readers']))

    assert normalized_book_name in name2meta
    assert normalized_book_name in snr_table
    assert normalized_book_name in voice_times

    errors = BookError()

    for file_name in dir_name.glob(extension):
        fname = file_name.stem
        if not fname.endswith('_64kb'):
            errors.name_collision.add(fname)
            continue

        fname = fname[:-5]

        if fname not in snr_table[normalized_book_name]:
            errors.no_match_snr.add(fname)
            continue

        if fname not in voice_times[normalized_book_name]:
            errors.no_match_vad.add(fname)
            continue

        if fname in speaker2file:
            speaker = speaker2file[fname]
        else:
            # TODO: check does not lead to errors
            match = [z for z in speaker2file.keys() if z.startswith(fname)]
            if len(match) != 1:
                errors.no_match_speaker.add(fname)
                continue
            speaker = speaker2file[match[0]]

        if speaker is None:
            errors.no_match_speaker.add(fname)
            continue

        if len(speaker) != 1:
            errors.multiple_speakers.add(fname)
            continue

        speaker = speaker[0]
        if int(speaker) in test_speakers:
            errors.test_speakers.add(fname)
            continue

        errors.ok += 1

        target = file_name.parent / (file_name.stem + '.json')
        data = copy.deepcopy(name2meta[normalized_book_name])
        del data['speaker_data']
        data['speaker'] = speaker
        #data['meta']['totaltimesecs'] = voice_times[normalized_book_name][fname]
        del data['meta']['totaltime']
        del data['meta']['trancription_status']
        meta = data['meta']
        del data['meta']
        data['book_meta'] = meta

        data['snr'] = round(snr_table[normalized_book_name][fname], 4)
        data['voice_activity'] = [(round(x[0], 4), round(x[1], 4))
                                  for x in voice_activities[normalized_book_name][fname]]

        #with open(target, 'w') as fout:
        #    out = json.dumps(data, indent=1)
        #    fout.write(out)

    return errors


def get_voice_activities(vad_preprocessed):
    voice_times = {}
    voice_activities = {}

    total_time = 0.0
    scaler = 80.0 / 1000
    with open(vad_preprocessed, 'r') as f:
        segments = json.loads(f.read())

        for k, v in segments.items():
            dir_name, fname = k.split('/')
            fname = fname[:-4]  # cut vad

            dir_name = canoninize_name(dir_name)
            if dir_name not in voice_times:
                voice_times[dir_name] = {}
                voice_activities[dir_name] = {}

            voice_times[dir_name][fname] = v[1] * scaler
            voice_activities[dir_name][fname] = [
                (round(scaler * x[0], 2), round(scaler * x[1], 2)) for x in v[0]]
            total_time += v[1]

    total_time *= scaler
    print('total time in VAD files, hours', total_time / 60 / 60)
    return voice_times, voice_activities


def get_duplicates(path):
    with open(path, 'r') as f:
        duplicates = json.loads(f.read())

    duplicates_to_remove = set()

    for duplicate_cluster in duplicates:
        for d in duplicate_cluster[1:]:
            duplicates_to_remove.add(d)
    return duplicates_to_remove


if __name__ == '__main__':
    args = get_args()

    with open(args.test_speakers, 'r') as f:
        test_speakers = json.loads(f.read())['test_speakers']

    duplicates_to_remove = get_duplicates(args.title_duplicates)
    print(f'Removing {len(duplicates_to_remove)} duplicates')

    snr_table = read_snr(args.snr_preprocessed)
    voice_times, voice_activities = get_voice_activities(args.vad_preprocessed)
    name2json = parse_downloaded_jsons(args.librivox_dir, duplicates_to_remove)

    total, matched = 0, 0
    unmatched_names = set()
    test_names = set()

    dir_paths = list(pathlib.Path(args.librivox_processed).glob('*'))

    for d in dir_paths:
        assert d.is_dir()

    aggregated_errors = BookError()

    for dir_path in dir_paths:
        total += 1

        book_name = str(dir_path.name)
        normalized_book_name = canoninize_name(book_name)

        if normalized_book_name in name2json:
            matched += 1
            errors = process_dir(normalized_book_name, dir_path, name2json,
                                 voice_times, voice_activities, snr_table, test_speakers)
            aggregated_errors.update(errors)
        else:
            unmatched_names.add(normalized_book_name)

    print('Done, flushing stats...')
    with open('processing_results.json', 'w') as f:
        results = {
            'unmatched_books': list(unmatched_names),
            'test_books': list(test_names),
            'file_stats': {
                'ok': aggregated_errors.ok,
                'no_match_speaker': len(aggregated_errors.no_match_speaker),
                'no_match_vad': len(aggregated_errors.no_match_vad),
                'no_match_snr': len(aggregated_errors.no_match_snr),
                'name_collision': len(aggregated_errors.name_collision),
                'test_speakers': len(aggregated_errors.test_speakers),
                'fuzzy_matched_speaker': len(aggregated_errors.fuzzy_matched_speaker),
            },
            'files_excluded': {
                'no_match_speaker': list(aggregated_errors.no_match_speaker),
                'no_match_vad': list(aggregated_errors.no_match_vad),
                'no_match_snr': list(aggregated_errors.no_match_snr),
                'name_collision': list(aggregated_errors.name_collision),
                'test_speakers': list(aggregated_errors.test_speakers),
                'fuzzy_matched_speaker': list(aggregated_errors.fuzzy_matched_speaker),

            }
        }

        f.write(json.dumps(results, indent=1))
