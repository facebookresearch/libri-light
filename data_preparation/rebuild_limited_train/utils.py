# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import pathlib
from collections import namedtuple
import torchaudio
import shutil


Speaker = namedtuple('Speaker', ['id', 'gender', 'subset'])
FileRecord = namedtuple(
    'FileRecord', ['fname', 'length', 'speaker', 'book', 'text_file'])


def get_speakers(speaker_path):
    all_speakers = []
    with open(speaker_path) as f:
        for line in f:
            if line.startswith(';'):
                continue

            line = line.split('|')
            speaker_id, gender, subset = [x.strip() for x in line[0:3]]
            speaker_id = int(speaker_id)

            assert subset in ['test-clean', 'train-clean-360', 'train-clean-100',
                              'test-other', 'dev-clean', 'train-other-500', 'dev-other'], subset

            speaker = Speaker(id=speaker_id, gender=gender, subset=subset)
            all_speakers.append(speaker)
    return all_speakers


def get_filelength(fname):
    info = torchaudio.info(fname)[0]
    return info.length


def traverse_tree(root, ext='flac'):
    fnames = pathlib.Path(root).rglob(f"*.{ext}")
    fnames = sorted(list(fnames))

    lengths = []
    for file in fnames:
        file = str(file.resolve())
        length = get_filelength(file)
        lengths.append(length)

    return list(zip(fnames, lengths))


def get_speaker_fname(fname):
    stemmed = fname.stem
    speaker, book, seq = stemmed.split('-')
    return int(speaker), int(book)


def full_records(speakers, fname2length, subset_name=None):
    all_records = []

    speakers = dict((speaker.id, speaker) for speaker in speakers)

    for fname, length in fname2length:
        speaker, book = get_speaker_fname(fname)
        assert speaker in speakers, f'Unknown speaker! {speaker}'

        speaker = speakers[speaker]

        if subset_name is not None:
            assert subset_name == speaker.subset
        # hacky
        text_file = fname.parent / f'{speaker.id}-{book}.trans.txt'
        frecord = FileRecord(speaker=speaker, length=length,
                             fname=fname, book=book, text_file=text_file)
        all_records.append(frecord)

    return all_records


def get_histogram(records, lambda_key, lambda_value):
    from collections import defaultdict
    key_value = defaultdict(int)

    for record in records:
        key = lambda_key(record)
        value = lambda_value(record)

        key_value[key] += value

    return key_value


def materialize(records, target_dir, tag=None, move=False):
    target_dir = pathlib.Path(target_dir)

    to_copy = set()
    to_move = set()

    for record in records:
        # outline:
        # target_dir / speaker / book / file
        if tag is None:
            target_book_dir = target_dir / \
                str(record.speaker.id) / str(record.book)
        else:
            target_book_dir = target_dir / tag / \
                str(record.speaker.id) / str(record.book)
        target_book_dir.mkdir(exist_ok=True, parents=True)

        if not move:
            to_copy.add((record.fname, target_book_dir / record.fname.name))
        else:
            to_move.add((record.fname, target_book_dir / record.fname.name))

        to_copy.add((record.text_file, target_book_dir / record.text_file.name))

    to_copy = sorted(list(to_copy))
    for src, dst in to_copy:
        shutil.copy(src, dst)

    if len(to_move) > 0:
        to_move = sorted(list(to_move))
        for src, dst in to_move:
            shutil.move(src, dst)


def print_stats(records):
    def lambda_speaker(r): return r.speaker.id
    def lambda_time(r): return r.length / 16000.0

    speaker_time = get_histogram(
        records, lambda_key=lambda_speaker, lambda_value=lambda_time)
    print(f'Unique speakers: {len(speaker_time)}')
    times = speaker_time.values()
    min_time, max_time, mean_time, total_time = min(
        times), max(times), sum(times) / len(times), sum(times)
    min_time, max_time, mean_time, total_time = map(
        int, [min_time, max_time, mean_time, total_time])
    print(
        f'Min/Mean/Max/Total, seconds: {min_time}/{mean_time}/{max_time}/{total_time}')
    print(f'n_utterances: {len(records)}')
