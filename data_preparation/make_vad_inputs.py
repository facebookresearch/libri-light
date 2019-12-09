# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
from pathlib import Path
import torchaudio
import progressbar
import argparse
import torch
import tqdm


def findAllSeqs(dirName,
                extension='.flac',
                loadCache=False):
    r"""
    Lists all the sequences with the given extension in the dirName directory.
    Output:
        outSequences, speakers

        outSequence
        A list of tuples seq_path, speaker where:
            - seq_path is the relative path of each sequence relative to the
            parent directory
            - speaker is the corresponding speaker index

        outSpeakers
        The speaker labels (in order)

    The speaker labels are organized the following way
    \dirName
        \speaker_label
            \..
                ...
                seqName.extension
    """
    cache_path = os.path.join(dirName, '_seqs_cache.txt')
    if loadCache:
        try:
            outSequences, speakers = torch.load(cache_path)
            print(f'Loaded from cache {cache_path} successfully')
            return outSequences, speakers
        except OSError as err:
            print(f'Ran in an error while loading {cache_path}: {err}')
        print('Could not load cache, rebuilding')

    if dirName[-1] != os.sep:
        dirName += os.sep
    prefixSize = len(dirName)
    speakersTarget = {}
    outSequences = []
    for root, dirs, filenames in tqdm.tqdm(os.walk(dirName)):
        filtered_files = [f for f in filenames if f.endswith(extension)]

        if len(filtered_files) > 0:
            speakerStr = root[prefixSize:].split(os.sep)[0]
            if speakerStr not in speakersTarget:
                speakersTarget[speakerStr] = len(speakersTarget)
            speaker = speakersTarget[speakerStr]
            for filename in filtered_files:
                full_path = os.path.join(root[prefixSize:], filename)
                outSequences.append((speaker, full_path))
    outSpeakers = [None for x in speakersTarget]
    for key, index in speakersTarget.items():
        outSpeakers[index] = key
    try:
        torch.save((outSequences, outSpeakers), cache_path)
        print(f'Saved cache file at {cache_path}')
    except OSError as err:
        print(f'Ran in an error while saving {cache_path}: {err}')
    return outSequences, outSpeakers


def get_file_duration_ms(path_file):
    info = torchaudio.info(path_file)[0]
    return 1000*(info.length // (info.rate))


def get_lst(path_db, file_list):

    bar = progressbar.ProgressBar(maxval=len(file_list))
    bar.start()

    path_db = Path(path_db)
    out = []

    for index, file_name in enumerate(file_list):

        bar.update(index)
        full_path = str(path_db / file_name)
        duration = get_file_duration_ms(full_path)
        out.append((full_path, full_path, int(duration)))

    bar.finish()
    return out


def save_lst(data, path_out):

    with open(path_out, 'w') as file:
        for id, path, val in data:
            file.write(' '.join((id, path, str(val))) + '\n')


def reorder_vad(path_vad, lst):

    path_vad = Path(path_vad)

    for id, full_path_wav, _ in lst:

        full_path_vad = (path_vad / id).with_suffix('.vad')
        full_path_out = Path(full_path_wav).with_suffix('.vad')
        full_path_vad.replace(full_path_out)

        full_path_vad.with_suffix('.fwt').unlink(missing_ok=True)
        full_path_vad.with_suffix('.tsc').unlink(missing_ok=True)
        full_path_vad.with_suffix('.sts').unlink(missing_ok=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Build the vad inputs")

    parser.add_argument('path_db', type=str,
                        help="Path to the dataset directory")
    parser.add_argument('path_out', type=str)
    parser.add_argument('--ignore_cache', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--extension', type=str, default='.wav')

    args = parser.parse_args()

    seqList, _ = findAllSeqs(args.path_db, extension=args.extension,
                             loadCache=not args.ignore_cache)
    if args.debug:
        seqList = seqList[:10]

    seqList = [i[1] for i in seqList]

    vad_data = get_lst(args.path_db, seqList)
    save_lst(vad_data, args.path_out)
