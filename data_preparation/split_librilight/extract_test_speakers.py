# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import pathlib


def get_args():
    parser = argparse.ArgumentParser(
        description="Prepare list of speakers that appear in Librispeech test/dev datasets")
    parser.add_argument('--librispeech_meta', type=str, required=True)
    parser.add_argument('--output', type=str, default='test_speakers.json')

    args = parser.parse_args()
    return args


def extract_holdout_speakers(librispeech_meta_path):
    chapters_file = pathlib.Path(librispeech_meta_path) / 'SPEAKERS.TXT'
    speakers_to_exclude = []

    with open(chapters_file, 'r') as f:
        for line in f.readlines():
            if line.startswith(';'):
                continue
            line = line.split("|")
            speaker_id, subset = int(line[0].strip()), line[2].strip()
            assert subset in ['train-other-500', 'train-clean-100',
                              'dev-other', 'dev-clean', 'test-other', 'test-clean', 'train-clean-360'], subset

            if subset in ['dev-other', 'dev-clean', 'test-other', 'test-clean']:
                speakers_to_exclude.append(speaker_id)

    return sorted(speakers_to_exclude)


if __name__ == '__main__':
    args = get_args()

    speakers_to_exclude = extract_holdout_speakers(args.librispeech_meta)
    print('Speakers to exclude: ', len(speakers_to_exclude))

    with open(args.output, 'w') as f:
        f.write(json.dumps(dict(
            test_speakers=speakers_to_exclude,
        ), indent=1))
