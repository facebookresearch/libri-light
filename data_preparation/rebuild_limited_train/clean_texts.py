# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import pathlib

"""
Cleans the *.txt files, removing the texts that correspond to the flac's not in the directory.
Assumes Librispeech-like outline.
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    txt_names = pathlib.Path(args.dataset).rglob(f"*.txt")
    txt_names = sorted(list(txt_names))

    for txt_name in txt_names:
        print(txt_name)

        parent_dir = txt_name.parent

        siblings = list(parent_dir.glob('*.flac'))
        assert len(siblings) > 0

        sibling_names = set([s.stem for s in siblings])

        with open(txt_name, 'r') as f:
            txt = f.readlines()

        with open(txt_name, 'w') as f:
            for l in txt:
                if l.split()[0] in sibling_names:
                    f.write(l)
