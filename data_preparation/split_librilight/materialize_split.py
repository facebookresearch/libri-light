# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import json
import pathlib
import shutil
import os
import multiprocessing


def _apply(task):
    file, args, action = task

    src = pathlib.Path(args.src_dir)
    dst = pathlib.Path(args.dst_dir)

    file = pathlib.Path(file)
    with open(file, 'r') as f:
        meta = json.loads(f.read())
        speaker = meta['speaker']

    dst_dir = dst / speaker / file.parent.name
    dst_dir.mkdir(exist_ok=True, parents=True)

    # move/copy json file
    src_file = src / file.parent.name / file.name
    dst_file = dst_dir / file.name
    action(src_file, dst_file)

    # move/copy flac file
    src_file = src / file.parent.name / (file.stem + '.flac')
    dst_file = dst_dir / (file.stem + '.flac')
    action(src_file, dst_file)


def get_args():
    parser = argparse.ArgumentParser(
        description="A script to copy prepared data splits to releasable folders")
    parser.add_argument('--src_dir', type=str, required=True)
    parser.add_argument('--dst_dir', type=str, required=True)
    parser.add_argument('--json', type=str, required=True)
    parser.add_argument('--minus', type=str, action='append', default=[])
    parser.add_argument('--n_workers', type=int, default=16)
    parser.add_argument('--mode', type=str,
                        choices=['copy', 'print'], default='print')

    args = parser.parse_args()

    assert args.json and args.dst_dir

    return args


# lambda-functions are un-pickable
def _print(src, dst):
    print(src, '->', dst)


if __name__ == '__main__':
    args = get_args()

    with open(args.json, 'r') as f:
        files = json.loads(f.read())['files']

    files_minus = []
    for fname in args.minus:
        with open(fname, 'r') as f:
            files_minus.extend(json.loads(f.read())['files'])

    files_minus = set(files_minus)

    if args.mode == 'copy':
        action = shutil.copy
    elif args.mode == 'print':
        action = _print

    tasks = [(file, args, action) for file in files if file not in files_minus]

    with multiprocessing.Pool(processes=args.n_workers) as pool:
        pool.map(_apply, tasks)
