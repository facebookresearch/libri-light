# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import subprocess
import argparse
import multiprocessing


def unzip(args):

    if args.path_out is None:
        args.path_out = args.path_in

    if not os.path.isdir(args.path_out):
        os.mkdir(args.path_out)

    files_in = [f for f in os.listdir(args.path_in)
                if os.path.splitext(f)[1] == '.zip']

    print(f"{len(files_in)} files found")

    for file_name in files_in:
        full_path_in = os.path.join(args.path_in, file_name)
        full_path_out = os.path.join(args.path_out,
                                     os.path.splitext(file_name)[0])

        if not os.path.isdir(full_path_out):
            os.mkdir(full_path_out)

        subprocess.run(["unzip", full_path_in, "-d", full_path_out])


def _convert_dir(task):
    dir_name, args = task
    valid_formats = ['.mp3', '.ogg', '.flac', '.wav']

    full_path_in = os.path.join(args.path_in, dir_name)
    files_list = [f for f in os.listdir(full_path_in)
                  if os.path.splitext(f)[1] in valid_formats]

    full_path_out = os.path.join(args.path_out, dir_name)
    if not os.path.isdir(full_path_out):
        os.mkdir(full_path_out)

    for file_name in files_list:
        base_name, format = os.path.splitext(file_name)
        path_out_file = os.path.join(
            full_path_out, base_name + args.format)
        path_in_file = os.path.join(full_path_in, file_name)

        subprocess.run(["ffmpeg", "-i", path_in_file,
                        "-ac", "1",
                        "-ar", str(args.sample_rate), path_out_file],
                       stdout=subprocess.DEVNULL)


def convert(args, n_processes=16):

    if args.path_out is None:
        args.path_out = args.path_in

    if not os.path.isdir(args.path_out):
        os.mkdir(args.path_out)

    dirs_in = [f for f in os.listdir(args.path_in)
               if os.path.isdir(os.path.join(args.path_in, f))]
    print(f"{len(dirs_in)} books found")

    pool = multiprocessing.Pool(processes=n_processes)
    pool.map(_convert_dir, [(dir_name, args) for dir_name in dirs_in])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Unzip and Convert Libri-Light')
    subparsers = parser.add_subparsers(dest='command')

    parser_unzip = subparsers.add_parser('unzip',
                                         help='Unzip the Libri-Light dataset')
    parser_unzip.add_argument('path_in', type=str)
    parser_unzip.add_argument('-o', '--path_out', type=str, default=None)

    parser_convert = subparsers.add_parser('convert',
                                           help="Convert the "
                                           "Librilight_dataset into the "
                                           "desired format.")
    parser_convert.add_argument('path_in', type=str)
    parser_convert.add_argument('-o', '--path_out', type=str, default=None)
    parser_convert.add_argument('-f', '--format', type=str, default=".flac")
    parser_convert.add_argument('-s', '--sample_rate', type=int, default=16000)
    parser_convert.add_argument('-j', '--n_processes', type=int, default=16,
                                help="Number of worker processes")

    args = parser.parse_args()

    if args.command == 'unzip':
        unzip(args)
    elif args.command == 'convert':
        convert(args)
