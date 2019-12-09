import pathlib
import soundfile as sf
import numpy as np
import json
import multiprocessing
import argparse


def save(seq, fname, index):
    output = np.hstack(seq)
    file_name = fname.parent / (fname.stem + f"_{index:04}.wav")
    fname.parent.mkdir(exist_ok=True, parents=True)
    sf.write(file_name, output, 16000)


def cut_sequence(pathSeq, vad, path_out, target_len_sec):
    data, samplerate = sf.read(pathSeq)

    assert len(data.shape) == 1
    assert samplerate == 16000

    to_stitch = []
    scaler = 16000
    length_accumulated = 0.0

    i = 0
    for start, end in vad:
        start_index = int(start * scaler)
        end_index = int(end * scaler)
        slice = data[start_index:end_index]

        # if a slice is longer than target_len_sec, we put it entirely in it's own piece
        if length_accumulated + (end - start) > target_len_sec and length_accumulated > 0:
            save(to_stitch, path_out, i)
            to_stitch = []
            i += 1
            length_accumulated = 0

        to_stitch.append(slice)
        length_accumulated += end - start

    if to_stitch:
        save(to_stitch, path_out, i)


def cut_book(task):
    path_book, root_out, target_len_sec = task

    speaker = pathlib.Path(path_book.parent.name)

    for i, meta_file_path in enumerate(path_book.glob('*.json')):
        with open(meta_file_path, 'r') as f:
            meta = json.loads(f.read())
        book_id = meta['book_meta']['id']
        vad = meta['voice_activity']

        sound_file = meta_file_path.parent / (meta_file_path.stem + '.flac')

        path_out = root_out / speaker / book_id / (meta_file_path.stem)
        cut_sequence(sound_file, vad, path_out, target_len_sec)


def cut(input_dir,
        output_dir,
        target_len_sec=30,
        n_process=32):

    list_dir = pathlib.Path(input_dir).glob('*/*')
    list_dir = [x for x in list_dir if x.is_dir()]

    print(f"{len(list_dir)} directories detected")
    print(f"Launching {n_process} processes")

    pool = multiprocessing.Pool(processes=n_process)
    tasks = [(path_book, output_dir, target_len_sec) for path_book in list_dir]

    pool.map(cut_book, tasks)


def parse_args():

    parser = argparse.ArgumentParser(description="Cut a dataset in small "
                                     "sequences using VAD files")
    parser.add_argument('--input_dir', type=str, default=None,
                        help="Path to the input directory", required=True)
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Path to the output directory", required=True)

    parser.add_argument('--target_len_sec', type=int, default=60,
                        help="Target time, in seconds of each output sequence"
                             "(default is 60)")
    parser.add_argument('--n_workers', type=int, default=32,
                        help="Number of parallel worker processes")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pathlib.Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    cut(args.input_dir, args.output_dir, args.target_len_sec, args.n_workers)
