# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import sys
import time
import os
import numpy as np
from scipy.io import wavfile
import multiprocessing
import argparse


_INT16_MAX_VALUE = float(np.abs(np.iinfo(np.int16).min))
_INT32_MAX_VALUE = float(np.abs(np.iinfo(np.int32).min))


def convert_wav_buf_f32(data):
    if data.dtype == np.float32:
        pass
    elif data.dtype == np.int16:
        data = data.astype(np.float32) / _INT16_MAX_VALUE
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / _INT32_MAX_VALUE
    else:
        raise ValueError(
            "Expecting dtype to be float32/int16/int32"
            + "current type is {}".format(str(data.dtype))
        )
    return data.astype(np.float32)


def cal_signal_power(start, end, audio, fs):
    signal_start = int(round(float(start) * fs))
    signal_length = int(round(float(end) * fs))
    signal_energy = np.sum(
        np.power(audio[signal_start: signal_start + signal_length], 2)
    )
    return signal_energy, signal_length


def calculate_snr(sample, vad, fs=16000, noise_th=0.995, speech_th=0.8, vad_window_ms=80):
    sample = convert_wav_buf_f32(sample)
    sample_chunk = np.split(sample, range(
        0, len(sample), int(vad_window_ms * fs / 1000)))
    speech_chunk = []
    noise_chunk = []
    leftover_chunk = []
    speech_continue_chunk = 2  # heuristic, 240ms
    for x, v in zip(sample_chunk, vad):
        if v < speech_th or speech_continue_chunk >= 0:
            speech_chunk.append(x)
            if v < speech_th:
                speech_continue_chunk = 2
            else:
                speech_continue_chunk -= 1
        elif v > noise_th:
            noise_chunk.append(x)
        else:
            leftover_chunk.append(x)
    speech_chunk = np.concatenate(speech_chunk)
    speech_energy = np.sum(np.power(speech_chunk, 2))
    speech_time = len(speech_chunk)/fs
    speech_power = speech_energy/speech_time
    if len(noise_chunk) == 0:
        print("no noise?", file=sys.stderr)
        return [float('nan'), speech_power, float('nan')]
    noise_chunk = np.concatenate(noise_chunk)
    leftover_chunk = np.concatenate(leftover_chunk)
    noise_energy = np.sum(np.power(noise_chunk, 2))
    noise_time = len(noise_chunk)/fs
    noise_power = noise_energy/noise_time
    snr = 10 * np.log10((speech_power)/noise_power)
    return [snr, speech_power, noise_power]


def calculate_file_snr(file_name, speech_th, noise_th):
    vad_file = file_name[:-4] + '.vad'
    if not os.path.exists(vad_file):
        return file_name, None
    try:
        fs, signal = wavfile.read(file_name)
    except:
        print("ignoring {}, wrong format".format(file_name), file=sys.stderr)
        return file_name, None
    with open(vad_file, 'r') as fh:
        vad_string = fh.read()
    vad = np.fromstring(vad_string, sep=' ')
    return file_name, calculate_snr(signal, vad, speech_th=speech_th, noise_th=noise_th, fs=fs)


def cal_snr_librivox(file_name):
    return calculate_file_snr(file_name, speech_th=0.8, noise_th=0.995)


def mp_file_snr(lst_file, records=None, nproc=60):
    with open(lst_file, 'r') as fh:
        fnames = [line.split()[0] for line in fh]
    if records is not None:
        with open(records, 'r') as fh:
            existing_fname = [line.split()[0] for line in fh]
        fnames = set(fnames) - set(existing_fname)
    print("loaded {} file to process".format(len(fnames)), file=sys.stderr)
    pool = multiprocessing.Pool(nproc)
    print("processing librivox format", file=sys.stderr)
    it = pool.imap_unordered(cal_snr_librivox, fnames)
    st = time.time()
    cnt = 0
    for fname, snr_fields in it:
        cnt += 1
        if snr_fields is not None:
            sys.stdout.write('{}\t{}\t{}\t{}\n'.format(
                *([fname] + snr_fields)))
        if cnt % 1000 == 0 and cnt != 0:
            dur = time.time() - st
            print("{} file/s".format(cnt / dur), file=sys.stderr)
            st = time.time()
            cnt = 0


if __name__ == "__main__":
    usage = """
    example: python calculate_snr.py librivox.lst > snr_output.tsv
    """
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument("wav_list", type=str,
                        help="list path to wavs. oneline per file")
    parser.add_argument("--resume_from", type=str,
                        help="if specified, all entries in the resume-from file will be skipped")
    parser.add_argument("--numproc", type=int, default=40,
                        help="num of processes")
    args = parser.parse_args()
    mp_file_snr(args.wav_list, records=args.resume_from, nproc=args.numproc)
