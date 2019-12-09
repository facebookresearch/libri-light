# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List

import torch

from torch.utils.data import Dataset
import torchaudio
from copy import deepcopy
import time
from pathlib import Path


def parse_ctc_labels_from_root(root, letters_path="./WER_data/letters.lst"):
    letter2index = {}
    index2letter = {}

    with open(letters_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip()

            letter2index[line] = i
            index2letter[i] = line

    result = {}

    for file in Path(root).rglob("*.txt"):
        with open(file, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                p = line.find(' ')
                assert p > 0
                fname = line[:p]

                chars = line[p+1:].replace(' ', '|').lower()
                decoded = []

                for c in chars:
                    decoded.append(letter2index[c])
                result[fname] = decoded

    return result, len(letter2index), (letter2index, index2letter)


def find_seqs(dir_name, extension='.flac'):
    sequences = []
    for file in Path(dir_name).rglob('*' + extension):
        speaker = file.parent.parent.stem
        sequences.append((speaker, file))

    speakers = set(x[0] for x in sequences)
    return sequences, speakers


class SingleSequenceDataset(Dataset):

    def __init__(self,
                 root: str,
                 labels: Dict[str, List[int]]):
        """
        root {str} -- Directory that contains the dataset files
        labels {Dict[str, List[int]]} -- Dict mapping a filename (without extension) to a list of 
                integer-encoded labels. 
        """
        self.seq_names, _ = find_seqs(root)
        self.labels_dict = deepcopy(labels)

        self.seq_offsets = [0]
        self.labels = []
        self.label_offsets = [0]
        self.data = []
        self.max_len_wave = 0
        self.max_len_labels = 0

        self.load_seqs()

    def load_seqs(self):
        data = []

        start_time = time.time()
        for _, seq in self.seq_names:
            name = Path(seq).stem
            wave = torchaudio.load(seq)[0].view(-1)
            data.append((name, wave))

        data.sort()

        temp_data = []
        total_size = 0
        for name, wave in data:
            self.labels.extend(self.labels_dict[name])
            self.label_offsets.append(len(self.labels))
            self.max_len_labels = max(
                self.max_len_labels, len(self.labels_dict[name]))
            wave_length = wave.size(0)
            self.max_len_wave = max(self.max_len_wave, wave_length)
            total_size += wave_length
            temp_data.append(wave)
            self.seq_offsets.append(self.seq_offsets[-1] + wave_length)

        self.data = torch.cat(temp_data, dim=0)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

        print(f'Loaded {len(self.label_offsets)} sequences '
              f'in {time.time() - start_time:.2f} seconds')
        print(f'max_len_wave: {self.max_len_wave}')
        print(f'max_len_labels: {self.max_len_labels}')
        print(f'Total size dataset {total_size / (16000 * 3600)} hours')

    def __getitem__(self, idx):
        wave_tart = self.seq_offsets[idx]
        wave_end = self.seq_offsets[idx + 1]
        labels_start = self.label_offsets[idx]
        labels_end = self.label_offsets[idx + 1]

        wave_len = wave_end - wave_tart
        label_len = labels_end - labels_start

        wave = torch.zeros((1, self.max_len_wave))
        labels = torch.zeros((self.max_len_labels), dtype=torch.long)

        wave[0, :wave_len] = self.data[wave_tart:wave_end]
        labels[:label_len] = self.labels[labels_start:labels_end]

        return wave,  torch.tensor([wave_len], dtype=torch.long), labels,  torch.tensor([label_len], dtype=torch.long)

    def __len__(self):
        return len(self.seq_offsets) - 1
