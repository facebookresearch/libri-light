# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import os
import torchaudio
import torch
import time
import json
import progressbar
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import jiwer
from pathlib import Path

from WER_src.letter_ctc import LetterClassifier, CTCLetterCriterion, cut_data
from WER_src.simple_dataset import SingleSequenceDataset, parse_ctc_labels_from_root, find_seqs
from WER_src.wl_decoder import WlDecoder

from CPC_loader import load_cpc_features, get_features_state_dict


def prepare_data(data, put_on_cuda=True):
    if put_on_cuda:
        data = [x.cuda() for x in data]

    seq, seq_size, labels, label_lengths = data
    seq_size = seq_size.view(-1)
    label_lengths = label_lengths.view(-1)
    seq = cut_data(seq, seq_size)

    return seq, seq_size, labels, label_lengths


def train_step(loader,
               criterion,
               optimizer,
               downsampling_factor):

    criterion.train()
    avg_loss = 0
    n_items = 0

    for data in loader:
        optimizer.zero_grad()
        seq, seq_size, labels, labels_size = prepare_data(data)
        seq_size = seq_size // downsampling_factor
        loss = criterion(seq, seq_size, labels, labels_size)
        loss.sum().backward()

        avg_loss += loss.mean().item()
        n_items += 1
        optimizer.step()

    return avg_loss / n_items


def val_step(loader,
             criterion,
             downsampling_factor):
    criterion.eval()
    avg_loss = 0
    n_items = 0

    for data in loader:
        with torch.no_grad():
            seq, seq_size, phone, label_size = prepare_data(data)
            seq_size = seq_size / downsampling_factor
            loss = criterion(seq, seq_size, phone, label_size)
            avg_loss += loss.mean().item()
            n_items += 1

    return avg_loss / n_items


class Worker:
    def __init__(self, lm_weight, index2letter, in_q, out_q):
        self.index2letter = index2letter
        self.lm_weight = lm_weight
        self.in_q, self.out_q = in_q, out_q

    def __call__(self):
        decoder = WlDecoder(lm_weight=self.lm_weight)
        while True:
            task = self.in_q.get()
            if task is None:
                break
            wer = self.get_wer(task, decoder)
            self.out_q.put(wer)
            self.in_q.task_done()

    def get_wer(self, task, decoder):
        predictions, labels_slice = task

        letters = decoder.predictions(predictions)
        decoded = ''.join(letters)
        truth = [self.index2letter[x] for x in labels_slice.tolist()]
        truth = [(x if x != '|' else ' ') for x in truth]
        truth = ''.join(truth)

        wer = jiwer.wer(truth, decoded)
        return wer


def eval_wer(loader,
             criterion,
             lm_weight,
             index2letter,
             n_processes=32):

    criterion.eval()

    bar = progressbar.ProgressBar(len(loader))
    bar.start()

    task_q, result_q = mp.JoinableQueue(), mp.Queue()
    processes = []
    for _ in range(n_processes):
        p = mp.Process(target=Worker(
            lm_weight, index2letter, task_q, result_q))
        p.start()
        processes.append(p)

    tasks_fed = 0
    mean_wer = 0.0
    results = 0.0

    for index, data in enumerate(loader):
        bar.update(index)
        batch_size = data[0].size(0)
        tasks_fed += batch_size

        with torch.no_grad():
            seq, seq_lengths, labels, label_lengths = prepare_data(
                data, put_on_cuda=False)
            seq = seq.cuda()

            predictions = criterion.letter_classifier(
                seq).log_softmax(dim=-1).cpu()

        for k in range(batch_size):
            p_ = predictions[k, :, :]
            labels_ = (labels[k, :label_lengths[k]])
            task_q.put((p_,  labels_))

        task_q.join()
        while not result_q.empty():
            mean_wer += result_q.get()
            results += 1
        assert results == tasks_fed
    bar.finish()

    for _ in processes:
        task_q.put(None)

    for p in processes:
        p.join()

    mean_wer /= results
    return mean_wer


def run(train_loader,
        val_loader,
        criterion,
        optimizer,
        downsampling_factor,
        n_epochs,
        path_checkpoint):

    print(f"Starting the training for {n_epochs} epochs")
    best_loss = float('inf')

    for epoch in range(n_epochs):
        loss_train = train_step(train_loader, criterion,
                                optimizer, downsampling_factor)

        print(f"Epoch {epoch} loss train : {loss_train}")

        loss_val = val_step(val_loader, criterion, downsampling_factor)
        print(f"Epoch {epoch} loss val : {loss_val}")

        if loss_val < best_loss:
            state_dict = criterion.state_dict()
            torch.save(state_dict, path_checkpoint)
            best_loss = loss_val


def get_eval_args(args):
    pathArgsTraining = os.path.join(args.output, "args_training.json")
    with open(pathArgsTraining, 'rb') as f:
        data = json.load(f)

    args.path_checkpoint = data["path_checkpoint"]
    args.dropout = data.get("dropout", False)
    return args


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args):
    description = "Training and evaluation of a letter classifier on top of a pre-trained CPC model. "
    "Please specify at least one `path_wer` (to calculate WER) or `path_train` and `path_val` (for training)."

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--path_checkpoint', type=str)
    parser.add_argument('--path_train', default=None, type=str)
    parser.add_argument('--path_val', default=None, type=str)
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--downsampling_factor', type=int, default=160)

    parser.add_argument('--lr', type=float, default=2e-04)
    parser.add_argument('--output', type=str, default='out',
                        help="Output directory")
    parser.add_argument('--p_dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--lm_weight', type=float, default=2.0)
    parser.add_argument('--path_wer',
                        help="For computing the WER on specific sequences",
                        action='append')
    parser.add_argument('--letters_path', type=str,
                        default='WER_data/letters.lst')

    args = parser.parse_args(args=args)

    if not args.path_wer and not (args.path_train and args.path_val):
        print('Please specify at least one `path_wer` (to calculate WER) or `path_train` and `path_val` (for training).')

    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    # creating models before reading the datasets
    with open(args.letters_path) as f:
        n_chars = len(f.readlines())

    state_dict = torch.load(args.path_checkpoint)
    feature_maker = load_cpc_features(state_dict)
    feature_maker.cuda()
    hidden = feature_maker.get_output_dim()

    letter_classifier = LetterClassifier(
        feature_maker,
        hidden, n_chars, p_dropout=args.p_dropout if hasattr(args, 'p_dropout') else 0.0)

    criterion = CTCLetterCriterion(letter_classifier, n_chars)
    criterion.cuda()
    criterion = torch.nn.DataParallel(criterion)

    # Checkpoint file where the model should be saved
    path_checkpoint = os.path.join(args.output, 'checkpoint.pt')

    if args.path_train and args.path_val:
        set_seed(args.seed)

        char_labels_val, n_chars, _ = parse_ctc_labels_from_root(
            args.path_val, letters_path="./WER_data/letters.lst")
        print(f"Loading the validation dataset at {args.path_val}")
        dataset_val = SingleSequenceDataset(args.path_val, char_labels_val)
        val_loader = DataLoader(dataset_val, batch_size=args.batch_size,
                                shuffle=False)

        # train dataset
        char_labels_train, n_chars, _ = parse_ctc_labels_from_root(
            args.path_train, letters_path="./WER_data/letters.lst")

        print(f"Loading the training dataset at {args.path_train}")
        dataset_train = SingleSequenceDataset(
            args.path_train, char_labels_train)
        train_loader = DataLoader(dataset_train, batch_size=args.batch_size,
                                  shuffle=True)

        # Optimizer
        g_params = list(criterion.parameters())
        optimizer = torch.optim.Adam(g_params, lr=args.lr)

        args_path = os.path.join(args.output, "args_training.json")
        with open(args_path, 'w') as file:
            json.dump(vars(args), file, indent=2)

        run(train_loader, val_loader, criterion,
            optimizer, args.downsampling_factor, args.n_epochs, path_checkpoint)

    if args.path_wer:
        args = get_eval_args(args)

        state_dict = torch.load(path_checkpoint)
        criterion.load_state_dict(state_dict)
        criterion = criterion.module
        criterion.eval()

        args_path = os.path.join(args.output, "args_validation.json")
        with open(args_path, 'w') as file:
            json.dump(vars(args), file, indent=2)

        for path_wer in args.path_wer:
            print(f"Loading the validation dataset at {path_wer}")

            char_labels_wer, _, (letter2index, index2letter) = parse_ctc_labels_from_root(
                path_wer, letters_path="./WER_data/letters.lst")
            dataset_eval = SingleSequenceDataset(path_wer, char_labels_wer)
            eval_loader = DataLoader(
                dataset_eval, batch_size=args.batch_size, shuffle=False)

            wer = eval_wer(eval_loader,
                           criterion,
                           args.lm_weight,
                           index2letter)
            print(f'WER: {wer}')


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
