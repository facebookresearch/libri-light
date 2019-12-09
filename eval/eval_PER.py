# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import os
import subprocess
import sys
import json
import random
from pathlib import Path
from CPC_loader import load_cpc_features, get_features_state_dict
import PER_src.simplePhonemLearner as per_src
from torch.utils.data import DataLoader
from PER_src.seq_alignment import per_step
import torch


def find_all_files(path_dir, extension):
    out = []
    for root, dirs, filenames in os.walk(path_dir):
        for f in filenames:
            if f.endswith(extension):
                out.append(((str(Path(f).stem)), os.path.join(root, f)))
    return out


def parse_phone_labels(path_labels):
    with open(path_labels, 'r') as file:
        data = file.readlines()

    out = {}
    for line in data:
        words = line.split()
        out[words[0]] = [int(x) for x in words[1:]]
    return out


def get_n_phones(path_phone_converter):
    with open(path_phone_converter, 'rb') as file:
        return len(json.load(file))


def filter_seq(seq_list, path_selection):
    with open(path_selection, 'r') as file:
        selection = [p.replace('\n', '') for p in file.readlines()]

    selection.sort()
    seq_list.sort(key=lambda x: Path(x).stem)
    output, index = [], 0
    for x in seq_list:
        seq = str(Path(x).stem)
        while index < len(selection) and seq > selection[index]:
            index += 1
        if index == len(selection):
            break
        if seq == selection[index]:
            output.append(x)
    return output


def run_training(trainLoader,
                 valLoader,
                 model,
                 criterion,
                 optimizer,
                 downsamplingFactor,
                 nEpochs,
                 pathCheckpoint):

    print(f"Starting the training for {nEpochs} epochs")
    bestLoss = 2000000

    for epoch in range(nEpochs):
        lossTrain = per_src.trainStep(trainLoader, model, criterion,
                                      optimizer, downsamplingFactor)

        print(f"Epoch {epoch} loss train : {lossTrain}")

        lossVal = per_src.valStep(valLoader, model, criterion,
                                  downsamplingFactor)
        print(f"Epoch {epoch} loss val : {lossVal}")

        if lossVal < bestLoss:
            bestLoss = lossVal
            stateDict = {'classifier': criterion.module.state_dict(),
                         'model': get_features_state_dict(model.module),
                         'bestLoss': bestLoss}
            torch.save(stateDict, pathCheckpoint)


def train(args):

    # Output Directory
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    name = f"_{args.name}" if args.command == "per" else ""
    pathLogs = os.path.join(args.output, f'logs_{args.command}{name}.txt')
    tee = subprocess.Popen(["tee", pathLogs], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())

    nPhones = get_n_phones(args.path_phone_converter)
    phoneLabels = parse_phone_labels(args.pathPhone)
    inSeqs = find_all_files(args.pathDB, args.file_extension)

    # Model
    downsamplingFactor = 160
    state_dict = torch.load(args.pathCheckpoint)
    featureMaker = load_cpc_features(state_dict)
    hiddenGar = featureMaker.get_output_dim()
    featureMaker.cuda()
    featureMaker = torch.nn.DataParallel(featureMaker)

    # Criterion
    phoneCriterion = per_src.CTCPhoneCriterion(hiddenGar, nPhones, args.LSTM,
                                               seqNorm=args.seqNorm,
                                               dropout=args.dropout,
                                               reduction=args.loss_reduction)
    phoneCriterion.cuda()
    phoneCriterion = torch.nn.DataParallel(phoneCriterion)

    # Datasets
    if args.command == 'train' and args.pathTrain is not None:
        seqTrain = filter_seq(args.pathTrain, inSeqs)
    else:
        seqTrain = inSeqs

    if args.pathVal is None:
        random.shuffle(seqTrain)
        sizeTrain = int(0.9 * len(seqTrain))
        seqTrain, seqVal = seqTrain[:sizeTrain], seqTrain[sizeTrain:]
    elif args.pathVal is not None:
        seqVal = filter_seq(args.pathVal, inSeqs)
        print(len(seqVal), len(inSeqs), args.pathVal)

    if args.debug:
        seqVal = seqVal[:100]

    print(f"Loading the validation dataset at {args.pathDB}")
    datasetVal = per_src.SingleSequenceDataset(args.pathDB, seqVal,
                                               phoneLabels, inDim=args.in_dim)

    valLoader = DataLoader(datasetVal, batch_size=args.batchSize,
                           shuffle=True)

    # Checkpoint file where the model should be saved
    pathCheckpoint = os.path.join(args.output, 'checkpoint.pt')

    featureMaker.optimize = True
    if args.freeze:
        featureMaker.eval()
        featureMaker.optimize = False
        for g in featureMaker.parameters():
            g.requires_grad = False

    if args.debug:
        print("debug")
        random.shuffle(seqTrain)
        seqTrain = seqTrain[:1000]
        seqVal = seqVal[:100]

    print(f"Loading the training dataset at {args.pathDB}")
    datasetTrain = per_src.SingleSequenceDataset(args.pathDB, seqTrain,
                                                 phoneLabels,
                                                 inDim=args.in_dim)

    trainLoader = DataLoader(datasetTrain, batch_size=args.batchSize,
                             shuffle=True)

    # Optimizer
    g_params = list(phoneCriterion.parameters())
    if not args.freeze:
        print("Optimizing model")
        g_params += list(featureMaker.parameters())

    optimizer = torch.optim.Adam(g_params, lr=args.lr,
                                 betas=(args.beta1, args.beta2),
                                 eps=args.epsilon)

    pathArgs = os.path.join(args.output, "args_training.json")
    with open(pathArgs, 'w') as file:
        json.dump(vars(args), file, indent=2)

    run_training(trainLoader, valLoader, featureMaker, phoneCriterion,
                 optimizer, downsamplingFactor, args.nEpochs, pathCheckpoint)


def per(args):

    # Load the model
    state_dict = torch.load(args.path_checkpoint)
    feature_maker = load_cpc_features(state_dict["model"])
    feature_maker.cuda()
    feature_maker.eval()
    hidden_gar = feature_maker.get_output_dim()

    # Get the model training configuration
    path_config = Path(args.path_checkpoint).parent / "args_training.json"
    with open(path_config, 'rb') as file:
        config_training = json.load(file)

    n_phones = get_n_phones(config_training["path_phone_converter"])
    phone_criterion = per_src.CTCPhoneCriterion(hidden_gar, n_phones, config_training["LSTM"],
                                                seqNorm=config_training["seqNorm"],
                                                dropout=config_training["dropout"],
                                                reduction=config_training["loss_reduction"])
    phone_criterion.load_state_dict(state_dict["classifier"])
    phone_criterion.cuda()
    downsamplingFactor = 160

    # dataset
    inSeqs = find_all_files(args.pathDB, args.file_extension)
    phoneLabels = parse_phone_labels(args.pathPhone)

    datasetVal = per_src.SingleSequenceDataset(args.pathDB, inSeqs,
                                               phoneLabels, inDim=1)
    valLoader = DataLoader(datasetVal, batch_size=args.batchSize,
                           shuffle=False)

    per_step(valLoader, feature_maker, phone_criterion, downsamplingFactor)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description='Trainer')

    subparsers = parser.add_subparsers(dest='command')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('pathDB', type=str)
    parser_train.add_argument('pathPhone', type=str)
    parser_train.add_argument('path_phone_converter', type=str)
    parser_train.add_argument('pathCheckpoint', type=str)
    parser_train.add_argument('--freeze', action='store_true',
                              help="Freeze the CPC layers")
    parser_train.add_argument('--pathTrain', default=None, type=str)
    parser_train.add_argument('--pathVal', default=None, type=str)
    parser_train.add_argument('--file_extension', type=str, default=".flac")
    parser_train.add_argument('--batchSize', type=int, default=8)
    parser_train.add_argument('--nEpochs', type=int, default=30)
    parser_train.add_argument('--beta1', type=float, default=0.9)
    parser_train.add_argument('--beta2', type=float, default=0.999)
    parser_train.add_argument('--epsilon', type=float, default=1e-08)
    parser_train.add_argument('--lr', type=float, default=2e-04)
    parser_train.add_argument('-o', '--output', type=str, default='out',
                              help="Output directory")
    parser_train.add_argument('--debug', action='store_true')
    parser_train.add_argument('--no_pretraining', action='store_true')
    parser_train.add_argument('--LSTM', action='store_true')
    parser_train.add_argument('--seqNorm', action='store_true')
    parser_train.add_argument('--kernelSize', type=int, default=8)
    parser_train.add_argument('--dropout', action='store_true')
    parser_train.add_argument('--in_dim', type=int, default=1)
    parser_train.add_argument('--loss_reduction', type=str, default='mean',
                              choices=['mean', 'sum'])

    parser_per = subparsers.add_parser('per')
    parser_per.add_argument('path_checkpoint', type=str)
    parser_per.add_argument('pathDB',
                            help="For computing the PER on another dataset",
                            type=str)
    parser_per.add_argument('pathPhone',
                            help="For computing the PER on specific sequences",
                            default=None)
    parser_per.add_argument('--batchSize', type=int, default=8)
    parser_per.add_argument('--debug', action='store_true')
    parser_per.add_argument('--file_extension', type=str, default=".flac")
    parser_per.add_argument('--name', type=str, default="0")

    args = parser.parse_args()

    if args.command == 'train':
        train(args)
    elif args.command == 'per':
        per(args)
