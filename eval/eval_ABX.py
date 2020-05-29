# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import sys
import torch
import json
import os
import numpy as np
import ABX_src.abx_group_computation as abx_g
import ABX_src.abx_iterators as abx_it
from CPC_loader import load_cpc_features, build_feature_from_file
from pathlib import Path


def find_all_files(path_dir, extension):
    out = []
    for root, dirs, filenames in os.walk(path_dir):
        for f in filenames:
            if f.endswith(extension):
                out.append(((str(Path(f).stem)), os.path.join(root, f)))
    return out


def reduce_sparse_data(quotient, divisor):
    return quotient / (1e-08 * (divisor == 0) + divisor)


def load_pt(x):
    data = torch.load(x, 'cpu')
    assert(len(data.size()) == 2)
    return data


def load_npy(x):
    data = torch.tensor(np.load(x))
    assert(len(data.size()) == 2)
    return data


def ABX(feature_function,
        path_item_file,
        seq_list,
        distance_mode,
        step_feature,
        modes,
        cuda=False,
        max_x_across=5,
        max_size_group=30):

    # ABX dataset
    ABXDataset = abx_it.ABXFeatureLoader(path_item_file, seq_list,
                                         feature_function, step_feature, True)

    if cuda:
        ABXDataset.cuda()

    # Distance function
    distance_function = abx_g.get_distance_function_from_name(distance_mode)

    # Output
    scores = {}

    # ABX within
    if 'within' in modes:
        print("Computing ABX within speakers...")
        ABXIterator = ABXDataset.get_iterator('within', max_size_group)
        group_confusion = abx_g.get_abx_scores_dtw_on_group(ABXIterator,
                                                            distance_function,
                                                            ABXIterator.symmetric)
        n_data = group_confusion._values().size(0)
        index_ = torch.sparse.LongTensor(group_confusion._indices(),
                                         torch.ones((n_data),
                                                    dtype=torch.float),
                                         group_confusion.size())
        divisor_context = torch.sparse.sum(index_, dim=3).to_dense()
        group_confusion = torch.sparse.sum(group_confusion, dim=3).to_dense()
        group_confusion = reduce_sparse_data(group_confusion, divisor_context)
        S, p1, p2 = group_confusion.size()

        index_speaker = divisor_context > 0
        divisor_speaker = index_speaker.sum(dim=0)
        phone_confusion = reduce_sparse_data(group_confusion.sum(dim=0),
                                             divisor_speaker)

        scores['within'] = (phone_confusion.sum() /
                            (divisor_speaker > 0).sum()).item()
        print(f"...done. ABX within : {scores['within']}")

    # ABX across
    if 'across' in modes:
        print("Computing ABX across speakers...")
        ABXIterator = ABXDataset.get_iterator('across', max_size_group)
        ABXIterator.max_x = max_x_across
        group_confusion = abx_g.get_abx_scores_dtw_on_group(ABXIterator,
                                                            distance_function,
                                                            ABXIterator.symmetric)
        n_data = group_confusion._values().size(0)
        index_ = torch.sparse.LongTensor(group_confusion._indices(),
                                         torch.ones((n_data),
                                                    dtype=torch.float),
                                         group_confusion.size())
        divisor_context = torch.sparse.sum(index_, dim=[3, 4]).to_dense()
        group_confusion = torch.sparse.sum(
            group_confusion, dim=[3, 4]).to_dense()
        group_confusion = reduce_sparse_data(group_confusion, divisor_context)
        S, p1, p2 = group_confusion.size()

        index_speaker = divisor_context > 0
        divisor_speaker = index_speaker.sum(dim=0)
        phone_confusion = reduce_sparse_data(group_confusion.sum(dim=0),
                                             divisor_speaker)
        scores['across'] = (phone_confusion.sum() /
                            (divisor_speaker > 0).sum()).item()
        print(f"...done. ABX across : {scores['across']}")

    return scores


def parse_args(argv):

    parser = argparse.ArgumentParser(description='ABX metric')

    parser.add_argument('path_data', type=str,
                        help="Path to directory containing the data")
    parser.add_argument('path_item_file', type=str,
                        help="Path to the .item file")
    parser.add_argument('--path_checkpoint', type=str, default=None,
                        help="Path to a CPC checkpoint. If set, the apply the "
                        "model to the input data to compute the features")
    parser.add_argument('--file_extension', type=str, default='.pt',
                        choices=['.pt', '.npy', '.wav', '.flac', '.mp3'])
    parser.add_argument('--feature_size', type=float, default=0.01,
                        help="Size (in s) of one feature")
    parser.add_argument('--cuda', action='store_true',
                        help="Use the GPU to compute distances")
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'within', 'across'],
                        help="Choose the mode of the ABX score to compute")
    parser.add_argument('--distance_mode', type=str, default='cosine',
                        choices=['euclidian', 'cosine', 'kl', 'kl_symmetric'],
                        help="Choose the kind of distance to use to compute "
                        "the ABX score.")
    parser.add_argument("--max_size_group", type=int, default=10,
                        help="Max size of a group while computing the"
                             "ABX score. A small value will make the code "
                             "faster but less precise.")
    parser.add_argument("--max_x_across", type=int, default=5,
                        help="When computing the ABX across score, maximum"
                             "number of speaker X to sample per couple A,B. "
                             " A small value will make the code faster but "
                             "less precise.")
    parser.add_argument("--out", type=str, default=None,
                        help="Path where the results should be saved")

    # multi-gpu / multi-node
    return parser.parse_args(argv)


def main(argv):

    args = parse_args(argv)

    if args.path_checkpoint is None:
        if args.file_extension == '.pt':
            feature_function = load_pt
        elif args.file_extension == '.npy':
            feature_function = load_npy
    else:
        state_dict = torch.load(args.path_checkpoint)
        feature_maker = load_cpc_features(state_dict)
        feature_maker.cuda()
        def feature_function(
            x): return build_feature_from_file(x, feature_maker)

    # Modes
    if args.mode == 'all':
        modes = ["within", "across"]
    else:
        modes = [args.mode]

    step_feature = 1 / args.feature_size

    # Get the list of sequences
    seq_list = find_all_files(args.path_data, args.file_extension)

    scores = ABX(feature_function, args.path_item_file,
                 seq_list, args.distance_mode,
                 step_feature, modes,
                 cuda=args.cuda,
                 max_x_across=args.max_x_across,
                 max_size_group=args.max_size_group)

    out_dir = Path(args.path_checkpoint).parent if args.out is None \
        else Path(args.out)
    out_dir.mkdir(exist_ok=True)

    path_score = out_dir / 'ABX_scores.json'
    with open(path_score, 'w') as file:
        json.dump(scores, file, indent=2)

    path_args = out_dir / 'ABX_args.json'
    with open(path_args, 'w') as file:
        json.dump(vars(args), file, indent=2)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
