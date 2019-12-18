# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from pathlib import Path
import sys
import json

import metadata_completion.utilities as ut
from metadata_completion.GenreScrapper import gather_all_genres
from metadata_completion.ReaderScapper import update_all_speaker_data
from metadata_completion.genre_folding import UNIQUE_GENRE_FOLDING, \
    SUPER_GENDER_FOLDING, \
    SUPER_GENDER_ORDERING
from metadata_completion.DuplicateSearch import get_books_duplicates
from metadata_completion.text_cleaner import clean_all_text_data


def parse_args():
    parser = argparse.ArgumentParser(description="Upgrade LibriBIG's metadata")
    parser.add_argument('--path_metadata', type=str,
                        help="Path to the directory containing the metadata",
                        default="/checkpoint/mriviere/LibriVox")
    parser.add_argument('--ignore_cache', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser_out = parser.add_mutually_exclusive_group(required=False)
    parser_out.add_argument('--out_dir', type=str, default=None,
                            help="Path to the output directory")
    parser_out.add_argument('-i', '--in_place', action='store_true')
    return parser


def main(argv):
    parser = parse_args()
    args = parser.parse_args(argv)

    if args.in_place:
        path_out = Path(args.path_metadata)
    elif args.out_dir is not None:
        path_out = Path(args.out_dir)
        Path.mkdir(path_out, exist_ok=True)
    else:
        print(f"You must input either an output directory or activate the "
              "inplace flag")
        parser.print_help()
        sys.exit()

    path_cache = path_out / ".cache"
    Path.mkdir(path_cache, exist_ok=True)

    path_global_data_dir = path_out / "global"
    Path.mkdir(path_global_data_dir, exist_ok=True)

    # Get the list of all metadata
    print("Gathering the list of metadata")
    path_cache_metadata = path_cache / "metadata.pkl"
    list_metadata = ut.load_cache(path_cache_metadata,
                                  ut.get_all_metadata,
                                  args=(args.path_metadata,),
                                  ignore_cache=args.ignore_cache)

    if args.debug:
        list_metadata = list_metadata[:10]

    # Retrieve the genres
    genre_list = gather_all_genres(args.path_metadata,
                                   list_metadata)

    ut.get_updated_metadata(genre_list, args.path_metadata, path_out, "genre")

    # Fold the genres
    reverse_folding_unique = ut.build_reverse_folding(UNIQUE_GENRE_FOLDING)
    reverse_folding_super = ut.build_reverse_folding(SUPER_GENDER_FOLDING)
    final_reverse_folding = ut.combine_reverse_foldings(reverse_folding_super,
                                                        reverse_folding_unique)

    # Convert the "dramatic reading" option into a binary tag
    has_dramatic_reading = [(name, 'Dramatic Readings' in vals)
                            for name, vals in genre_list]
    ut.get_updated_metadata(has_dramatic_reading, path_out,
                            path_out, 'Dramatic Readings')
    genre_list = [(name, ut.remove_tag(vals, 'Dramatic Readings', 'Undefined'))
                  for name, vals in genre_list]

    #dramatric_reading = [(name, ut.has_tag(tag_str, tag))]
    folded_genres = [(name, ut.remove_multiple_tags(ut.apply_folding('+'.join(vals),
                                                                     final_reverse_folding),
                                                    SUPER_GENDER_ORDERING))
                     for name, vals in genre_list]

    ut.get_updated_metadata(folded_genres, path_out,
                            path_out, "meta_genre")

    # Retrieve the readers names
    update_all_speaker_data(list_metadata, args.path_metadata, path_out)

    # Look for duplicates
    duplicate_list = get_books_duplicates(args.path_metadata, list_metadata)
    path_out_duplicates = path_global_data_dir / "duplicates.json"
    print(f"Saving the duplicates index at {path_out_duplicates}")
    with open(path_out_duplicates, 'w') as file:
        json.dump(duplicate_list, file, indent=2)

    # Clean text data when possible
    text_status = clean_all_text_data(list_metadata, args.path_metadata,
                                      str(path_out))
    ut.get_updated_metadata(text_status, path_out,
                            path_out, "trancription_status")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
