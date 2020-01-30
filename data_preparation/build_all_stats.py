# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from pathlib import Path
import metadata_completion.utilities as ut
import plot

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Build the statistics on LibriBig")
    parser.add_argument('path_data', type=str,
                        help="Path to the directory containing the data")
    parser.add_argument('out_dir', type=str,
                        help="Path to the output directory")
    parser.add_argument('--ignore_cache', action='store_true')
    args = parser.parse_args()

    # Build the output directory
    args.out_dir = Path(args.out_dir)
    Path.mkdir(args.out_dir, exist_ok=True)

    # Build the cache directory
    path_cache = args.out_dir / ".cache"
    Path.mkdir(path_cache, exist_ok=True)

    # Get the list of all metadata
    print("Gathering the list of metadata")
    path_cache_metadata = path_cache / "metadata.pkl"
    list_metadata = ut.load_cache(path_cache_metadata,
                                  ut.get_all_metadata,
                                  args=(args.path_data, ".json"),
                                  ignore_cache=args.ignore_cache)
    print(f"{len(list_metadata)} files found")

    # Get the genre statistics
    print("Building the genre statistics")
    path_genre_stats = path_cache / "meta_genre_stats.json"
    genre_data = ut.load_cache(path_genre_stats,
                               ut.get_hour_tag_repartition,
                               args=(list_metadata,
                                     "meta_genre", ".flac"),
                               ignore_cache=args.ignore_cache)

    path_tags_hist = args.out_dir / "meta_genres.png"
    plot.plot_pie(genre_data, str(path_tags_hist),
                  title="Genre's categories (in hours)")

    print("done.")

    # Get the speaker statistics
    print("Building the speaker statistics")
    path_speaker_cache = path_cache / "speaker_stats.json"
    speaker_data = ut.load_cache(path_speaker_cache,
                                 ut.get_speaker_hours_data,
                                 args=(list_metadata,
                                       ".flac"))

    speaker_hours = [x for _, x in speaker_data.items()]
    path_speaker_hist = args.out_dir / "speaker_data.png"
    n_bins = 100
    plot.plot_hist(speaker_hours, n_bins, str(path_speaker_hist),
                   title="Time spoken per speaker",
                   y_label="Number of speakers", normalized=False,
                   y_scale='log', x_label="Time spoken in hours")
    print("done.")
