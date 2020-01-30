# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from pathlib import Path
import pickle
import json
import torchaudio
import progressbar
import argparse
import os
import matplotlib
matplotlib.use('agg')


def get_all_metadata(path_dir, suffix="_metadata.json"):
    out = []
    for root, dirs, filenames in os.walk(path_dir):
        for f in filenames:
            if f.endswith(suffix):
                out.append(os.path.join(root, f))
    return out


def get_base_name_from_metadata(path):
    return os.path.basename(path)[:-14]


def get_zip_name(pathMetadata):
    return f'{get_base_name_from_metadata(pathMetadata)}.zip'


def get_wav_name(pathMetadata):
    return get_base_name_from_metadata(pathMetadata).replace('64kb_mp3', 'wav')


def get_txt_name(pathMetadata):
    return f'{get_base_name_from_metadata(pathMetadata)}_text.txt'


def get_speaker_data_name(pathMetadata):
    return f'{get_base_name_from_metadata(pathMetadata)}_speaker_data.json'


def getJSON(pathJSON):
    with open(pathJSON, 'rb') as file:
        return json.load(file)


def get_updated_metadata(update, path_dir_in, path_dir_out, tag):

    print(f"Updating metadata with tag {tag}")
    n_items = len(update)
    bar = progressbar.ProgressBar(maxval=n_items)
    bar.start()

    for index, item in enumerate(update):
        bar.update(index)
        metadada_name, new_value = item
        full_path = Path(path_dir_in) / metadada_name
        with open(str(full_path), 'rb') as file:
            data = json.load(file)
        data[tag] = new_value
        out_path = Path(path_dir_out) / metadada_name
        with open(str(out_path), 'w') as file:
            data = json.dump(data, file, indent=2)
    bar.finish()


def save_cache(path_cache, data):
    path_cache = Path(path_cache)
    print(f"Saving a cache at {path_cache}")
    extension = path_cache.suffix
    if extension == ".json":
        with open(path_cache, 'w') as file:
            return json.dump(data, file, indent=2)
    elif extension == ".pkl":
        with open(path_cache, 'wb') as file:
            pickle.dump(data, file)
    else:
        raise ValueError(f"{extension} : Invalid format")


def load_cache(path_cache, fallback_function, args=None,
               save=True, ignore_cache=False):

    path_cache = Path(path_cache)
    if not path_cache.is_file() or ignore_cache:
        print(f"No cache found at {path_cache}")
    else:
        print(f"Loading the cached data at {path_cache}...")
        extension = path_cache.suffix
        if extension == ".json":
            try:
                with open(path_cache, 'rb') as file:
                    return json.load(file)
            except json.decoder.JSONDecodeError:
                print("Invalid cache.")
        elif extension == ".pkl":
            try:
                with open(path_cache, 'rb') as file:
                    return pickle.load(file)
            except pickle.UnpicklingError:
                print("Invalid cache.")
        else:
            raise ValueError(f"{extension} : Invalid format")
    out = fallback_function(*args)
    if save:
        save_cache(path_cache, out)
    return out


def strToHours(inputStr):

    hours, minutes, sec = map(float, inputStr.split(':'))
    return hours + minutes / 60.0 + sec / 3600.0


def getTotalTime(path_dir, list_metadata):

    totTime = 0

    for metadata in list_metadata:

        fullPath = os.path.join(path_dir, metadata)

        with open(fullPath) as file:
            data = json.load(file)

        try:
            size = strToHours(data['totaltime'])
            totTime += size
        except:
            continue

    return totTime


def get_speakers(pathSpeakerdata):

    with open(pathSpeakerdata, 'rb') as file:
        data = json.load(file)

    outData = set()

    if data["names"] is None or data["readers"] is None:
        return outData

    for items in data["readers"]:
        if items is not None:
            outData |= set(items)

    return outData


def get_all_speakers(path_dir, list_metadata):

    outSpeakers = set()
    for metadata in list_metadata:

        fullPath = os.path.join(path_dir, get_speaker_data_name(metadata))
        outSpeakers |= get_speakers(fullPath)

    return outSpeakers


def get_speaker_data(path_dir, list_metadata, pathWav):
    speakerTalk = {}
    nData = len(list_metadata)
    multiples = 0

    bar = progressbar.ProgressBar(maxval=nData)
    bar.start()
    for nM, metadataName in enumerate(list_metadata):

        bar.update(nM)
        zipName = get_zip_name(metadataName)
        wavName = zipName.replace("64kb_mp3.zip", "wav")
        speakerData = getJSON(os.path.join(path_dir,
                                           get_speaker_data_name(metadataName)))

        dirWav = os.path.join(pathWav, wavName)
        if not os.path.isdir(dirWav):
            continue

        listWav = [f'{f}.wav' for f in speakerData["names"]]

        for index, wavFile in enumerate(listWav):

            locPath = os.path.join(dirWav, wavFile)
            if not os.path.isfile(locPath):
                continue

            info = torchaudio.info(locPath)
            size = (info[0].length / info[0].rate) / 3600

            speakers = speakerData['readers'][index]

            if speakers is None:
                speakers = ['null']

            if len(speakers) > 1:
                multiples += size

            for IDspeaker in speakers:
                if IDspeaker not in speakerTalk:
                    speakerTalk[IDspeaker] = 0

                speakerTalk[IDspeaker] += size

    bar.finish()
    return speakerTalk, multiples


def get_speaker_hours_data(list_metadata, audio_extension):

    speakerTalk = {}
    nData = len(list_metadata)

    bar = progressbar.ProgressBar(maxval=nData)
    bar.start()

    for index, pathMetadata in enumerate(list_metadata):
        bar.update(index)
        with open(pathMetadata, 'rb') as file:
            locMetadata = json.load(file)

        speaker_name = locMetadata['speaker']

        path_audio_data = os.path.splitext(pathMetadata)[0] + audio_extension

        info = torchaudio.info(path_audio_data)[0]
        totAudio = info.length / (info.rate * 3600.)

        if speaker_name is None:
            speaker_name = 'null'

        if speaker_name not in speakerTalk:
            speakerTalk[speaker_name] = 0

        speakerTalk[speaker_name] += totAudio

    bar.finish()

    return speakerTalk


def get_hour_tag_repartition(list_metadata, tagName,
                             audio_extension):

    nItems = len(list_metadata)
    tags = {}

    bar = progressbar.ProgressBar(maxval=nItems)
    bar.start()

    for index, pathMetadata in enumerate(list_metadata):
        bar.update(index)
        with open(pathMetadata, 'rb') as file:
            locMetadata = json.load(file)

        value = locMetadata['book_meta'][tagName]

        path_audio_data = os.path.splitext(pathMetadata)[0] + audio_extension

        info = torchaudio.info(path_audio_data)[0]
        totAudio = info.length / (info.rate * 3600.)

        if value is None:
            value = 'null'

        if not isinstance(value, list):
            value = [value]

        full_tag = '+'.join(value)

        if full_tag not in tags:
            tags[full_tag] = 0

        tags[full_tag] += totAudio

    bar.finish()
    return tags


def get_tag_list(tagStats):
    out = set()
    for x in tagStats:
        out = out.union(set(x.split('+')))
    return out


def combine_reverse_foldings(f1, f2):
    r"""
    Compute f1 o f2
    """

    return {x: f1.get(f2[x], f2[x]) for x in f2}


def build_reverse_folding(gender_folding):
    out = {}
    for key, val_list in gender_folding.items():
        for val in val_list:
            out[val] = key

    return out


def apply_folding(tag_str, reverse_folding):
    tag_list = tag_str.split('+')

    new_tags = []

    for tag in tag_list:
        t = reverse_folding.get(tag, tag)
        if t not in new_tags:
            new_tags.append(t)
    new_tags.sort()
    return '+'.join(new_tags)


def remove_tag(tag_list, bad_tag, rescue_tag):
    out = [x for x in tag_list if x != bad_tag]
    if len(out) == 0:
        out = [rescue_tag]
    return out


def remove_multiple_tags(tag_str, order):
    tag_list = tag_str.split('+')
    return order[min([order.index(t) for t in tag_list])]


def get_metdata_from_id(path_dir, list_metadata, ID):

    for index, name_metadata in enumerate(list_metadata):
        pathMetadata = os.path.join(path_dir, name_metadata)
        with open(pathMetadata, 'r') as file:
            data = json.load(file)

        if data["id"] == ID:
            return data

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset tools')
    subparsers = parser.add_subparsers(dest='command')

    parser_info = subparsers.add_parser('info')
    parser_info.add_argument('path_dir', type=str)

    args = parser.parse_args()

    if args.command == 'info':
        print("*"*50)
        print(f"{args.path_dir} INFO :")
        print("*"*50)
        list_metadata = get_all_metadata(args.path_dir)
        print(f"{len(list_metadata)} books found")
        speakerList = get_all_speakers(args.path_dir, list_metadata)
        print(f"{len(speakerList)} speakers")
        time = getTotalTime(args.path_dir, list_metadata)
        print(f"{time} hours of data")
