from pathlib import Path
import pickle
import json
import numpy as np
import torchaudio
import progressbar
import argparse
import os
import matplotlib
matplotlib.use('agg')


def get_all_metadata(path_dir):
    return [f for f in os.listdir(path_dir) if len(f) > 14 and
            f[-14:] == "_metadata.json"]


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


def hasSpeakerData(pathSpeakerdata):

    if not os.path.isfile(pathSpeakerdata):
        return False

    with open(pathSpeakerdata, 'rb') as file:
        data = json.load(file)

    if data["names"] is None or data["readers"] is None:
        return False

    for item in data["readers"]:
        if item is None:
            return False

    return True


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


def getAllFileMultiVersion(path_dir, list_metadata):

    output = []
    for metadata_name in list_metadata:
        fullPath = os.path.join(path_dir, metadata_name)

        with open(fullPath, 'rb') as file:
            genre = json.load(file)["genre"]

        if genre is None:
            continue

        if "Multi-version (Weekly and Fortnightly poetry)" in genre:
            output.append(metadata_name)

    return output


def getFilesWithTag(path_dir, list_metadata, tagName, tagValues):

    output = []
    for metadata_name in list_metadata:
        fullPath = os.path.join(path_dir, metadata_name)

        with open(fullPath, 'rb') as file:
            tagData = json.load(file)[tagName]

        if not isinstance(tagData, list):
            tagData = [tagData]

        if len(set(tagValues).intersection(set(tagData))) > 0:
            output.append(metadata_name)

    return output


def getFilesWithWordInTitle(path_dir, list_metadata, word):
    output = []
    for metadata_name in list_metadata:
        fullPath = os.path.join(path_dir, metadata_name)

        with open(fullPath, 'rb') as file:
            title = json.load(file)["title"].lower()

        if title.find(word) >= 0:
            output.append(metadata_name)

    return output


def getFilesStats(path_dir, list_metadata):

    hasZip = 0
    hasTxt = 1
    hasSpeaker = 2

    nFiles = len(list_metadata)

    statsArray = np.zeros((nFiles, 3), dtype=int)
    index = 0

    for metadataFile in list_metadata:

        zip = get_zip_name(metadataFile)
        statsArray[index][hasZip] = int(
            os.path.isfile(os.path.join(path_dir, zip)))

        txt = get_txt_name(metadataFile)
        statsArray[index][hasTxt] = int(
            os.path.isfile(os.path.join(path_dir, txt)))

        spe = get_speaker_data_name(metadataFile)
        statsArray[index][hasSpeaker] = int(
            hasSpeakerData(os.path.join(path_dir, spe)))

        index += 1

    print(f'Number of files : {index}')
    print(f'Has zip data : {np.sum(statsArray[:, hasZip]) / float(index)}')
    print(f'Has txt data : {np.sum(statsArray[:, hasTxt]) / float(index)}')
    print(
        f'Has speaker data : {np.sum(statsArray[:, hasSpeaker]) / float(index)}')
    print(
        f'Has all: {np.sum(statsArray[:,0] * statsArray[:,1] * statsArray[:,2] > 0) / float(index)}')


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


def getSpeakers(pathSpeakerdata):

    with open(pathSpeakerdata, 'rb') as file:
        data = json.load(file)

    outData = set()

    if data["names"] is None or data["readers"] is None:
        return outData

    for items in data["readers"]:
        if items is not None:
            outData |= set(items)

    return outData


def getAllSpeakers(path_dir, list_metadata):

    outSpeakers = set()
    for metadata in list_metadata:

        fullPath = os.path.join(path_dir, get_speaker_data_name(metadata))
        outSpeakers |= getSpeakers(fullPath)

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


def get_speaker_hours_data(path_dir, list_metadata, pathWav):

    speakerTalk = {}
    nData = len(list_metadata)
    allSizes = []
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

            allSizes.append(size)

            if len(speakers) > 1:
                multiples += size

            for IDspeaker in speakers:
                if IDspeaker not in speakerTalk:
                    speakerTalk[IDspeaker] = 0

                speakerTalk[IDspeaker] += size

    bar.finish()
    print(f"Multiple size {multiples}")

    return speakerTalk


def getMissingTranscriptList(path_dir, list_metadata):

    output = []
    for nM, metadataName in enumerate(list_metadata):

        txtName = get_txt_name(metadataName)
        if not os.path.isfile(os.path.join(path_dir, txtName)):
            output.append(metadataName)

    return output


def count(path_dir, list_metadata, key):

    output = []
    nData = len(list_metadata)
    bar = progressbar.ProgressBar(maxval=nData)
    bar.start()
    for nM, metadataName in enumerate(list_metadata):

        bar.update(nM)
        fullPath = os.path.join(path_dir, metadataName)
        with open(fullPath, 'rb') as file:
            data = json.load(file)

        urlTextSource = data["url_text_source"]
        if urlTextSource.find(key) >= 0:
            output.append((metadataName, urlTextSource))

    bar.finish()
    return output


def getStatus(path_dir, list_metadata):

    nTextData = 0
    for item in list_metadata:
        textFileName = get_txt_name(item)
        if os.path.isfile(os.path.join(path_dir, textFileName)):
            nTextData += 1

    print(f"{nTextData} text data found for {len(list_metadata)} items")


def get_hour_tag_repartition(path_dir, list_metadata, tagName, pathWav):

    nItems = len(list_metadata)
    tags = {}

    bar = progressbar.ProgressBar(maxval=nItems)
    bar.start()

    for index, name_metadata in enumerate(list_metadata):
        bar.update(index)
        pathMetadata = os.path.join(path_dir, name_metadata)
        with open(pathMetadata, 'rb') as file:
            locMetadata = json.load(file)

        value = locMetadata[tagName]

        # Get the number of hours associated to theses data
        baseName = get_wav_name(name_metadata)

        path_dirWav = os.path.join(pathWav, baseName)

        if not os.path.isdir(path_dirWav):
            continue

        wavList = [f for f in os.listdir(path_dirWav)
                   if os.path.splitext(f)[1] == ".wav"]

        totAudio = 0
        for item in wavList:
            info = torchaudio.info(os.path.join(path_dirWav, item))[0]
            totAudio += info.length / (info.rate * 3600.)

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


def get_tags_dependencies(tagStats, tagList):

    out = {x: {} for x in tagList}

    unique_symbol = 'unique'
    assert(unique_symbol not in tagList)

    for key, val in tagStats.items():

        tagList = key.split('+')
        nTags = len(tagList)

        if nTags == 1:
            tag = tagList[0]
            if unique_symbol not in out[tag]:
                out[tag][unique_symbol] = 0
            out[tag][unique_symbol] += val

        else:
            for i in range(nTags):
                ti = tagList[i]
                for j in range(i+1, nTags):
                    tj = tagList[j]
                    if ti not in out[tj]:
                        out[tj][ti] = 0
                    if tj not in out[ti]:
                        out[ti][tj] = 0

                    out[ti][tj] += val
                    out[tj][ti] += val
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


def get_txt_status(path_dir, list_metadata):
    from HathitrustParser import isHathitrustUrl

    bar = progressbar.ProgressBar(maxval=len(list_metadata))
    bar.start()
    output = {}

    for index, name_metadata in enumerate(list_metadata):
        bar.update(index)
        path_metadata = os.path.join(path_dir, name_metadata)
        pathTxt = os.path.join(path_dir, get_txt_name(name_metadata))

        if not os.path.isfile(pathTxt):
            status = 'absent'

        else:
            with open(path_metadata, 'rb') as file:
                url = json.load(file)["url_text_source"]

            if url is None:
                status = 'absent'
            elif isHathitrustUrl(url):
                status = 'noisy'
            else:
                status = 'clean'

        output[name_metadata] = status

    bar.finish()
    return output


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
        speakerList = getAllSpeakers(args.path_dir, list_metadata)
        print(f"{len(speakerList)} speakers")
        time = getTotalTime(args.path_dir, list_metadata)
        print(f"{time} hours of data")
