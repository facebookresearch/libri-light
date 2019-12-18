# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from html.parser import HTMLParser
import requests
import json
import os
from .utilities import get_speaker_data_name
from copy import deepcopy
import progressbar


class ReaderScrapper(HTMLParser):

    def __init__(self):

        self.readerName = None
        self.inHeader = False
        self.getData = False
        super(ReaderScrapper, self).__init__()

    def handle_starttag(self, tag, attrs):
        if tag == "div":
            if ('class', 'page author-page') in attrs:
                self.inHeader = True

        if tag == "h1" and self.inHeader:
            self.getData = True

    def handle_endtag(self, tag):
        if tag == "div":
            self.inHeader = False
        if tag == "h1":
            self.getData = False

    def handle_data(self, data):
        if self.getData:
            self.readerName = data


def get_librivox_reader_from_id(readerID):
    url = f"https://librivox.org/reader/{readerID}"
    parser = ReaderScrapper()
    req = requests.get(url)
    parser.feed(str(req._content))
    return parser.readerName


def updateDataWithNames(speakerData, idMatch):

    newData = deepcopy(speakerData)

    if speakerData["readers"] is None:
        newData["readers_names"] = None
        return newData

    newData["readers_names"] = []

    for item in speakerData["readers"]:
        if item is not None:
            for ID in item:
                if ID not in idMatch:
                    try:
                        idMatch[ID] = get_librivox_reader_from_id(ID)
                    except RuntimeError:
                        idMatch[ID] = None

    for item in speakerData["readers"]:
        if item is None:
            newData["readers_names"].append(None)
        else:
            all_names = []
            for ID in item:
                all_names.append(idMatch[ID])
            newData["readers_names"].append(all_names)

    return newData


def update_all_speaker_data(listMetadata, pathInDir, pathOutDir):

    print("Updating the speaker data, this is going to be looong....")
    pathInDir = os.path.abspath(pathInDir)
    pathOutDir = os.path.abspath(pathOutDir)
    assert(pathInDir != pathOutDir)

    if not os.path.isdir(pathOutDir):
        os.mkdir(pathOutDir)

    bar = progressbar.ProgressBar(maxval=len(listMetadata))
    bar.start()

    idMatch = {None: None}

    for index, pathMetadata in enumerate(listMetadata):
        bar.update(index)

        pathSpeakerData = get_speaker_data_name(pathMetadata)
        fullPathSpeakerData = os.path.join(pathInDir, pathSpeakerData)
        with open(fullPathSpeakerData, 'rb') as file:
            speakerData = json.load(file)

        outData = updateDataWithNames(speakerData, idMatch)
        pathOutData = os.path.join(pathOutDir, pathSpeakerData)

        assert(fullPathSpeakerData != pathOutData)

        with open(pathOutData, 'w') as file:
            json.dump(outData, file, indent=2)

    bar.finish()


if __name__ == "__main__":
    pathIn = "/checkpoint/mriviere/LibriVox/"
    pathOut = "/checkpoint/mriviere/LibriVox_updatedSpeakers/"
    update_all_speaker_data(pathIn, pathOut)
