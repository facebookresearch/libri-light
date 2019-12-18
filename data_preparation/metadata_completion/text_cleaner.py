# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from text_retrieval.guttenberg import is_guttenberg_url
from .utilities import get_txt_name
import os
import json
import progressbar
import sys
sys.path.append('..')


def loadData(pathFile):

    with open(pathFile, 'r') as file:
        data = file.readlines()

    indexStartProject = -1
    indexProducedBy = -1
    indexEndProject = -1
    for index, line in enumerate(data):
        if indexStartProject < 0:
            value = line.replace(' ', '').find("***START")
            if value >= 0:
                indexStartProject = index
            elif line.find("CONTENTS") >= 0:
                indexStartProject = index
            else:
                continue

        value = line.replace(' ', '').find("***END")
        if value >= 0:
            indexEndProject = index
            break

        if indexProducedBy < 0:
            value = line.find("Produced by")
            if value >= 0:
                indexProducedBy = index

    if indexStartProject < 0:
        return None

    if indexEndProject < 0:
        indexEndProject = len(data)

    startIndex = indexProducedBy + 1 if indexProducedBy > 0 \
        else indexStartProject + 1
    while startIndex < len(data) and data[startIndex] == '\n':
        startIndex += 1

    return ''.join(data[startIndex:indexEndProject])


def find404Error(pathFile):
    with open(pathFile, 'r') as file:
        data = file.readlines()

    return len(data) == 1 and \
        data[0] == "<h1>404 Not Found</h1><p>File not found.</p>"


def clean_all_text_data(metadataList, pathInDir, pathOutDir):

    pathInDir = os.path.abspath(pathInDir)
    pathOutDir = os.path.abspath(pathOutDir)

    if pathInDir == pathOutDir:
        raise ValueError("Can't save the data in the same directory \
                          as the originals")

    bar = progressbar.ProgressBar(maxval=len(metadataList))
    bar.start()
    nCleaned = 0
    nMissing = 0
    nNotWorking = 0
    emptyTxt = []
    out = []

    for index, metadataName in enumerate(metadataList):
        bar.update(index)
        textFileName = get_txt_name(metadataName)
        pathInFile = os.path.join(pathInDir, textFileName)
        outPathFile = os.path.join(pathOutDir, textFileName)

        if not os.path.isfile(pathInFile):
            status = "missing"
            nMissing += 1
        else:

            assert(pathInFile != outPathFile)

            with open(os.path.join(pathInDir, metadataName), 'rb') as file:
                urlSource = json.load(file)["url_text_source"]

            if not is_guttenberg_url(urlSource):
                os.popen(f'cp {pathInFile} {outPathFile}')
                status = "clear"
            else:
                outData = loadData(pathInFile)

                if outData is None:
                    nNotWorking += 1
                    if find404Error(pathInFile):
                        emptyTxt.append(pathInFile)
                        status = "missing"
                    else:
                        status = "noisy"
                else:
                    with open(outPathFile, 'w') as file:
                        file.write(outData)
                    status = "clear"
        out.append((metadataName, status))
        nCleaned += 1

    bar.finish()
    print(f"Out of {len(metadataList)} items")
    print(f"{nCleaned} files were cleaned and saved to {pathOutDir}")
    print(f"{nNotWorking} files didn't match the good format among which {len(emptyTxt)} were empty")
    print(f"{nMissing} files were missing")
    return out


if __name__ == "__main__":

    pathDirData = "/checkpoint/mriviere/LibriVox/"
    pathOutData = "/checkpoint/mriviere/LibriVox_cleanTxt/"

    if not os.path.isdir(pathOutData):
        os.mkdir(pathOutData)

    clean_all_text_data(pathDirData, pathOutData)

    # pathTestFile = "/checkpoint/mriviere/LibriVox/sadhana_realisation_librivox_64kb_mp3_text.txt"
    # print(find404Error(pathTestFile))
