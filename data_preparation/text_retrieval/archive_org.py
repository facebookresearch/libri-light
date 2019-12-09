# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
from internetarchive import get_item, download


def download_text_data(textID, outDir):

    item = get_item(textID)
    namesFile = []
    for data in item.files:
        name = data['name']
        if os.path.splitext(name)[1] == ".txt":
            namesFile.append(name)

    if len(namesFile) == 0:
        return False, []

    return download(textID, files=namesFile, destdir=outDir), namesFile


def get_archive_id(textURL):

    indexStart = textURL.find("archive.org/details/") \
        + len("archive.org/details/")
    if indexStart < 0:
        return False

    indexEnd = textURL[indexStart:].find("/")
    if indexEnd < 0:
        return textURL[indexStart:]
    return textURL[indexStart:(indexStart + indexEnd)]


def get_archive_org_text_data(url):

    ID = get_archive_id(url)
    tmpDir = "tmp"
    status, fileNames = download_text_data(ID, tmpDir)

    if len(fileNames) == 0:
        raise RuntimeError("Invalid URL")

    fullText = ""
    for fileName in fileNames:
        fullPath = os.path.join(tmpDir, os.path.join(ID, fileName))
        with open(fullPath, 'r', encoding="ISO-8859-1") as file:
            data = file.read()

        os.remove(fullPath)
        fullText += data.replace('\\n', '\n') + '\n'

    return fullText


def is_archive_org_url(url):
    if url.find("https://archive.org/stream/") == 0 \
            or url.find("http://archive.org/stream/") == 0:
        url = url.replace("archive.org/stream/", "archive.org/details/")
    return url.find("archive.org/details/") >= 0


if __name__ == "__main__":

    testID = "completepoetical00byro"
    outDIR = "tmp"
    fullURL = "https://archive.org/details/slaveryourtimes00tolsiala/page/n8"
    print(get_archive_org_text_data(fullURL))
