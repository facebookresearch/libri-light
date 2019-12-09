# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
import os
from html.parser import HTMLParser
import requests
import progressbar


class GenreScapper(HTMLParser):

    def __init__(self):
        super(GenreScapper, self).__init__()
        self.genre = None
        self.inGoodTag = False
        self.takeData = False
        self.getTitle = False

    def handle_starttag(self, tag, attr):
        if tag == "p":
            if ('class', 'book-page-genre') in attr:
                self.inGoodTag = True
        if tag == "span" and self.inGoodTag:
            self.getTitle = True

    def handle_endtag(self, tag):
        if tag == "p":
            self.inGoodTag = False
            self.getTitle = False

    def handle_data(self, data):
        if self.getTitle:
            if data.find("Genre") >= 0:
                self.takeData = True
                self.getTitle = False
        elif self.takeData and self.genre is None:
            self.genre = data

    def getGenre(self):
        if self.genre is None:
            return None
        allGenres = self.genre.replace('\\', '').split(',')
        output = []
        for item in allGenres:
            if len(item) == 0 or item == ' ':
                continue
            output.append(item.lstrip().rstrip())

        if len(output) == 0:
            return None
        return output


def getGenreFromMetadata(metadata):

    urlLibriVoxPage = metadata["url_librivox"]
    parser = GenreScapper()
    req = requests.get(urlLibriVoxPage)
    parser.feed(str(req._content))
    return parser.getGenre()


def gather_all_genres(pathDIR, metadataList):
    out = []
    print("Retrieving all books' genres...")
    bar = progressbar.ProgressBar(maxval=len(metadataList))
    bar.start()
    for index, fileName in enumerate(metadataList):

        bar.update(index)
        pathMetadata = os.path.join(pathDIR, fileName)
        with open(pathMetadata, 'rb') as file:
            metadata = json.load(file)

        try:
            genre = getGenreFromMetadata(metadata)
        except KeyboardInterrupt:
            break
        except:
            genre = None

        out.append((fileName, genre))

    bar.finish()
    return out
