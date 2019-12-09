# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from html.parser import HTMLParser
import requests


class HathitrustParser(HTMLParser):

    def __init__(self):

        self.nextUrl = None
        self.tmpUrl = None
        self.text = ""
        self.getNextP = False
        self.getTextData = False
        self.emptyPage = False
        super(HathitrustParser, self).__init__()

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for name, value in attrs:
                if name == "href":
                    self.tmpUrl = value

        if tag == "div" and ("id", "mdpPage") in attrs:
            self.getNextP = True

        if tag == "div" and ("id", "mdpTextEmpty") in attrs:
            self.emptyPage = True

        if tag == "p" and self.getNextP:
            self.getTextData = True

    def handle_data(self, data):
        if self.tmpUrl is not None and data.find("Next Page") >= 0:
            self.nextUrl = self.tmpUrl
        if self.getTextData:
            self.text += data

    def handle_endtag(self, tag):
        if tag == "a":
            self.tmpUrl = None
        elif tag == "div":
            self.getNextP = False
        elif tag == "p":
            self.getTextData = False
            self.getNextP = False


class CatalogParser(HTMLParser):

    def __init__(self):
        self.candidatesID = []
        super(CatalogParser, self).__init__()

    def handle_starttag(self, tag, attrs):

        attrs = dict(attrs)
        if tag == "a":
            if attrs["href"].find("handle.net") >= 0:
                self.candidatesID.append(attrs["data-hdl"])


def load_whole_book(bookID):

    baseUrl = "https://babel.hathitrust.org/cgi/ssd?"
    nextUrl = f"{baseUrl}id={bookID};page=ssd;view=plaintext;seq=1;num="

    fullText = ""

    while True:
        parserChapter = HathitrustParser()
        req = requests.get(nextUrl)
        parserChapter.feed(req._content.decode('utf-8'))
        if parserChapter.nextUrl is None:
            break
        nextUrl = f"https://babel.hathitrust.org{parserChapter.nextUrl}"

        if not parserChapter.emptyPage:
            fullText += parserChapter.text

    return fullText


def is_hathitrust_url(url):
    return url.find("hathitrust.org") >= 0


def load_hathitrust_book(url):

    candidatesID = None
    if url.find("catalog.hathitrust.org") >= 0:
        catalogParser = CatalogParser()
        req = requests.get(url)
        catalogParser.feed(req._content.decode('utf-8'))

        if len(catalogParser.candidatesID) == 0:
            raise RuntimeError("Invalid url")

        candidatesID = catalogParser.candidatesID

    else:
        key = "cgi/ssd?"
        startOffset = url.find(key)

        if startOffset < 0:
            raise RuntimeError("Invalid url")

        startOffset += len(key)
        markers = url[startOffset:].split(';')

        for data in markers:
            name, value = data.split('=')
            if name == "id":
                candidatesID = [value]
                break

        if candidatesID is None:
            raise RuntimeError("Invalid url")

    text = None
    for id in candidatesID:

        try:
            text = load_whole_book(id)
        except RuntimeError:
            continue

    if text is None:
        raise RuntimeError("Couldn't find any transcription")

    return text


if __name__ == "__main__":

    url1 = "https://babel.hathitrust.org/cgi/ssd?id=coo.31924074296884;page=ssd;view=plaintext;seq=110;num=104"
    url2 = "http://catalog.hathitrust.org/Record/002242980"
    print(load_hathitrust_book(url1))
