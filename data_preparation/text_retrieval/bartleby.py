# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from html.parser import HTMLParser
import requests
import time


class BarthelebyParser(HTMLParser):

    from enum import Enum
    GLOBAL_STATUS = Enum('STATUS', 'NONE IN_TITLE IN_CHAPTER')
    LOCAL_STATUS = Enum('STATUS', 'NONE PARAGRAPH')

    def __init__(self):
        super(BarthelebyParser, self).__init__()
        self.text = ""
        self.global_status = BarthelebyParser.GLOBAL_STATUS.NONE
        self.local_status = BarthelebyParser.LOCAL_STATUS.NONE
        self.title = ""
        self.ignore = False
        self.textFound = False

    def handle_comment(self, tag):
        if tag.find("BEGIN CHAPTERTITLE") >= 0:
            self.global_status = BarthelebyParser.GLOBAL_STATUS.IN_TITLE
        elif tag.find("END CHAPTERTITLE") >= 0:
            if not self.global_status == BarthelebyParser.GLOBAL_STATUS.IN_TITLE:
                raise RuntimeError("Page of invalid format")
            self.global_status = BarthelebyParser.GLOBAL_STATUS.NONE
        elif tag.find("BEGIN CHAPTER") >= 0 or tag.find("END MAIN HEADER CODE") >= 0:
            self.global_status = BarthelebyParser.GLOBAL_STATUS.IN_CHAPTER
            self.local_status = BarthelebyParser.LOCAL_STATUS.NONE
            self.textFound = True
        elif tag.find("END CHAPTER") >= 0 or tag.find("AMAZON") >= 0:
            if not self.global_status == BarthelebyParser.GLOBAL_STATUS.IN_CHAPTER:
                raise RuntimeError("Page of invalid format")
            self.global_status = BarthelebyParser.GLOBAL_STATUS.NONE

    def handle_data(self, data):
        if self.global_status == BarthelebyParser.GLOBAL_STATUS.IN_TITLE:
            if self.local_status == BarthelebyParser.LOCAL_STATUS.PARAGRAPH:
                self.title += data
        elif self.global_status == BarthelebyParser.GLOBAL_STATUS.IN_CHAPTER:
            if self.local_status == BarthelebyParser.LOCAL_STATUS.PARAGRAPH and not self.ignore:
                self.text += data

    def handle_starttag(self, tag, attrs):
        if self.global_status == BarthelebyParser.GLOBAL_STATUS.IN_TITLE:
            if tag == 'b':
                self.local_status = BarthelebyParser.LOCAL_STATUS.PARAGRAPH
        elif self.global_status == BarthelebyParser.GLOBAL_STATUS.IN_CHAPTER:
            if tag == 'tr':
                if self.local_status != BarthelebyParser.LOCAL_STATUS.NONE:
                    self.text += '\n'
                self.local_status = BarthelebyParser.LOCAL_STATUS.PARAGRAPH
            if tag == "i":
                self.ignore = True

    def handle_endtag(self, tag):
        if self.global_status == BarthelebyParser.GLOBAL_STATUS.IN_TITLE and tag == 'b':
            self.local_status = BarthelebyParser.LOCAL_STATUS.NONE
        if tag == "i":
            self.ignore = False

    def getCleanText(self):
        return self.text.replace('\\n', '\n').replace("\\'", "'")


class BarthelebyTitleParser(HTMLParser):

    def __init__(self):
        super(BarthelebyTitleParser, self).__init__()
        self.titleFound = False
        self.load = False
        self.title = ""

    def handle_starttag(self, tag, attr):
        if tag == "title":
            self.titleFound = True
            self.load = True

    def handle_endtag(self, tag):
        if tag == "title":
            self.load = False

    def handle_data(self, data):
        if self.load:
            self.title = data


def get_bartheleby_data(url):

    extension = url.split('.')[-1]
    isUniquePage = extension == 'html'

    def loadText(locUrl):
        parser = BarthelebyParser()
        req = requests.get(locUrl)
        parser.feed(str(req._content))
        time.sleep(1)
        if not parser.textFound:
            return None
        return parser.title + '\n' + '\n' + parser.getCleanText()

    if not isUniquePage:

        # Load title
        parser = BarthelebyTitleParser()
        req = requests.get(url)
        parser.feed(str(req._content))

        if not parser.titleFound:
            raise RuntimeError("No title found")

        fullText = parser.title + '\n' + '\n'

        if url[-1] != '/':
            url += '/'
        data = url.split('/')

        try:
            int(data[-2])
        except ValueError:
            raise RuntimeError("Invalid url")

        index = 1
        while True:
            nextUrl = f"{url}{index}.html"
            textData = loadText(nextUrl)
            if textData is None:
                break
            fullText += '\n\n' + textData
            index += 1

        return fullText

    text = loadText(url)
    if text is None:
        raise RuntimeError("Couldn't find the page")
    return text


def is_bartheleby_url(url):
    return url.find("bartleby.com") >= 0


if __name__ == "__main__":

    url = "https://www.bartleby.com/95/1.html"
    data = get_bartheleby_data(url)
    with open('coin.txt', 'w') as file:
        file.write(data)
    #parser = BarthelebyParser()
    #req = requests.get(url)
    # parser.feed(str(req._content))
    # print(parser.title)
    #print(parser.text.replace('\\n', '\n').replace("\\'", "'"))
