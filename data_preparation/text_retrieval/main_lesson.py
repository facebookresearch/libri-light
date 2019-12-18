# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from html.parser import HTMLParser
import requests


def get_tag_value_in_url(url, tag):
    baseUrl = "display.php?"
    argBegin = url.find(baseUrl)
    if argBegin < 0:
        raise RuntimeError("Invalid url")

    argBegin += len(baseUrl)
    args = url[argBegin:].split('&')

    for item in args:
        if item.find(f'{tag}=') == 0:
            return item.split(f'{tag}=')[1]

    raise RuntimeError("{tag} not found")


def get_full_url(author, book, chapter):
    return f"http://www.gatewaytotheclassics.com/browse/display.php?author={author}&book={book}&story={chapter}"


class ToCParser(HTMLParser):

    def __init__(self):

        self.chaptersList = []
        self.inLinkBlock = False
        super(ToCParser, self).__init__()

    def handle_starttag(self, tag, attrs):
        if tag == "div" and ("class", "lhlink") in attrs:
            self.inLinkBlock = True
        elif tag == 'a' and self.inLinkBlock:
            for name, value in attrs:
                if name == "href":
                    self.chaptersList.append(
                        get_tag_value_in_url(value, 'story'))

    def handle_endtag(self, tag):
        if tag == "div":
            self.inLinkBlock = False


class ChapterParser(HTMLParser):

    def __init__(self):

        self.text = ""
        self.title = None
        self.getData = False
        self.getTitle = False
        super(ChapterParser, self).__init__()

    def handle_starttag(self, tag, attrs):
        if tag == "h1" and ("align", "CENTER") in attrs:
            self.getTitle = True

        if tag == "table":
            self.getData = False

    def handle_endtag(self, tag):
        if tag == "h1":
            self.getTitle = False
            self.getData = True
        if tag == "table" and self.title is not None:
            self.getData = True

    def handle_data(self, data):
        if self.getTitle:
            self.title = data.replace("\\", "")
        elif self.getData:
            self.text += data.replace('\\n', '\n').replace("\\", "")

    def get_full_text(self):

        if self.title is None:
            raise RuntimeError("No title found")

        return self.title + '\n' + self.text


def get_all_text_from_main_lesson(url):

    book = get_tag_value_in_url(url, 'book')
    author = get_tag_value_in_url(url, 'author')

    tocUrl = get_full_url(author, book, '_contents')

    parserToC = ToCParser()
    req = requests.get(tocUrl)
    parserToC.feed(str(req._content))

    fullText = ""

    for chapterName in parserToC.chaptersList:

        txtUrl = get_full_url(author, book, chapterName)
        parserChapter = ChapterParser()
        req = requests.get(txtUrl)
        parserChapter.feed(str(req._content))
        fullText += parserChapter.get_full_text()

    return fullText


def is_main_lesson_url(url):
    return url.find("mainlesson.com") >= 0


if __name__ == "__main__":
    url = "http://www.gatewaytotheclassics.com/browse/display.php?author=marshall&book=beowulf&story=_contents"
    print(get_all_text_from_main_lesson(url))
