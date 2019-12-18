# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import requests
import urllib.request
from html.parser import HTMLParser
import os
import sys
from termcolor import colored
import json
import progressbar


import argparse
from text_retrieval import get_text_data


class RequestPBar():
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


class MyHTMLParser(HTMLParser):

    def __init__(self):

        self.tableFound = False
        self.tableBegin = False

        self.isInBody = False
        self.nChapters = 0

        self.chapterNames = []
        self.chapterReaders = []

        self.currNameFound = True
        self.currIDFound = True

        super(MyHTMLParser, self).__init__()

    def check_speaker_data(self):

        if len(self.chapterNames) == 0:
            raise RuntimeError("No chapter found")
        if len(self.chapterNames) != len(self.chapterReaders):
            raise RuntimeError("Invalid speaker data")
        if None in self.chapterNames:
            raise RuntimeError("Invalid speaker data")
        if None in self.chapterReaders:
            raise RuntimeError("Invalid speaker data")

    def handle_starttag(self, tag, attrs):
        if tag == "table" and attrs[0] == ('class', 'chapter-download'):
            if self.tableFound:
                raise RuntimeError("Two speakers tables ??")
            self.tableBegin = True
            self.tableFound = True

        if not self.tableBegin:
            return

        if tag == 'tbody':
            self.isInBody = True

        if tag == 'tr' and self.isInBody:
            self.nChapters += 1
            self.chapterNames.append(None)
            self.chapterReaders.append(None)
            self.currNameFound = False
            self.currIDFound = False

        if tag == 'a':
            if len(attrs) == 2 and attrs[1] == ("class", "chapter-name"):
                if self.currNameFound:
                    raise RuntimeError("Two names for the same chapter !")
                name = attrs[0][1].split('/')[-1]
                self.chapterNames[-1] = os.path.splitext(name)[0]
                self.currNameFound = True
            elif len(attrs) == 1:
                _, link = attrs[0]
                if link.find('https://librivox.org/reader/') == 0:
                    _size = len('https://librivox.org/reader/')
                    if self.currIDFound:
                        self.chapterReaders[-1].append(link[_size:])
                    self.chapterReaders[-1] = [link[_size:]]
                    self.currIDFound = True

    def handle_endtag(self, tag):
        if tag == "table" and self.tableBegin:
            self.tableBegin = False

        if not self.tableBegin:
            return

        if tag == 'tbody':
            self.isInBody = False

    def handle_data(self, data):
        pass


def get_reader_data(url):
    parser = MyHTMLParser()
    req = requests.get(url)
    parser.feed(str(req._content))
    return parser.chapterNames, parser.chapterReaders


def import_page(baseUrl, offset, limit, dirOut, language):

    urlRequest = f'{baseUrl}/?offset={offset}&format=json&limit={limit}'
    response = requests.get(urlRequest).json()

    if "error" in response:
        return -1

    if "books" not in response:
        raise RuntimeError(f"Invalid url {baseUrl}")

    bookList = response['books']
    fullSize = 0

    if len(bookList) == 0:
        return -1

    for bookData in bookList:
        if bookData['language'] != language:
            continue

        title = bookData['title']
        print(f'Loading title {title}...')
        chapterReaders, chaptersNames = None, None
        try:
            chaptersNames, chapterReaders = \
                get_reader_data(bookData['url_librivox'])
        except:
            print(colored(f'Error when loading title {title} metadata', 'red'))
            print(colored(sys.exc_info(), 'red'))

        name = os.path.splitext(os.path.basename(bookData['url_zip_file']))[0]

        outSpeaker = os.path.join(dirOut, f'{name}_speaker_data.json')
        with open(outSpeaker, 'w') as file:
            json.dump({"names": chaptersNames,
                       "readers": chapterReaders},
                      file, indent=2)

        print(f'{title}\'s speaker data loaded')

        outMetadata = os.path.join(dirOut, f'{name}_metadata.json')
        with open(outMetadata, 'w') as file:
            json.dump(bookData, file, indent=2)
        print(f'{title}\'s metadata loaded')
        try:
            txtData = get_text_data(bookData['url_text_source'])
            outTxt = os.path.join(dirOut, f'{name}_text.txt')
            with open(outTxt, 'w') as file:
                file.write(txtData)
                print('... text data loaded')
            fullSize += os.path.getsize(outTxt)
        except:
            print(colored(f'Error when loading {title}\'s text data', 'red'))

        print(f'Loading audio data at {bookData["url_zip_file"]}')
        outPath = os.path.join(dirOut, name + ".zip")
        if not os.path.isfile(outPath):
            try:
                d = urllib.request.urlopen(bookData['url_zip_file'])
                fullSize += int(d.info()['Content-Length'])
                urllib.request.urlretrieve(bookData['url_zip_file'], outPath,
                                           RequestPBar())
            except KeyboardInterrupt:
                if os.path.isfile(outPath):
                    os.remove(outPath)
                sys.exit()
            except:
                if os.path.isfile(outPath):
                    os.remove(outPath)

        print('')

    return fullSize


def get_size_page(baseUrl, offset, limit, language):

    urlRequest = f'{baseUrl}/?offset={offset}&format=json&limit={limit}'
    response = requests.get(urlRequest).json()

    if "error" in response:
        return -1

    if "books" not in response:
        raise RuntimeError(f"Invalid url {baseUrl}")

    outSize = 0

    bookList = response['books']

    if len(bookList) == 0:
        return -1

    for bookData in bookList:
        if bookData['language'] != language:
            continue
        title = bookData['title']
        print(f'Loading title {title}...')

        try:
            d = urllib.request.urlopen(bookData['url_zip_file'])
            size = int(d.info()['Content-Length'])
            outSize += size
        except KeyboardInterrupt:
            sys.exit()
        except:
            print(colored(f'Error when loading title {title} metadata', 'red'))
            print(colored(sys.exc_info(), 'red'))
            continue

        print(f'..{title} loaded')

    return outSize


def load_tmp(pathTmp):

    with open(pathTmp, 'r') as file:
        data = file.readlines()[0]
    return int(data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Dowload the librivox \
                                     database to a given location. Each element \
                                     will be loaded inside NAME.zip along with \
                                     3 other files: NAME_metadata.json, NAME_speaker_data.json, \
                                     and NAME_text.txt.")
    parser.add_argument('output_dir', type=str, help='Path to the output \
                        directory')
    parser.add_argument('--language', type=str, default='English',
                        help='Desired language')
    parser.add_argument('--getSize', action='store_true')
    parser.add_argument('--maxSize', type=float, default=-1,
                        help="Maximum size to load in Tb")
    parser.add_argument('--startOffset', type=int, default=0)
    parser.add_argument('--maxOffset', type=int, default=-1)

    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    pathTmp = os.path.join(args.output_dir, 'tmp.txt')
    pathStatus = os.path.join(args.output_dir, 'status.txt')

    if os.path.isfile(pathTmp) and args.startOffset == 0:
        offset = load_tmp(pathTmp)
    else:
        offset = max(0, args.startOffset)

    offsetStep = 200
    url = 'https://librivox.org/api/feed/audiobooks'
    fullSize = 0
    pathSize = os.path.join(args.output_dir, 'size.txt')
    maxSize = args.maxSize * 1e12 if args.maxSize > 0 else -1

    if args.maxSize > 0:
        print(f'Limiting the downloaded size to {args.maxSize} Tb')

    if os.path.isfile(pathSize):
        fullSize = load_tmp(pathSize)

    while True:
        if args.getSize:
            size = get_size_page(url,
                                 offset,
                                 offsetStep,
                                 args.language)
        else:
            size = import_page(url,
                               offset,
                               offsetStep,
                               args.output_dir,
                               args.language)
        if size < 0:
            break
        fullSize += size
        with open(pathSize, 'w') as file:
            file.write(str(fullSize))
        offset += offsetStep
        with open(pathTmp, 'w') as file:
            file.write(str(offset))

        with open(pathStatus, 'w') as file:
            if args.maxOffset > 0:
                file.write(f'{(offset / args.maxOffset) * 100}%')
            else:
                file.write(f'{offset / 130}%')

        if maxSize > 0 and fullSize >= maxSize:
            break

        if args.maxOffset > 0 and offset >= args.maxOffset:
            break
