# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import requests


def is_guttenberg_url(url):
    return url.find('http://www.gutenberg.org') == 0 or \
        url.find('https://www.gutenberg.org') == 0 or \
        url.find("http://gutenberg.org") == 0


def get_guttenberg_data(url):
    txtID = url.split('/')[-1]
    targetURL = f'http://www.gutenberg.org/cache/epub/{txtID}/pg{txtID}.txt'
    return requests.get(targetURL)._content.decode("utf-8")
