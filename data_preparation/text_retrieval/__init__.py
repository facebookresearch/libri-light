# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .archive_org import is_archive_org_url, get_archive_org_text_data
from .main_lesson import is_main_lesson_url, get_all_text_from_main_lesson
from .hathitrust import is_hathitrust_url, load_hathitrust_book
from .bartleby import is_bartheleby_url, get_bartheleby_data
from .guttenberg import is_guttenberg_url, get_guttenberg_data


def get_text_data(url):
    if is_guttenberg_url(url):
        return get_guttenberg_data(url)
    elif is_archive_org_url(url):
        return get_archive_org_text_data(url)
    elif is_bartheleby_url(url):
        return get_bartheleby_data(url)
    elif is_main_lesson_url(url):
        return get_all_text_from_main_lesson(url)
    elif is_hathitrust_url(url):
        return load_hathitrust_book(url)
    else:
        raise RuntimeError(f'Unknown web API {url}')
