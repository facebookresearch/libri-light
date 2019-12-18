# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math
import os
import struct
import sys

import numpy as np
from wav2letter.common import Dictionary, create_word_dict, load_words, tkn_to_idx
from wav2letter.decoder import (
    CriterionType,
    DecoderOptions,
    KenLM,
    SmearingMode,
    Trie,
    WordLMDecoder,
)


class WlDecoder:
    """
    Wav2Letter-based decoder. Follows the official examples for the python bindings, 
    see https://github.com/facebookresearch/wav2letter/blob/master/bindings/python/examples/decoder_example.py
    """

    def __init__(self,
                 lm_weight=2.0,
                 lexicon_path="WER_data/lexicon.txt",
                 token_path="WER_data/letters.lst",
                 lm_path="WER_data/4-gram.bin"):
        lexicon = load_words(lexicon_path)
        word_dict = create_word_dict(lexicon)

        self.token_dict = Dictionary(token_path)
        self.lm = KenLM(lm_path, word_dict)

        self.sil_idx = self.token_dict.get_index("|")
        self.unk_idx = word_dict.get_index("<unk>")
        self.token_dict.add_entry("#")
        self.blank_idx = self.token_dict.get_index('#')

        self.trie = Trie(self.token_dict.index_size(), self.sil_idx)
        start_state = self.lm.start(start_with_nothing=False)

        for word, spellings in lexicon.items():
            usr_idx = word_dict.get_index(word)
            _, score = self.lm.score(start_state, usr_idx)
            for spelling in spellings:
                # max_reps should be 1; using 0 here to match DecoderTest bug
                spelling_idxs = tkn_to_idx(
                    spelling, self.token_dict, max_reps=0)
                self.trie.insert(spelling_idxs, usr_idx, score)

        self.trie.smear(SmearingMode.MAX)
        self.opts = DecoderOptions(
            beam_size=2500, beam_threshold=100.0, lm_weight=lm_weight,
            word_score=2.0, unk_score=-math.inf, log_add=False, sil_weight=-1, criterion_type=CriterionType.CTC)

    def collapse(self, prediction):
        result = []

        for p in prediction:
            if result and p == result[-1]:
                continue
            result.append(p)

        blank = '#'
        space = '|'

        result = [x for x in result if x != blank]
        result = [(x if x != space else ' ') for x in result if x != blank]
        return result

    def predictions(self, emissions):
        t, n = emissions.size()

        emissions = emissions.cpu().numpy()
        decoder = WordLMDecoder(
            self.opts, self.trie, self.lm, self.sil_idx, self.blank_idx, self.unk_idx, [])
        results = decoder.decode(emissions.ctypes.data, t, n)

        prediction = [self.token_dict.get_entry(
            x) for x in results[0].tokens if x >= 0]
        prediction = self.collapse(prediction)

        return prediction
