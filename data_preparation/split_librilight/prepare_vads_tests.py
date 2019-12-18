# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from prepare_vads import split_vad


class TestSplit(unittest.TestCase):
    def test_all_silence(self):
        p_silence = [1.0] * 100
        segments = split_vad(silence_probs=p_silence,
                             p_silence_threshold=0.999, len_threshold=6)

        self.assertFalse(segments)

    def test_all_speech(self):
        p_silence = [0.0] * 100
        segments = split_vad(silence_probs=p_silence,
                             p_silence_threshold=0.999, len_threshold=6)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0], (0, 100))

    def test_half_speech(self):
        p_silence = [1.0] * 50 + [0.0] * 50
        segments = split_vad(silence_probs=p_silence,
                             p_silence_threshold=0.999, len_threshold=6)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0], (50, 100))

    def test_short_speech(self):
        """Speech segment shorter than len_threshold"""
        p_silence = [1.0] * 50 + [0.0] * 5 + [1.0] * 50
        segments = split_vad(silence_probs=p_silence,
                             p_silence_threshold=0.999, len_threshold=6)

        self.assertFalse(segments)

    def test_short_silence(self):
        """Silence segment shorter than len_threshold"""
        p_silence = [0.0] * 50 + [1.0] * 5 + [0.0] * 50
        segments = split_vad(silence_probs=p_silence,
                             p_silence_threshold=0.999, len_threshold=6)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0], (0, 105))

    def test_few_segments(self):
        #  positions 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19    20..29     30..39
        p_silence = [0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
                     0, 0, 1, 1, 1, 1, 1, 1] + [0] * 10 + [1] * 10
        # 1st:       | .         short silence                 | <- taken
        # 2nd:                                                                      ^ those 10, taken
        self.assertEqual(len(p_silence), 40)

        segments = split_vad(silence_probs=p_silence,
                             p_silence_threshold=0.999, len_threshold=6)

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0], (0, 14))
        self.assertEqual(segments[1], (20, 30))

    def test_final_silence(self):
        p_silence = [1.0] * 50
        p_silence[40] = 0
        segments = split_vad(silence_probs=p_silence,
                             p_silence_threshold=0.999, len_threshold=6)

        self.assertFalse(segments)


if __name__ == '__main__':
    unittest.main()
