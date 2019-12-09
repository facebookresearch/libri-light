# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest
from nose.tools import eq_, ok_
import numpy as np
from cutDB import cutWithVAD, greedyMerge


class TestCutDB(unittest.TestCase):

    def setUp(self):

        self.vad = np.array([0.8, 0.9, 1., 0.49, 0.4, 0.2,
                             0.1, 1., 1., 0.9, 0.0, 0.99])
        self.stepVAD = 3
        self.data = np.array(list(range(36)))

    def testCutDB(self):
        outCuts = list(cutWithVAD(self.data, self.vad, 0.5, self.stepVAD))
        expectedOutput = [(0, 9), (21, 30), (33, 36)]

        eq_(len(outCuts), len(expectedOutput))
        for index in range(len(outCuts)):
            eq_(outCuts[index], expectedOutput[index])

    def testGreedyMerge(self):

        cutsIndex = [(0, 9), (21, 30), (33, 36), (24, 49), (53, 117),
                     (201, 222), (230, 240)]
        sizeMultiplier = 0.5
        targetSize = 20

        mergeIndexes = greedyMerge(cutsIndex, targetSize, sizeMultiplier)
        expectedOutput = [(46, [(0, 9), (21, 30), (33, 36), (24, 49)]),
                          (64, [(53, 117)]),
                          (31, [(201, 222), (230, 240)])]

        eq_(len(mergeIndexes), len(expectedOutput))
        for index in range(len(mergeIndexes)):
            eq_(mergeIndexes[index][0], expectedOutput[index][0])
            eq_(len(mergeIndexes[index][1]), len(expectedOutput[index][1]))
            size = len(mergeIndexes[index][1])
            for p in range(size):
                eq_(mergeIndexes[index][1][p], expectedOutput[index][1][p])
