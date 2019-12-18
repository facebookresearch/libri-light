# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest
import torch
from nose.tools import eq_, ok_
from . import abx_group_computation
from . import abx_iterators
import numpy as np
import math


class TestDistancesDTW(unittest.TestCase):

    def testDTWFunction(self):
        X = torch.tensor([[[0, 1], [0, 0], [1, 1], [42, 42]],
                          [[0, 2], [0, 1], [1, 1], [-1, 0]],
                          [[0, 0], [0, 1], [0, 0], [21, 211]]],
                         dtype=torch.float)

        X_size = torch.tensor([3, 4, 2])

        Y = torch.tensor([[[0, 1], [1, 2], [0, 0]]], dtype=torch.float)
        Y_size = torch.tensor([3])

        distance_mode = abx_group_computation.get_euclidian_distance_batch
        dist = abx_group_computation.get_distance_group_dtw(X, Y,
                                                            X_size, Y_size,
                                                            distance_function=distance_mode)
        eq_(dist.size(), (3, 1))
        expected_dist = [[(math.sqrt(2)) / 2], [3 / 4],
                         [(2 + math.sqrt(2)) / 3]]
        for i in range(3):
            ok_(abs(expected_dist[i][0] - dist[i].item()) < 1e-4)

    def testThetaDTWFunctionSymetric(self):
        A = torch.tensor([[[0, 1], [0, 0], [1, 1], [42, 42]],
                          [[0, 2], [0, 1], [1, 1], [-1, 0]],
                          [[0, 0], [0, 1], [0, 0], [21, 211]]],
                         dtype=torch.float)
        A_size = torch.tensor([3, 4, 2])
        B = torch.tensor([[[0, 1], [1, 2], [0, 0]]], dtype=torch.float)
        B_size = torch.tensor([3])

        distance_mode = abx_group_computation.get_euclidian_distance_batch
        symetric = True
        theta = abx_group_computation.get_theta_group_dtw(A, B, A, A_size,
                                                          B_size, A_size,
                                                          distance_mode,
                                                          symetric)
        eq_(theta, 0.5)


class testSingularityNormalization(unittest.TestCase):

    def testCosineNormalized(self):
        x = torch.tensor([[[1., 0., 0., 0.], [0., 0., 0., 0.]],
                          [[0., 0., -1., 0.], [0.5, -0.5, 0.5, -0.5]]])
        y = torch.tensor(
            [[-0.5, -0.5, -0.5, 0.5], [0., 0., 0., 0.], [0., 1., 0., 0.]])
        norm_x = []
        for i in range(2):
            norm_x.append(abx_iterators.normalize_with_singularity(x[i]).view(1, 2, 5))
        norm_x = torch.cat(norm_x, dim=0)
        norm_y = abx_iterators.normalize_with_singularity(y).view(1, 3, 5)
        dist = abx_group_computation.get_cosine_distance_batch(norm_x, norm_y)

        eq_(dist.size(), (2, 1, 2, 3))
        ok_(abs(dist[0, 0, 0, 0] - 0.6667) < 1e-4)
        ok_(abs(dist[0, 0, 0, 1] - 1.) < 1e-4)
        ok_(abs(dist[0, 0, 0, 2] - 0.5) < 1e-4)

        ok_(abs(dist[0, 0, 1, 0] - 1) < 1e-4)
        ok_(abs(dist[0, 0, 1, 1]) < 1e-4)
        ok_(abs(dist[0, 0, 1, 2] - 1) < 1e-4)

        ok_(abs(dist[1, 0, 0, 0] - 0.3333) < 1e-4)
        ok_(abs(dist[1, 0, 0, 1] - 1.) < 1e-4)
        ok_(abs(dist[1, 0, 0, 2] - 0.5) < 1e-4)

        ok_(abs(dist[1, 0, 1, 0]-0.6667) < 1e-4)
        ok_(abs(dist[1, 0, 1, 1] - 1.) < 1e-4)
        ok_(abs(dist[1, 0, 1, 2] - 0.6667) < 1e-4)


class testGroupMaker(unittest.TestCase):

    def test1DGroupMaker(self):

        data = [[0], [1], [2], [3], [4], [2], [2], [2]]
        order = [0]
        out_index, out_data = abx_iterators.get_features_group(data, order)

        expected_index = [0, 1, 2, 5, 6, 7, 3, 4]
        eq_(out_index, expected_index)

        expected_output = [(0, 1), (1, 2), (2, 6), (6, 7), (7, 8)]
        eq_(out_data, expected_output)

    def test2DGroupMaker(self):

        data = [[0, 1], [1, 2], [2, 3], [3, 3],
                [4, 0], [2, 2], [4, 2], [2, 2], [0, 3]]

        order = [1, 0]
        out_index, out_data = abx_iterators.get_features_group(data, order)
        expected_index = [4, 0, 1, 5, 7, 6, 8, 2, 3]
        eq_(out_index, expected_index)
        expected_output = [[(0, 1)],
                           [(1, 2)],
                           [(2, 3), (3, 5), (5, 6)],
                           [(6, 7), (7, 8), (8, 9)]]
        eq_(out_data, expected_output)

    def test3DGroupMaker(self):

        data = [[0, 0, 0, 1],
                [41, 1, 0, 2],
                [-23, 0, 3, 1],
                [220, 1, -2, 3],
                [40, 2, 1, 0],
                [200, 0, 0, 1]]

        order = [1, 3, 2]
        out_index, out_data = abx_iterators.get_features_group(data, order)
        expected_index = [0, 5, 2, 1, 3, 4]
        eq_(out_index, expected_index)

        expected_output = [[[(0, 2), (2, 3)]], [
            [(3, 4)], [(4, 5)]], [[(5, 6)]]]
        eq_(out_data, expected_output)


class testItemLoader(unittest.TestCase):

    def testLoadItemFile(self):
        path_item_file = "test_data/dummy_item_file.item"
        out, context_match, phone_match, speaker_match = \
            abx_iterators.load_item_file(path_item_file)

        eq_(len(out), 4)
        eq_(len(phone_match), 5)
        eq_(len(speaker_match), 3)

        expected_phones = {'n': 0, 'd': 1, 'ih': 2,
                           's': 3, 'dh': 4}
        eq_(phone_match, expected_phones)

        expected_speakers = {'8193': 0, '2222': 1, '12': 2}
        eq_(speaker_match, expected_speakers)

        expected_context = {'ae+d': 0, 'n+l': 1, 'l+n': 2, 'ih+s': 3,
                            'n+ax': 4, 'ax+dh': 5, 's+ax': 6}
        eq_(context_match, expected_context)

        expected_output = {'2107': [[0.3225, 0.5225, 0, 0, 0],
                                    [0.4225, 0.5925, 1, 1, 1],
                                    [1.1025, 1.2925, 6, 4, 2]],
                           '42':  [[0.4525, 0.6525, 1, 1, 1],
                                   [0.5225, 0.7325, 2, 2, 0],
                                   [0.5925, 0.8725, 3, 0, 0]],
                           '23':  [[0.6525, 1.1025, 4, 3, 0],
                                   [0.7325, 1.1925, 4, 3, 1]],
                           '407': [[0.8725, 1.2425, 5, 3, 1]]}

        eq_(expected_output, out)

    def testLoadWithinItemFile(self):
        path_item_file = "test_data/dummy_item_within.item"
        out, context_match, phone_match, speaker_match = \
            abx_iterators.load_item_file(path_item_file)

        expected_output = {'2107': [[0., 0.2, 0, 0, 0],
                                    [0.3225, 0.5225, 1, 0, 0],
                                    [0.6, 0.75, 1, 0, 0],
                                    [0.4225, 0.5925, 2, 1, 1]],
                           '42':  [[0.4525, 0.6525, 2, 1, 1],
                                   [0.1301, 0.2501, 2, 2, 1],
                                   [0.5225, 0.7325, 2, 1, 0],
                                   [0.0025, 0.3561, 3, 1, 1],
                                   [0.5925, 0.8725, 3, 1, 0]]}
        eq_(expected_output, out)


class testABXFeatureLoader(unittest.TestCase):

    def setUp(self):
        self.stepFeature = 10

    def dummy_feature_maker(path_file, *args):
        data = torch.tensor(np.load(path_file))
        assert(len(data.size()) == 1)
        return data.view(-1, 1)

    def testBaseLoader(self):
        seqList = [('2107', 'test_data/2107.npy'),
                   ('42', 'test_data/42.npy'),
                   ('23', 'test_data/23.npy'),
                   ('407', 'test_data/407.npy')]

        dataset = abx_iterators.ABXFeatureLoader("test_data/dummy_item_file.item",
                                                 seqList,
                                                 testABXFeatureLoader.dummy_feature_maker,
                                                 self.stepFeature,
                                                 False)
        print(dataset.features)
        eq_(dataset.feature_dim, 1)
        eq_(len(dataset), 9)
        eq_(len(dataset.data.size()), 2)
        eq_(len(dataset.data), 16)
        data, size, coords = dataset[0]
        eq_(size, 1)
        eq_(coords, (0, 0, 0))
        eq_(data.tolist(), [[3]])

        data, size, coords = dataset[3]
        eq_(size, 1)
        eq_(coords, (1, 1, 1))
        eq_(data.tolist(), [[5]])

    def testWithinIterator(self):
        seqList = [('2107', 'test_data/2107.npy'),
                   ('42', 'test_data/42.npy')]
        dataset = abx_iterators.ABXFeatureLoader("test_data/dummy_item_within.item",
                                                 seqList,
                                                 testABXFeatureLoader.dummy_feature_maker,
                                                 self.stepFeature,
                                                 False)
        iterator = dataset.get_iterator('within', 40)
        eq_(iterator.index_csp, [0, 1, 2, 6, 3, 4, 5, 8, 7])
        eq_(iterator.groups_csp, [[[(0, 1)]], [[(1, 3)]], [
            [(3, 4)], [(4, 6), (6, 7)]], [[(7, 8)], [(8, 9)]]])
        eq_(len(iterator), 1)

        it = iter(iterator)
        c1, a_01, b_01, x_01 = next(it)
        eq_(c1, (1, 1, 2, 2))
        a_1, s_a = a_01
        eq_(s_a.tolist(), [1, 1])
        eq_(a_1.tolist(), [[[4.]], [[5.]]])
        eq_(x_01[0].tolist(), a_1.tolist())
        eq_(x_01[1].tolist(), s_a.tolist())
        eq_(b_01[0].tolist(), [[[1.]]])
        eq_(b_01[1].item(), 1)

        eq_(next(it, False), False)
        eq_(iterator.get_board_size(), (2, 3, 3, 4))
