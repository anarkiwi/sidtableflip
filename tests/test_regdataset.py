import os
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch

from sidtableflip.regdataset import RegDataset


class FakeArgs:
    def __init__(self, sequence_length=128, diffq=64):
        self.reglog = None
        self.reglogs = ""
        self.sequence_length = sequence_length
        self.max_files = 1
        self.diffq = diffq
        self.token_csv = None


class TestRegDatasetLoader(unittest.TestCase):
    def test_make_tokens(self):
        loader = RegDataset(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": 1, "val": 1, "diff": 1},
                {"reg": 1, "val": 1, "diff": 1},
                {"reg": 1, "val": 2, "diff": 1},
            ]
        )
        tokens_df = pd.DataFrame(
            [
                {"reg": 1, "val": 1, "diff": 1, "n": 0},
                {"reg": 1, "val": 2, "diff": 1, "n": 1},
            ]
        )
        self.assertTrue(tokens_df.equals(loader._make_tokens([test_df])))

    def test_highbitmask(self):
        loader = RegDataset(FakeArgs())
        self.assertEqual(loader.highbitmask(7), 128)
        self.assertEqual(loader.highbitmask(4), 240)
        self.assertEqual(loader.highbitmask(1), 254)

    def test_maskregbits(self):
        loader = RegDataset(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": 1, "val": 255},
                {"reg": 1, "val": 128},
            ]
        )
        loader._maskregbits(test_df, 1, 1)
        mask_df = pd.DataFrame(
            [
                {"reg": 1, "val": 254},
                {"reg": 1, "val": 128},
            ]
        )
        self.assertTrue(mask_df.equals(test_df))

    def test_squeeze_changes(self):
        loader = RegDataset(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"clock": 0, "reg": 1, "val": 1},
                {"clock": 1, "reg": 1, "val": 1},
                {"clock": 2, "reg": 2, "val": 1},
                {"clock": 3, "reg": 2, "val": 2},
            ]
        )
        squeeze_df = pd.DataFrame(
            [
                {"clock": 0, "reg": 1, "val": 1},
                {"clock": 2, "reg": 2, "val": 1},
                {"clock": 3, "reg": 2, "val": 2},
            ]
        )
        self.assertTrue(squeeze_df.equals(loader._squeeze_changes(test_df)))

    def test_loader(self):
        sequence_length = 2
        dfs_n = torch.LongTensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        loader = RegDataset(FakeArgs(sequence_length))
        loader.dfs_n = dfs_n
        self.assertEqual(len(loader), len(dfs_n) - sequence_length)
        loader_output = [[j.tolist() for j in loader[i]] for i in range(len(loader))]
        self.assertEqual(len(loader_output), len(loader))
        self.assertEqual(
            [
                [[1, 2], [2, 3]],
                [[2, 3], [3, 4]],
                [[3, 4], [4, 5]],
                [[4, 5], [5, 6]],
                [[5, 6], [6, 7]],
                [[6, 7], [7, 8]],
                [[7, 8], [8, 9]],
                [[8, 9], [9, 10]],
            ],
            loader_output,
        )
