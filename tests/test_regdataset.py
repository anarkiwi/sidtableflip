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
                {"reg": 1, "val": 1, "diff": 1, "n": 1},
                {"reg": 1, "val": 2, "diff": 1, "n": 2},
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

    def test_add_voice_reg(self):
        loader = RegDataset(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"clock": 0, "reg": 0, "val": 255},
                {"clock": 2, "reg": 9, "val": 255},
                {"clock": 4, "reg": 0, "val": 1},
            ],
            dtype=pd.Int64Dtype(),
        )
        add_df = pd.DataFrame(
            [
                {"clock": 0, "reg": 0, "val": 255},
                {"clock": 1, "reg": -2, "val": 1},
                {"clock": 2, "reg": 2, "val": 255},
                {"clock": 3, "reg": -2, "val": 0},
                {"clock": 4, "reg": 0, "val": 1},
            ],
            dtype=pd.Int64Dtype(),
        )
        self.assertTrue(add_df.equals(loader._add_voice_reg(test_df)))

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

    def test_combine_reg(self):
        loader = RegDataset(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"clock": 0, "reg": 1, "val": 1},
                {"clock": 32, "reg": 1, "val": 1},
                {"clock": 64, "reg": 2, "val": 1},
                {"clock": 80, "reg": 2, "val": 2},
                {"clock": 128, "reg": 1, "val": 2},
            ],
            dtype=pd.UInt64Dtype(),
        )
        combine_df = pd.DataFrame(
            [
                {"clock": 0, "reg": 1, "val": 1},
                {"clock": 32, "reg": 1, "val": 1},
                {"clock": 80, "reg": 1, "val": 513},
                {"clock": 128, "reg": 1, "val": 514},
            ],
            dtype=pd.UInt64Dtype(),
        )
        self.assertTrue(
            combine_df.equals(
                loader._combine_reg(test_df, 1, 16).astype(pd.UInt64Dtype())
            )
        )

    def test_combine_vreg(self):
        loader = RegDataset(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"clock": 0, "reg": 0, "val": 1},
                {"clock": 64, "reg": 2, "val": 2},
                {"clock": 80, "reg": 4, "val": 4},
            ],
            dtype=pd.UInt64Dtype(),
        )
        combine_df = pd.DataFrame(
            [
                {"clock": 0, "reg": 0, "val": 1},
                {"clock": 64, "reg": 0, "val": (2 << (2 * 8)) + 1},
                {"clock": 80, "reg": 0, "val": (2 << (2 * 8)) + 1 + (4 << (4 * 8))},
            ],
            dtype=pd.UInt64Dtype(),
        )
        self.assertTrue(combine_df.equals(loader._combine_vreg(test_df, 0)))

    def test_rotate_voice_augment(self):
        loader = RegDataset(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"clock": 0, "reg": 0, "val": 1},
                {"clock": 8, "reg": 4, "val": 1},
                {"clock": 12, "reg": 11, "val": 2},
                {"clock": 16, "reg": 23, "val": 1 + 4},
                {"clock": 32, "reg": 7, "val": 2},
                {"clock": 64, "reg": 14, "val": 3},
            ],
            dtype=pd.Int64Dtype(),
        )
        rotate_df = pd.DataFrame(
            [
                {"clock": 0, "reg": 0, "val": 1},
                {"clock": 8, "reg": -3, "val": 1},
                {"clock": 12, "reg": -3, "val": 513},
                {"clock": 16, "reg": 23, "val": 1 + 4},
                {"clock": 32, "reg": 7, "val": 2},
                {"clock": 64, "reg": 14, "val": 3},
                {"clock": 0, "reg": 7, "val": 1},
                {"clock": 8, "reg": -3, "val": 256},
                {"clock": 12, "reg": -3, "val": 131328},
                {"clock": 16, "reg": 23, "val": 2 + 1},
                {"clock": 32, "reg": 14, "val": 2},
                {"clock": 64, "reg": 0, "val": 3},
                {"clock": 0, "reg": 14, "val": 1},
                {"clock": 8, "reg": -3, "val": 65536},
                {"clock": 12, "reg": -3, "val": 65538},
                {"clock": 16, "reg": 23, "val": 4 + 2},
                {"clock": 32, "reg": 0, "val": 2},
                {"clock": 64, "reg": 7, "val": 3},
            ],
            dtype=pd.Int64Dtype(),
        )
        result_df = pd.concat(loader._rotate_voice_augment(test_df)).reset_index(
            drop=True
        )
        self.assertTrue(rotate_df.equals(result_df), result_df)
