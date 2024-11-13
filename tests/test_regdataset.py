import os
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch

from sidtableflip.regdataset import RegDataset


class FakeArgs:
    def __init__(self, reglogs, sequence_length):
        self.reglog = None
        self.reglogs = reglogs
        self.sequence_length = sequence_length
        self.max_files = 1
        self.diffq = 64
        self.token_csv = None


class TestRegDatasetLoader(unittest.TestCase):
    def test_loader(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_log_name = os.path.join(tmpdir, "log.txt")
            regdata = [
                (1, 0, 2, 3),
                (5, 0, 5, 6),
                (12, 0, 8, 9),
                (22, 0, 11, 12),
                (29, 0, 13, 14),
            ]
            with open(test_log_name, "w") as log:
                for reg_tuple in regdata:
                    log.write(" ".join([str(x) for x in reg_tuple]) + "\n")
            args = FakeArgs(test_log_name, 2)
            loader = RegDataset(args)

            self.assertEqual(loader.highbitmask(7), 128)
            self.assertEqual(loader.highbitmask(4), 240)
            self.assertEqual(loader.highbitmask(1), 254)

            test_df = pd.DataFrame(
                [
                    {"clock": 1, "reg": 1, "val": 1},
                    {"clock": 2, "reg": 2, "val": 2},
                    {"clock": 3, "reg": 1, "val": 1},  # dropped as no-op
                    {"clock": 8192 + 2, "reg": 1, "val": 2},
                ],
                dtype=pd.Int64Dtype(),
            )
            squeeze_df = loader._squeeze_changes(test_df)
            compare_df = pd.DataFrame(
                [
                    {"clock": 1, "reg": 1, "val": 1},
                    {"clock": 2, "reg": 2, "val": 2},
                    {"clock": 8192 + 2, "reg": 1, "val": 2},
                ],
                dtype=pd.Int64Dtype(),
            )
            self.assertTrue(compare_df.equals(squeeze_df), (compare_df, squeeze_df))

            compare_df = pd.DataFrame(
                [
                    {"diff": 64, "reg": 1, "val": 1},
                    {"diff": 64, "reg": 2, "val": 2},
                    {"diff": 4096, "reg": -1, "val": 0},
                    {"diff": 64, "reg": 1, "val": 2},
                ],
                dtype=pd.Int64Dtype(),
            )
            compare_df["diff"] = compare_df["diff"].astype(pd.UInt64Dtype())
            quantize_df = loader._quantize_diff(squeeze_df)[compare_df.columns]
            self.assertTrue(compare_df.equals(quantize_df), (quantize_df, compare_df))

            results = [(i.tolist(), j.tolist()) for i, j in loader]
            self.assertEqual(
                [([0, 1], [1, 2]), ([1, 2], [2, 3])],
                results,
            )
            tokens = [tuple([int(i) for i in x]) for x in loader.tokens.values]
            self.assertEqual(
                [(5, 6, 64, 0), (8, 9, 64, 1), (11, 12, 64, 2), (13, 14, 64, 3)], tokens
            )

            unquantized_df = pd.DataFrame(
                [
                    {"clock": 1000, "reg": 1, "val": 1},
                    {"clock": 1016, "reg": 2, "val": 2},
                    {"clock": 1032, "reg": 3, "val": 3},
                    {"clock": 2000, "reg": 4, "val": 4},
                    {"clock": 2016, "reg": 5, "val": 5},
                    {"clock": 3000, "reg": 6, "val": 6},
                ]
            )
            diffmin = 8
            quantized_df = loader._quantize_longdiff(unquantized_df, diffmin=diffmin)
            compare_df = pd.DataFrame(
                [
                    {"reg": 1, "val": 1, "diff": 8},
                    {"reg": 2, "val": 2, "diff": 8},
                    {"reg": 3, "val": 3, "diff": 8},
                    {"reg": -1, "val": 0, "diff": 976},
                    {"reg": 4, "val": 4, "diff": 8},
                    {"reg": 5, "val": 5, "diff": 8},
                    {"reg": -1, "val": 0, "diff": 984},
                    {"reg": 6, "val": 6, "diff": 8},
                ],
                dtype=pd.Int64Dtype(),
            )
            self.assertTrue(quantized_df.astype(pd.Int64Dtype()).equals(compare_df))
            self.assertEqual(
                unquantized_df["clock"].diff().sum(),
                quantized_df["diff"].sum() - diffmin,
            )
