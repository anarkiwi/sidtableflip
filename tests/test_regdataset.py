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
                (4, 0, 5, 6),
                (7, 0, 8, 9),
                (10, 0, 11, 12),
                (7, 0, 13, 14),
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
                dtype=np.uint32,
            )
            squeeze_df = loader._squeeze_changes(test_df)
            compare_df = pd.DataFrame(
                [
                    {"clock": 1, "reg": 1, "val": 1},
                    {"clock": 2, "reg": 2, "val": 2},
                    {"clock": 8192 + 2, "reg": 1, "val": 2},
                ],
                dtype=np.uint32,
            )
            self.assertTrue(compare_df.equals(squeeze_df), (compare_df, squeeze_df))

            compare_df = pd.DataFrame(
                [
                    {"diff": 64, "reg": 1, "val": 1},
                    {"diff": 8192, "reg": 2, "val": 2},
                    {"diff": 64, "reg": 1, "val": 2},
                ],
                dtype=np.uint32,
            )
            quantize_df = loader._quantize_diff(squeeze_df)[compare_df.columns]
            self.assertTrue(compare_df.equals(quantize_df), quantize_df)

            results = [(i.tolist(), j.tolist()) for i, j in loader]
            self.assertEqual(
                [([0, 1], [1, 2]), ([1, 2], [2, 3])],
                results,
            )
            tokens = [tuple([int(i) for i in x]) for x in loader.tokens.values]
            self.assertEqual(
                [(5, 6, 64, 0), (8, 9, 64, 1), (11, 12, 64, 2), (13, 14, 64, 3)], tokens
            )
