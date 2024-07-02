import os
import tempfile
import unittest
import torch

from sidtableflip.regdataset import RegDataset


class FakeArgs:
    def __init__(self, reglogs, sequence_length):
        self.reglogs = reglogs
        self.sequence_length = sequence_length
        self.max_files = 1
        self.diffq = 64


class TestRegDatasetLoader(unittest.TestCase):
    def test_loader(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_log_name = os.path.join(tmpdir, "log.txt")
            regdata = [
                (1, 0, 2, 3),
                (4, 0, 5, 6),
                (7, 0, 8, 9),
                (10, 0, 11, 12),
                (7, 0, 8, 9),
                (4, 0, 5, 6),
            ]
            with open(test_log_name, "w") as log:
                for reg_tuple in regdata:
                    log.write(" ".join([str(x) for x in reg_tuple]) + "\n")
            args = FakeArgs(test_log_name, 2)
            loader = RegDataset(args)
            results = [(i.tolist(), j.tolist()) for i, j in loader]
            self.assertEqual(
                [
                    ([0, 1], [1, 2]),
                    ([1, 2], [2, 3]),
                    ([2, 3], [3, 2]),
                    ([3, 2], [2, 1]),
                ],
                results,
            )
            tokens = [tuple([int(i) for i in x]) for x in loader.tokens.values]
            self.assertEqual(
                [(1, 2, 3, 0), (1, 5, 6, 1), (1, 8, 9, 2), (1, 11, 12, 3)], tokens
            )
