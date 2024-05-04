#!/usr/bin/python3

import logging
import glob
import torch
import numpy as np
import pandas as pd


class RegDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.dfs = None
        self.dfs_n = None
        self.tokens = None
        self._load()
        self.n_vocab = len(self.tokens)
        self.n_words = len(self.dfs_n)
        logging.info(f"n_vocab: {self.n_vocab}, n_words {self.n_words}")

    def _read_df(self, name):
        logging.info(f"loading {name}")
        df = pd.read_csv(
            name,
            sep=" ",
            names=["clock", "chipno", "reg", "val"],
            dtype={
                "clock": np.uint64,
                "chipno": np.uint8,
                "reg": np.uint8,
                "val": np.uint8,
            },
        )
        assert df["reg"].min() >= 0
        df = df[df["chipno"] == 0]
        df = df[["clock", "reg", "val"]]
        df["diff"] = df["clock"].diff().fillna(0).astype(np.uint64)
        df = df[["diff", "reg", "val"]]
        return df

    def _maskreg(self, df, reg, valmask):
        mask = df["reg"] == reg
        df.loc[mask, ["val"]] = df[mask]["val"] & valmask

    def _downsample_df(self, df):
        # resample diffs to 128
        df["diff"] = (df["diff"].floordiv(128) * 128).astype(np.uint32)
        self._maskreg(df, 23, (2**8 - 1) - 2**3)
        # drop 4 low bits in pulse width
        for reg in (2, 9, 16):
            self._maskreg(df, reg, 240)
        # 21 filter cutoff low
        # 22 filter cutoff high
        # 23 filter res + route
        # 24 filter mode + vol
        return df

    def _load(self):
        self.dfs = [
            self._downsample_df(self._read_df(name))
            for name in sorted(glob.glob(self.args.reglogs))
        ]
        self.tokens = (
            pd.concat(self.dfs).drop_duplicates().sort_values(["reg", "val", "diff"])
        )
        self.tokens.reset_index(drop=True, inplace=True)
        self.tokens["n"] = self.tokens.index
        self.dfs = pd.concat(
            [
                df.merge(self.tokens, on=["reg", "val", "diff"], how="left")
                for df in self.dfs
            ]
        )
        self.dfs_n = torch.tensor(self.dfs["n"].values, dtype=torch.long)

    def __len__(self):
        return len(self.dfs) - self.args.sequence_length

    def __getitem__(self, index):
        def slice(n):
            return self.dfs_n[n : n + self.args.sequence_length]

        return (slice(index), slice(index + 1))


class RegDatasetSingle(RegDataset):
    def __len__(self):
        return (
            int(len(self.dfs) / self.args.sequence_length) - 1
        ) * self.args.sequence_length

    def __getitem__(self, index):
        def slice(n):
            return self.dfs_n[n : n + self.args.sequence_length]

        return (slice(index), self.dfs_n[index + self.args.sequence_length + 1])
