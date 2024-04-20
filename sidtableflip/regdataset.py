#!/usr/bin/python3

import logging
import glob
import torch
import numpy as np
import pandas as pd


class RegDataset(torch.utils.data.Dataset):
    regwidths = {
        3: 4,  # v1 PWM high
        10: 4,  # v2 PWM high
        17: 4,  # v3 PWM high,
        21: 3,  # filter cutoff low
    }

    def __init__(self, args):
        self.args = args
        self.dfs = None
        self.dfs_n = None
        self.tokens = None
        self._load()
        self.n_vocab = len(self.tokens)
        self.n_words = len(self.dfs_n)
        logging.info(f"n_vocab: {self.n_vocab}, n_words {self.n_words}")

    def _squeeze_diffs(self, df, diff_cols, fill_value=0):
        return df.loc[
            (df[diff_cols].shift(fill_value=fill_value) != df[diff_cols]).any(axis=1)
        ]

    def _rightsize_regs(self, df):
        for reg, width in self.regwidths.items():
            mask = df["reg"] == reg
            df.loc[mask, "val"] = df[mask]["val"] & 2**width
        mask = df["reg"] == 23
        # clear fltex
        df.loc[mask, "val"] = df[mask]["val"] & ((2**8 - 1) - 2**3)
        return df

    def _read_df(self, name):
        logging.info(f"loading {name}")
        df = pd.read_csv(
            name,
            sep=" ",
            names=["clock_offset", "reg", "val"],
            dtype={"clock_offset": np.uint64, "reg": np.uint8, "val": np.uint8},
        )
        df["clock"] = df["clock_offset"].cumsum()
        assert df["reg"].min() >= 0
        df = df[["clock", "reg", "val"]]
        reg_dfs = []
        reg_cols = ["reg", "val"]
        df = self._rightsize_regs(df)
        for reg in sorted(df.reg.unique()):
            reg_df = df[df["reg"] == reg]
            reg_df = self._squeeze_diffs(reg_df, reg_cols)
            reg_dfs.append(reg_df)
        df = pd.concat(reg_dfs).sort_values("clock")
        df["diff"] = df["clock"].diff().fillna(0).astype(np.uint64)
        df = df[["diff", "reg", "val"]]
        return df

    def _downsample_df(self, df, ds={"diff": 128}):
        for col, val in ds.items():
            df[col] = (df[col].floordiv(val) * val).astype(np.uint32)
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
            [df.merge(self.tokens, on=["reg", "val", "diff"]) for df in self.dfs]
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
