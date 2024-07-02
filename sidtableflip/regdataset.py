#!/usr/bin/python3

import logging
import glob
import random
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

    def _maskregbits(self, df, reg, bits):
        self._maskreg(df, reg, 255 - (2**bits - 1))

    def _downsample_df(self, df):
        # resample diffs to 128
        df["diff"] = (df["diff"].floordiv(64) * 64).astype(np.uint32) + 1
        # 21 filter cutoff low
        # 22 filter cutoff high
        # 23 filter res + route
        # 24 filter mode + vol
        return df

    def _load(self):
        random.seed(0)
        globbed = list(glob.glob(self.args.reglogs))
        files = []
        while len(files) < self.args.max_files and globbed:
            file = random.choice(globbed)
            files.append(file)
            globbed.remove(file)
        self.dfs = [self._downsample_df(self._read_df(name)) for name in sorted(files)]
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
        self.dfs_n = torch.LongTensor(self.dfs["n"].values)

    def __len__(self):
        return len(self.dfs) - self.args.sequence_length

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError

        def slice_n(n):
            return self.dfs_n[n : n + self.args.sequence_length]

        return (slice_n(index), slice_n(index + 1))


def get_loader(args, dataset):
    return torch.utils.data.DataLoader(
        dataset, shuffle=args.shuffle, batch_size=args.batch_size, pin_memory=True
    )
