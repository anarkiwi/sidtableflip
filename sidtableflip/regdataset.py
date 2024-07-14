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
        # keep only chipno 0
        df = df[df["chipno"] == 0]
        df = df[["clock", "reg", "val"]]
        return df

    def _maskreg(self, df, reg, valmask):
        mask = df["reg"] == reg
        df.loc[mask, ["val"]] = df[mask]["val"] & valmask

    def highbitmask(self, bits):
        return 255 - (2**bits - 1)

    def _maskregbits(self, df, reg, bits):
        self._maskreg(df, reg, self.highbitmask(bits))

    def _squeeze_changes(self, df):
        diff_cols = df.reg.unique()
        reg_df = (
            df.pivot(columns="reg", values="val").ffill().fillna(0).astype(np.uint8)
        )
        reg_df = reg_df.loc[
            (reg_df[diff_cols].shift(fill_value=0) != reg_df[diff_cols]).any(axis=1)
        ]
        return reg_df.join(df)[["clock", "reg", "val"]].reset_index(drop=True)

    def _downsample_diff(self, df_diff, diffq):
        return (df_diff["diff"].floordiv(diffq).clip(lower=1) * diffq).astype(np.uint32)

    def _quantize_diff(self, df):
        df["diff"] = df["clock"].diff().fillna(0).astype(np.uint64)
        for diffq in (self.args.diffq**2,):
            mask = df["diff"] > diffq
            df.loc[mask, ["diff"]] = self._downsample_diff(df, diffq)
        df["diff"] = self._downsample_diff(df, self.args.diffq)
        return df

    def _downsample_df(self, df):
        for v in range(3):
            v_offset = v * 7
            # keep high 4 bits, of PCM low
            self._maskregbits(df, 2 + v_offset, 4)
            # keep high 7 bits of freq low
            # self._maskregbits(df, v_offset, 1)
        # discard low 4 bits of filter cutoff.
        df = df[df["reg"] != 21].copy()
        # keep high 4 bits of filter cutoff
        # self._maskregbits(df, 22, 4)
        # 23 filter res + route
        # 24 filter mode + vol
        df = self._squeeze_changes(df)
        df = self._quantize_diff(df)
        return df[["diff", "reg", "val"]]

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
        if self.args.token_csv:
            self.tokens.to_csv(self.args.token_csv)
        self.dfs = pd.concat(
            [
                df.merge(self.tokens, on=["reg", "val", "diff"], how="left")
                for df in self.dfs
            ]
        )
        self.dfs_n = torch.LongTensor(self.dfs["n"].values)

    def __len__(self):
        return len(self.dfs) - self.args.sequence_length

    def slice_n(self, n):
        return self.dfs_n[n : n + self.args.sequence_length]

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError

        return (self.slice_n(index), self.slice_n(index + 1))


def get_loader(args, dataset):
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=args.shuffle,
        pin_memory=True,
        batch_size=args.batch_size,
    )
