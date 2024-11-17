import logging
import glob
import os
import random
import torch
import numpy as np
import pandas as pd

TOKEN_KEYS = ["reg", "val", "diff"]
DELAY_REG = -1


class RegDataset(torch.utils.data.Dataset):
    def __init__(self, args, logger=logging):
        self.args = args
        self.logger = logger
        self.dfs = None
        self.dfs_n = None
        self.tokens = None
        self._load()
        self.n_vocab = len(self.tokens)
        self.n_reg_val_vocab = len(self.reg_val_tokens)
        self.n_words = len(self.dfs_n)
        self.logger.info(
            f"n_vocab: {self.n_vocab}, n_reg_val_vocab {self.n_reg_val_vocab}, n_words {self.n_words}"
        )

    def _read_df(self, name):
        self.logger.info(f"loading {name}")
        df = pd.read_csv(
            name,
            sep=" ",
            names=["clock", "chipno", "reg", "val"],
            dtype={
                "clock": pd.UInt64Dtype(),
                "chipno": pd.UInt8Dtype(),
                "reg": pd.Int8Dtype(),
                "val": pd.UInt16Dtype(),
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
            df.pivot(columns="reg", values="val")
            .ffill()
            .fillna(0)
            .astype(pd.Int16Dtype())
        )
        reg_df = reg_df.loc[
            (reg_df[diff_cols].shift(fill_value=0) != reg_df[diff_cols]).any(axis=1)
        ]
        return reg_df.join(df)[["clock", "reg", "val"]].reset_index(drop=True)

    def _downsample_diff(self, df_diff, diffq):
        return (df_diff["diff"].floordiv(diffq).clip(lower=1) * diffq).astype(
            pd.UInt64Dtype()
        )

    def _quantize_reg(self, df, lb, diffmax):
        m = (df["reg"] == lb) | (df["reg"] == (lb + 1))
        h_df = df[m].copy()
        df = df[~m]
        h_df["lb"] = h_df[h_df["reg"] == lb]["val"]
        h_df["lb"] = h_df["lb"].ffill().fillna(0)
        h_df["hb"] = h_df[h_df["reg"] == (lb + 1)]["val"] * 256
        h_df["hb"] = h_df["hb"].ffill().fillna(0)
        h_df["val"] = (h_df["hb"] + h_df["lb"]).astype(pd.UInt16Dtype())
        h_df["reg"] = int(lb)
        h_df["clock"] = h_df["clock"].floordiv(diffmax) * int(diffmax)
        h_df = h_df.drop(["hb", "lb"], axis=1)
        h_df = h_df.drop_duplicates(subset=["clock"], keep="last")
        df = pd.concat([df, h_df]).sort_values(["clock"]).reset_index(drop=True)
        return df

    def _quantize_longdiff(self, df, diffmin, diffmax):
        for v in range(3):
            offset = v * 7
            # frequency, PCM
            for reg in (0, 2):
                df = self._quantize_reg(df, reg + offset, diffmax)
        # filter cutoff
        df = self._quantize_reg(df, 22, diffmax)
        df["diff"] = df["clock"].diff().shift(-1).fillna(0).astype(pd.Int64Dtype())
        # add delay rows
        m = df["diff"] >= diffmax
        long_df = df[m].copy()
        df.loc[m, "diff"] = diffmin
        long_df["reg"] = DELAY_REG
        long_df["val"] = 0
        long_df["clock"] += diffmin
        df = pd.concat([df, long_df]).sort_values(["clock"]).reset_index(drop=True)
        # move delay to DELAY_REG
        df["delaymarker"] = (
            (df["reg"] == DELAY_REG)
            .astype(pd.Int64Dtype())
            .diff(periods=1)
            .astype(pd.Int64Dtype())
            .cumsum()
            .cumsum()
            .shift(1)
            .fillna(0)
        )
        df["markerdelay"] = df.groupby("delaymarker")["diff"].transform("sum")
        df["markercount"] = df.groupby("delaymarker")["diff"].transform("count")
        df.loc[df["reg"] != DELAY_REG, ["diff"]] = 0
        df["diff"] = df["markerdelay"] - (df["markercount"] * diffmin)
        df.loc[df["reg"] != DELAY_REG, ["diff"]] = diffmin
        df = df.drop(["clock", "delaymarker", "markerdelay", "markercount"], axis=1)
        return df

    def _quantize_diff(self, df, diffmin=8, diffmax=128):
        df = self._quantize_longdiff(df, diffmin, diffmax)
        for diffq_pow in (2, 3, 4, 5):
            diffq = self.args.diffq**diffq_pow
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
        return df[TOKEN_KEYS]

    def _make_tokens(self, dfs):
        tokens = pd.concat(dfs).drop_duplicates().sort_values(TOKEN_KEYS)
        tokens.reset_index(drop=True, inplace=True)
        tokens["n"] = tokens.index
        tokens = tokens.sort_values(["n"])
        return tokens

    def _load(self):
        if self.args.reglog:
            self.dfs = [self._downsample_df(self._read_df(self.args.reglog))]
            self.tokens = pd.read_csv(
                self.args.token_csv, dtype=pd.Int64Dtype(), index_col=0
            )
        else:
            files = []
            random.seed(0)
            for reglogs in self.args.reglogs.split(","):
                globbed = list(glob.glob(reglogs))
                while len(files) < self.args.max_files and globbed:
                    file = random.choice(globbed)
                    files.append(file)
                    globbed.remove(file)
            random.seed()
            self.dfs = [
                self._downsample_df(self._read_df(name)) for name in sorted(files)
            ]
            self.tokens = self._make_tokens(self.dfs)
            if self.args.token_csv:
                self.logger.info("writing %s", self.args.token_csv)
                self.tokens.to_csv(self.args.token_csv)
        self.reg_val_tokens = (
            pd.concat(self.dfs)[["reg", "val"]]
            .drop_duplicates()
            .sort_values(["reg", "val"])
        )
        self.dfs = pd.concat(self.dfs)
        dfs = self.dfs.merge(self.tokens, on=TOKEN_KEYS, how="left")
        missing_tokens = dfs[dfs["n"].isna()].drop_duplicates()[TOKEN_KEYS].copy()
        if len(missing_tokens):
            for row in missing_tokens.itertuples():
                token_cond = (
                    (self.dfs["reg"] == row.reg)
                    & (self.dfs["val"] == row.val)
                    & (self.dfs["diff"] == row.diff)
                )
                nodiffs = self.tokens[
                    (self.tokens["reg"] == row.reg) & (self.tokens["val"] == row.val)
                ]
                if len(nodiffs) == 0:
                    self.dfs = self.dfs[~token_cond]
                    self.logger.info(
                        "reg %u val %u has no token, dropping", row.reg, row.val
                    )
                    continue
                diff2 = (nodiffs["diff"].astype(pd.Int64Dtype()) - row.diff).abs()
                mindiff = nodiffs[diff2 == diff2.min()].iloc[0]["diff"]
                self.dfs.loc[token_cond, ["diff"]] = mindiff
                self.logger.info(
                    "replacing reg %u val %u diff %u with diff %u",
                    row.reg,
                    row.val,
                    row.diff,
                    mindiff,
                )
            dfs = self.dfs.merge(self.tokens, on=TOKEN_KEYS, how="left")
        self.dfs = dfs
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
        num_workers=4,  # os.cpu_count(),
    )
