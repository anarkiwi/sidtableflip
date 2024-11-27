import logging
import glob
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
        self.n_vocab = 0
        self.n_words = 0

    def _read_df(self, name):
        self.logger.info(f"loading {name}")
        df = pd.read_csv(
            name,
            sep=" ",
            names=["clock", "irq_diff", "nmi_diff", "chipno", "reg", "val"],
            dtype={
                "clock": pd.UInt64Dtype(),
                "irq_diff": pd.UInt64Dtype(),
                "nmi_diff": pd.UInt64Dtype(),
                "chipno": pd.UInt8Dtype(),
                "reg": pd.Int8Dtype(),
                "val": pd.UInt16Dtype(),
            },
        )
        assert df["reg"].min() >= 0
        df["irq"] = df["clock"].astype(pd.Int64Dtype()) - df["irq_diff"]
        # keep only chipno 0
        df = df[df["chipno"] == 0]
        df = df[["clock", "reg", "val"]]
        return df

    def _make_tokens(self, dfs):
        tokens = pd.concat(dfs).drop_duplicates().sort_values(TOKEN_KEYS)
        tokens.reset_index(drop=True, inplace=True)
        tokens["n"] = tokens.index
        tokens = tokens.sort_values(["n"])
        return tokens

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

    def _combine_val(self, reg_df, reg, reg_range):
        origcols = reg_df.columns
        for i in range(reg_range):
            reg_df[str(i)] = reg_df[reg_df["reg"] == (reg + i)]["val"]
            reg_df[str(i)] = reg_df[str(i)].ffill().fillna(0)
            reg_df[str(i)] = np.left_shift(reg_df[str(i)].values, int(8 * i))
        reg_df.loc[:, "val"] = 0
        reg_df.loc[:, "reg"] = reg
        for i in range(reg_range):
            reg_df["val"] = reg_df["val"].astype(pd.UInt64Dtype()) + reg_df[str(i)]
        return reg_df[origcols]

    def _combine_reg(self, df, reg, diffmax=128):
        origcols = df.columns
        cond = (df["reg"] == reg) | (df["reg"] == (reg + 1))
        reg_df = df[cond].copy()
        df = df[~cond]
        reg_df = self._combine_val(reg_df, reg, 2)
        reg_df["clockdiff"] = (
            reg_df["clock"].astype(pd.Int64Dtype()).diff(-1).abs().fillna(diffmax + 1)
        )
        reg_df = reg_df[reg_df["clockdiff"] > diffmax]
        reg_df = reg_df[origcols]
        df = pd.concat([df, reg_df]).sort_values(["clock"]).reset_index(drop=True)
        return df

    def _combine_vreg(self, df, reg, reg_range=7):
        origcols = df.columns
        df["val"] = df["val"].astype(pd.UInt64Dtype())
        cond = (df["reg"] >= reg) & (df["reg"] < (reg + reg_range))
        reg_df = df[cond].copy()
        df = df[~cond]
        reg_df = self._combine_val(reg_df, reg, reg_range)
        reg_df = reg_df[origcols]
        df = pd.concat([df, reg_df]).sort_values(["clock"]).reset_index(drop=True)
        return df

    def _combine_regs(self, df, diffmax=128):
        for v in range(3):
            v_offset = v * 7
            for reg in (0, 2):
                df = self._combine_reg(df, reg + v_offset, diffmax)
        df = self._combine_reg(df, 21)
        return df

    def _combine_vregs(self, df):
        for v in range(3):
            v_offset = v * 7
            df = self._combine_vreg(df, v_offset)
        return df

    def _downsample_diff(self, df_diff, diffq):
        return (df_diff["diff"].floordiv(diffq).clip(lower=1) * diffq).astype(
            pd.UInt64Dtype()
        )

    def _quantize_diff(self, df):
        for diffq_pow in (2, 3, 4, 5):
            diffq = self.args.diffq**diffq_pow
            mask = df["diff"] > diffq
            df.loc[mask, ["diff"]] = self._downsample_diff(df, diffq)
        df["diff"] = self._downsample_diff(df, self.args.diffq)
        return df

    def _quantize_longdiff(self, df, diffmin, diffmax):
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

    def _downsample_df(self, df, diffmin=8, diffmax=128):
        df = self._squeeze_changes(df)
        df = self._combine_regs(df)
        df = self._combine_vregs(df)
        df = self._quantize_longdiff(df, diffmin, diffmax)
        df = self._quantize_diff(df)
        return df[TOKEN_KEYS].astype(pd.Int64Dtype())

    def _make_tokens(self, dfs):
        tokens = pd.concat(dfs).drop_duplicates().sort_values(TOKEN_KEYS)
        tokens.reset_index(drop=True, inplace=True)
        tokens["n"] = tokens.index
        tokens = tokens.sort_values(["n"])
        return tokens

    def load(self):
        if self.args.reglog:
            self.dfs = [self._downsample_df(self._read_df(self.args.reglog))]
            self.tokens = pd.read_csv(
                self.args.token_csv, dtype=pd.Int64Dtype(), index_col=0
            )
            self.dfs = pd.concat(self.dfs).merge(self.tokens, on=TOKEN_KEYS, how="left")
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
            self.dfs = pd.concat(self.dfs).merge(self.tokens, on=TOKEN_KEYS, how="left")
            if self.args.token_csv:
                self.logger.info("writing %s", self.args.token_csv)
                self.tokens.to_csv(self.args.token_csv)
            if self.args.dataset_csv:
                self.dfs.to_csv(self.args.dataset_csv)
        self.dfs_n = torch.LongTensor(self.dfs["n"].values)
        self.n_vocab = len(self.tokens)
        self.n_words = len(self.dfs_n)
        self.logger.info(f"n_vocab: {self.n_vocab}, n_words {self.n_words}")

    def __len__(self):
        return len(self.dfs_n) - self.args.sequence_length

    def slice_n(self, n):
        return self.dfs_n[n : n + self.args.sequence_length]

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError

        return (self.slice_n(index), self.slice_n(index + 1))


def get_loader(args, dataset):
    dataset.load()
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=args.shuffle,
        pin_memory=True,
        batch_size=args.batch_size,
        num_workers=4,  # os.cpu_count(),
    )
