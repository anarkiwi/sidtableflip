#!/usr/bin/env python3

import argparse
import os
import pytorch_lightning as pl
from torchtune.utils import get_logger
from regdataset import RegDataset, get_loader
from args import add_args
from model import get_model


def train(model, dataloader, args):
    tb_logger = pl.loggers.TensorBoardLogger(args.tb_logs, "sidtableflip")
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        default_root_dir=os.path.dirname(args.model_state),
        precision=args.trainer_precision,
        enable_checkpointing=True,
        logger=tb_logger,
    )
    ckpt_path = None
    if os.path.exists(args.model_state):
        ckpt_path = args.model_state
    trainer.fit(model, dataloader, ckpt_path=ckpt_path)
    return model


def main():
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()
    logger = get_logger("INFO")
    dataset = RegDataset(args, logger=logger)
    dataloader = get_loader(args, dataset)
    model = get_model(dataset, args) # , mode="max-autotune")
    train(model, dataloader, args)


if __name__ == "__main__":
    main()
