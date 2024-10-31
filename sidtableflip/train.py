#!/usr/bin/env python3

import argparse
import os
import pytorch_lightning as pl
import torch
from torchtune.utils import get_logger
from regdataset import RegDataset, get_loader
from args import add_args
from model import get_model


class SaveCallback(pl.callbacks.Callback):
    def __init__(self, args, logger, model):
        self.args = args
        self.logger = logger
        self.model = model

    def on_train_epoch_end(self, trainer, pl_module):
        self.logger.info("saving to %s", self.args.model_state)
        torch.save([self.model.state_dict()], self.args.model_state)


def train(model, dataset, dataloader, args, logger):
    callback = SaveCallback(args, logger, model)
    tb_logger = pl.loggers.TensorBoardLogger(args.tb_logs, name="sidtableflip")
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        default_root_dir=os.path.dirname(args.model_state),
        precision=args.trainer_precision,
        callbacks=[callback],
        enable_checkpointing=False,
        logger=tb_logger,
    )
    trainer.fit(model, dataloader)
    return model


def main():
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()
    logger = get_logger("INFO")
    dataset = RegDataset(args, logger=logger)
    dataloader = get_loader(args, dataset)
    model = get_model(dataset, args)
    train(model, dataset, dataloader, args, logger)


if __name__ == "__main__":
    main()
