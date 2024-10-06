#!/usr/bin/env python3

import argparse
import os
import pytorch_lightning as pl
import torch
from regdataset import RegDataset, get_loader
from args import add_args
from model import get_model


def train(model, dataset, dataloader, args):
    trainer = pl.Trainer(
        max_epochs=args.max_epochs, default_root_dir=os.path.dirname(args.model_state)
    )
    trainer.fit(model, dataloader)
    torch.save([model.state_dict()], args.model_state)
    return model


def main():
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()

    dataset = RegDataset(args)
    dataloader = get_loader(args, dataset)
    model = get_model(dataset, args)
    train(model, dataset, dataloader, args)


if __name__ == "__main__":
    main()
