#!/usr/bin/python3

import argparse
import logging
import torch
import numpy as np
from regdataset import RegDataset
from args import add_args
from model import Model


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()

    dataset = RegDataset(args)
    model = Model(dataset)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters())
    loss_f = torch.nn.CrossEntropyLoss(reduction="sum")
    loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=args.batch_size
    )

    for epoch in range(args.max_epochs):
        state_h, state_c = model.init_state(args.sequence_length)
        model.train()
        for batch, (x, y) in enumerate(loader):
            optimizer.zero_grad()
            y_pred, (state_h, state_c) = model(
                x.to(device), (state_h.to(device), state_c.to(device))
            )
            loss = loss_f(y_pred.transpose(1, 2), y.to(device))
            state_h = state_h.detach()
            state_c = state_c.detach()
            loss.backward()
            optimizer.step()
            progress = (batch * args.batch_size) / dataset.n_words * 100
            logging.info(
                f"epoch {epoch} batch {batch} ({progress: .2f}%): loss: {loss.item(): .2f}"
            )

    best_model = model.state_dict()
    torch.save([best_model], args.model_state)


if __name__ == "__main__":
    main()
