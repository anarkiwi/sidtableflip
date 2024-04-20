#!/usr/bin/python3

import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from args import add_args
from model import SingleModel
from regdataset import RegDatasetSingle


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()

    dataset = RegDatasetSingle(args)
    model = SingleModel(dataset)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=args.batch_size
    )
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    best_model = None
    best_loss = np.inf
    for epoch in range(args.max_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch.to(device))
            loss = loss_fn(y_pred, y_batch.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        loss = 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                y_pred = model(X_batch.to(device))
                loss += loss_fn(y_pred, y_batch.to(device))
            if loss < best_loss:
                best_loss = loss
                best_model = model.state_dict()
            logging.info(f"Epoch {epoch}: Cross-entropy: {loss: .4f}")

    torch.save([best_model], args.single_model)


if __name__ == "__main__":
    main()
