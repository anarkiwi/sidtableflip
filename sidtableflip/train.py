#!/usr/bin/env python3

import argparse
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
from regdataset import RegDataset, get_loader
from args import add_args
from model import TransformerModel


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_float32_matmul_precision("high")

    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()

    dataset = RegDataset(args)
    dataloader = get_loader(args, dataset)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.compile(
        TransformerModel(dataset, sequence_length=args.sequence_length)
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    last_log = None
    for epoch in range(args.max_epochs):
        running_loss = 0
        for batch, input_seq_target_seq in enumerate(dataloader):
            input_seq, target_seq = input_seq_target_seq
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            outputs = model(input_seq)
            target_seq = target_seq.contiguous().view(-1)
            outputs = outputs.view(-1, dataset.n_vocab)
            loss = criterion(outputs, target_seq.view(-1))
            loss.sum().backward()
            optimizer.step()
            optimizer.zero_grad()
            now = time.time()
            if last_log is None or now - last_log > 10:
                progress = (batch * args.batch_size) / dataset.n_words * 100
                last_log = now
                logging.info(
                    f"epoch {epoch} batch {batch} ({progress: .2f}%): loss: {loss.item(): .2f}"
                )
            running_loss += loss.detach().cpu().numpy()
        epoch_loss = running_loss / len(dataloader)
        logging.info(f"Epoch {epoch} loss: {epoch_loss:.3f}")

    best_model = model.state_dict()
    torch.save([best_model], args.model_state)


if __name__ == "__main__":
    main()
