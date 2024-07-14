#!/usr/bin/env python3

import argparse
import logging
import time
from torch import optim, save
from torch.nn import CrossEntropyLoss
from regdataset import RegDataset, get_loader
from args import add_args
from model import get_device, get_model


def train(model, device, dataset, dataloader, args):
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = CrossEntropyLoss()
    model.train()
    last_log = None
    for epoch in range(args.max_epochs):
        running_loss = 0
        for batch, input_seq_target_seq in enumerate(dataloader):
            input_seq, target_seq = input_seq_target_seq
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            outputs = model(input_seq)
            outputs = outputs.view(-1, dataset.n_vocab)
            target_seq = target_seq.contiguous().view(-1)
            loss = criterion(outputs, target_seq.view(-1))
            loss.backward()
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
    return model


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()

    dataset = RegDataset(args)
    dataloader = get_loader(args, dataset)

    device = get_device()
    model = get_model(dataset, device, args)
    model = train(model, device, dataset, dataloader, args)
    best_model = model.state_dict()
    save([best_model], args.model_state)


if __name__ == "__main__":
    main()
