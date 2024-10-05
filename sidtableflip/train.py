#!/usr/bin/env python3

import argparse
import time
from torch import optim, save, no_grad
from torch.nn import CrossEntropyLoss
from torchtune.utils import get_logger
from regdataset import RegDataset, get_loader
from args import add_args
from model import get_device, get_model


def evaluate(model, device, dataset, dataloader):
    model.eval()
    total_loss = 0.0
    criterion = CrossEntropyLoss()
    with no_grad():
        for batch, input_seq_target_seq in enumerate(dataloader):
            input_seq, target_seq = input_seq_target_seq
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            outputs = model(input_seq)
            outputs = outputs.view(-1, dataset.n_vocab)
            target_seq = target_seq.contiguous().view(-1)
            total_loss += len(input_seq) * criterion(outputs, target_seq).item()
    return total_loss / (len(dataset.dfs_n) - 1)


def train(model, device, dataset, dataloader, args):
    logger = get_logger("INFO")
    optimizer = {
        "adam": optim.Adam(model.parameters(), lr=args.learning_rate),
        "sgd": optim.SGD(model.parameters(), lr=args.learning_rate),
    }.get(args.optimizer)
    criterion = CrossEntropyLoss()
    last_log = None
    for epoch in range(args.max_epochs):
        model.train()
        running_loss = 0
        for batch, input_seq_target_seq in enumerate(dataloader):
            optimizer.zero_grad()
            model.zero_grad()
            input_seq, target_seq = input_seq_target_seq
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            outputs = model(input_seq)
            outputs = outputs.view(-1, dataset.n_vocab)
            target_seq = target_seq.contiguous().view(-1)
            loss = criterion(outputs, target_seq)
            loss.backward()
            optimizer.step()
            now = time.time()
            if last_log is None or now - last_log > 10:
                progress = (batch * args.batch_size) / dataset.n_words * 100
                last_log = now
                logger.info(
                    f"epoch {epoch} batch {batch} ({progress: .2f}%): loss: {loss.item(): .2f}"
                )
            running_loss += loss.detach().cpu().numpy()
        epoch_loss = running_loss / len(dataloader)
        logger.info(
            f"Epoch {epoch} running loss: {epoch_loss:.3f}",
        )
        save([model.state_dict()], args.model_state)
        # val_loss = evaluate(model, device, dataset, dataloader)
        # logging.info(
        #    f"Epoch {epoch} validation loss {val_loss:.3f}"
        # )
    return model


def main():
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()

    dataset = RegDataset(args)
    dataloader = get_loader(args, dataset)

    device = get_device()
    model = get_model(dataset, device, args)
    train(model, device, dataset, dataloader, args)


if __name__ == "__main__":
    main()
