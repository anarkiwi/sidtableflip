#!/usr/bin/env python3

import argparse
import logging
import random
import time
import torch
import pandas as pd
from regdataset import RegDataset
import torch.nn.functional as F
from args import add_args
from model import TransformerModel
from sidwav import write_samples


# TODO: add variation rather than most probable
def sample_next(predictions):
    probabilities = F.softmax(predictions[:, -1, :], dim=-1).cpu()
    next_token = torch.argmax(probabilities)
    return int(next_token.cpu())


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_float32_matmul_precision("high")

    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()

    dataset = RegDataset(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.compile(
        TransformerModel(dataset, sequence_length=args.sequence_length)
    ).to(device)
    best_model = torch.load(args.model_state)[0]
    model.load_state_dict(best_model)
    model.eval()

    # TODO: CLI prompt input.
    n = len(dataset.dfs_n)
    random.seed(time.time())
    start = random.randint(0, n)
    prompt = dataset.dfs_n[start:][: args.sequence_length].unsqueeze(0)
    states = []

    for _ in range(args.output_length):
        prompt = prompt.to(device)
        with torch.no_grad():
            predictions = model(prompt)
        prompt = torch.roll(prompt.to("cpu"), -1)
        state = sample_next(predictions)
        prompt[-1][0] = state
        states.append(state)

    df = pd.DataFrame(states, columns=["n"]).merge(dataset.tokens, on="n", how="left")
    write_samples(df, args.wav)


if __name__ == "__main__":
    main()
