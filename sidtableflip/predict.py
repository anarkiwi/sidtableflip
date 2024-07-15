#!/usr/bin/env python3

import argparse
import logging
import random
import time
import pandas as pd
from regdataset import RegDataset
from torch import argmax, roll, load, no_grad
import torch.nn.functional as F
from args import add_args
from model import get_device, get_model
from sidwav import write_samples


# TODO: add variation rather than most probable
def sample_next(predictions):
    probabilities = F.softmax(predictions[:, -1, :], dim=-1).cpu()
    return int(argmax(probabilities))


def generate(dataset, model, device, prompt, args):
    states = []
    cycles = 0

    while cycles < args.output_cycles:
        prompt = prompt.to(device)
        with no_grad():
            predictions = model(prompt)
        prompt = roll(prompt.to("cpu"), -1)
        state = sample_next(predictions)
        diff = dataset.tokens[dataset.tokens["n"] == state].iloc[0]["diff"]
        cycles += diff
        prompt[0][-1] = state
        states.append(state)

    return states


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()

    dataset = RegDataset(args)
    device = get_device()
    model = get_model(dataset, device, args)
    best_model = load(args.model_state)[0]
    model.load_state_dict(best_model)
    model.eval()

    # TODO: CLI prompt input.
    n = len(dataset.dfs_n)
    random.seed(time.time())
    start = random.randint(0, n)
    prompt = dataset.dfs_n[start:][: args.sequence_length].unsqueeze(0)
    states = generate(dataset, model, device, prompt, args)

    df = pd.DataFrame(states, columns=["n"]).merge(dataset.tokens, on="n", how="left")
    if args.csv:
        df.to_csv(args.csv)
    write_samples(df, args.wav)


if __name__ == "__main__":
    main()
