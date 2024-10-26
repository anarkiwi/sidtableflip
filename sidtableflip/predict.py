#!/usr/bin/env python3

import argparse
import random
import time
import pandas as pd
from torchtune.utils import get_logger
from torch import argmax, roll, load, no_grad
import torch.nn.functional as F
from args import add_args
from model import get_device, get_model
from regdataset import RegDataset
from sidwav import write_samples


# TODO: add variation rather than most probable
def sample_next(predictions):
    probabilities = F.softmax(predictions[:, -1, :], dim=-1).cpu()
    return int(argmax(probabilities))


def generate(logger, dataset, model, device, prompt, args):
    states = []
    cycles = 0
    last_log = 0

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
        progress = cycles / float(args.output_cycles) * 100
        now = time.time()
        if now - last_log > 10:
            last_log = now
            logger.info("%.2f%%", progress)

    return states


def main():
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()
    logger = get_logger("INFO")
    dataset = RegDataset(args, logger=logger)
    device = get_device()
    model = get_model(dataset, args).to(device)
    best_model = load(args.model_state, weights_only=True)[0]
    model.load_state_dict(best_model)
    model.eval()

    # TODO: CLI prompt input.
    random.seed(time.time())
    start = random.randint(0, dataset.n_words)
    logger.info("starting at %u / %u", start, dataset.n_words)
    prompt = dataset.dfs_n[start:][: args.sequence_length].unsqueeze(0)
    states = generate(logger, dataset, model, device, prompt, args)

    df = pd.DataFrame(states, columns=["n"]).merge(dataset.tokens, on="n", how="left")
    if args.csv:
        df.to_csv(args.csv)
    write_samples(df, args.wav)


if __name__ == "__main__":
    main()
