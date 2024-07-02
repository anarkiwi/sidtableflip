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


def sample_next(predictions):
    probabilities = F.softmax(predictions[:, -1, :], dim=-1).cpu()
    next_token = torch.argmax(probabilities)
    return int(next_token.cpu())


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()

    dataset = RegDataset(args)
    model = TransformerModel(dataset, sequence_length=args.sequence_length)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_model = torch.load(args.model_state)[0]
    model.load_state_dict(best_model)
    model.eval()

    n = len(dataset.dfs_n)
    random.seed(time.time())
    start = random.randint(0, n)
    prompt = dataset.dfs_n[start:][: args.sequence_length].unsqueeze(0)
    # states = dataset.dfs_n[:args.sequence_length].tolist()
    states = dataset.dfs_n[:100].tolist()
    print("start", start)

    for _ in range(args.output_length):
        prompt = prompt.to(device)
        with torch.no_grad():
            predictions = model(prompt)
        prompt = prompt.to("cpu")
        # prompt = torch.roll(prompt.to("cpu"), -1)
        state = sample_next(predictions)
        z = prompt.squeeze(0).tolist() + [state]
        prompt = torch.tensor(z[-args.sequence_length :]).unsqueeze(0)
        # prompt[-1][0] = state
        states.append(state)

    df = pd.DataFrame(states, columns=["n"]).merge(dataset.tokens, on="n", how="left")
    write_samples(df, args.wav)


if __name__ == "__main__":
    main()
