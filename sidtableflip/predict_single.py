#!/usr/bin/python3

import argparse
import logging
import random
import pandas as pd
import numpy as np
import torch
from args import add_args
from model import SingleModel
from regdataset import RegDatasetSingle
from sidwav import write_samples


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()
    dataset = RegDatasetSingle(args)
    model = SingleModel(dataset)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    n_chars = dataset.n_words

    best_model = torch.load(args.single_model)[0]
    model.load_state_dict(best_model)
    model.eval()
    states = []

    for _ in range(2):
        pattern = dataset.dfs[random.randint(0, n_chars - args.output_length) :][
            : args.output_length
        ]["n"].tolist()
        x = torch.tensor(np.reshape(pattern, (1, len(pattern))), dtype=torch.long)
        with torch.no_grad():
            for _ in range(args.output_length):
                prediction = model(x.to(device))
                state = int(prediction.argmax())
                states.append(state)
                x = torch.roll(x, -1, 1)
                x[0][-1] = state

    df = pd.DataFrame(states, columns=["n"]).merge(dataset.tokens, on="n", how="left")
    write_samples(df, args.wav)


if __name__ == "__main__":
    main()
