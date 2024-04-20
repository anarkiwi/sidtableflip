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
    rows = []

    for i in range(2):
        pattern = dataset.dfs[random.randint(0, n_chars - args.output_length) :][
            : args.output_length
        ]["n"].tolist()
        with torch.no_grad():
            for i in range(args.output_length):
                x = torch.tensor(
                    np.reshape(pattern, (1, len(pattern))), dtype=torch.long
                )
                prediction = model(x.to(device))
                index = int(prediction.argmax())
                rows.append(index)
                pattern.append(index)
                pattern = pattern[1:]

    df = pd.DataFrame(rows, columns=["n"]).merge(dataset.tokens, on="n")
    write_samples(df, args.wav)


if __name__ == "__main__":
    main()
