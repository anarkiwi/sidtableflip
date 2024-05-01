#!/usr/bin/python3

import argparse
import logging
import random
import torch
import pandas as pd
from regdataset import RegDataset
from args import add_args
from model import Model
from sidwav import write_samples


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()

    dataset = RegDataset(args)
    model = Model(dataset)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_model = torch.load(args.model_state)[0]
    model.load_state_dict(best_model)
    model.eval()

    state_h, state_c = model.init_state(1)
    prompt = dataset.dfs_n[random.randint(0, len(dataset.dfs_n)) :][
        : args.sequence_length
    ].unsqueeze(1)
    states = []

    for _ in range(args.output_length):
        y_pred, (state_h, state_c) = model(
            prompt.to(device), (state_h.to(device), state_c.to(device))
        )
        last_word_logits = y_pred[0][-1]
        prompt = torch.roll(prompt, -1)
        state = int(last_word_logits.argmax())
        prompt[-1][0] = state
        states.append(state)

    df = pd.DataFrame(states, columns=["n"]).merge(dataset.tokens, on="n", how="left")
    write_samples(df, args.wav)


if __name__ == "__main__":
    main()
