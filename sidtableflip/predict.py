#!/usr/bin/env python3

import argparse
import random
import time
import pandas as pd
from torchtune.utils import get_logger
import torch
import torch.nn.functional as F
from args import add_args
from model import get_device, get_model
from regdataset import RegDataset
from sidwav import write_samples, sidq


class Predictor:
    def __init__(self, args, model, device, prompt):
        self.args = args
        self.model = model
        self.prompt = prompt.to(device)

    def predict(self):
        for _ in range(self.args.sequence_length):
            with torch.no_grad():
                outputs = self.model(self.prompt).view(
                    -1, self.model.tok_embeddings.num_embeddings
                )
            self.prompt = torch.roll(self.prompt, -1)
            state = torch.argmax(outputs, dim=1)[-1]
            self.prompt[0][-1] = state
        return self.prompt.detach().squeeze(0)


def state_df(states, dataset):
    return pd.DataFrame(states, columns=["n"]).merge(dataset.tokens, on="n", how="left")


def generate(logger, dataset, model, device, prompt, prompt_from, args):
    stats = []
    cycles = 0
    prompt_cycles = 0
    from_offset = 0
    predictor = torch.compile(Predictor)(args, model, device, prompt)

    if args.include_prompt:
        states = prompt.squeeze(0).tolist()
        prompt_df = state_df(states, dataset)
        prompt_cycles = prompt_df["diff"].sum()
        logger.info(
            "prompt lasts %u cycles %.2f seconds", prompt_cycles, prompt_cycles * sidq()
        )

    while cycles < args.output_cycles:
        new_states = predictor.predict()
        prompt_compare = prompt_from[from_offset:][: args.sequence_length]
        states.extend(new_states.tolist())
        df = state_df(states, dataset)
        cycles = df["diff"].sum() - prompt_cycles
        write_samples(df, args.wav)
        progress = cycles / float(args.output_cycles) * 100
        acc = "unknown"
        if prompt_compare.shape == new_states.shape:
            acc = (new_states == prompt_compare).float().mean()
            acc = "%2.2f" % acc
        logger.info(
            "generated %9.u cycles %6.2f seconds accuracy %s %6.2f%%",
            cycles,
            cycles * sidq(),
            acc,
            progress,
        )
        from_offset += args.sequence_length

    clock = df["diff"].cumsum()
    df = df[clock <= args.output_cycles]
    cycles = df["diff"].sum()
    logger.info(
        "finalized %9.u cycles %6.2f seconds %6.2f%%", cycles, cycles * sidq(), 100
    )
    write_samples(df, args.wav)
    if args.csv:
        df.to_csv(args.csv)


def main():
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()
    logger = get_logger("INFO")
    dataset = RegDataset(args, logger=logger)
    device = get_device()
    model = get_model(dataset, args).to(device)
    best_model = torch.load(args.model_state, weights_only=True, map_location=device)[0]
    model.load_state_dict(best_model)
    model.eval()

    if args.start_n is None:
        random.seed(time.time())
        start = random.randint(0, dataset.n_words)
    else:
        start = args.start_n
    logger.info("starting at %u / %u", start, dataset.n_words)
    prompt = dataset.dfs_n[start:][: args.sequence_length].unsqueeze(0).to(device)
    prompt_from = dataset.dfs_n[start:].to(device)
    generate(logger, dataset, model, device, prompt, prompt_from, args)


if __name__ == "__main__":
    main()
