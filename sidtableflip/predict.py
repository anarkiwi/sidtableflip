#!/usr/bin/env python3

import argparse
import random
import time
import pandas as pd
from torchtune.utils import get_logger
from torchtune.generation._generation import generate
import torch
import torch.nn.functional as F
from args import add_args
from model import get_device, get_model, Model
from regdataset import RegDataset
from sidwav import write_samples, sidq


class Predictor:
    def __init__(self, args, model, device, prompt):
        self.args = args
        self.model = model
        self.prompt = prompt.clone().to(device)

    @torch.inference_mode()
    def predict(self):
        for _ in range(self.args.sequence_length):
            outputs, _logits = generate(
                self.model.model, self.prompt, max_generated_tokens=1
            )
            self.prompt = torch.roll(self.prompt, -1)
            self.prompt[0][-1] = outputs[0][-1]
        return self.prompt.detach().squeeze(0)


def state_df(states, dataset):
    return pd.DataFrame(states, columns=["n"]).merge(dataset.tokens, on="n", how="left")


def generate_sequence(logger, dataset, model, device, prompt, prompt_from, args):
    states = []
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
        if args.csv:
            df.to_csv(args.csv)
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
    logger.info("loading %s", args.model_state)
    model = torch.compile(
        Model.load_from_checkpoint(args.model_state), mode="max-autotune"
    )
    model.eval()

    if args.start_n is None:
        random.seed(time.time())
        start = random.randint(0, dataset.n_words)
    else:
        start = args.start_n
    logger.info("starting at %u / %u", start, dataset.n_words)
    prompt = dataset.dfs_n[start:][: args.sequence_length].unsqueeze(0).to(device)
    prompt_from = dataset.dfs_n[start + 1 :].to(device)
    generate_sequence(logger, dataset, model, device, prompt, prompt_from, args)


if __name__ == "__main__":
    main()
