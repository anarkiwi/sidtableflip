#!/usr/bin/env python3

import argparse
import glob
import os
import random
import pandas as pd
from torchtune.utils import get_logger
from torchtune.generation import generate as generate
import torch
import torchmetrics
from args import add_args
from model import get_device, Model
from regdataset import RegDataset
from sidwav import write_samples, sidq


class Predictor:
    def __init__(self, args, model, device, prompt):
        self.args = args
        self.model = model
        self.device = device
        self.prompt = prompt.clone().to(device)
        self.rng = torch.Generator(device=self.device)
        self.rng.seed()

    def next_prompt(self, output, n):
        self.prompt = torch.roll(self.prompt, -n)
        self.prompt[0][-n:] = output[0][-n:]

    def predict_n_tt(self, n, temperature, top_k):
        output, _logits = generate(
            self.model.model,
            self.prompt.clone(),
            max_generated_tokens=n,
            pad_id=-999,
            top_k=top_k,
            temperature=temperature,
            rng=self.rng,
        )
        self.next_prompt(output, n)

    def predict_nano(self, temperature, top_k):
        logits = self.model.model(self.prompt.clone())
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_state = torch.multinomial(probs, num_samples=1)
        self.next_prompt(next_state, 1)

    @torch.inference_mode()
    def predict(self, temperature=1.0, top_k=None):
        for _ in range(self.args.sequence_length):
            self.predict_nano(temperature, top_k)
            # self.predict_n_tt(1, temperature, top_k)
        # self.predict_n_tt(self.args.sequence_length, temperature, top_k)
        return self.prompt.clone().detach().squeeze(0)


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
        prompt_compare = prompt_from[from_offset:][: args.sequence_length]
        new_states = predictor.predict()
        states.extend(new_states.tolist())
        df = state_df(states, dataset)
        cycles = df["diff"].sum() - prompt_cycles
        if args.csv:
            df.to_csv(args.csv)
        write_samples(df, args.wav, dataset.reg_widths)
        progress = cycles / float(args.output_cycles) * 100
        acc = "unknown"
        if prompt_compare.shape == new_states.shape:
            acc = torchmetrics.functional.classification.multiclass_accuracy(
                new_states,
                prompt_compare,
                dataset.n_vocab,
                validate_args=False,
            )
            acc = "%3.3f" % acc
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
    write_samples(df, args.wav, dataset.reg_widths)
    if args.csv:
        df.to_csv(args.csv)


def main():
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()
    logger = get_logger("INFO")
    dataset = RegDataset(args, logger=logger)
    dataset.load()
    device = get_device()
    ckpt = args.model_state
    if not ckpt:
        ckpts = sorted(
            [
                (os.path.getmtime(p), p)
                for p in glob.glob(f"{args.tb_logs}/**/*ckpt", recursive=True)
            ]
        )
        ckpt = ckpts[-1][1]
    logger.info("loading %s", ckpt)
    model = Model.load_from_checkpoint(ckpt)  # pylint: disable=no-value-for-parameter
    model = torch.compile(model, mode="max-autotune")
    model.eval()
    # TODO: kv cache makes it slower...
    # with device:
    #    model.model.setup_caches(
    #        1,
    #        torch.float32,
    #    )

    if args.start_n is None:
        start = random.randint(0, dataset.n_words)
    else:
        start = args.start_n
    logger.info("starting at %u / %u", start, dataset.n_words)
    prompt = dataset.dfs_n[start:][: args.sequence_length].unsqueeze(0).to(device)
    prompt_from = dataset.dfs_n[start + args.sequence_length :].to(device)
    generate_sequence(logger, dataset, model, device, prompt, prompt_from, args)


if __name__ == "__main__":
    main()
