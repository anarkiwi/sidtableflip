import torch
from torchtune.models.llama2._component_builders import llama2
from torchtune.models.phi3._component_builders import phi3


def get_phi3(dataset, args):
    intermediate = args.intermediate if args.intermediate else args.embed

    return phi3(
        vocab_size=dataset.n_vocab,
        num_layers=args.layers,
        num_heads=args.heads,
        num_kv_heads=args.kv_heads,
        embed_dim=args.embed,
        intermediate_dim=intermediate,
        max_seq_len=args.max_sequence_length,
        attn_dropout=args.attn_dropout,
        norm_eps=args.norm_eps,
    )


def get_llama2(dataset, args):
    return llama2(
        vocab_size=dataset.n_vocab,
        num_layers=args.layers,
        num_heads=args.heads,
        num_kv_heads=args.kv_heads,
        embed_dim=args.embed,
        max_seq_len=args.max_sequence_length,
        attn_dropout=args.attn_dropout,
        norm_eps=args.norm_eps,
    )


MODEL_GETTERS = {
    "llama2": get_llama2,
    "phi3": get_phi3,
}


def get_model(dataset, device, args):
    torch.set_float32_matmul_precision("high")
    model = MODEL_GETTERS[args.model](dataset, args)
    model = torch.compile(
        model,
        options={"triton.cudagraphs": True},
        fullgraph=True,
    ).to(device)
    return model


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
