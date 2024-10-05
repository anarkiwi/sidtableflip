import torch
from torchtune.models.llama2._component_builders import llama2


def get_model(dataset, device, args):
    torch.set_float32_matmul_precision("high")
    model = llama2(
        vocab_size=dataset.n_vocab,
        num_layers=args.layers,
        num_heads=args.heads,
        num_kv_heads=args.heads,
        embed_dim=args.embed,
        max_seq_len=args.max_sequence_length,
        attn_dropout=0.0,
        norm_eps=1e-5,
    )
    model = torch.compile(
        model,
        options={"triton.cudagraphs": True},
        fullgraph=True,
    ).to(device)
    return model


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
