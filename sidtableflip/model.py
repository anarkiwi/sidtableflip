import math
import torch
from torch import nn
import torch.nn.functional as F

# from fairseq.modules.positional_encoding import PositionalEncoding
# https://github.com/pytorch/examples/blob/main/word_language_model/model.py


def model_div_term(d_model):
    return torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * -(math.log(10000.0) / d_model)
    )


class PositionalEncoding(nn.Module):
    """Positional encoding.

    Args:
        d_model: Embedding dimension.
        dropout_rate: Dropout rate.
        max_len: Maximum input length.
    """

    def __init__(self, d_model, max_len, device, dropout_rate):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = nn.Dropout(p=dropout_rate)
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = model_div_term(self.d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=device, dtype=torch.float32)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input tensor B X T X C
        Returns:
            torch.Tensor: Encoded tensor B X T X C
        """
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Transformer):
    def __init__(
        self,
        dataset,
        device,
        embed_dim=256,
        num_layers=4,
        num_heads=2,
        sequence_length=128,
        max_len=128 * 10,
        dropout=0.2,
    ):
        decoder = nn.Linear(embed_dim, dataset.n_vocab)
        super().__init__(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=sequence_length,
            num_encoder_layers=num_layers,
            custom_decoder=decoder,
            device=device,
            batch_first=True,
        )
        self.pos_encoder = PositionalEncoding(embed_dim, max_len, device, dropout)
        self.input_emb = nn.Embedding(dataset.n_vocab, embed_dim)
        self.src_mask = self.generate_square_subsequent_mask(
            sequence_length, device=device
        )
        self.init_weights()

    def init_weights(self, initrange=0.1):
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src):
        src = self.input_emb(src)
        src = self.pos_encoder(src)
        output = self.encoder(src, mask=self.src_mask, is_causal=True)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)


def get_model(dataset, device, args):
    torch.set_float32_matmul_precision("high")
    model = torch.compile(
        TransformerModel(
            dataset,
            device,
            sequence_length=args.sequence_length,
            num_heads=args.heads,
            num_layers=args.layers,
            embed_dim=args.embed,
            max_len=args.max_sequence_length,
        ),
        options={"triton.cudagraphs": True},
        fullgraph=True,
    ).to(device)
    return model


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
