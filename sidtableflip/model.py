import math
import torch
from torch import nn
import torch.nn.functional as F

# https://github.com/pytorch/examples/blob/main/word_language_model/model.py


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, max_len, dropout):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[: x.size(0), :]
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
        self.pos_encoder = PositionalEncoding(embed_dim, dataset.n_vocab, dropout)
        self.input_emb = nn.Embedding(dataset.n_vocab, embed_dim)
        self.src_mask = self.generate_square_subsequent_mask(
            sequence_length, device=device
        )
        self.ninp_sqrt = math.sqrt(embed_dim)
        self.init_weights()

    def init_weights(self, initrange=0.1):
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src):
        src = self.input_emb(src) * self.ninp_sqrt
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
        ),
        options={"triton.cudagraphs": True},
        fullgraph=True,
    ).to(device)
    return model


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
