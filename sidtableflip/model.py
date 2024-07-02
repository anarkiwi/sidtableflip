import math
import torch
from torch import nn


def generate_square_subsequent_mask(sz):
    """
    Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


class Model(nn.Module):
    def __init__(self, dataset, embedding_dim=128, lstm_size=256, num_layers=2):
        super().__init__()
        self.lstm_size = lstm_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(
            num_embeddings=dataset.n_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.lstm_size, dataset.n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return (
            torch.zeros(self.num_layers, sequence_length, self.lstm_size),
            torch.zeros(self.num_layers, sequence_length, self.lstm_size),
        )


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self, dataset, embed_dim=128, num_layers=2, num_heads=2, sequence_length=128
    ):
        super().__init__()
        vocab_size = dataset.n_vocab
        self.pos_encoder = PositionalEncoding(
            max_len=sequence_length, d_model=embed_dim
        )
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=num_layers,
        )
        self.linear = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        emb = self.emb(x)

        # Generate input sequence mask with shape (SEQUENCE_LENGTH, SEQUENCE_LENGTH)
        input_mask = generate_square_subsequent_mask(x.size(1)).to(x.device)

        x = self.pos_encoder(emb)
        x = self.decoder(x, memory=x, tgt_mask=input_mask, memory_mask=input_mask)
        x = self.dropout(x)
        out = self.linear(x)
        return out
