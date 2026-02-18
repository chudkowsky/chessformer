import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Creating positional encodings
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class ChessTransformer(nn.Module):
    def __init__(self, inp_dict: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Chess-former'

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer Encoder Layers
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # Embedding layers for input
        self.embedding = nn.Embedding(inp_dict, d_model)
        self.x_embedding = nn.Embedding(8, d_model)
        self.y_embedding = nn.Embedding(8, d_model)
        self.d_model = d_model

        # Output heads
        # from_head / to_head: per-square scalar logit  → (B, 64)
        self.from_head = nn.Linear(d_model, 1)
        self.to_head   = nn.Linear(d_model, 1)
        # promo_head: mean-pooled board representation → 4 logits (q=0, r=1, b=2, n=3)
        self.promo_head = nn.Linear(d_model, 4)

        # Initialization of weights
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.x_embedding.weight.data.uniform_(-initrange, initrange)
        self.y_embedding.weight.data.uniform_(-initrange, initrange)
        for head in (self.from_head, self.to_head, self.promo_head):
            head.bias.data.zero_()
            head.weight.data.uniform_(-initrange, initrange)

    def forward(self, board: Tensor, src_mask: Tensor = None):
        """
        Args:
            board: (B, 64) long tensor of piece indices
        Returns:
            from_logits:  (B, 64)  — logit for each square being the source
            to_logits:    (B, 64)  — logit for each square being the destination
            promo_logits: (B, 4)   — logit for each promotion piece (q/r/b/n)
        """
        board_emb = self.embedding(board)

        # Generating input for positional embeddings
        batch_size = board.shape[0]
        x_inp = torch.arange(8).unsqueeze(0).repeat(batch_size, 8).to(board.device)
        y_inp = torch.arange(8).repeat_interleave(8).unsqueeze(0).repeat(batch_size, 1).to(board.device)

        x_emb = self.x_embedding(x_inp)
        y_emb = self.y_embedding(y_inp)

        # Combining embeddings
        combined_emb = board_emb + x_emb + y_emb

        # Scaling and passing through transformer  — (B, 64, d_model)
        combined_emb = combined_emb * math.sqrt(self.d_model)
        enc = self.transformer_encoder(combined_emb.permute(1, 0, 2)).permute(1, 0, 2)

        # Per-square heads  (B, 64, 1) → squeeze → (B, 64)
        from_logits = self.from_head(enc).squeeze(-1)
        to_logits   = self.to_head(enc).squeeze(-1)

        # Global promo head: mean-pool across the 64 squares → (B, d_model) → (B, 4)
        promo_logits = self.promo_head(enc.mean(dim=1))

        return from_logits, to_logits, promo_logits
