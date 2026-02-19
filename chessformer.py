import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from attention import ShawRelativePositionBias, SmolgenBias, ChessTransformerBlock

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Creating positional encodings
        # Changed: shape [1, max_len, d_model] for batch_first format (was [max_len, 1, d_model])
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        # Changed: index dim 1 instead of dim 0 for batch_first format
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class ChessTransformer(nn.Module):
    def __init__(self, inp_dict: int, out_dict: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Chess-former'

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer Encoder Layers
        # Changed: batch_first=True enables Flash Attention and removes need for permute()
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # Embedding layers for input
        self.embedding = nn.Embedding(inp_dict, d_model)
        self.x_embedding = nn.Embedding(8, d_model)
        self.y_embedding = nn.Embedding(8, d_model)
        self.d_model = d_model

        # Output linear layer
        self.linear_output = nn.Linear(d_model, out_dict)

        # Changed: pre-compute x/y coordinate tensors (shape [1, 64]) once instead of every forward pass
        # x_coords: [0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7, ...] (column index for each square)
        # y_coords: [0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, ...] (row index for each square)
        self.register_buffer('x_coords', torch.arange(8).unsqueeze(0).repeat(1, 8))
        self.register_buffer('y_coords', torch.arange(8).repeat_interleave(8).unsqueeze(0))

        # Initialization of weights
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.x_embedding.weight.data.uniform_(-initrange, initrange)
        self.y_embedding.weight.data.uniform_(-initrange, initrange)
        self.linear_output.bias.data.zero_()
        self.linear_output.weight.data.uniform_(-initrange, initrange)

    def forward(self, board: Tensor, src_mask: Tensor = None) -> Tensor:
        board_emb = self.embedding(board)

        # Changed: use pre-computed coordinate buffers, expand() creates a view without copying memory
        x_emb = self.x_embedding(self.x_coords.expand(board.shape[0], -1))
        y_emb = self.y_embedding(self.y_coords.expand(board.shape[0], -1))

        # Combining embeddings
        combined_emb = board_emb + x_emb + y_emb

        # Scaling and passing through transformer
        combined_emb = combined_emb * math.sqrt(self.d_model)
        # Changed: removed .permute() calls - no longer needed with batch_first=True
        output = self.transformer_encoder(combined_emb)

        # Applying linear layer to the output
        output = self.linear_output(output)
        return output


# --- V2: Enhanced architecture with Shaw RPE, Smolgen, WDLP ---

NUM_AUX_FEATURES = 14  # material(1) + check(1) + castling(4) + en_passant(8)


class SourceDestPolicyHead(nn.Module):
    """Bilinear source-destination policy head.

    Computes attention between source-square queries and destination-square
    keys, producing a [B, 64, 64] logit matrix where entry (i, j) scores
    moving from square i to square j.
    """

    def __init__(self, d_model: int, d_policy: int = 128) -> None:
        super().__init__()
        self.from_proj = nn.Linear(d_model, d_policy)
        self.to_proj = nn.Linear(d_model, d_policy)
        self.scale = d_policy ** -0.5
        self.promo_head = nn.Linear(d_model, 4)  # Q, R, B, N

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: [B, 64, d_model]
        Returns:
            policy_logits: [B, 64, 64] — source-destination move scores
            promo_logits:  [B, 64, 4]  — promotion piece scores per source square
        """
        q = self.from_proj(x)
        k = self.to_proj(x)
        policy = torch.bmm(q, k.transpose(1, 2)) * self.scale
        promo = self.promo_head(x)
        return policy, promo


class WDLPValueHead(nn.Module):
    """Win/Draw/Loss + Ply prediction value head.

    Mean-pools 64 latent tokens, then predicts:
    - WDL: 3-class probability distribution (softmax)
    - Ply: expected game length (non-negative via softplus, auxiliary signal)
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.wdl_linear = nn.Linear(d_model, 3)
        self.ply_linear = nn.Linear(d_model, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: [B, 64, d_model]
        Returns:
            wdl: [B, 3] — win/draw/loss probabilities (sum to 1)
            ply: [B, 1] — predicted game length (non-negative)
        """
        pooled = self.norm(x.mean(dim=1))
        wdl = torch.softmax(self.wdl_linear(pooled), dim=-1)
        ply = torch.nn.functional.softplus(self.ply_linear(pooled))
        return wdl, ply


class ChessTransformerV2(nn.Module):
    """Enhanced chess transformer with Shaw RPE, smolgen, and dual heads.

    Improvements over V1:
    - Shaw relative position encoding (topology-aware, not sinusoidal)
    - Smolgen dynamic attention biases (content-dependent)
    - Pre-norm transformer blocks (more stable training)
    - Source-destination policy head (structured 64x64 logit matrix)
    - WDLP value head (win/draw/loss + ply prediction)
    - Auxiliary input features (material, check, castling, en passant)
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        d_hid: int = 1024,
        nlayers: int = 12,
        d_policy: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_type = "Chess-former-v2"
        self.d_model = d_model

        # Input embeddings (same as V1)
        self.embedding = nn.Embedding(13, d_model)
        self.x_embedding = nn.Embedding(8, d_model)
        self.y_embedding = nn.Embedding(8, d_model)

        # Auxiliary feature fusion: project (d_model + 14) → d_model
        self.feature_proj = nn.Linear(d_model + NUM_AUX_FEATURES, d_model)

        # Pre-computed coordinate buffers (same as V1)
        self.register_buffer(
            "x_coords", torch.arange(8).unsqueeze(0).repeat(1, 8)
        )
        self.register_buffer(
            "y_coords", torch.arange(8).repeat_interleave(8).unsqueeze(0)
        )

        # Attention biases
        self.shaw_rpe = ShawRelativePositionBias(nhead)
        self.smolgen = SmolgenBias(d_model, nhead)

        # Transformer backbone (pre-norm blocks)
        self.layers = nn.ModuleList(
            [ChessTransformerBlock(d_model, nhead, d_hid) for _ in range(nlayers)]
        )
        self.final_norm = nn.LayerNorm(d_model)

        # Dual output heads
        self.policy_head = SourceDestPolicyHead(d_model, d_policy)
        self.value_head = WDLPValueHead(d_model)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.x_embedding.weight.data.uniform_(-initrange, initrange)
        self.y_embedding.weight.data.uniform_(-initrange, initrange)

    def forward(
        self,
        board: Tensor,
        features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            board:    [B, 64] int tensor — piece indices (0-12)
            features: [B, 14] float tensor — auxiliary features, or None

        Returns:
            policy_logits: [B, 64, 64] — source-destination move scores
            promo_logits:  [B, 64, 4]  — promotion piece scores
            wdl:           [B, 3]      — win/draw/loss probabilities
            ply:           [B, 1]      — predicted game length
        """
        B = board.shape[0]

        # Embeddings (same as V1)
        board_emb = self.embedding(board)
        x_emb = self.x_embedding(self.x_coords.expand(B, -1))
        y_emb = self.y_embedding(self.y_coords.expand(B, -1))
        combined = board_emb + x_emb + y_emb

        # Fuse auxiliary features if provided
        if features is not None:
            # Broadcast [B, 14] → [B, 64, 14] and concatenate
            feat_expanded = features.unsqueeze(1).expand(-1, 64, -1)
            combined = self.feature_proj(torch.cat([combined, feat_expanded], dim=-1))

        combined = combined * math.sqrt(self.d_model)
        combined = self.dropout(combined)

        # Compute attention biases (once, shared across all layers)
        shaw_bias = self.shaw_rpe()                    # [nhead, 64, 64]
        smolgen_bias = self.smolgen(combined)           # [B, nhead, 64, 64]
        attn_bias = smolgen_bias + shaw_bias.unsqueeze(0)

        # Transformer backbone
        x = combined
        for layer in self.layers:
            x = layer(x, attn_bias=attn_bias)
        x = self.final_norm(x)

        # Dual heads
        policy_logits, promo_logits = self.policy_head(x)
        wdl, ply = self.value_head(x)

        return policy_logits, promo_logits, wdl, ply
