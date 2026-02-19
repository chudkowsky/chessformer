"""Shared utilities for model loading, device detection, preprocessing, and loss.

Consolidates duplicated logic from train_model.py, inference_test.py,
play_gui.py, and selfplay_loop.py into a single module.
"""

from __future__ import annotations

import sys

import chess
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import chessformer
from chess_loader import PIECE_TO_INDEX, compute_features
from chess_moves_to_input_data import get_board_str
from chessformer import ChessTransformer, ChessTransformerV2

# Allow unpickling old models saved under the name 'transformer'
sys.modules["transformer"] = chessformer


def detect_device(preference: str = "auto") -> torch.device:
    """Resolve device string to a torch.device.

    Args:
        preference: "auto", "cuda", "mps", or "cpu".

    Returns:
        Resolved torch.device.
    """
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    path: str, device: torch.device
) -> tuple[nn.Module, str, dict | None]:
    """Load a V1 or V2 model checkpoint.

    Handles:
    - V2 checkpoints: {'version': 'v2', 'state_dict': ..., 'config': ...}
    - V1 state dicts (including RL adapter prefix stripping)
    - V1 pickled model objects

    Args:
        path:   Path to .pth checkpoint.
        device: Target device.

    Returns:
        (model, version, config) where version is 'v1' or 'v2',
        and config is the V2 config dict (None for V1).
    """
    obj = torch.load(path, weights_only=False, map_location="cpu")

    if isinstance(obj, dict) and obj.get("version") == "v2":
        cfg = obj["config"]
        model = ChessTransformerV2(**cfg, dropout=0.0)
        model.load_state_dict(obj["state_dict"])
        return model.to(device), "v2", cfg

    if isinstance(obj, dict):
        sd = obj
        # RL checkpoint: keys prefixed with "adapter.transformer."
        prefix = "adapter.transformer."
        if any(k.startswith(prefix) for k in sd):
            sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        # Infer V1 architecture from state dict
        if "embedding.weight" not in sd:
            raise ValueError(f"Unrecognised checkpoint format in {path}")
        inp_dict = sd["embedding.weight"].shape[0]
        out_dict = sd["linear_output.weight"].shape[0]
        d_model = sd["embedding.weight"].shape[1]
        d_hid = sd["transformer_encoder.layers.0.linear1.weight"].shape[0]
        nlayers = sum(1 for k in sd if k.endswith(".self_attn.in_proj_weight"))
        nhead = max(1, d_model // 64)
        model = ChessTransformer(
            inp_dict, out_dict, d_model, nhead, d_hid, nlayers, dropout=0.0
        )
        model.load_state_dict(sd)
        return model.to(device), "v1", None

    # Pickled model object (legacy V1)
    if isinstance(obj, nn.Module):
        return obj.to(device), "v1", None

    raise ValueError(f"Cannot load model from {path}")


def preprocess_board(
    board: chess.Board, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Convert chess.Board to V2 model inputs.

    Board is always oriented for the current player (flipped if black to move).

    Returns:
        (board_tensor[1,64], features_tensor[1,14])
    """
    board_str = get_board_str(board, white_side=board.turn)
    board_pieces = [PIECE_TO_INDEX[p] for p in board_str]
    features = compute_features(board_str)
    return (
        torch.tensor([board_pieces], dtype=torch.long, device=device),
        torch.tensor([features], dtype=torch.float32, device=device),
    )


def preprocess_board_v1(
    board: chess.Board, device: torch.device
) -> Tensor:
    """Convert chess.Board to V1 model input (board only, no features).

    Returns:
        board_tensor[1,64]
    """
    board_str = get_board_str(board, white_side=board.turn)
    board_pieces = [PIECE_TO_INDEX[p] for p in board_str]
    return torch.tensor([board_pieces], dtype=torch.long, device=device)


def compute_loss_v2(
    model: ChessTransformerV2,
    boards: Tensor,
    features: Tensor,
    from_sq: Tensor,
    to_sq: Tensor,
    wdl_target: Tensor,
) -> Tensor:
    """V2 multi-task loss: policy cross-entropy + 0.5 * WDL cross-entropy."""
    policy_logits, _promo, wdl_pred, _ply = model(boards, features)
    B = boards.shape[0]
    policy_loss = F.cross_entropy(policy_logits.reshape(B, -1), from_sq * 64 + to_sq)
    wdl_loss = -(wdl_target * torch.log(wdl_pred + 1e-8)).sum(dim=-1).mean()
    return policy_loss + 0.5 * wdl_loss
