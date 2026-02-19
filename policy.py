"""
policy.py — legal-move policy distribution for ChessFormer.

Converts raw model logits into a proper probability distribution π(m)
over legal chess moves, including promotion handling.

Coordinate systems
------------------
python-chess:   square = rank*8 + file   (rank 0 = rank-1, file 0 = a)
model (ours):   square = file + (7-rank)*8  (row 0 = rank-8, always current-player POV)

When it is Black's turn the board is flipped vertically (rank r → 7-r) so
the model always sees its own pieces at the bottom two rows.
"""

import chess
import torch
import torch.nn.functional as F
from typing import List, Tuple

# Promotion piece → promo_logits index
PROMO_PIECE_TO_IDX = {
    chess.QUEEN:  0,
    chess.ROOK:   1,
    chess.BISHOP: 2,
    chess.KNIGHT: 3,
}


def chess_sq_to_model_idx(sq: int, white_turn: bool) -> int:
    """
    Convert a python-chess square index to the model's flat board index.

    Args:
        sq:         python-chess square (0=a1 … 63=h8)
        white_turn: True if it is White's turn (no flip needed)

    Returns:
        Model board index in [0, 63]
    """
    file = chess.square_file(sq)   # 0-7  (a=0)
    rank = chess.square_rank(sq)   # 0-7  (rank-1 = 0)
    if not white_turn:
        rank = 7 - rank            # flip for Black's perspective
    return file + (7 - rank) * 8


def legal_move_policy(
    board: chess.Board,
    from_logits: torch.Tensor,
    to_logits: torch.Tensor,
    promo_logits: torch.Tensor,
) -> Tuple[List[chess.Move], torch.Tensor, torch.Tensor]:
    """
    Compute a softmax policy distribution over all legal moves.

    Scoring
    -------
    score(m) = from_logits[f] + to_logits[t]
               + promo_logits[p]  (only when m is a promotion)

    where f and t are model-perspective square indices for the move's
    source and destination squares.

    Args:
        board:        python-chess Board (any position, any side to move)
        from_logits:  (64,) float tensor — raw source-square logits
        to_logits:    (64,) float tensor — raw destination-square logits
        promo_logits: (4,)  float tensor — raw promotion-piece logits
                      order: queen=0, rook=1, bishop=2, knight=3

    Returns:
        moves:     list of chess.Move in the same order as probs/log_probs
        probs:     (N,) tensor — π(m), softmax over legal moves
        log_probs: (N,) tensor — log π(m), for use in policy-gradient / RL losses
    """
    white_turn = (board.turn == chess.WHITE)
    legal = list(board.legal_moves)

    if not legal:
        empty = torch.zeros(0, device=from_logits.device)
        return [], empty, empty

    scores = []
    for move in legal:
        f = chess_sq_to_model_idx(move.from_square, white_turn)
        t = chess_sq_to_model_idx(move.to_square,   white_turn)
        score = from_logits[f] + to_logits[t]
        if move.promotion is not None:
            p = PROMO_PIECE_TO_IDX[move.promotion]
            score = score + promo_logits[p]
        scores.append(score)

    scores     = torch.stack(scores)          # (N,)
    log_probs  = F.log_softmax(scores, dim=0) # (N,)
    probs      = log_probs.exp()              # (N,)

    return legal, probs, log_probs


def sample_move(
    board: chess.Board,
    from_logits: torch.Tensor,
    to_logits: torch.Tensor,
    promo_logits: torch.Tensor,
) -> Tuple[chess.Move, torch.Tensor]:
    """
    Sample a move from the policy distribution and return its log-probability.

    Args:
        board, from_logits, to_logits, promo_logits: same as legal_move_policy

    Returns:
        move:     the sampled chess.Move
        log_prob: scalar tensor — log π(move), needed for REINFORCE / PPO
    """
    moves, _, log_probs = legal_move_policy(board, from_logits, to_logits, promo_logits)
    idx = torch.distributions.Categorical(logits=log_probs).sample()
    return moves[idx.item()], log_probs[idx]


def rate_move(
    board: chess.Board,
    move: chess.Move,
    from_logits: torch.Tensor,
    to_logits: torch.Tensor,
    promo_logits: torch.Tensor,
) -> float:
    """
    Score how much the model agrees with a move that was played.

    Returns q ∈ [0, 1] = π(move) / π(best legal move).
      1.0 → model's top choice was played (excellent)
      ~0  → model strongly disagreed (poor)
    """
    moves, probs, _ = legal_move_policy(board, from_logits, to_logits, promo_logits)
    best_prob = probs.max().item()
    if best_prob <= 0:
        return 0.0
    for m, p in zip(moves, probs):
        if m == move:
            return p.item() / best_prob
    return 0.0  # move wasn't legal (shouldn't happen)


def greedy_move(
    board: chess.Board,
    from_logits: torch.Tensor,
    to_logits: torch.Tensor,
    promo_logits: torch.Tensor,
) -> chess.Move:
    """
    Return the highest-scoring legal move (argmax of the policy).
    """
    moves, probs, _ = legal_move_policy(board, from_logits, to_logits, promo_logits)
    return moves[probs.argmax().item()]


# --- V2: Source-destination 64x64 policy ---


def legal_move_policy_v2(
    board: chess.Board,
    policy_logits: torch.Tensor,
    promo_logits: torch.Tensor,
) -> Tuple[List[chess.Move], torch.Tensor, torch.Tensor]:
    """Compute softmax policy from V2 model's 64x64 source-destination logits.

    score(m) = policy_logits[from_sq, to_sq]
               + promo_logits[from_sq, promo_idx]  (only for promotions)

    Args:
        board:         python-chess Board
        policy_logits: (64, 64) source-destination logit matrix
        promo_logits:  (64, 4) per-source promotion piece logits

    Returns:
        moves, probs, log_probs — same contract as legal_move_policy
    """
    white_turn = (board.turn == chess.WHITE)
    legal = list(board.legal_moves)

    if not legal:
        empty = torch.zeros(0, device=policy_logits.device)
        return [], empty, empty

    scores = []
    for move in legal:
        f = chess_sq_to_model_idx(move.from_square, white_turn)
        t = chess_sq_to_model_idx(move.to_square, white_turn)
        score = policy_logits[f, t]
        if move.promotion is not None:
            p = PROMO_PIECE_TO_IDX[move.promotion]
            score = score + promo_logits[f, p]
        scores.append(score)

    scores = torch.stack(scores)
    log_probs = F.log_softmax(scores, dim=0)
    probs = log_probs.exp()

    return legal, probs, log_probs


def greedy_move_v2(
    board: chess.Board,
    policy_logits: torch.Tensor,
    promo_logits: torch.Tensor,
) -> chess.Move:
    """Return highest-scoring legal move from V2 policy."""
    moves, probs, _ = legal_move_policy_v2(board, policy_logits, promo_logits)
    return moves[probs.argmax().item()]
