"""Trajectory data for Phase 2 diffusion training.

Extracts (current_position, future_position) pairs from PGN game
trajectories. The diffusion model learns to predict future board
latents given the current position as conditioning.

Usage:
    from trajectory_loader import get_trajectory_dataloader

    train_loader, test_loader = get_trajectory_dataloader(
        pgn_path="full_datasets/filtered_1500_elo.pgn",
        horizon=4,
        batch_size=64,
    )

Each batch contains:
    current_board:    [B, 64]  int   — piece indices for current position
    current_features: [B, 14]  float — auxiliary features for current
    future_board:     [B, 64]  int   — piece indices for position H moves ahead
    future_features:  [B, 14]  float — auxiliary features for future
"""

import chess
import chess.pgn
import torch
from torch.utils.data import Dataset, DataLoader

from chess_loader import PIECE_TO_INDEX, compute_features
from chess_moves_to_input_data import get_board_str


def extract_trajectories(
    pgn_path: str,
    horizon: int = 4,
    min_elo: int = 1500,
    max_trajectories: int | None = None,
) -> list[tuple[str, str]]:
    """Extract (current, future) board string pairs from a PGN file.

    For each position P in a game, pairs it with position P+horizon
    (the board state `horizon` half-moves later in the same game).

    Both board strings are 64-char representations oriented from
    the current player's perspective (flipped for black via get_board_str).

    Args:
        pgn_path: Path to PGN file.
        horizon: Number of half-moves to look ahead (default 4 = 2 full moves).
        min_elo: Minimum Elo for both players (Lichess games).
        max_trajectories: Stop after collecting this many pairs.

    Returns:
        List of (current_board_str, future_board_str) pairs.
    """
    trajectories: list[tuple[str, str]] = []
    games_used = 0

    with open(pgn_path) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            # Filter by Elo (same criteria as pgn_to_training_data.py)
            event = game.headers.get("Event", "")
            is_selfplay = event.startswith("SF-SelfPlay")

            if not is_selfplay:
                try:
                    white_elo = int(game.headers.get("WhiteElo", "0"))
                    black_elo = int(game.headers.get("BlackElo", "0"))
                except ValueError:
                    continue
                if white_elo < min_elo or black_elo < min_elo:
                    continue
                if "Blitz" in event or "Bullet" in event:
                    continue

            # Collect all board states in the game
            board = game.board()
            states = [get_board_str(board, white_side=board.turn)]
            for move in game.mainline_moves():
                board.push(move)
                states.append(get_board_str(board, white_side=board.turn))

            # Create trajectory pairs: (state[i], state[i + horizon])
            for i in range(len(states) - horizon):
                trajectories.append((states[i], states[i + horizon]))

            games_used += 1
            if games_used % 1000 == 0:
                print(
                    f"Games: {games_used}, trajectories: {len(trajectories)}"
                )

            if max_trajectories and len(trajectories) >= max_trajectories:
                trajectories = trajectories[:max_trajectories]
                break

    print(
        f"Done! Games: {games_used}, trajectories: {len(trajectories)}, "
        f"horizon: {horizon}"
    )
    return trajectories


class TrajectoryDataset(Dataset):
    """Board-level trajectory dataset for diffusion training.

    Each example provides current and future board states (as integer
    tensors + features). During Phase 2 training, both are encoded by
    the frozen V2 backbone to obtain latent representations.
    """

    def __init__(self, trajectory_pairs: list[tuple[str, str]]) -> None:
        current_boards = []
        current_features = []
        future_boards = []
        future_features = []

        for current_str, future_str in trajectory_pairs:
            current_boards.append([PIECE_TO_INDEX[p] for p in current_str])
            current_features.append(compute_features(current_str))
            future_boards.append([PIECE_TO_INDEX[p] for p in future_str])
            future_features.append(compute_features(future_str))

        self.current_boards = torch.tensor(current_boards, dtype=torch.long)
        self.current_features = torch.tensor(
            current_features, dtype=torch.float32
        )
        self.future_boards = torch.tensor(future_boards, dtype=torch.long)
        self.future_features = torch.tensor(
            future_features, dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.current_boards)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.current_boards[idx],
            self.current_features[idx],
            self.future_boards[idx],
            self.future_features[idx],
        )


def get_trajectory_dataloader(
    pgn_path: str,
    horizon: int = 4,
    batch_size: int = 64,
    min_elo: int = 1500,
    max_trajectories: int | None = None,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Create train/test dataloaders from PGN trajectory pairs."""
    pairs = extract_trajectories(
        pgn_path, horizon, min_elo, max_trajectories
    )
    dataset = TrajectoryDataset(pairs)

    test_len = min(5000, int(len(dataset) * 0.1))
    train_set, test_set = torch.utils.data.random_split(
        dataset, [len(dataset) - test_len, test_len]
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    return train_loader, test_loader
