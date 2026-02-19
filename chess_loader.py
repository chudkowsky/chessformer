import torch
from torch.utils.data import Dataset, DataLoader


# --- V2 constants ---

PIECE_TO_INDEX = {
    '.': 0, 'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12,
}

# Piece values for material balance (uppercase = current player, lowercase = opponent)
PIECE_VALUES = {
    'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9,
    'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9,
}
MAX_MATERIAL = 39.0  # Q + 2R + 2B + 2N + 8P

PROMO_PIECES = {'q': 0, 'r': 1, 'b': 2, 'n': 3}

def square_num(sq: str) -> int:
    """
    Converts chess square notation to a numerical index.
    """
    sq = sq.lower()
    return (ord(sq[0]) - ord('a')) + (8 - int(sq[1])) * 8

def parse_pos_lists(list_file, num_pos=None):
    """
    Parses a file containing chess positions and moves, converting them to numerical representations.
    """
    if isinstance(num_pos, float):
        num_pos = int(num_pos)
    if not isinstance(num_pos, int):
        num_pos = int(1e9)  # Default to a large number if not specified

    with open(list_file, 'r') as file:
        pos = [line for i, line in enumerate(file) if i < num_pos or num_pos < 0]

    boards, new_moves = [], []
    for line in pos:
        if not line:
            continue

        board, new_move = line.strip().split()
        piece_to_index = {'.': 0, 'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
                  'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12}
        board = [piece_to_index[p] for p in board]  # Convert pieces to integers

        new_move = new_move[:2], new_move[2:]  # Split move into start and end squares
        new_move = square_num(new_move[0]), square_num(new_move[1])  # Convert squares to indices

        boards.append(board)
        new_moves.append(new_move)

    return boards, new_moves

class ChessDataset(Dataset):
    """
    A custom PyTorch Dataset for chess positions and moves.
    """
    def __init__(self, boards, moves):
        self.boards = [torch.tensor(board, dtype=torch.long) for board in boards]
        self.moves = [torch.tensor(mv, dtype=torch.long) for mv in moves]

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        board = self.boards[idx]
        move = self.moves[idx]
        return board, move

def get_dataloader(pos_file, batch_size=32, num_workers=0, num_pos=None):
    """
    Creates dataloaders for training and testing datasets.
    """
    boards, moves = parse_pos_lists(pos_file, num_pos=num_pos)
    dataset = ChessDataset(boards, moves)

    test_len = min(5000, int(len(dataset) * 0.1))
    dataset, testset = torch.utils.data.random_split(dataset, [len(dataset) - test_len, test_len])

    # Changed: pin_memory=True for faster CPU→GPU transfer (DMA direct copy)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

    return dataloader, testloader


# --- V2: Extended data pipeline with features and WDL ---


def compute_features(board_str: str) -> list[float]:
    """Compute auxiliary features from 64-char board string.

    Returns 14 floats: material_balance(1) + check(1) + castling(4) + en_passant(8).
    Only material_balance is computable from board string alone;
    remaining features require FEN-level info and default to zero.
    """
    material = sum(PIECE_VALUES.get(c, 0) for c in board_str)
    return [material / MAX_MATERIAL] + [0.0] * 13


def result_to_wdl(result: float) -> tuple[float, float, float]:
    """Convert game result to WDL one-hot (from current player's perspective).

    1.0 → win, 0.5 → draw, 0.0 → loss.
    """
    if result > 0.75:
        return (1.0, 0.0, 0.0)
    if result > 0.25:
        return (0.0, 1.0, 0.0)
    return (0.0, 0.0, 1.0)


def parse_move(move_str: str) -> tuple[int, int, int]:
    """Parse UCI move to (from_sq, to_sq, promo_idx).

    promo_idx: 0-3 for Q/R/B/N promotion, -1 if no promotion.
    """
    from_sq = square_num(move_str[:2])
    to_sq = square_num(move_str[2:4])
    promo_idx = PROMO_PIECES.get(move_str[4:5], -1)
    return from_sq, to_sq, promo_idx


def parse_pos_lists_v2(
    list_file: str, num_pos: int | None = None
) -> tuple[list, list, list, list]:
    """Parse position file with optional result column.

    Supports both formats:
      <64-char board> <uci_move>           (V1 compat, uniform WDL)
      <64-char board> <uci_move> <result>  (V2 with game outcome)

    Returns: (boards, moves, features, wdl_targets)
    """
    if isinstance(num_pos, float):
        num_pos = int(num_pos)
    if not isinstance(num_pos, int):
        num_pos = int(1e9)

    boards: list[list[int]] = []
    moves: list[tuple[int, int, int]] = []
    features: list[list[float]] = []
    wdl_targets: list[tuple[float, float, float]] = []

    with open(list_file, 'r') as file:
        for i, line in enumerate(file):
            if i >= num_pos:
                break
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            board_str = parts[0]
            move_str = parts[1]
            result = float(parts[2]) if len(parts) > 2 else None

            boards.append([PIECE_TO_INDEX[p] for p in board_str])
            moves.append(parse_move(move_str))
            features.append(compute_features(board_str))
            wdl_targets.append(
                result_to_wdl(result) if result is not None
                else (1 / 3, 1 / 3, 1 / 3)
            )

    return boards, moves, features, wdl_targets


class ChessDatasetV2(Dataset):
    """Dataset for V2 model: board + features + from/to targets + WDL."""

    def __init__(self, boards, moves, features, wdl_targets):
        self.boards = torch.tensor(boards, dtype=torch.long)
        self.from_sq = torch.tensor([m[0] for m in moves], dtype=torch.long)
        self.to_sq = torch.tensor([m[1] for m in moves], dtype=torch.long)
        self.promo = torch.tensor([m[2] for m in moves], dtype=torch.long)
        self.features = torch.tensor(features, dtype=torch.float32)
        self.wdl = torch.tensor(wdl_targets, dtype=torch.float32)

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        return (
            self.boards[idx],
            self.features[idx],
            self.from_sq[idx],
            self.to_sq[idx],
            self.promo[idx],
            self.wdl[idx],
        )


def get_dataloader_v2(pos_file, batch_size=32, num_workers=0, num_pos=None):
    """Creates train/test dataloaders for ChessTransformerV2."""
    boards, moves, features, wdl_targets = parse_pos_lists_v2(
        pos_file, num_pos=num_pos
    )
    dataset = ChessDatasetV2(boards, moves, features, wdl_targets)

    test_len = min(5000, int(len(dataset) * 0.1))
    dataset, testset = torch.utils.data.random_split(
        dataset, [len(dataset) - test_len, test_len]
    )

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=num_workers,
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=num_workers,
    )
    return dataloader, testloader


if __name__ == '__main__':
    # Example usage of the get_dataloader function
    dataloader, testloader = get_dataloader('path_to_your_pgn_file.txt')
    print("Dataloader and Testloader created.")
