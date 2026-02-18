import torch
from torch.utils.data import Dataset, DataLoader

# Promotion piece → index mapping (used in both loader and policy)
PROMO_TO_IDX = {'q': 0, 'r': 1, 'b': 2, 'n': 3}
IDX_TO_PROMO = {v: k for k, v in PROMO_TO_IDX.items()}

def square_num(sq: str) -> int:
    """
    Converts chess square notation to a flat board index (0-63).
    File a-h → 0-7, rank 8→row0 … rank 1→row7.
    """
    sq = sq.lower()
    return (ord(sq[0]) - ord('a')) + (8 - int(sq[1])) * 8

def parse_pos_lists(list_file, num_pos=None):
    """
    Parses a file of chess positions and moves.

    Each line: '<64-char board> <uci_move>'
    where uci_move is 4 chars (e.g. 'e2e4') or 5 chars for promotions ('e7e8q').

    Returns boards and moves where each move is (from_sq, to_sq, promo_idx).
    promo_idx is 0-3 (q/r/b/n) for promotions, -1 for normal moves.
    """
    if isinstance(num_pos, float):
        num_pos = int(num_pos)
    if not isinstance(num_pos, int):
        num_pos = int(1e9)  # Default to a large number if not specified

    with open(list_file, 'r') as file:
        pos = [line for i, line in enumerate(file) if i < num_pos or num_pos < 0]

    piece_to_index = {'.': 0, 'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
                      'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12}

    boards, new_moves = [], []
    for line in pos:
        if not line:
            continue

        board_str, move_str = line.strip().split()
        board = [piece_to_index[p] for p in board_str]

        from_sq   = square_num(move_str[:2])
        to_sq     = square_num(move_str[2:4])
        promo_idx = PROMO_TO_IDX.get(move_str[4], -1) if len(move_str) > 4 else -1

        boards.append(board)
        new_moves.append((from_sq, to_sq, promo_idx))

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

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

    return dataloader, testloader

if __name__ == '__main__':
    # Example usage of the get_dataloader function
    dataloader, testloader = get_dataloader('path_to_your_pgn_file.txt')
    print("Dataloader and Testloader created.")
