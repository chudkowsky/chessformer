import torch
import chess
from chess_loader import ChessDataset
from chessformer import ChessTransformer
from chess_moves_to_input_data import get_board_str, switch_player, switch_move
from policy import greedy_move, legal_move_policy
from torch.utils.data import DataLoader
from copy import deepcopy
import time

# Configuration
MODEL = "2000_elo_pos_engine_3head.pth"

# Model and device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = torch.load(f'models/{MODEL}', map_location="cpu").to(device)

# Preprocessing function
def preprocess(board):
    """
    Converts a chess board state to a tensor representation.
    """
    board_str = get_board_str(board, white_side=board.turn)
    piece_to_index = {'.': 0, 'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
                      'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12}
    board_pieces = [piece_to_index[p] for p in board_str]
    return torch.tensor([board_pieces], dtype=torch.long).to(device)

def postprocess_valid(output, board: chess.Board, rep_mv=""):
    """
    Converts model output to the best legal chess move (UCI string).

    output is the 3-tuple (from_logits, to_logits, promo_logits) returned
    by ChessTransformer.forward, each with a batch dimension of 1.
    rep_mv is an optional move string to avoid (repetition avoidance).
    """
    start = time.time()
    from_logits, to_logits, promo_logits = output
    # Strip the batch dimension → (64,), (64,), (4,)
    from_logits  = from_logits[0]
    to_logits    = to_logits[0]
    promo_logits = promo_logits[0]

    moves, probs, _ = legal_move_policy(board, from_logits, to_logits, promo_logits)

    # Sort by descending probability, skip the repetition move if provided
    order = probs.argsort(descending=True)
    for idx in order:
        move = moves[idx.item()]
        mv_str = move.uci()
        if mv_str != rep_mv:
            print(f'Move: {mv_str} = {board.san(move)}')
            print('Completed in:', str(time.time() - start))
            return mv_str

    print('Completed in:', str(time.time() - start))
    return None

# Main execution
if __name__ == '__main__':
    # Sample chess game data
    input_data = ["Pe2e4", "pe7e5", "Ng1f3"]
    board = chess.Board()

    for move in input_data:
        mv = move if len(move) == 4 else move[1:]
        board.push(chess.Move.from_uci(mv))

    count = 0
    while not (board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw()):
        input_tensors = preprocess(board)
        count += 1
        with torch.no_grad():
            output = model(input_tensors)
        uci_move = postprocess_valid(output, board)
        board.push(chess.Move.from_uci(uci_move))
        print(f'Predicted {count}\n', board)
