import torch
import chess
import sys
import chessformer
sys.modules['transformer'] = chessformer
from chessformer import ChessTransformer, ChessTransformerV2
from chess_moves_to_input_data import get_board_str, switch_player, switch_move
from model_utils import (
    detect_device,
    load_model,
    preprocess_board as preprocess_v2,
    preprocess_board_v1 as preprocess,
)
from policy import greedy_move_v2
from torch.utils.data import DataLoader
from copy import deepcopy
import time

# Configuration
MODEL = "2000_elo_pos_engine.pth"

# Changed: --device flag to override auto-detection (auto/cuda/mps/cpu)
import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
_args = _parser.parse_args()

device = detect_device(_args.device)

# Auto-detect V1/V2 from checkpoint format
model, _model_version, _model_cfg = load_model(f'models/{MODEL}', device)

# Helper functions for postprocessing
def sq_to_str(sq):
    """
    Converts a square index to algebraic notation.
    """
    return chr(ord('a') + sq % 8) + str(8 - sq // 8)

def postprocess_valid(output, board: chess.Board, rep_mv=""):
    """
    Converts model output to a valid chess move.
    """
    start = time.time()
    single_output = output[0].tolist()
    all_moves = []
    for i, st_sqr in enumerate(single_output):
        for j, end_sq in enumerate(single_output):
            if i != j:
                all_moves.append(((i, st_sqr[0]), (j, end_sq[1])))

    all_moves.sort(key=lambda x: x[0][1] + x[1][1], reverse=True)
    legal_moves = [str(move) for move in board.legal_moves]

    for mv in all_moves:
        mv_str = sq_to_str(mv[0][0]) + sq_to_str(mv[1][0])
        if not board.turn:
            mv_str = switch_move(mv_str, wht_turn=board.turn, normal_format=True)
        if mv_str in legal_moves and mv_str != rep_mv:
            print(f'Move: {mv_str} = {chess.Board.san(board, chess.Move.from_uci(mv_str))}')
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
        count += 1
        with torch.no_grad():
            if _model_version == 'v2':
                board_t, feat_t = preprocess_v2(board, device)
                policy_logits, promo_logits, wdl, ply = model(board_t, feat_t)
                move = greedy_move_v2(board, policy_logits[0], promo_logits[0])
                uci_move = move.uci()
                print(f'WDL: W={wdl[0,0]:.2f} D={wdl[0,1]:.2f} L={wdl[0,2]:.2f}')
            else:
                input_tensors = preprocess(board, device)
                output = model(input_tensors)
                uci_move = postprocess_valid(output, board)
        board.push(chess.Move.from_uci(uci_move))
        print(f'Predicted {count}\n', board)
