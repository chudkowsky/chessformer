import chess
import chess.engine
import glob
import math
import os
import tarfile
import torch
from inference_test import preprocess, postprocess_valid
from copy import deepcopy

# Model Configuration
MODEL = "2000_elo_pos_engine_3head.pth"

# Device setup for model
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = torch.load(f'models/{MODEL}', map_location="cpu").to(device)

# Stockfish setup — extract tar if present
_base = os.path.dirname(os.path.abspath(__file__))
for _tar_path in glob.glob(os.path.join(_base, "*stockfish*.tar*")):
    if not os.path.isdir(os.path.join(_base, "stockfish")):
        with tarfile.open(_tar_path) as _tf:
            _tf.extractall(_base)
        for _bin in glob.glob(os.path.join(_base, "stockfish", "stockfish*")):
            if os.path.isfile(_bin):
                os.chmod(_bin, 0o755)
        print(f"Stockfish extracted to: {os.path.join(_base, 'stockfish')}")
    break

_sf_path = input("Path to Stockfish binary (or press Enter to skip): ").strip()
try:
    sf_engine = chess.engine.SimpleEngine.popen_uci(_sf_path) if _sf_path else None
except FileNotFoundError:
    sf_engine = None
    print("Stockfish not found at that path. Move rating disabled.")


def rate_move_sf(board_before: chess.Board, move: chess.Move, depth: int = 15):
    """Rate a move using Stockfish. Returns (cp_loss, label)."""
    if sf_engine is None:
        return None, ""
    info_before = sf_engine.analyse(board_before, chess.engine.Limit(depth=depth))
    best_cp = info_before["score"].white().score(mate_score=10000)
    board_after = board_before.copy()
    board_after.push(move)
    info_after = sf_engine.analyse(board_after, chess.engine.Limit(depth=depth))
    actual_cp = info_after["score"].white().score(mate_score=10000)
    if board_before.turn == chess.WHITE:
        cp_loss = best_cp - actual_cp
    else:
        cp_loss = actual_cp - best_cp
    cp_loss = max(0, cp_loss)
    if cp_loss <= 10:
        label = "Excellent (!)"
    elif cp_loss <= 25:
        label = "Good"
    elif cp_loss <= 50:
        label = "Inaccuracy (?!)"
    elif cp_loss <= 100:
        label = "Mistake (?)"
    else:
        label = "Blunder (??)"
    return cp_loss, label


LABELS = ["Excellent (!)", "Good", "Inaccuracy (?!)", "Mistake (?)", "Blunder (??)"]

# move_log entries: (who: "You"|"AI", cp_loss: int, label: str)
move_log = []


def print_summary():
    if not move_log:
        return
    print("\n" + "=" * 44)
    print("           GAME SUMMARY")
    print("=" * 44)
    for who in ["You", "AI"]:
        moves = [(cp, lbl) for (w, cp, lbl) in move_log if w == who]
        if not moves:
            continue
        counts = {lbl: 0 for lbl in LABELS}
        for cp, lbl in moves:
            counts[lbl] += 1
        avg_cp = sum(cp for cp, _ in moves) / len(moves)
        accuracy = max(0.0, min(100.0, 100 * math.exp(-avg_cp / 150)))
        print(f"\n  {who} ({len(moves)} moves)")
        print(f"  {'Excellent (!)':18s} {counts['Excellent (!)']}")
        print(f"  {'Good':18s} {counts['Good']}")
        print(f"  {'Inaccuracy (?!)':18s} {counts['Inaccuracy (?!)']}")
        print(f"  {'Mistake (?)':18s} {counts['Mistake (?)']}")
        print(f"  {'Blunder (??)':18s} {counts['Blunder (??)']}")
        print(f"  {'Avg cp loss':18s} {avg_cp:.1f}")
        print(f"  {'Accuracy':18s} {accuracy:.1f}%")
    print("=" * 44)


# Initialize the chess board and move history
board = chess.Board()
made_moves = []

# Function to handle player's move input
def get_player_move(board):
    while True:
        move_input = input("Your move (in SAN or UCI format): ")
        try:
            move = board.parse_san(move_input)
        except (chess.InvalidMoveError, chess.IllegalMoveError):
            try:
                move = chess.Move.from_uci(move_input.lower())
            except (chess.InvalidMoveError, chess.IllegalMoveError):
                print("Invalid move. Please try again.")
                continue

        if move in board.legal_moves:
            board_before = board.copy()
            board.push(move)
            made_moves.append(move.uci())
            cp_loss, label = rate_move_sf(board_before, move)
            if label:
                move_log.append(("You", cp_loss, label))
                print(f"Your move rating: {label} (cp loss: {cp_loss})")
            break

    return board

# Game Loop
while not (board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw()):
    print(board)
    print("\nMove history:", made_moves)

    # Determining who the AI is playing as
    ai_player = input("Is the AI playing as white (w) or black (b)? ").lower()
    if ai_player in ['b', 'w']:
        ai_player = ai_player == 'b'
        break

    print("Invalid choice. Please enter 'w' for white or 'b' for black.")

# Play as human if AI is set to play as black
if ai_player:
    board = get_player_move(board)

count = 0
try:
    while not (board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw()):
        # AI's turn
        input_tensors = preprocess(board)
        count += 1

        def predict_move(rep_mv=""):
            with torch.no_grad():
                output = model(input_tensors)
            uci_mv = postprocess_valid(output, board, rep_mv=rep_mv)
            return uci_mv

        uci_move = predict_move()

        # avoiding 3-move repetition
        temp_board = deepcopy(board)
        temp_board.push(chess.Move.from_uci(uci_move))
        if temp_board.can_claim_threefold_repetition():
            uci_move = predict_move(rep_mv=uci_move)

        # Prioritize checkmate move if available
        for move in board.legal_moves:
            temp_board = deepcopy(board)
            temp_board.push(move)
            if temp_board.is_checkmate():
                uci_move = move.uci()
                break

        # Execute AI's move
        move = chess.Move.from_uci(uci_move)
        board_before_ai = board.copy()
        board.push(move)
        made_moves.append(move.uci())
        cp_loss, label = rate_move_sf(board_before_ai, move)
        if label:
            move_log.append(("AI", cp_loss, label))

        # Display board and move history
        print(board)
        rating_str = f"  →  {label} (cp loss: {cp_loss})" if label else ""
        print(f"Predicted move {count}: {move}{rating_str}")
        print("\nMove history:", made_moves)

        if not (board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw()):
            board = get_player_move(board)
except (KeyboardInterrupt, EOFError):
    pass
finally:
    print_summary()
    if sf_engine:
        sf_engine.quit()
