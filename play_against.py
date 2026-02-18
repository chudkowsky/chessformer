import chess
import chess.engine
import glob
import math
import os
import tarfile
import time
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
    players = list(dict.fromkeys(w for w, _, _ in move_log))  # unique, insertion order
    print("\n" + "=" * 44)
    print("           GAME SUMMARY")
    print("=" * 44)
    for who in players:
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


# ── shared state ────────────────────────────────────────────────────────────
board      = chess.Board()
made_moves = []
move_log   = []


def is_over():
    return (board.is_checkmate() or board.is_stalemate()
            or board.is_insufficient_material() or board.can_claim_draw())


def ai_make_move(who: str):
    """Run the model for the current position, push the move, log the rating."""
    input_tensors = preprocess(board)

    def predict(rep_mv=""):
        with torch.no_grad():
            output = model(input_tensors)
        return postprocess_valid(output, board, rep_mv=rep_mv)

    uci_move = predict()

    # Avoid 3-move repetition
    tmp = deepcopy(board)
    tmp.push(chess.Move.from_uci(uci_move))
    if tmp.can_claim_threefold_repetition():
        uci_move = predict(rep_mv=uci_move)

    # Prioritize checkmate
    for m in board.legal_moves:
        tmp = deepcopy(board)
        tmp.push(m)
        if tmp.is_checkmate():
            uci_move = m.uci()
            break

    move        = chess.Move.from_uci(uci_move)
    board_before = board.copy()
    board.push(move)
    made_moves.append(move.uci())
    cp_loss, label = rate_move_sf(board_before, move)
    if label:
        move_log.append((who, cp_loss, label))
    return move, cp_loss, label


def sf_make_move(who: str):
    """Have Stockfish play the current position, push the move, log the rating."""
    result = sf_engine.play(board, chess.engine.Limit(depth=15))
    move = result.move
    board_before = board.copy()
    board.push(move)
    made_moves.append(move.uci())
    cp_loss, label = rate_move_sf(board_before, move)
    if label:
        move_log.append((who, cp_loss, label))
    return move, cp_loss, label


def get_player_move():
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


# ── mode selection ───────────────────────────────────────────────────────────
print("\n1. Human vs AI")
print("2. AI vs AI")
if sf_engine:
    print("3. Model vs Stockfish")
_valid = ("1", "2", "3") if sf_engine else ("1", "2")
while True:
    mode = input("Select mode: ").strip()
    if mode in _valid:
        break
    print(f"Please enter {' or '.join(_valid)}.")

# ── AI vs AI ─────────────────────────────────────────────────────────────────
if mode == "2":
    delay_str = input("Delay between moves in seconds (default 1): ").strip()
    delay = float(delay_str) if delay_str else 1.0

    count = 0
    try:
        while not is_over():
            count += 1
            who  = "White" if board.turn == chess.WHITE else "Black"
            move, cp_loss, label = ai_make_move(who)
            rating_str = f"  →  {label} (cp loss: {cp_loss})" if label else ""
            print(board)
            print(f"Move {count} ({who}): {move}{rating_str}")
            print("\nMove history:", made_moves)
            if delay > 0 and not is_over():
                time.sleep(delay)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        print_summary()
        if sf_engine:
            sf_engine.quit()

# ── Model vs Stockfish ───────────────────────────────────────────────────────
elif mode == "3":
    while True:
        model_color = input("Model plays as (w)hite or (b)lack? ").lower()
        if model_color in ("w", "b"):
            break
        print("Please enter 'w' or 'b'.")
    model_is_white = (model_color == "w")

    delay_str = input("Delay between moves in seconds (default 1): ").strip()
    delay = float(delay_str) if delay_str else 1.0

    count = 0
    try:
        while not is_over():
            count += 1
            if (board.turn == chess.WHITE) == model_is_white:
                move, cp_loss, label = ai_make_move("Model")
                who = "Model"
            else:
                move, cp_loss, label = sf_make_move("Stockfish")
                who = "Stockfish"
            rating_str = f"  →  {label} (cp loss: {cp_loss})" if label else ""
            print(board)
            print(f"Move {count} ({who}): {move}{rating_str}")
            print("\nMove history:", made_moves)
            if delay > 0 and not is_over():
                time.sleep(delay)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        print_summary()
        if sf_engine:
            sf_engine.quit()

# ── Human vs AI ──────────────────────────────────────────────────────────────
else:
    print(board)
    while True:
        ai_color = input("Is the AI playing as white (w) or black (b)? ").lower()
        if ai_color in ("w", "b"):
            break
        print("Please enter 'w' or 'b'.")
    ai_is_black = (ai_color == "b")

    if ai_is_black:
        get_player_move()

    count = 0
    try:
        while not is_over():
            count += 1
            move, cp_loss, label = ai_make_move("AI")
            rating_str = f"  →  {label} (cp loss: {cp_loss})" if label else ""
            print(board)
            print(f"Predicted move {count}: {move}{rating_str}")
            print("\nMove history:", made_moves)
            if not is_over():
                get_player_move()
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        print_summary()
        if sf_engine:
            sf_engine.quit()
