"""Parse a PGN file and produce training data for ChessFormer.

Output format (one line per position):
    <64-char board string> <uci_move> <result>

Result is from the current player's perspective: 1.0 = win, 0.5 = draw, 0.0 = loss.
Board is always oriented so the current player plays as white (flipped for black).

For Lichess games: only games where both players have Elo >= MIN_ELO are
included, and Blitz/Bullet games are skipped.

For self-play games (Event header starts with "SF-SelfPlay"): all games are
included regardless of Elo or time control.
"""

import chess
import chess.pgn
import sys
from chess_moves_to_input_data import get_board_str, switch_move

MIN_ELO = 1500
INPUT_PGN = "lichess_db_standard_rated_2013-01.pgn"
OUTPUT_DIR = "full_datasets"
OUTPUT_FILE = f"{OUTPUT_DIR}/elo_{MIN_ELO}_pos.txt"


# Changed: added max_positions parameter to stop after collecting enough data
def process_pgn(pgn_path, out_path, min_elo=MIN_ELO, max_positions=None):
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    games_used = 0
    games_skipped = 0
    positions_written = 0

    is_stdin = pgn_path == "/dev/stdin"
    pgn_file = sys.stdin if is_stdin else open(pgn_path)
    seekable = not is_stdin

    with open(out_path, "w") as out:
        while True:
            # Fast path: read headers first (skips move parsing), then only
            # parse the full game if it passes the Elo/event filter.
            if seekable:
                offset = pgn_file.tell()
                headers = chess.pgn.read_headers(pgn_file)
                if headers is None:
                    break
            else:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                headers = game.headers

            event = headers.get("Event", "")
            is_selfplay = event.startswith("SF-SelfPlay")

            if not is_selfplay:
                # Filter by Elo
                try:
                    white_elo = int(headers.get("WhiteElo", "0"))
                    black_elo = int(headers.get("BlackElo", "0"))
                except ValueError:
                    games_skipped += 1
                    continue

                if white_elo < min_elo or black_elo < min_elo:
                    games_skipped += 1
                    if games_skipped % 50000 == 0:
                        total = games_used + games_skipped
                        print(f"Scanned {total} games, used: {games_used}, positions: {positions_written}")
                    continue

                # Skip Blitz/Bullet
                if "Blitz" in event or "Bullet" in event:
                    games_skipped += 1
                    continue

            # Game passes filter — parse moves
            if seekable:
                pgn_file.seek(offset)
                game = chess.pgn.read_game(pgn_file)

            # Parse game result: 1-0 → 1.0 (white win), 0-1 → 0.0, 1/2-1/2 → 0.5
            result_str = game.headers.get("Result", "*")
            result_map = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}
            base_result = result_map.get(result_str, 0.5)

            board = game.board()
            for move in game.mainline_moves():
                board_str = get_board_str(board, white_side=board.turn)
                uci = move.uci()
                uci_adjusted = switch_move(uci, wht_turn=board.turn, normal_format=True)
                # Result from current player's perspective (flip for black)
                result = base_result if board.turn else 1.0 - base_result
                out.write(f"{board_str} {uci_adjusted} {result}\n")
                positions_written += 1
                board.push(move)

            games_used += 1
            if games_used % 1000 == 0:
                total = games_used + games_skipped
                print(f"Games used: {games_used} / {total} scanned, positions: {positions_written}")

            # Changed: stop early if we've collected enough positions
            if max_positions and positions_written >= max_positions:
                print(f"\nReached {max_positions:,} positions limit, stopping.")
                break

    if pgn_file is not sys.stdin:
        pgn_file.close()
    print(f"\nDone! Games used: {games_used} (skipped: {games_skipped}), positions: {positions_written}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    pgn = sys.argv[1] if len(sys.argv) > 1 else INPUT_PGN
    out = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_FILE
    elo = int(sys.argv[3]) if len(sys.argv) > 3 else MIN_ELO
    # Changed: optional 4th arg for max positions (e.g. 10000000 for 10M)
    max_pos = int(sys.argv[4]) if len(sys.argv) > 4 else None
    process_pgn(pgn, out, min_elo=elo, max_positions=max_pos)
