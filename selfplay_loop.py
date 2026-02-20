"""Self-play training loop for ChessTransformerV2.

Generates games where the V2 model plays against itself, writes positions
in the standard training format, then trains on the collected data.
Repeats for multiple generations (generate -> train -> generate -> ...).

Usage (standalone):
    uv run selfplay_loop.py --model models/2500_elo_pos_engine_v2.pth \
        --generations 10 --games-per-gen 100 --epochs-per-gen 2

Usage (from train_model.py):
    uv run train_model.py 2500 --phase 3 \
        --backbone-model models/2500_elo_pos_engine_v2.pth \
        --generations 10 --games-per-gen 100
"""

from __future__ import annotations

import random
import time
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import chess
import torch
from torch import Tensor
from torch.amp import GradScaler, autocast

from chess_loader import get_dataloader_v2
from chess_moves_to_input_data import get_board_str, switch_move
from chessformer import ChessTransformerV2
from model_utils import compute_loss_v2, detect_device, find_latest_model, load_model, preprocess_board
from openings import sample_opening
from policy import sample_move_v2


# --- Config ---


@dataclass(frozen=True)
class SelfPlayConfig:
    """All parameters for one self-play training run."""

    model_path: str
    output_dir: str
    generations: int
    games_per_gen: int
    epochs_per_gen: int
    batch_size: int
    lr: float
    temp_schedule: list[tuple[float, int]]
    max_moves: int
    resign_threshold: float
    resign_count: int
    draw_threshold: float
    draw_count: int
    buffer_size: int
    use_diffusion: bool
    mix_supervised: str | None
    mix_ratio: float
    eval_games: int
    device: str
    mcts_sims: int            # 0 = disabled (raw policy), >0 = MCTS simulations per move
    cpuct: float              # MCTS exploration constant
    dirichlet_alpha: float    # Dirichlet noise parameter (lower = spikier)
    dirichlet_epsilon: float  # Noise mixing weight (0 = no noise, 0.25 = AlphaZero default)


# --- Pure helpers ---


def parse_temp_schedule(s: str) -> list[tuple[float, int]]:
    """Parse "1.5:10,1.0:25,0.3:999" -> [(1.5, 10), (1.0, 25), (0.3, 999)]."""
    result = []
    for part in s.split(","):
        temp_str, ply_str = part.strip().split(":")
        result.append((float(temp_str), int(ply_str)))
    return result


def get_temperature(schedule: list[tuple[float, int]], ply: int) -> float:
    """Return temperature for the given ply based on schedule."""
    for temp, until_ply in schedule:
        if ply < until_ply:
            return temp
    return schedule[-1][0]


RESULT_MAP = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


def format_time(seconds: float) -> str:
    """Format seconds as '1h 23m 45s', dropping zero-value leading units."""
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def mcts_sample_move(
    visit_policy: dict[chess.Move, float],
    temperature: float,
    rng: random.Random,
) -> chess.Move:
    """Sample a move from MCTS visit-count policy with temperature."""
    moves = list(visit_policy.keys())
    counts = [visit_policy[m] for m in moves]

    if temperature <= 0 or len(moves) == 1:
        return moves[max(range(len(counts)), key=lambda i: counts[i])]

    powered = [c ** (1.0 / temperature) for c in counts]
    total = sum(powered)
    probs = [p / total for p in powered]
    return rng.choices(moves, weights=probs, k=1)[0]


# --- Game generation ---


def generate_game(
    model: ChessTransformerV2,
    device: torch.device,
    opening: tuple[str, str, str, list[str]],
    config: SelfPlayConfig,
    rng: random.Random,
) -> list[str]:
    """Play one complete self-play game, return list of training lines.

    Each line: '<64-char board> <uci_move> <result>'
    Result is filled in retroactively once the game ends.
    """
    _eco, _name, _fen, opening_moves = opening

    board = chess.Board()
    for uci in opening_moves:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            break  # stop at first illegal opening move
        board.push(move)
    opening_ply = board.ply()

    positions: list[tuple[str, str, bool]] = []  # (board_str, uci_move, was_white)
    resign_counter = {chess.WHITE: 0, chess.BLACK: 0}
    draw_counter = 0
    resigned_side: chess.Color | None = None

    # MCTS searcher (created once per game, reused across moves)
    mcts_searcher: MCTS | None = None
    if config.mcts_sims > 0:
        from mcts import MCTS

        mcts_searcher = MCTS(
            model, device,
            num_simulations=config.mcts_sims,
            cpuct=config.cpuct,
            use_diffusion=config.use_diffusion,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=config.dirichlet_epsilon,
        )

    model.eval()
    with torch.no_grad():
        while not board.is_game_over(claim_draw=True) and board.ply() < config.max_moves:
            ply_since_opening = board.ply() - opening_ply
            temperature = get_temperature(config.temp_schedule, ply_since_opening)
            was_white = board.turn == chess.WHITE

            # Record position before making the move
            board_str = get_board_str(board, white_side=board.turn)

            if mcts_searcher is not None:
                # MCTS move selection
                visit_policy, wdl_tuple = mcts_searcher.search(board)
                if not visit_policy:
                    break  # no legal moves
                move = mcts_sample_move(visit_policy, temperature, rng)
                loss_prob = wdl_tuple[2]
                draw_prob = wdl_tuple[1]
            else:
                # Raw policy sampling (original behavior)
                board_t, feat_t = preprocess_board(board, device)
                policy_logits, promo_logits, wdl, _ply = model(
                    board_t, feat_t, use_diffusion=config.use_diffusion
                )
                try:
                    move, _log_prob = sample_move_v2(
                        board, policy_logits[0], promo_logits[0], temperature
                    )
                except ValueError:
                    break
                loss_prob = wdl[0, 2].item()
                draw_prob = wdl[0, 1].item()

            # UCI from model perspective (always as-white) -> adjust for actual side
            uci_adjusted = switch_move(
                move.uci(), wht_turn=board.turn, normal_format=True
            )
            positions.append((board_str, uci_adjusted, was_white))

            # Adjudication: resignation
            if loss_prob > config.resign_threshold:
                resign_counter[board.turn] += 1
            else:
                resign_counter[board.turn] = 0

            if resign_counter[board.turn] >= config.resign_count:
                resigned_side = board.turn
                break

            # Adjudication: draw
            if draw_prob > config.draw_threshold:
                draw_counter += 1
            else:
                draw_counter = 0

            if draw_counter >= config.draw_count:
                break  # draw by adjudication

            board.push(move)

    # Determine result
    if resigned_side is not None:
        result_str = "0-1" if resigned_side == chess.WHITE else "1-0"
    elif draw_counter >= config.draw_count:
        result_str = "1/2-1/2"
    elif board.ply() >= config.max_moves:
        result_str = "1/2-1/2"
    else:
        outcome = board.outcome(claim_draw=True)
        result_str = outcome.result() if outcome else "1/2-1/2"

    base_result = RESULT_MAP[result_str]

    # Build training lines with result from current player's perspective
    lines = []
    for board_str, uci_move, was_white in positions:
        result = base_result if was_white else 1.0 - base_result
        lines.append(f"{board_str} {uci_move} {result}")

    return lines


def generate_games(
    model: ChessTransformerV2,
    device: torch.device,
    config: SelfPlayConfig,
    generation: int,
) -> str:
    """Generate N games and write to output file.

    Returns path to the generated data file.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / f"gen_{generation:03d}.txt"

    all_lines: list[str] = []
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    rng = random.Random(generation * 1000)

    start = time.time()
    for i in range(config.games_per_gen):
        opening = sample_opening(rng)
        lines = generate_game(model, device, opening, config, rng)
        all_lines.extend(lines)

        # Count result from the lines (first position is always white's perspective)
        if lines:
            first_result = float(lines[0].split()[-1])
            if first_result > 0.75:
                results["1-0"] += 1
            elif first_result > 0.25:
                results["1/2-1/2"] += 1
            else:
                results["0-1"] += 1

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            print(
                f"  Generated {i + 1}/{config.games_per_gen} games "
                f"({len(all_lines)} positions, {format_time(elapsed)})"
            )

    with open(data_path, "w") as f:
        for line in all_lines:
            f.write(line + "\n")

    elapsed = time.time() - start
    print(
        f"  Done: {config.games_per_gen} games, {len(all_lines)} positions -> {data_path}"
    )
    print(
        f"  Results: +{results['1-0']} ={results['1/2-1/2']} -{results['0-1']} "
        f"({format_time(elapsed)})"
    )
    return str(data_path)


# --- Replay buffer ---


def build_training_data(
    config: SelfPlayConfig,
    generation: int,
) -> str:
    """Concatenate recent generations + optional supervised data into one training file."""
    output_dir = Path(config.output_dir)
    start_gen = max(0, generation - config.buffer_size + 1)

    all_lines: list[str] = []
    for g in range(start_gen, generation + 1):
        gen_path = output_dir / f"gen_{g:03d}.txt"
        with open(gen_path) as f:
            all_lines.extend(f.readlines())

    selfplay_count = len(all_lines)

    # Mix supervised data
    if config.mix_supervised and config.mix_ratio > 0:
        ratio = min(config.mix_ratio, 0.99)
        num_supervised = int(selfplay_count * ratio / (1 - ratio))
        with open(config.mix_supervised) as f:
            supervised_lines = f.readlines()
        mix_rng = random.Random(generation)
        sampled = mix_rng.sample(
            supervised_lines, min(num_supervised, len(supervised_lines))
        )
        all_lines.extend(sampled)

    # Shuffle
    random.Random(generation).shuffle(all_lines)

    combined_path = output_dir / f"combined_gen_{generation:03d}.txt"
    with open(combined_path, "w") as f:
        f.writelines(all_lines)

    print(
        f"  Training data: {selfplay_count} selfplay"
        + (f" + {len(all_lines) - selfplay_count} supervised" if config.mix_supervised else "")
        + f" = {len(all_lines)} total"
    )
    return str(combined_path)


# --- Training ---


def train_on_selfplay(
    model: ChessTransformerV2,
    data_path: str,
    config: SelfPlayConfig,
    device: torch.device,
) -> float:
    """Train model on self-play data for K epochs. Returns final train loss."""
    dataloader, testloader = get_dataloader_v2(
        data_path, batch_size=config.batch_size, num_workers=4,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda") if use_amp else None

    for epoch in range(1, config.epochs_per_gen + 1):
        model.train()
        total_loss = 0.0
        for batch_data in dataloader:
            boards, features, from_sq, to_sq, _promo, wdl_target = [
                x.to(device) for x in batch_data
            ]
            if use_amp:
                with autocast("cuda"):
                    loss = compute_loss_v2(model, boards, features, from_sq, to_sq, wdl_target)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = compute_loss_v2(model, boards, features, from_sq, to_sq, wdl_target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
            total_loss += loss.item()

        avg_train = total_loss / max(len(dataloader), 1)

        # Eval
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_data in testloader:
                boards, features, from_sq, to_sq, _promo, wdl_target = [
                    x.to(device) for x in batch_data
                ]
                loss = compute_loss_v2(model, boards, features, from_sq, to_sq, wdl_target)
                test_loss += loss.item()

        avg_test = test_loss / max(len(testloader), 1)
        print(
            f"    Epoch {epoch}/{config.epochs_per_gen} | "
            f"Train: {avg_train:.4f} | Test: {avg_test:.4f}"
        )

    return avg_train


# --- Evaluation ---


def evaluate_model(
    current: ChessTransformerV2,
    baseline: ChessTransformerV2,
    device: torch.device,
    num_games: int,
) -> tuple[int, int, int]:
    """Play evaluation games between current and baseline (greedy, no temperature).

    Returns (wins, draws, losses) from current model's perspective.
    """
    from policy import greedy_move_v2

    wins, draws, losses = 0, 0, 0
    current.eval()
    baseline.eval()

    for game_idx in range(num_games):
        board = chess.Board()
        rng = random.Random(game_idx)
        opening = sample_opening(rng)
        for uci in opening[3]:
            board.push(chess.Move.from_uci(uci))

        # current plays white in even games, black in odd
        current_is_white = game_idx % 2 == 0

        with torch.no_grad():
            while not board.is_game_over(claim_draw=True) and board.ply() < 200:
                is_white_turn = board.turn == chess.WHITE
                active_model = (
                    current if is_white_turn == current_is_white else baseline
                )

                board_t, feat_t = preprocess_board(board, device)
                policy, promo, _wdl, _ply = active_model(board_t, feat_t)
                move = greedy_move_v2(board, policy[0], promo[0])
                board.push(move)

        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            draws += 1
        elif outcome.winner == chess.WHITE:
            if current_is_white:
                wins += 1
            else:
                losses += 1
        else:
            if not current_is_white:
                wins += 1
            else:
                losses += 1

    return wins, draws, losses


# --- Main loop ---


def _resolve_model_path(path: str) -> str:
    """Resolve 'latest' to newest model in models/, otherwise return as-is."""
    if path == "latest":
        found = find_latest_model()
        if found is None:
            raise FileNotFoundError("No models found in models/ directory")
        return found
    return path


def selfplay_loop(config: SelfPlayConfig) -> None:
    """Main self-play training loop."""
    config = dataclasses.replace(config, model_path=_resolve_model_path(config.model_path))
    device = detect_device(config.device)

    model, _version, model_cfg = load_model(config.model_path, device)
    num_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*60}")
    print(f"  Self-Play Training")
    print(f"{'='*60}")
    print(f"  Model:        {config.model_path}")
    print(f"  Parameters:   {num_params:,}")
    print(f"  Device:       {device}")
    print(f"{'─'*60}")
    print(f"  Generations:  {config.generations}")
    print(f"  Games/gen:    {config.games_per_gen}")
    print(f"  Epochs/gen:   {config.epochs_per_gen}")
    print(f"  Batch size:   {config.batch_size}")
    print(f"  LR:           {config.lr}")
    print(f"  Buffer:       {config.buffer_size} generations")
    print(f"  Temperature:  {config.temp_schedule}")
    if config.mcts_sims > 0:
        print(f"  MCTS:         {config.mcts_sims} sims, cpuct={config.cpuct}, "
              f"Dir(α={config.dirichlet_alpha}, ε={config.dirichlet_epsilon})")
    if config.mix_supervised:
        print(f"  Supervised:   {config.mix_ratio:.0%} from {config.mix_supervised}")
    if config.eval_games > 0:
        print(f"  Eval games:   {config.eval_games} vs baseline")
    print(f"{'='*60}\n")

    # Baseline for evaluation (frozen copy of initial model)
    baseline = None
    if config.eval_games > 0:
        baseline, _, _ = load_model(config.model_path, device)

    # Best-model gating: track best weights, revert if new model is worse
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_gen = "baseline"
    accepted = 0

    output_dir = Path(config.output_dir)

    for gen in range(config.generations):
        print(f"\n{'='*60}")
        print(f"  Generation {gen + 1}/{config.generations}")
        print(f"{'='*60}\n")

        # 1. Generate games (always with current best model)
        print("[1/4] Generating self-play games...")
        generate_games(model, device, config, gen)

        # 2. Build training dataset
        print("\n[2/4] Building training dataset...")
        combined_path = build_training_data(config, gen)

        # 3. Train
        print(f"\n[3/4] Training ({config.epochs_per_gen} epochs)...")
        train_on_selfplay(model, combined_path, config, device)

        # 4. Save checkpoint (always, even if rejected later)
        ckpt_path = output_dir / f"model_gen_{gen:03d}.pth"
        torch.save(
            {
                "version": "v2",
                "state_dict": model.state_dict(),
                "config": model_cfg,
                "selfplay_generation": gen,
            },
            ckpt_path,
        )
        print(f"\n  Checkpoint saved: {ckpt_path}")

        # 5. Evaluate and gate
        if baseline is not None and config.eval_games > 0:
            print(f"\n[4/4] Evaluating ({config.eval_games} games vs baseline)...")
            w, d, l = evaluate_model(model, baseline, device, config.eval_games)
            total = w + d + l
            score = (w + 0.5 * d) / total if total > 0 else 0.5
            # Approximate Elo difference
            if 0 < score < 1:
                import math
                elo_diff = -400 * math.log10(1 / score - 1)
            else:
                elo_diff = 400 if score >= 1 else -400
            print(f"  Result: +{w} ={d} -{l} (approx. {elo_diff:+.0f} Elo vs baseline)")

            # Gating: accept only if wins > losses
            if w > l:
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_gen = f"gen_{gen:03d}"
                accepted += 1
                print(f"  -> ACCEPTED (best = {best_gen}, {accepted} accepted so far)")
            else:
                model.load_state_dict(best_state)
                print(f"  -> REJECTED, reverted to {best_gen}")

    # Save best model back to models/ directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    best_path = Path(config.model_path)
    save_name = best_path.stem + "_selfplay" + best_path.suffix
    save_path = models_dir / save_name
    torch.save(
        {"version": "v2", "state_dict": best_state, "config": model_cfg},
        save_path,
    )

    print(f"\n{'='*60}")
    print(f"Self-play training complete! ({accepted}/{config.generations} generations accepted)")
    print(f"  Best model: {best_gen}")
    print(f"  Saved to: {save_path}")
    print(f"{'='*60}")


# --- CLI ---


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Self-play training for ChessTransformerV2")
    parser.add_argument("--model", required=True,
                        help='Path to V2 model checkpoint, or "latest" for newest in models/')
    parser.add_argument("--output-dir", default="selfplay_data", help="Output directory")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--games-per-gen", type=int, default=200)
    parser.add_argument("--epochs-per-gen", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--temp-schedule", type=str, default="1.5:10,1.0:25,0.3:999",
        help='Temperature schedule as "temp:until_ply,..." (default: 1.5:10,1.0:25,0.3:999)',
    )
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--buffer-size", type=int, default=5)
    parser.add_argument("--mix-supervised", type=str, default=None)
    parser.add_argument("--mix-ratio", type=float, default=0.5)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--mcts-sims", type=int, default=0,
                        help="MCTS simulations per move (0=disabled, raw policy)")
    parser.add_argument("--cpuct", type=float, default=1.25,
                        help="MCTS exploration constant")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3,
                        help="Dirichlet noise alpha (lower=spikier, default: 0.3)")
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.25,
                        help="Dirichlet noise weight (0=off, 0.25=AlphaZero default)")
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"],
    )
    args = parser.parse_args()

    config = SelfPlayConfig(
        model_path=args.model,
        output_dir=args.output_dir,
        generations=args.generations,
        games_per_gen=args.games_per_gen,
        epochs_per_gen=args.epochs_per_gen,
        batch_size=args.batch_size,
        lr=args.lr,
        temp_schedule=parse_temp_schedule(args.temp_schedule),
        max_moves=args.max_moves,
        resign_threshold=0.95,
        resign_count=3,
        draw_threshold=0.80,
        draw_count=5,
        buffer_size=args.buffer_size,
        use_diffusion=False,
        mix_supervised=args.mix_supervised,
        mix_ratio=args.mix_ratio,
        eval_games=args.eval_games,
        device=args.device,
        mcts_sims=args.mcts_sims,
        cpuct=args.cpuct,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
    )
    selfplay_loop(config)


if __name__ == "__main__":
    main()
