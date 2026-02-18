import chess
import chess.engine
import glob
import math
import os
import tarfile
import torch
import pygame
import sys
from inference_test import preprocess, postprocess_valid
from copy import deepcopy

# --- Constants ---
SQ_SIZE   = 80
BOARD_SIZE = SQ_SIZE * 8
BAR_W     = 64          # quality bar on the left
STATUS_H  = 40
WIN_W     = BAR_W + BOARD_SIZE
WIN_SIZE  = (WIN_W, BOARD_SIZE + STATUS_H)

# Colors
LIGHT_SQ       = (240, 217, 181)
DARK_SQ        = (181, 136, 99)
SELECTED_LIGHT = (186, 202, 68)
SELECTED_DARK  = (170, 186, 58)
LAST_MOVE_LIGHT = (205, 210, 106)
LAST_MOVE_DARK  = (170, 162, 58)
CHECK_COLOR    = (235, 97, 80)
STATUS_BG      = (48, 48, 48)
TEXT_COLOR     = (220, 220, 220)
BTN_COLOR      = (70, 130, 180)
BTN_HOVER      = (90, 150, 200)
BTN_TEXT       = (255, 255, 255)
WHITE_PIECE_COLOR = (255, 255, 255)
BLACK_PIECE_COLOR = (0, 0, 0)
BAR_BG         = (28, 28, 28)

PIECE_UNICODE = {
    'R': '\u2656', 'N': '\u2658', 'B': '\u2657', 'Q': '\u2655', 'K': '\u2654', 'P': '\u2659',
    'r': '\u265c', 'n': '\u265e', 'b': '\u265d', 'q': '\u265b', 'k': '\u265a', 'p': '\u265f',
}

# --- Model setup ---
MODEL = "2000_elo_pos_engine_3head.pth"

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = torch.load(f'models/{MODEL}', map_location="cpu").to(device)
model.eval()

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
    _sf_engine = chess.engine.SimpleEngine.popen_uci(_sf_path) if _sf_path else None
except FileNotFoundError:
    _sf_engine = None
    print("Stockfish not found at that path. Move rating disabled.")

_delay_str = input("AI vs AI delay in seconds (default 1): ").strip()
_ai_vs_ai_delay = float(_delay_str) if _delay_str else 1.0

LABELS = ["Excellent (!)", "Good", "Inaccuracy (?!)", "Mistake (?)", "Blunder (??)"]


def print_summary(move_log):
    if not move_log:
        return
    players = list(dict.fromkeys(w for w, _, _ in move_log))
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


def _quality_color(q: float):
    """Map q ∈ [0,1] → RGB: 0=red, 0.5=yellow, 1=green."""
    r = int(220 * (1.0 - q))
    g = int(200 * q)
    return (r, g, 30)


class ChessGUI:
    def __init__(self):
        self.sf_engine = _sf_engine
        pygame.init()
        self.screen = pygame.display.set_mode(WIN_SIZE)
        pygame.display.set_caption("Chessformer")
        self.clock = pygame.time.Clock()

        # Fonts
        self.piece_font   = self._init_piece_font(56)
        self.label_font   = pygame.font.SysFont("sans", 13)
        self.status_font  = pygame.font.SysFont("sans", 20)
        self.btn_font     = pygame.font.SysFont("sans", 24, bold=True)
        self.title_font   = pygame.font.SysFont("sans", 48, bold=True)
        self.summary_font = pygame.font.SysFont("sans", 17)

        # Game state
        self.board        = chess.Board()
        self.made_moves   = []
        self.selected_sq  = None
        self.legal_dests  = set()
        self.last_move    = None
        self.flipped      = False
        self.ai_is_black  = True
        self.ai_vs_ai     = False
        self.ai_vs_ai_delay = _ai_vs_ai_delay
        self.next_ai_move_at = 0       # pygame ticks for next AI move
        self.game_started = False
        self.game_over    = False
        self.status_text  = "Choose your color"
        self.summary_done = False

        # Move quality history: list of (q ∈ [0,1], is_white_move)
        self.move_quality = []
        # Stockfish move log: list of (who, cp_loss, label)
        self.move_log = []

        # Start screen buttons — three buttons centred in the window
        cx  = WIN_W // 2
        bw, bh, gap = 110, 50, 20
        total = 3 * bw + 2 * gap
        x0 = cx - total // 2
        self.white_btn    = pygame.Rect(x0,            300, bw, bh)
        self.black_btn    = pygame.Rect(x0 + bw + gap, 300, bw, bh)
        self.ai_vs_ai_btn = pygame.Rect(x0 + 2*(bw+gap), 300, bw, bh)

    def _init_piece_font(self, size):
        for name in ["DejaVu Sans", "Noto Sans Symbols2", "Noto Sans Symbols",
                      "Symbola", "FreeSerif", "Segoe UI Symbol", "Arial Unicode MS"]:
            font = pygame.font.SysFont(name, size)
            if font.get_height() > 0:
                return font
        return pygame.font.SysFont(None, size)

    # --- Coordinate helpers ---

    def rc_to_square(self, row, col):
        if self.flipped:
            return chess.square(7 - col, row)
        return chess.square(col, 7 - row)

    def square_to_rc(self, sq):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        if self.flipped:
            return r, 7 - f
        return 7 - r, f

    def _board_x(self, col):
        """Pixel x of column, accounting for the left bar."""
        return BAR_W + col * SQ_SIZE

    # --- Move quality ---

    def _eval_move(self, board_before: chess.Board, move: chess.Move) -> float:
        """Rate a move using Stockfish. Returns q ∈ [0, 1] (1=excellent, 0=blunder)."""
        if self.sf_engine is None:
            return 0.5
        info_before = self.sf_engine.analyse(board_before, chess.engine.Limit(depth=12))
        best_cp = info_before["score"].white().score(mate_score=10000)
        board_after = board_before.copy()
        board_after.push(move)
        info_after = self.sf_engine.analyse(board_after, chess.engine.Limit(depth=12))
        actual_cp = info_after["score"].white().score(mate_score=10000)
        if board_before.turn == chess.WHITE:
            cp_loss = best_cp - actual_cp
        else:
            cp_loss = actual_cp - best_cp
        cp_loss = max(0, cp_loss)
        return max(0.0, 1.0 - cp_loss / 200.0)

    # --- Drawing ---

    def _sq_color(self, row, col, sq):
        light = (row + col) % 2 == 0
        if self.board.is_check() and sq == self.board.king(self.board.turn):
            return CHECK_COLOR
        if sq == self.selected_sq:
            return SELECTED_LIGHT if light else SELECTED_DARK
        if self.last_move and sq in (self.last_move.from_square, self.last_move.to_square):
            return LAST_MOVE_LIGHT if light else LAST_MOVE_DARK
        return LIGHT_SQ if light else DARK_SQ

    def _draw_piece(self, piece, rect):
        sym = PIECE_UNICODE[piece.symbol()]
        fg      = WHITE_PIECE_COLOR if piece.color == chess.WHITE else BLACK_PIECE_COLOR
        outline = BLACK_PIECE_COLOR if piece.color == chess.WHITE else WHITE_PIECE_COLOR
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    s = self.piece_font.render(sym, True, outline)
                    self.screen.blit(s, s.get_rect(center=(rect.centerx + dx, rect.centery + dy)))
        s = self.piece_font.render(sym, True, fg)
        self.screen.blit(s, s.get_rect(center=rect.center))

    def draw_board(self):
        for row in range(8):
            for col in range(8):
                sq   = self.rc_to_square(row, col)
                rect = pygame.Rect(self._board_x(col), row * SQ_SIZE, SQ_SIZE, SQ_SIZE)

                pygame.draw.rect(self.screen, self._sq_color(row, col, sq), rect)

                # Legal move dots
                if sq in self.legal_dests:
                    overlay = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
                    center  = (SQ_SIZE // 2, SQ_SIZE // 2)
                    if self.board.piece_at(sq):
                        pygame.draw.circle(overlay, (0, 0, 0, 50), center, SQ_SIZE // 2 - 2, 6)
                    else:
                        pygame.draw.circle(overlay, (0, 0, 0, 50), center, SQ_SIZE // 6)
                    self.screen.blit(overlay, rect.topleft)

                piece = self.board.piece_at(sq)
                if piece:
                    self._draw_piece(piece, rect)

        # Coordinate labels
        for i in range(8):
            file_idx = i if not self.flipped else 7 - i
            label = chr(ord('a') + file_idx)
            color = LIGHT_SQ if i % 2 == 0 else DARK_SQ
            s = self.label_font.render(label, True, color)
            self.screen.blit(s, (self._board_x(i) + SQ_SIZE - s.get_width() - 2,
                                 BOARD_SIZE - s.get_height() - 2))

            rank_idx = i if self.flipped else 7 - i
            label = str(rank_idx + 1)
            color = DARK_SQ if i % 2 == 0 else LIGHT_SQ
            s = self.label_font.render(label, True, color)
            self.screen.blit(s, (BAR_W + 2, i * SQ_SIZE + 2))

    def draw_quality_bar(self):
        """Left panel: per-move progress bar (0→1) with numerical value."""
        pygame.draw.rect(self.screen, BAR_BG, (0, 0, BAR_W, BOARD_SIZE))

        hdr = self.label_font.render("quality", True, (120, 120, 120))
        self.screen.blit(hdr, hdr.get_rect(center=(BAR_W // 2, 9)))

        if not self.move_quality:
            pygame.draw.line(self.screen, (60, 60, 60),
                             (BAR_W - 1, 0), (BAR_W - 1, BOARD_SIZE))
            return

        SEG_H   = 20
        pad     = 5
        inner_w = BAR_W - pad * 2
        max_vis = (BOARD_SIZE - 20) // SEG_H
        visible = self.move_quality[-max_vis:]
        n       = len(visible)
        start_y = BOARD_SIZE - n * SEG_H     # newest at bottom

        for i, (q, is_white) in enumerate(visible):
            y = start_y + i * SEG_H

            # Track background (empty bar)
            pygame.draw.rect(self.screen, (45, 45, 45),
                             (pad, y + 2, inner_w, SEG_H - 4), border_radius=3)

            # Filled portion — proportional to q
            fill_w = max(1, int(inner_w * q))
            pygame.draw.rect(self.screen, _quality_color(q),
                             (pad, y + 2, fill_w, SEG_H - 4), border_radius=3)

            # Numerical value centred over the bar
            val_str = f"{q:.2f}"
            s = self.label_font.render(val_str, True, (240, 240, 240))
            self.screen.blit(s, s.get_rect(center=(BAR_W // 2, y + SEG_H // 2)))

        # Divider
        pygame.draw.line(self.screen, (60, 60, 60),
                         (BAR_W - 1, 0), (BAR_W - 1, BOARD_SIZE))

    def draw_status(self):
        bar = pygame.Rect(0, BOARD_SIZE, WIN_W, STATUS_H)
        pygame.draw.rect(self.screen, STATUS_BG, bar)
        s = self.status_font.render(self.status_text, True, TEXT_COLOR)
        self.screen.blit(s, s.get_rect(center=bar.center))

    def draw_start_screen(self):
        self.screen.fill((40, 40, 40))
        cx = WIN_W // 2
        s = self.title_font.render("Chessformer", True, TEXT_COLOR)
        self.screen.blit(s, s.get_rect(center=(cx, 150)))
        s = self.status_font.render("Choose your color", True, TEXT_COLOR)
        self.screen.blit(s, s.get_rect(center=(cx, 220)))
        mouse = pygame.mouse.get_pos()
        for btn, label in [(self.white_btn, "White"), (self.black_btn, "Black"),
                           (self.ai_vs_ai_btn, "AI vs AI")]:
            color = BTN_HOVER if btn.collidepoint(mouse) else BTN_COLOR
            pygame.draw.rect(self.screen, color, btn, border_radius=8)
            s = self.btn_font.render(label, True, BTN_TEXT)
            self.screen.blit(s, s.get_rect(center=btn.center))
        self.draw_status()

    # --- Game logic ---

    def update_status(self):
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            self.status_text = f"Checkmate! {winner} wins."
            self.game_over = True
        elif self.board.is_stalemate():
            self.status_text = "Draw by stalemate."
            self.game_over = True
        elif self.board.is_insufficient_material():
            self.status_text = "Draw — insufficient material."
            self.game_over = True
        elif self.board.can_claim_draw():
            self.status_text = "Draw (50-move rule / repetition)."
            self.game_over = True
        elif self.board.is_check():
            turn = "White" if self.board.turn else "Black"
            self.status_text = f"{turn} to move — Check!"
        else:
            turn = "White" if self.board.turn else "Black"
            self.status_text = f"{turn} to move"

    def is_ai_turn(self):
        if self.ai_vs_ai:
            return True
        if self.ai_is_black:
            return self.board.turn == chess.BLACK
        return self.board.turn == chess.WHITE

    def ai_move(self, who: str = "AI"):
        if self.game_over:
            return
        input_tensors = preprocess(self.board)

        def predict(rep_mv=""):
            with torch.no_grad():
                output = model(input_tensors)
            return postprocess_valid(output, self.board, rep_mv=rep_mv)

        uci = predict()
        if uci is None:
            moves = list(self.board.legal_moves)
            if moves:
                uci = moves[0].uci()
            else:
                return

        # Avoid 3-move repetition
        tmp = deepcopy(self.board)
        tmp.push(chess.Move.from_uci(uci))
        if tmp.can_claim_threefold_repetition():
            alt = predict(rep_mv=uci)
            if alt is not None:
                uci = alt

        # Prioritize checkmate
        for move in self.board.legal_moves:
            tmp = deepcopy(self.board)
            tmp.push(move)
            if tmp.is_checkmate():
                uci = move.uci()
                break

        move      = chess.Move.from_uci(uci)
        was_white = (self.board.turn == chess.WHITE)
        quality   = self._eval_move(self.board, move)
        self.board.push(move)
        self.move_quality.append((quality, was_white))
        self.made_moves.append(move.uci())
        self.last_move = move
        if self.sf_engine is not None:
            cp_loss = round((1.0 - quality) * 200)
            if cp_loss <= 10:    label = "Excellent (!)"
            elif cp_loss <= 25:  label = "Good"
            elif cp_loss <= 50:  label = "Inaccuracy (?!)"
            elif cp_loss <= 100: label = "Mistake (?)"
            else:                label = "Blunder (??)"
            self.move_log.append((who, cp_loss, label))

    def handle_click(self, pos):
        if self.game_over or self.is_ai_turn():
            return False

        # Ignore clicks inside the left quality bar
        bx = pos[0] - BAR_W
        if bx < 0:
            return False

        col, row = bx // SQ_SIZE, pos[1] // SQ_SIZE
        if not (0 <= row < 8 and 0 <= col < 8):
            return False
        sq = self.rc_to_square(row, col)

        if self.selected_sq is not None:
            if sq in self.legal_dests:
                move = chess.Move(self.selected_sq, sq)
                if move not in self.board.legal_moves:
                    move = chess.Move(self.selected_sq, sq, promotion=chess.QUEEN)
                if move in self.board.legal_moves:
                    was_white = (self.board.turn == chess.WHITE)
                    quality   = self._eval_move(self.board, move)
                    self.board.push(move)
                    self.move_quality.append((quality, was_white))
                    self.made_moves.append(move.uci())
                    self.last_move = move
                    if self.sf_engine is not None:
                        cp_loss = round((1.0 - quality) * 200)
                        if cp_loss <= 10:    label = "Excellent (!)"
                        elif cp_loss <= 25:  label = "Good"
                        elif cp_loss <= 50:  label = "Inaccuracy (?!)"
                        elif cp_loss <= 100: label = "Mistake (?)"
                        else:                label = "Blunder (??)"
                        self.move_log.append(("You", cp_loss, label))
                    self.selected_sq = None
                    self.legal_dests = set()
                    return True

            piece = self.board.piece_at(sq)
            if piece and piece.color == self.board.turn:
                self.selected_sq = sq
                self.legal_dests = {m.to_square for m in self.board.legal_moves
                                    if m.from_square == sq}
            else:
                self.selected_sq = None
                self.legal_dests = set()
        else:
            piece = self.board.piece_at(sq)
            if piece and piece.color == self.board.turn:
                self.selected_sq = sq
                self.legal_dests = {m.to_square for m in self.board.legal_moves
                                    if m.from_square == sq}
        return False

    # --- Summary overlay ---

    def draw_summary_screen(self):
        if not self.move_log:
            return

        LABEL_COLORS = {
            "Excellent (!)":   (80,  210,  80),
            "Good":            (140, 210,  80),
            "Inaccuracy (?!)": (220, 200,  60),
            "Mistake (?)":     (220, 140,  50),
            "Blunder (??)":    (220,  70,  70),
        }

        overlay = pygame.Surface(WIN_SIZE, pygame.SRCALPHA)
        overlay.fill((10, 10, 10, 215))
        self.screen.blit(overlay, (0, 0))

        title = self.btn_font.render("Stockfish Analysis", True, TEXT_COLOR)
        self.screen.blit(title, title.get_rect(center=(WIN_W // 2, 32)))
        pygame.draw.line(self.screen, (80, 80, 80), (30, 56), (WIN_W - 30, 56), 1)

        players  = list(dict.fromkeys(w for w, _, _ in self.move_log))
        n        = len(players)
        panel_w  = 290
        gap      = 24
        total_w  = n * panel_w + (n - 1) * gap
        start_x  = (WIN_W - total_w) // 2

        for i, who in enumerate(players):
            moves = [(cp, lbl) for (w, cp, lbl) in self.move_log if w == who]
            if not moves:
                continue
            counts  = {lbl: 0 for lbl in LABELS}
            for cp, lbl in moves:
                counts[lbl] += 1
            avg_cp   = sum(cp for cp, _ in moves) / len(moves)
            accuracy = max(0.0, min(100.0, 100 * math.exp(-avg_cp / 150)))

            px = start_x + i * (panel_w + gap)
            y  = 70

            s = self.status_font.render(f"{who}  —  {len(moves)} moves", True, TEXT_COLOR)
            self.screen.blit(s, s.get_rect(centerx=px + panel_w // 2, y=y))
            y += 30
            pygame.draw.line(self.screen, (60, 60, 60), (px, y), (px + panel_w, y), 1)
            y += 8

            for lbl in LABELS:
                s  = self.summary_font.render(lbl, True, LABEL_COLORS[lbl])
                s2 = self.summary_font.render(str(counts[lbl]), True, TEXT_COLOR)
                self.screen.blit(s,  (px + 8, y))
                self.screen.blit(s2, s2.get_rect(right=px + panel_w - 8, y=y))
                y += 26

            y += 4
            pygame.draw.line(self.screen, (50, 50, 50), (px, y), (px + panel_w, y), 1)
            y += 8

            s  = self.summary_font.render("Avg cp loss", True, (160, 160, 160))
            s2 = self.summary_font.render(f"{avg_cp:.1f}", True, TEXT_COLOR)
            self.screen.blit(s,  (px + 8, y))
            self.screen.blit(s2, s2.get_rect(right=px + panel_w - 8, y=y))
            y += 32

            acc_color = _quality_color(accuracy / 100)
            s  = self.btn_font.render(f"{accuracy:.1f}%", True, acc_color)
            s2 = self.label_font.render("accuracy", True, (140, 140, 140))
            self.screen.blit(s,  s.get_rect(centerx=px + panel_w // 2, y=y))
            self.screen.blit(s2, s2.get_rect(centerx=px + panel_w // 2, y=y + 30))

    # --- Main loop ---

    def run(self):
        running      = True
        need_ai_move = False

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if not self.game_started:
                        if self.white_btn.collidepoint(event.pos):
                            self.ai_is_black  = True
                            self.flipped      = False
                            self.game_started = True
                            self.update_status()
                        elif self.black_btn.collidepoint(event.pos):
                            self.ai_is_black  = False
                            self.flipped      = True
                            self.game_started = True
                            self.update_status()
                            need_ai_move = True
                        elif self.ai_vs_ai_btn.collidepoint(event.pos):
                            self.ai_vs_ai     = True
                            self.game_started = True
                            self.update_status()
                            self.next_ai_move_at = pygame.time.get_ticks()
                    else:
                        if self.handle_click(event.pos):
                            self.update_status()
                            if not self.game_over and self.is_ai_turn():
                                need_ai_move = True

            if not self.game_started:
                self.draw_start_screen()
            else:
                # Human vs AI: trigger move via flag
                if need_ai_move and not self.ai_vs_ai:
                    self.status_text = "AI is thinking..."
                    self.draw_quality_bar()
                    self.draw_board()
                    self.draw_status()
                    pygame.display.flip()
                    self.ai_move("AI")
                    self.update_status()
                    need_ai_move = False

                # AI vs AI: timer-driven moves
                if self.ai_vs_ai and not self.game_over:
                    now = pygame.time.get_ticks()
                    if now >= self.next_ai_move_at:
                        who = "White" if self.board.turn == chess.WHITE else "Black"
                        self.status_text = f"{who} is thinking..."
                        self.draw_quality_bar()
                        self.draw_board()
                        self.draw_status()
                        pygame.display.flip()
                        self.ai_move(who)
                        self.update_status()
                        self.next_ai_move_at = pygame.time.get_ticks() + int(self.ai_vs_ai_delay * 1000)

                # Print terminal summary once when game ends
                if self.game_over and not self.summary_done:
                    print_summary(self.move_log)
                    self.summary_done = True

                self.draw_quality_bar()
                self.draw_board()
                self.draw_status()
                if self.game_over:
                    self.draw_summary_screen()

            pygame.display.flip()
            self.clock.tick(30)

        if self.sf_engine:
            self.sf_engine.quit()
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    gui = ChessGUI()
    gui.run()
