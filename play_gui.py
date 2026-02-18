import chess
import torch
import pygame
import sys
from inference_test import preprocess, postprocess_valid
from copy import deepcopy

# --- Constants ---
SQ_SIZE = 80
BOARD_SIZE = SQ_SIZE * 8
STATUS_H = 40
WIN_SIZE = (BOARD_SIZE, BOARD_SIZE + STATUS_H)

# Colors
LIGHT_SQ = (240, 217, 181)
DARK_SQ = (181, 136, 99)
SELECTED_LIGHT = (186, 202, 68)
SELECTED_DARK = (170, 186, 58)
LAST_MOVE_LIGHT = (205, 210, 106)
LAST_MOVE_DARK = (170, 162, 58)
CHECK_COLOR = (235, 97, 80)
STATUS_BG = (48, 48, 48)
TEXT_COLOR = (220, 220, 220)
BTN_COLOR = (70, 130, 180)
BTN_HOVER = (90, 150, 200)
BTN_TEXT = (255, 255, 255)
WHITE_PIECE_COLOR = (255, 255, 255)
BLACK_PIECE_COLOR = (0, 0, 0)

PIECE_UNICODE = {
    'R': '\u2656', 'N': '\u2658', 'B': '\u2657', 'Q': '\u2655', 'K': '\u2654', 'P': '\u2659',
    'r': '\u265c', 'n': '\u265e', 'b': '\u265d', 'q': '\u265b', 'k': '\u265a', 'p': '\u265f',
}

# --- Model setup ---
MODEL = "2000_elo_pos_engine_best_test_whole.pth"

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = torch.load(f'models/{MODEL}', map_location="cpu").to(device)


class ChessGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(WIN_SIZE)
        pygame.display.set_caption("Chessformer")
        self.clock = pygame.time.Clock()

        # Fonts
        self.piece_font = self._init_piece_font(56)
        self.label_font = pygame.font.SysFont("sans", 14)
        self.status_font = pygame.font.SysFont("sans", 20)
        self.btn_font = pygame.font.SysFont("sans", 24, bold=True)
        self.title_font = pygame.font.SysFont("sans", 48, bold=True)

        # Game state
        self.board = chess.Board()
        self.made_moves = []
        self.selected_sq = None
        self.legal_dests = set()
        self.last_move = None
        self.flipped = False
        self.ai_is_black = True
        self.game_started = False
        self.game_over = False
        self.status_text = "Choose your color"

        # Start screen buttons
        self.white_btn = pygame.Rect(BOARD_SIZE // 2 - 150, 300, 120, 50)
        self.black_btn = pygame.Rect(BOARD_SIZE // 2 + 30, 300, 120, 50)

    def _init_piece_font(self, size):
        for name in ["DejaVu Sans", "Noto Sans Symbols2", "Noto Sans Symbols",
                      "Symbola", "FreeSerif", "Segoe UI Symbol", "Arial Unicode MS"]:
            font = pygame.font.SysFont(name, size)
            if font.get_height() > 0:
                return font
        return pygame.font.SysFont(None, size)

    # --- Coordinate conversion ---

    def rc_to_square(self, row, col):
        if self.flipped:
            return chess.square(7 - col, row)
        return chess.square(col, 7 - row)

    def square_to_rc(self, sq):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        if self.flipped:
            return r, 7 - f
        return 7 - r, f

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
        fg = WHITE_PIECE_COLOR if piece.color == chess.WHITE else BLACK_PIECE_COLOR
        outline = BLACK_PIECE_COLOR if piece.color == chess.WHITE else WHITE_PIECE_COLOR
        # Outline
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    s = self.piece_font.render(sym, True, outline)
                    self.screen.blit(s, s.get_rect(
                        center=(rect.centerx + dx, rect.centery + dy)))
        # Piece
        s = self.piece_font.render(sym, True, fg)
        self.screen.blit(s, s.get_rect(center=rect.center))

    def draw_board(self):
        for row in range(8):
            for col in range(8):
                sq = self.rc_to_square(row, col)
                rect = pygame.Rect(col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE)

                # Square
                pygame.draw.rect(self.screen, self._sq_color(row, col, sq), rect)

                # Legal move indicator
                if sq in self.legal_dests:
                    overlay = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
                    center = (SQ_SIZE // 2, SQ_SIZE // 2)
                    if self.board.piece_at(sq):
                        pygame.draw.circle(overlay, (0, 0, 0, 50),
                                           center, SQ_SIZE // 2 - 2, 6)
                    else:
                        pygame.draw.circle(overlay, (0, 0, 0, 50),
                                           center, SQ_SIZE // 6)
                    self.screen.blit(overlay, rect.topleft)

                # Piece
                piece = self.board.piece_at(sq)
                if piece:
                    self._draw_piece(piece, rect)

        # Coordinate labels
        for i in range(8):
            # File labels (bottom edge)
            file_idx = i if not self.flipped else 7 - i
            label = chr(ord('a') + file_idx)
            color = LIGHT_SQ if i % 2 == 0 else DARK_SQ
            s = self.label_font.render(label, True, color)
            self.screen.blit(s, (i * SQ_SIZE + SQ_SIZE - s.get_width() - 2,
                                 BOARD_SIZE - s.get_height() - 2))
            # Rank labels (left edge)
            rank_idx = i if self.flipped else 7 - i
            label = str(rank_idx + 1)
            color = DARK_SQ if i % 2 == 0 else LIGHT_SQ
            s = self.label_font.render(label, True, color)
            self.screen.blit(s, (2, i * SQ_SIZE + 2))

    def draw_status(self):
        bar = pygame.Rect(0, BOARD_SIZE, BOARD_SIZE, STATUS_H)
        pygame.draw.rect(self.screen, STATUS_BG, bar)
        s = self.status_font.render(self.status_text, True, TEXT_COLOR)
        self.screen.blit(s, s.get_rect(center=bar.center))

    def draw_start_screen(self):
        self.screen.fill((40, 40, 40))
        # Title
        s = self.title_font.render("Chessformer", True, TEXT_COLOR)
        self.screen.blit(s, s.get_rect(center=(BOARD_SIZE // 2, 150)))
        # Subtitle
        s = self.status_font.render("Choose your color", True, TEXT_COLOR)
        self.screen.blit(s, s.get_rect(center=(BOARD_SIZE // 2, 220)))
        # Buttons
        mouse = pygame.mouse.get_pos()
        for btn, label in [(self.white_btn, "White"), (self.black_btn, "Black")]:
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
        if self.ai_is_black:
            return self.board.turn == chess.BLACK
        return self.board.turn == chess.WHITE

    def ai_move(self):
        if self.game_over:
            return
        input_tensors = preprocess(self.board)

        def predict(rep_mv=""):
            with torch.no_grad():
                output = model(*input_tensors)
            return postprocess_valid(output, self.board, rep_mv=rep_mv)

        uci = predict()
        if uci is None:
            # Fallback: pick first legal move
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

        move = chess.Move.from_uci(uci)
        self.board.push(move)
        self.made_moves.append(move.uci())
        self.last_move = move

    def handle_click(self, pos):
        if self.game_over or self.is_ai_turn():
            return False
        col, row = pos[0] // SQ_SIZE, pos[1] // SQ_SIZE
        if not (0 <= row < 8 and 0 <= col < 8):
            return False
        sq = self.rc_to_square(row, col)

        if self.selected_sq is not None:
            if sq in self.legal_dests:
                # Try normal move, then promotion
                move = chess.Move(self.selected_sq, sq)
                if move not in self.board.legal_moves:
                    move = chess.Move(self.selected_sq, sq, promotion=chess.QUEEN)
                if move in self.board.legal_moves:
                    self.board.push(move)
                    self.made_moves.append(move.uci())
                    self.last_move = move
                    self.selected_sq = None
                    self.legal_dests = set()
                    return True
            # Re-select own piece or deselect
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

    # --- Main loop ---

    def run(self):
        running = True
        need_ai_move = False

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if not self.game_started:
                        if self.white_btn.collidepoint(event.pos):
                            self.ai_is_black = True
                            self.flipped = False
                            self.game_started = True
                            self.update_status()
                        elif self.black_btn.collidepoint(event.pos):
                            self.ai_is_black = False
                            self.flipped = True
                            self.game_started = True
                            self.update_status()
                            need_ai_move = True
                    else:
                        if self.handle_click(event.pos):
                            self.update_status()
                            if not self.game_over and self.is_ai_turn():
                                need_ai_move = True

            if not self.game_started:
                self.draw_start_screen()
            else:
                if need_ai_move:
                    self.status_text = "AI is thinking..."
                    self.draw_board()
                    self.draw_status()
                    pygame.display.flip()
                    self.ai_move()
                    self.update_status()
                    need_ai_move = False
                self.draw_board()
                self.draw_status()

            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    gui = ChessGUI()
    gui.run()
