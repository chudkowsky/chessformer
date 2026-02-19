"""
Opening book for self-play data generation.

Each entry is a tuple: (eco_code, name, fen, moves_uci)
- eco_code:   ECO classification string
- name:       Human-readable opening name
- fen:        Starting FEN (always the standard starting position here;
              moves_uci is applied on top of it)
- moves_uci:  List of UCI moves to reach the opening position

The standard starting FEN is used as the base for all openings.
moves_uci lists the moves that define the opening line; after these
are played the engine takes over with temperature sampling.
"""

import random

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# fmt: off
OPENINGS: list[tuple[str, str, str, list[str]]] = [
    # ------------------------------------------------------------------ #
    # Open Games (1. e4 e5)
    # ------------------------------------------------------------------ #
    ("C20", "King's Pawn Game",              START_FEN, ["e2e4", "e7e5"]),
    ("C21", "Center Game",                   START_FEN, ["e2e4", "e7e5", "d2d4", "e5d4"]),
    ("C23", "Bishop's Opening",              START_FEN, ["e2e4", "e7e5", "f1c4"]),
    ("C25", "Vienna Game",                   START_FEN, ["e2e4", "e7e5", "b1c3"]),
    ("C30", "King's Gambit",                 START_FEN, ["e2e4", "e7e5", "f2f4"]),
    ("C31", "King's Gambit Declined",        START_FEN, ["e2e4", "e7e5", "f2f4", "d7d5"]),
    ("C33", "King's Gambit Accepted",        START_FEN, ["e2e4", "e7e5", "f2f4", "e5f4"]),
    ("C40", "Latvian Gambit",                START_FEN, ["e2e4", "e7e5", "g1f3", "f7f5"]),
    ("C41", "Philidor Defense",              START_FEN, ["e2e4", "e7e5", "g1f3", "d7d6"]),
    ("C42", "Petrov Defense",                START_FEN, ["e2e4", "e7e5", "g1f3", "g8f6"]),
    ("C44", "Scotch Game",                   START_FEN, ["e2e4", "e7e5", "g1f3", "b8c6", "d2d4"]),
    ("C45", "Scotch Game: main line",        START_FEN, ["e2e4", "e7e5", "g1f3", "b8c6", "d2d4", "e5d4", "f3d4"]),
    ("C46", "Three Knights Game",            START_FEN, ["e2e4", "e7e5", "g1f3", "b8c6", "b1c3"]),
    ("C47", "Four Knights Game",             START_FEN, ["e2e4", "e7e5", "g1f3", "b8c6", "b1c3", "g8f6"]),
    ("C50", "Italian Game",                  START_FEN, ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"]),
    ("C51", "Evans Gambit",                  START_FEN, ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "b2b4"]),
    ("C53", "Italian: Classical",            START_FEN, ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3"]),
    ("C54", "Italian: Giuoco Piano",         START_FEN, ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3", "g8f6", "d2d4"]),
    ("C55", "Two Knights Defense",           START_FEN, ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"]),
    ("C56", "Two Knights: Fried Liver",      START_FEN, ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "f3g5", "d7d5", "e4d5", "c6a5"]),
    ("C60", "Ruy Lopez",                     START_FEN, ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]),
    ("C61", "Ruy Lopez: Bird Defense",       START_FEN, ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "c6d4"]),
    ("C62", "Ruy Lopez: Steinitz Defense",   START_FEN, ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "d7d6"]),
    ("C63", "Ruy Lopez: Schliemann",         START_FEN, ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "f7f5"]),
    ("C65", "Ruy Lopez: Berlin Defense",     START_FEN, ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "g8f6"]),
    ("C67", "Ruy Lopez: Berlin Endgame",     START_FEN, ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "g8f6", "e1g1", "f6e4", "d1e1", "e4d6", "f3e5", "c6e5"]),
    ("C68", "Ruy Lopez: Exchange",           START_FEN, ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5c6"]),
    ("C70", "Ruy Lopez: Morphy Defense",     START_FEN, ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4"]),
    ("C78", "Ruy Lopez: Archangel",          START_FEN, ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8b4"]),
    ("C80", "Ruy Lopez: Open",               START_FEN, ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f6e4"]),
    ("C84", "Ruy Lopez: Closed",             START_FEN, ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7"]),
    ("C88", "Ruy Lopez: Anti-Marshall",      START_FEN, ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7", "f1e1", "b7b5", "a4b3", "e8g8"]),
    ("C92", "Ruy Lopez: Marshall Attack",    START_FEN, ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7", "f1e1", "b7b5", "a4b3", "e8g8", "c2c3", "d7d5"]),

    # ------------------------------------------------------------------ #
    # Sicilian Defense (1. e4 c5)
    # ------------------------------------------------------------------ #
    ("B20", "Sicilian Defense",              START_FEN, ["e2e4", "c7c5"]),
    ("B21", "Sicilian: Grand Prix Attack",   START_FEN, ["e2e4", "c7c5", "b1c3", "b8c6", "f2f4"]),
    ("B22", "Sicilian: Alapin",              START_FEN, ["e2e4", "c7c5", "c2c3"]),
    ("B23", "Sicilian: Closed",              START_FEN, ["e2e4", "c7c5", "b1c3"]),
    ("B27", "Sicilian: Hungarian",           START_FEN, ["e2e4", "c7c5", "g1f3", "g7g6"]),
    ("B30", "Sicilian: Old Sicilian",        START_FEN, ["e2e4", "c7c5", "g1f3", "b8c6"]),
    ("B32", "Sicilian: Löwenthal",           START_FEN, ["e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4", "f3d4", "e7e5"]),
    ("B40", "Sicilian: Kan",                 START_FEN, ["e2e4", "c7c5", "g1f3", "e7e6"]),
    ("B41", "Sicilian: Kan main",            START_FEN, ["e2e4", "c7c5", "g1f3", "e7e6", "d2d4", "c5d4", "f3d4", "a7a6"]),
    ("B44", "Sicilian: Taimanov",            START_FEN, ["e2e4", "c7c5", "g1f3", "e7e6", "d2d4", "c5d4", "f3d4", "b8c6"]),
    ("B50", "Sicilian: 2.Nf3 d6",            START_FEN, ["e2e4", "c7c5", "g1f3", "d7d6"]),
    ("B54", "Sicilian: Dragon",              START_FEN, ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "g7g6"]),
    ("B57", "Sicilian: Classical",           START_FEN, ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "b8c6"]),
    ("B58", "Sicilian: Boleslavsky",         START_FEN, ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "e7e5"]),
    ("B60", "Sicilian: Richter-Rauzer",      START_FEN, ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "b8c6", "c1g5"]),
    ("B70", "Sicilian: Dragon main",         START_FEN, ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "g7g6", "f1e2", "f8g7"]),
    ("B72", "Sicilian: Scheveningen",        START_FEN, ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "e7e6"]),
    ("B80", "Sicilian: Najdorf",             START_FEN, ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"]),
    ("B85", "Sicilian: Najdorf 6.Be2",       START_FEN, ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6", "f1e2", "e7e6"]),
    ("B90", "Sicilian: Najdorf 6.Be3",       START_FEN, ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6", "c1e3"]),
    ("B96", "Sicilian: Najdorf Poisoned Pawn", START_FEN, ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6", "c1g5", "e7e6", "f2f4", "d8b6"]),

    # ------------------------------------------------------------------ #
    # French Defense (1. e4 e6)
    # ------------------------------------------------------------------ #
    ("C00", "French Defense",                START_FEN, ["e2e4", "e7e6"]),
    ("C01", "French: Exchange",              START_FEN, ["e2e4", "e7e6", "d2d4", "d7d5", "e4d5"]),
    ("C02", "French: Advance",               START_FEN, ["e2e4", "e7e6", "d2d4", "d7d5", "e4e5"]),
    ("C03", "French: Tarrasch",              START_FEN, ["e2e4", "e7e6", "d2d4", "d7d5", "b1d2"]),
    ("C10", "French: Rubinstein",            START_FEN, ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "d5e4"]),
    ("C11", "French: Classical",             START_FEN, ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6"]),
    ("C14", "French: Classical Winawer",     START_FEN, ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "f8b4"]),
    ("C18", "French: Winawer Poisoned Pawn", START_FEN, ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "f8b4", "e4e5", "c7c5", "a2a3", "b4c3", "b2c3", "d8c7"]),

    # ------------------------------------------------------------------ #
    # Caro-Kann (1. e4 c6)
    # ------------------------------------------------------------------ #
    ("B10", "Caro-Kann Defense",             START_FEN, ["e2e4", "c7c6"]),
    ("B12", "Caro-Kann: Advance",            START_FEN, ["e2e4", "c7c6", "d2d4", "d7d5", "e4e5"]),
    ("B13", "Caro-Kann: Exchange",           START_FEN, ["e2e4", "c7c6", "d2d4", "d7d5", "e4d5"]),
    ("B14", "Caro-Kann: Panov Attack",       START_FEN, ["e2e4", "c7c6", "d2d4", "d7d5", "e4d5", "c6d5", "c2c4"]),
    ("B15", "Caro-Kann: Classical",          START_FEN, ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4", "g8f6"]),
    ("B17", "Caro-Kann: Steinitz",           START_FEN, ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4", "b8d7"]),

    # ------------------------------------------------------------------ #
    # Pirc / Modern Defense
    # ------------------------------------------------------------------ #
    ("B06", "Modern Defense",                START_FEN, ["e2e4", "g7g6"]),
    ("B07", "Pirc Defense",                  START_FEN, ["e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6"]),
    ("B08", "Pirc: Classical",               START_FEN, ["e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6", "g1f3"]),
    ("B09", "Pirc: Austrian Attack",         START_FEN, ["e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6", "f2f4"]),

    # ------------------------------------------------------------------ #
    # Scandinavian (1. e4 d5)
    # ------------------------------------------------------------------ #
    ("B01", "Scandinavian Defense",          START_FEN, ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5"]),
    ("B01b", "Scandinavian: 3...Qd6",        START_FEN, ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5d6"]),

    # ------------------------------------------------------------------ #
    # Alekhine's Defense
    # ------------------------------------------------------------------ #
    ("B02", "Alekhine's Defense",            START_FEN, ["e2e4", "g8f6"]),
    ("B04", "Alekhine: Modern Variation",    START_FEN, ["e2e4", "g8f6", "e4e5", "f6d5", "d2d4", "d7d6", "g1f3"]),

    # ------------------------------------------------------------------ #
    # Queen's Gambit (1. d4 d5 2. c4)
    # ------------------------------------------------------------------ #
    ("D00", "Queen's Pawn Game",             START_FEN, ["d2d4", "d7d5"]),
    ("D01", "Richter-Veresov Attack",        START_FEN, ["d2d4", "d7d5", "b1c3", "g8f6", "c1g5"]),
    ("D02", "London System",                 START_FEN, ["d2d4", "d7d5", "g1f3", "g8f6", "c1f4"]),
    ("D06", "Queen's Gambit",                START_FEN, ["d2d4", "d7d5", "c2c4"]),
    ("D07", "QGD: Chigorin Defense",         START_FEN, ["d2d4", "d7d5", "c2c4", "b8c6"]),
    ("D10", "QGD: Slav",                     START_FEN, ["d2d4", "d7d5", "c2c4", "c7c6"]),
    ("D12", "QGD: Slav main line",           START_FEN, ["d2d4", "d7d5", "c2c4", "c7c6", "g1f3", "g8f6", "b1c3", "c8f5"]),
    ("D15", "QGD: Slav Gambit",              START_FEN, ["d2d4", "d7d5", "c2c4", "c7c6", "b1c3", "g8f6", "g1f3", "d5c4"]),
    ("D20", "QGA",                           START_FEN, ["d2d4", "d7d5", "c2c4", "d5c4"]),
    ("D25", "QGA: Classical",                START_FEN, ["d2d4", "d7d5", "c2c4", "d5c4", "g1f3", "g8f6", "e2e3", "e7e6"]),
    ("D30", "QGD",                           START_FEN, ["d2d4", "d7d5", "c2c4", "e7e6"]),
    ("D31", "QGD: Semi-Slav",                START_FEN, ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "c7c6"]),
    ("D32", "QGD: Tarrasch",                 START_FEN, ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "c7c5"]),
    ("D35", "QGD: Exchange Variation",       START_FEN, ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c4d5"]),
    ("D37", "QGD: 4.Nf3",                    START_FEN, ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "g1f3"]),
    ("D41", "QGD: Semi-Tarrasch",            START_FEN, ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "g1f3", "c7c5", "c4d5"]),
    ("D43", "QGD: Semi-Slav main",           START_FEN, ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "g1f3", "c7c6"]),
    ("D45", "QGD: Semi-Slav Botvinnik",      START_FEN, ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "g1f3", "c7c6", "e2e3", "b8d7", "f1d3", "d5c4", "d3c4", "b7b5"]),
    ("D50", "QGD: Semi-Slav Meran",          START_FEN, ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "g1f3", "c7c6", "e2e3", "b8d7", "f1d3", "d5c4", "d3c4", "b7b5", "c4d3"]),

    # ------------------------------------------------------------------ #
    # King's Indian Defense (1. d4 Nf6 2. c4 g6 3. Nc3 Bg7)
    # ------------------------------------------------------------------ #
    ("E60", "King's Indian Defense",         START_FEN, ["d2d4", "g8f6", "c2c4", "g7g6"]),
    ("E61", "KID: 3.Nc3",                    START_FEN, ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7"]),
    ("E62", "KID: Fianchetto",               START_FEN, ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "g1f3", "e8g8", "g2g3"]),
    ("E67", "KID: Fianchetto with ...d6",    START_FEN, ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "g1f3", "e8g8", "g2g3", "d7d6", "f1g2", "b8d7"]),
    ("E70", "KID: Averbakh",                 START_FEN, ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6"]),
    ("E73", "KID: Averbakh Variation",       START_FEN, ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6", "c1e3"]),
    ("E76", "KID: Four Pawns Attack",        START_FEN, ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6", "f2f4"]),
    ("E80", "KID: Sämisch",                  START_FEN, ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6", "f2f3"]),
    ("E84", "KID: Sämisch Panno",            START_FEN, ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6", "f2f3", "b8c6", "c1e3", "a7a6"]),
    ("E90", "KID: Classical",                START_FEN, ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6", "g1f3"]),
    ("E92", "KID: Classical with ...e5",     START_FEN, ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6", "g1f3", "e8g8", "f1e2", "e7e5"]),
    ("E97", "KID: Mar del Plata",            START_FEN, ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6", "g1f3", "e8g8", "f1e2", "e7e5", "e1g1", "b8c6"]),

    # ------------------------------------------------------------------ #
    # Nimzo-Indian (1. d4 Nf6 2. c4 e6 3. Nc3 Bb4)
    # ------------------------------------------------------------------ #
    ("E20", "Nimzo-Indian Defense",          START_FEN, ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4"]),
    ("E21", "Nimzo-Indian: 4.Nf3",           START_FEN, ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "g1f3"]),
    ("E32", "Nimzo-Indian: Classical",       START_FEN, ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "d1c2"]),
    ("E40", "Nimzo-Indian: Rubinstein",      START_FEN, ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "e2e3"]),
    ("E41", "Nimzo-Indian: Huebner",         START_FEN, ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "e2e3", "c7c5"]),
    ("E43", "Nimzo-Indian: Fischer",         START_FEN, ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "e2e3", "b7b6"]),
    ("E46", "Nimzo-Indian: Reshevsky",       START_FEN, ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "e2e3", "e8g8", "g1e2"]),
    ("E47", "Nimzo-Indian: 4.e3 d5",         START_FEN, ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "e2e3", "e8g8", "f1d3", "d7d5"]),
    ("E52", "Nimzo-Indian: Spassky",         START_FEN, ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "e2e3", "e8g8", "f1d3", "d7d5", "g1f3", "c7c5"]),
    ("E60b", "Nimzo-Indian: Samisch",        START_FEN, ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "a2a3"]),

    # ------------------------------------------------------------------ #
    # Queen's Indian (1. d4 Nf6 2. c4 e6 3. Nf3 b6)
    # ------------------------------------------------------------------ #
    ("E12", "Queen's Indian Defense",        START_FEN, ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6"]),
    ("E14", "Queen's Indian: 4.e3",          START_FEN, ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6", "e2e3", "c8b7"]),
    ("E15", "Queen's Indian: Petrosian",     START_FEN, ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6", "g2g3"]),
    ("E17", "Queen's Indian: Classical",     START_FEN, ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6", "g2g3", "c8b7", "f1g2", "f8e7"]),

    # ------------------------------------------------------------------ #
    # Catalan (1. d4 Nf6 2. c4 e6 3. Nf3 d5 4. g3)
    # ------------------------------------------------------------------ #
    ("E00", "Catalan Opening",               START_FEN, ["d2d4", "g8f6", "c2c4", "e7e6", "g2g3"]),
    ("E01", "Catalan: Closed",               START_FEN, ["d2d4", "g8f6", "c2c4", "e7e6", "g2g3", "d7d5", "f1g2", "f8e7"]),
    ("E06", "Catalan: Open",                 START_FEN, ["d2d4", "g8f6", "c2c4", "e7e6", "g2g3", "d7d5", "g1f3", "d5c4", "f1g2"]),

    # ------------------------------------------------------------------ #
    # Grünfeld Defense (1. d4 Nf6 2. c4 g6 3. Nc3 d5)
    # ------------------------------------------------------------------ #
    ("D80", "Grünfeld Defense",              START_FEN, ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5"]),
    ("D85", "Grünfeld: Exchange",            START_FEN, ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5", "c4d5", "f6d5", "e2e4", "d5c3", "b2c3", "f8g7"]),
    ("D87", "Grünfeld: Russian",             START_FEN, ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5", "c4d5", "f6d5", "e2e4", "d5c3", "b2c3", "f8g7", "f1c4"]),
    ("D94", "Grünfeld: 4.e3",                START_FEN, ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5", "e2e3"]),
    ("D97", "Grünfeld: Russian main line",   START_FEN, ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5", "g1f3", "f8g7", "d1b3"]),

    # ------------------------------------------------------------------ #
    # English Opening (1. c4)
    # ------------------------------------------------------------------ #
    ("A10", "English Opening",               START_FEN, ["c2c4"]),
    ("A15", "English: Anglo-Indian",         START_FEN, ["c2c4", "g8f6"]),
    ("A17", "English: Anglo-Indian 2.Nc3",   START_FEN, ["c2c4", "g8f6", "b1c3", "e7e6"]),
    ("A20", "English: 1...e5",               START_FEN, ["c2c4", "e7e5"]),
    ("A25", "English: Closed",               START_FEN, ["c2c4", "e7e5", "b1c3", "b8c6", "g2g3", "g7g6"]),
    ("A29", "English: Four Knights",         START_FEN, ["c2c4", "e7e5", "b1c3", "b8c6", "g1f3", "g8f6"]),
    ("A30", "English: Symmetrical",          START_FEN, ["c2c4", "c7c5"]),
    ("A34", "English: Symmetrical 2.Nc3",    START_FEN, ["c2c4", "c7c5", "b1c3", "g8f6", "g2g3"]),

    # ------------------------------------------------------------------ #
    # Reti Opening (1. Nf3)
    # ------------------------------------------------------------------ #
    ("A04", "Reti Opening",                  START_FEN, ["g1f3"]),
    ("A05", "Reti: 1...Nf6",                 START_FEN, ["g1f3", "g8f6"]),
    ("A06", "Reti: 1...d5",                  START_FEN, ["g1f3", "d7d5"]),
    ("A07", "King's Indian Attack",          START_FEN, ["g1f3", "d7d5", "g2g3"]),
    ("A08", "KIA with ...e5",                START_FEN, ["g1f3", "d7d5", "g2g3", "g8f6", "f1g2", "c7c5"]),

    # ------------------------------------------------------------------ #
    # Bird's Opening (1. f4)
    # ------------------------------------------------------------------ #
    ("A02", "Bird's Opening",                START_FEN, ["f2f4"]),
    ("A03", "Bird's: 1...d5",                START_FEN, ["f2f4", "d7d5"]),

    # ------------------------------------------------------------------ #
    # Dutch Defense (1. d4 f5)
    # ------------------------------------------------------------------ #
    ("A80", "Dutch Defense",                 START_FEN, ["d2d4", "f7f5"]),
    ("A84", "Dutch: 2.c4",                   START_FEN, ["d2d4", "f7f5", "c2c4"]),
    ("A87", "Dutch: Leningrad",              START_FEN, ["d2d4", "f7f5", "c2c4", "g8f6", "g2g3", "g7g6"]),
    ("A90", "Dutch: Classical",              START_FEN, ["d2d4", "f7f5", "c2c4", "g8f6", "g2g3", "e7e6", "f1g2", "f8e7"]),
    ("A92", "Dutch: Stonewall",              START_FEN, ["d2d4", "f7f5", "c2c4", "g8f6", "g2g3", "e7e6", "f1g2", "f8e7", "g1f3", "e8g8", "e1g1", "d7d5"]),

    # ------------------------------------------------------------------ #
    # Benoni Defense (1. d4 Nf6 2. c4 c5)
    # ------------------------------------------------------------------ #
    ("A56", "Benoni Defense",                START_FEN, ["d2d4", "g8f6", "c2c4", "c7c5"]),
    ("A60", "Modern Benoni",                 START_FEN, ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6", "b1c3", "e6d5", "c4d5", "d7d6"]),
    ("A65", "Benoni: 6.e4",                  START_FEN, ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6", "b1c3", "e6d5", "c4d5", "d7d6", "e2e4", "g7g6"]),
    ("A70", "Benoni: Classical",             START_FEN, ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6", "b1c3", "e6d5", "c4d5", "d7d6", "e2e4", "g7g6", "g1f3"]),

    # ------------------------------------------------------------------ #
    # Budapest Gambit
    # ------------------------------------------------------------------ #
    ("A51", "Budapest Gambit",               START_FEN, ["d2d4", "g8f6", "c2c4", "e7e5"]),

    # ------------------------------------------------------------------ #
    # Trompowsky Attack (1. d4 Nf6 2. Bg5)
    # ------------------------------------------------------------------ #
    ("A45", "Trompowsky Attack",             START_FEN, ["d2d4", "g8f6", "c1g5"]),

    # ------------------------------------------------------------------ #
    # Torre Attack (1. d4 Nf6 2. Nf3 e6 3. Bg5)
    # ------------------------------------------------------------------ #
    ("A46", "Torre Attack",                  START_FEN, ["d2d4", "g8f6", "g1f3", "e7e6", "c1g5"]),

    # ------------------------------------------------------------------ #
    # Bogo-Indian (1. d4 Nf6 2. c4 e6 3. Nf3 Bb4+)
    # ------------------------------------------------------------------ #
    ("E11", "Bogo-Indian Defense",           START_FEN, ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "f8b4"]),

    # ------------------------------------------------------------------ #
    # Miscellaneous
    # ------------------------------------------------------------------ #
    ("A00", "Grob's Attack",                 START_FEN, ["g2g4"]),
    ("A00b", "Sokolsky Opening",             START_FEN, ["b2b4"]),
    ("A00c", "Van't Kruijs Opening",         START_FEN, ["e2e3"]),
    ("A01", "Nimzowitsch-Larsen Attack",     START_FEN, ["b2b3"]),
    ("A40", "Modern Defense: 1.d4 g6",      START_FEN, ["d2d4", "g7g6"]),
    ("A41", "Old Indian Defense",            START_FEN, ["d2d4", "g7g6", "c2c4", "f8g7", "b1c3", "d7d6"]),
    ("A43", "Old Benoni",                    START_FEN, ["d2d4", "c7c5"]),
]
# fmt: on


def load_openings() -> list[tuple[str, str, str, list[str]]]:
    """Return the full list of (eco, name, fen, moves_uci) tuples."""
    return OPENINGS


def sample_opening(rng: random.Random | None = None) -> tuple[str, str, str, list[str]]:
    """Return one randomly chosen opening entry."""
    r = rng or random
    return r.choice(OPENINGS)
