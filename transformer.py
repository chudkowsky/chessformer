"""Compatibility shim for older pickled models.

The released model was saved with a module path named "transformer".
This file re-exports the current implementation from chessformer.py
so torch.load can resolve the class during unpickling.
"""

from chessformer import ChessTransformer, PositionalEncoding  # re-export for torch.load

__all__ = ["ChessTransformer", "PositionalEncoding"]
