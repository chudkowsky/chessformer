"""Monte Carlo Tree Search for ChessTransformerV2.

Inspired by alpha-zero-general (suragnair/alpha-zero-general), but works
directly with chess.Board and our model — no generic Game/NNet adapters needed.

Usage:
    mcts = MCTS(model, device, num_simulations=50, cpuct=1.25)
    visit_policy, root_wdl = mcts.search(board)
    # visit_policy: dict[chess.Move, float] — normalized visit counts
    # root_wdl: (win, draw, loss) probabilities from root forward pass
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import chess
import torch

from model_utils import preprocess_board
from policy import legal_move_policy_v2


@dataclass
class MCTSNode:
    """Single node in the MCTS search tree."""

    board: chess.Board
    parent: MCTSNode | None = None
    move: chess.Move | None = None
    children: dict[chess.Move, MCTSNode] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0

    def q_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def is_expanded(self) -> bool:
        return len(self.children) > 0


class MCTS:
    """AlphaZero-style MCTS using ChessTransformerV2 for priors and value."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        num_simulations: int = 50,
        cpuct: float = 1.25,
        use_diffusion: bool = False,
    ):
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.cpuct = cpuct
        self.use_diffusion = use_diffusion
        self._last_wdl: tuple[float, float, float] = (0.0, 0.5, 0.5)

    def search(
        self, board: chess.Board
    ) -> tuple[dict[chess.Move, float], tuple[float, float, float]]:
        """Run MCTS from position.

        Returns:
            (visit_policy, root_wdl) where visit_policy maps legal moves
            to normalized visit fractions, and root_wdl is (W, D, L) from
            the root node's neural network evaluation.
        """
        root = MCTSNode(board=board.copy())
        root_wdl = self._expand_and_evaluate(root)
        # Convert value back to WDL tuple: value = W - L, and W + D + L = 1
        # We stored the raw WDL from forward pass, so capture it directly
        root_wdl_tuple = self._last_wdl

        for _ in range(self.num_simulations):
            node = root
            # SELECT: traverse tree via UCB until unexpanded leaf
            while node.is_expanded() and not node.board.is_game_over():
                node = self._select_child(node)
            # EXPAND + EVALUATE
            value = self._expand_and_evaluate(node)
            # BACKUP
            self._backup(node, value)

        total = sum(c.visit_count for c in root.children.values())
        if total == 0:
            return {}, root_wdl_tuple
        policy = {
            move: child.visit_count / total
            for move, child in root.children.items()
        }
        return policy, root_wdl_tuple

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Pick child with highest UCB score."""
        total_visits = sum(c.visit_count for c in node.children.values())
        sqrt_total = math.sqrt(total_visits) if total_visits > 0 else 1.0

        best_score = -math.inf
        best_child = None
        for child in node.children.values():
            q = child.q_value()
            exploration = (
                self.cpuct * child.prior * sqrt_total / (1 + child.visit_count)
            )
            score = q + exploration
            if score > best_score:
                best_score = score
                best_child = child
        return best_child  # type: ignore[return-value]

    @torch.no_grad()
    def _expand_and_evaluate(self, node: MCTSNode) -> float:
        """Expand leaf node and return value from current player's perspective."""
        # Terminal position: ground-truth value
        if node.board.is_game_over():
            outcome = node.board.outcome()
            if outcome is None or outcome.winner is None:
                self._last_wdl = (0.0, 1.0, 0.0)
                return 0.0
            # Side to move is checkmated
            self._last_wdl = (0.0, 0.0, 1.0)
            return -1.0

        board_t, feat_t = preprocess_board(node.board, self.device)
        policy_logits, promo_logits, wdl, _ply = self.model(
            board_t, feat_t, use_diffusion=self.use_diffusion
        )

        # Store raw WDL for root node capture
        w, d, l = wdl[0, 0].item(), wdl[0, 1].item(), wdl[0, 2].item()
        self._last_wdl = (w, d, l)

        # Create children with neural network priors
        moves, probs, _log_probs = legal_move_policy_v2(
            node.board, policy_logits[0], promo_logits[0]
        )
        for move, prior in zip(moves, probs):
            child_board = node.board.copy()
            child_board.push(move)
            node.children[move] = MCTSNode(
                board=child_board,
                parent=node,
                move=move,
                prior=prior.item(),
            )

        # Value from current player's perspective: W - L
        return w - l

    @staticmethod
    def _backup(node: MCTSNode, value: float) -> None:
        """Propagate value up the tree, negating at each level."""
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value
            node = node.parent  # type: ignore[assignment]
