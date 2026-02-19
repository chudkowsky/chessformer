"""Monte Carlo Tree Search for ChessTransformerV2.

Inspired by alpha-zero-general (suragnair/alpha-zero-general), but works
directly with chess.Board and our model — no generic Game/NNet adapters needed.

Uses virtual-loss batching: collects multiple leaves per batch, evaluates them
in a single forward pass (~5-7x faster than one-at-a-time with 25+ sims).

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

from model_utils import preprocess_board, preprocess_boards_batch
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
    """AlphaZero-style MCTS using ChessTransformerV2 for priors and value.

    Virtual-loss batching: instead of evaluating one leaf at a time,
    collects `batch_size` leaves per round using virtual losses to diversify
    selection, then evaluates all in one forward pass. Cuts GPU round-trips
    from num_simulations down to num_simulations / batch_size.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        num_simulations: int = 50,
        cpuct: float = 1.25,
        use_diffusion: bool = False,
        batch_size: int = 8,
    ):
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.cpuct = cpuct
        self.use_diffusion = use_diffusion
        self.batch_size = batch_size
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
        self._expand_and_evaluate(root)
        root_wdl_tuple = self._last_wdl

        remaining = self.num_simulations
        while remaining > 0:
            batch = min(self.batch_size, remaining)

            # SELECT: collect leaves, applying virtual loss to diversify
            leaves: list[MCTSNode] = []
            for _ in range(batch):
                node = root
                while node.is_expanded() and not node.board.is_game_over():
                    node = self._select_child(node)
                # Virtual loss: discourage other sims from picking same node
                node.visit_count += 1
                node.value_sum -= 1.0
                leaves.append(node)

            # EXPAND + EVALUATE (batched)
            value_by_id: dict[int, float] = {}
            to_expand: list[MCTSNode] = []
            seen: set[int] = set()

            for node in leaves:
                nid = id(node)
                if nid in seen:
                    continue
                seen.add(nid)
                if node.board.is_game_over():
                    outcome = node.board.outcome()
                    value_by_id[nid] = (
                        0.0 if (outcome is None or outcome.winner is None) else -1.0
                    )
                elif node.is_expanded():
                    # Already expanded (by previous batch)
                    value_by_id[nid] = node.q_value()
                else:
                    to_expand.append(node)

            if to_expand:
                values = self._batch_expand_and_evaluate(to_expand)
                for node, val in zip(to_expand, values):
                    value_by_id[id(node)] = val

            # UNDO virtual loss + BACKUP
            for node in leaves:
                node.visit_count -= 1
                node.value_sum += 1.0
                self._backup(node, value_by_id[id(node)])

            remaining -= batch

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

    def _expand_node(
        self,
        node: MCTSNode,
        policy_logits: torch.Tensor,
        promo_logits: torch.Tensor,
    ) -> None:
        """Create child nodes with neural network priors."""
        moves, probs, _log_probs = legal_move_policy_v2(
            node.board, policy_logits, promo_logits
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

    @torch.no_grad()
    def _expand_and_evaluate(self, node: MCTSNode) -> float:
        """Expand single leaf node (used for root). Returns value W - L."""
        if node.board.is_game_over():
            outcome = node.board.outcome()
            if outcome is None or outcome.winner is None:
                self._last_wdl = (0.0, 1.0, 0.0)
                return 0.0
            self._last_wdl = (0.0, 0.0, 1.0)
            return -1.0

        board_t, feat_t = preprocess_board(node.board, self.device)
        policy_logits, promo_logits, wdl, _ply = self.model(
            board_t, feat_t, use_diffusion=self.use_diffusion
        )

        w, d, l = wdl[0, 0].item(), wdl[0, 1].item(), wdl[0, 2].item()
        self._last_wdl = (w, d, l)
        self._expand_node(node, policy_logits[0], promo_logits[0])
        return w - l

    @torch.no_grad()
    def _batch_expand_and_evaluate(
        self, nodes: list[MCTSNode]
    ) -> list[float]:
        """Expand multiple leaf nodes in a single batched forward pass."""
        if len(nodes) == 1:
            return [self._expand_and_evaluate(nodes[0])]

        boards = [node.board for node in nodes]
        boards_t, feats_t = preprocess_boards_batch(boards, self.device)

        policy_logits, promo_logits, wdl, _ply = self.model(
            boards_t, feats_t, use_diffusion=self.use_diffusion
        )

        values = []
        for i, node in enumerate(nodes):
            w, l = wdl[i, 0].item(), wdl[i, 2].item()
            self._expand_node(node, policy_logits[i], promo_logits[i])
            values.append(w - l)

        return values

    @staticmethod
    def _backup(node: MCTSNode, value: float) -> None:
        """Propagate value up the tree, negating at each level."""
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value
            node = node.parent  # type: ignore[assignment]
