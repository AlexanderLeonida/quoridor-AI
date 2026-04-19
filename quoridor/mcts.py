"""AlphaZero-style Monte Carlo Tree Search for Quoridor.

Implements PUCT-based MCTS with neural network guidance following
Silver et al. (2017) and Schrittwieser et al. (2020).

    - PUCT exploration with log-scaling c_puct (MuZero formula)
    - Dirichlet noise at root for exploration diversity
    - First Play Urgency (FPU) reduction for unvisited children
    - Temperature-controlled action selection from visit counts
    - Correct negamax value propagation for two-player zero-sum games

The search is single-threaded.  For Quoridor's 9×9 board with a
6-block / 64-filter net, 400–800 simulations per move runs in roughly
1–5 s on Apple Silicon (MPS) or a mid-range GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ai import zobrist
from .board import Board, Move
from .encoding import (
    ACTION_SPACE,
    action_to_move,
    canonical_view,
    encode_state,
    legal_action_mask,
)


# ======================================================================
# Evaluation cache (transposition table for NN forward passes)
# ======================================================================

class EvalCache:
    """Bounded cache of (policy_logits, value) keyed on Zobrist hash.

    Quoridor has real transpositions (e.g. pawn-then-wall vs wall-then-
    pawn), so within a single MCTS tree — and across successive searches
    in the same game — the same position is evaluated multiple times.
    This cache skips the NN forward on repeats.

    Safety: the Zobrist key fully identifies the board state (pawns,
    walls, walls-left, side-to-move), and ``encode_state`` is a pure
    function of the state, so cache hits are bit-identical to fresh
    forwards. Zero impact on learning.
    """

    __slots__ = ("_store", "_max_size", "hits", "misses")

    def __init__(self, max_size: int = 20_000):
        self._store: Dict[int, Tuple[np.ndarray, float]] = {}
        self._max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, key: int):
        val = self._store.get(key)
        if val is None:
            self.misses += 1
        else:
            self.hits += 1
        return val

    def put(self, key: int, value: Tuple[np.ndarray, float]) -> None:
        if len(self._store) >= self._max_size:
            # FIFO-ish eviction: drop the oldest half in bulk.
            for k in list(self._store.keys())[: self._max_size // 2]:
                del self._store[k]
        self._store[key] = value

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class MCTSConfig:
    """Hyperparameters for MCTS search."""

    num_simulations: int = 800
    """Number of simulations (tree traversals) per search call."""

    # --- PUCT exploration ---
    c_base: float = 19652.0
    c_init: float = 1.25

    # --- Root exploration noise ---
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    # --- First Play Urgency ---
    fpu_reduction: float = 0.25

    # --- Move limits ---
    max_moves: int = 200


# ======================================================================
# Node
# ======================================================================

class Node:
    """A single node in the MCTS tree.

    Each node stores statistics for the *incoming* edge (the action the
    parent took to reach this node).  ``value_sum`` is accumulated from
    the perspective of ``to_play`` — the player whose turn it is at
    this node.
    """

    __slots__ = (
        "visit_count",
        "value_sum",
        "prior",
        "children",
        "to_play",
        "is_terminal",
        "terminal_value",
    )

    def __init__(self, prior: float = 0.0):
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.prior: float = prior
        self.children: Dict[int, "Node"] = {}
        self.to_play: int = -1
        self.is_terminal: bool = False
        self.terminal_value: float = 0.0  # from to_play's perspective

    @property
    def value(self) -> float:
        """Mean value from ``to_play``'s perspective."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def expanded(self) -> bool:
        return len(self.children) > 0


# ======================================================================
# Internal helpers
# ======================================================================

def _ucb_score(parent: Node, child: Node, config: MCTSConfig) -> float:
    """Upper confidence bound for trees (PUCT variant)."""
    # Log-scaling c_puct as in MuZero / KataGo.
    pb_c = (
        math.log((parent.visit_count + config.c_base + 1) / config.c_base)
        + config.c_init
    )
    prior_score = (
        pb_c * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
    )
    if child.visit_count > 0:
        # child.value is from child.to_play's perspective (opponent).
        # Negate to get parent's perspective.
        value_score = -child.value
    else:
        # FPU: parent's estimated value minus a small reduction so that
        # un-visited children are tried but ranked below promising ones.
        value_score = parent.value - config.fpu_reduction
    return value_score + prior_score


def _select_child(node: Node, config: MCTSConfig) -> Tuple[int, "Node"]:
    """Pick the child with the highest UCB score."""
    best_score = -math.inf
    best_action = -1
    best_child: Optional[Node] = None
    for action, child in node.children.items():
        score = _ucb_score(node, child, config)
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
    assert best_child is not None
    return best_action, best_child


def _add_dirichlet_noise(node: Node, config: MCTSConfig) -> None:
    """Mix Dirichlet noise into root priors for exploration."""
    actions = list(node.children.keys())
    if not actions:
        return
    noise = np.random.dirichlet([config.dirichlet_alpha] * len(actions))
    eps = config.dirichlet_epsilon
    for a, eta in zip(actions, noise):
        child = node.children[a]
        child.prior = (1.0 - eps) * child.prior + eps * float(eta)


def _expand(
    node: Node,
    board: Board,
    policy_logits: np.ndarray,
    value: float,
) -> float:
    """Expand a leaf node, creating children with priors from the net.

    Returns the value of this node from ``board.turn``'s perspective.
    For terminal nodes the neural-net value is overridden by the true
    game outcome.
    """
    node.to_play = board.turn

    winner = board.winner()
    if winner is not None:
        node.is_terminal = True
        node.terminal_value = 1.0 if winner == board.turn else -1.0
        return node.terminal_value

    # Check for legal moves (should always exist if no winner).
    mask = legal_action_mask(board)
    if not mask.any():
        # Stalemate (shouldn't happen in Quoridor, but be safe).
        node.is_terminal = True
        node.terminal_value = 0.0
        return 0.0

    # Masked softmax → priors.
    logits = policy_logits.copy()
    logits[~mask] = -1e9
    logits -= logits.max()
    exp_l = np.exp(logits)
    priors = exp_l / exp_l.sum()

    for idx in range(ACTION_SPACE):
        if mask[idx]:
            node.children[idx] = Node(prior=float(priors[idx]))

    return value


def _evaluate(
    board: Board,
    net,
    device,
    cache: Optional[EvalCache] = None,
) -> Tuple[np.ndarray, float]:
    """Run the neural network on *board* and return (policy_logits, value).

    If *cache* is provided, transposed positions reuse prior forwards.
    """
    if cache is not None:
        key = zobrist(board)
        hit = cache.get(key)
        if hit is not None:
            return hit

    import torch

    state = encode_state(board)  # canonical (7, 9, 9)
    state_t = torch.from_numpy(state).unsqueeze(0).to(device)
    with torch.no_grad():
        p_logits, v = net(state_t)
    result = (p_logits.squeeze(0).cpu().numpy(), float(v.item()))

    if cache is not None:
        cache.put(key, result)
    return result


def _backup(
    search_path: List[Node],
    value: float,
    to_play: int,
) -> None:
    """Propagate *value* (from *to_play*'s perspective) up the path."""
    for node in reversed(search_path):
        node.value_sum += value if node.to_play == to_play else -value
        node.visit_count += 1


# ======================================================================
# Public API
# ======================================================================

def search(
    board: Board,
    net,
    config: MCTSConfig,
    device,
    *,
    add_noise: bool = True,
    cache: Optional[EvalCache] = None,
    reuse_root: Optional[Node] = None,
) -> Node:
    """Run a full MCTS search from *board*.  Returns the root ``Node``.

    Parameters
    ----------
    board : Board
        Current game state (not mutated).
    net : torch.nn.Module
        Policy/value network in eval mode.
    config : MCTSConfig
        Search hyper-parameters.
    device : torch.device
        Where *net* lives.
    add_noise : bool
        Whether to mix Dirichlet noise into the root priors (True for
        self-play, False for evaluation / competitive play).
    reuse_root : Node, optional
        A previously-searched node (typically the child corresponding to
        the last move played) whose accumulated statistics should be
        carried forward as the new root.  This is the canonical
        AlphaZero subtree-reuse optimisation: visits already invested
        under this node become "free" simulations, so the target total
        visit count is reached faster.
    """
    # --- establish root (fresh or inherited) ---
    if (
        reuse_root is not None
        and reuse_root.expanded
        and not reuse_root.is_terminal
    ):
        # Subtree reuse: node was expanded during the previous search.
        # Its priors are already the NN's (correct) predictions; its
        # visit counts and value sums are still valid statistics for
        # this position, so we just reseed Dirichlet noise at the new
        # root for exploration.
        root = reuse_root
        if add_noise:
            _add_dirichlet_noise(root, config)
    else:
        root = Node()
        policy_logits, value = _evaluate(board, net, device, cache=cache)
        value = _expand(root, board, policy_logits, value)

        if root.is_terminal:
            root.visit_count = 1
            root.value_sum = root.terminal_value
            return root

        root.visit_count = 1
        root.value_sum = value

        if add_noise:
            _add_dirichlet_noise(root, config)

    # --- simulations ---
    # Target the same TOTAL visit count whether the tree is fresh or
    # reused — this is where the speedup comes from.  A fresh search
    # ends with 1 + num_simulations visits, so that's the target.
    target_visits = 1 + config.num_simulations
    remaining = max(0, target_visits - root.visit_count)
    for _ in range(remaining):
        node = root
        scratch = board  # Board.apply returns new objects; *board* is safe.
        path: List[Node] = [root]

        # SELECT
        while node.expanded and not node.is_terminal:
            action, child = _select_child(node, config)
            _, _, _, _, _, _, flipped = canonical_view(scratch)
            move = action_to_move(action, flipped)
            scratch = scratch.apply(move)
            path.append(child)
            node = child

        # EXPAND & EVALUATE
        if node.is_terminal:
            leaf_value = node.terminal_value
        else:
            p_logits, v = _evaluate(scratch, net, device, cache=cache)
            leaf_value = _expand(node, scratch, p_logits, v)
            if node.is_terminal:
                leaf_value = node.terminal_value

        # BACKUP
        _backup(path, leaf_value, node.to_play)

    return root


def get_policy(root: Node, temperature: float = 1.0) -> np.ndarray:
    """Extract the action-probability distribution from visit counts.

    With ``temperature == 0`` the distribution is a one-hot on the most
    visited action.  With ``temperature == 1`` probabilities are
    proportional to visit counts.
    """
    policy = np.zeros(ACTION_SPACE, dtype=np.float32)
    if not root.children:
        return policy

    actions = np.array(list(root.children.keys()), dtype=np.int64)
    counts = np.array(
        [root.children[a].visit_count for a in actions], dtype=np.float64
    )

    if temperature == 0:
        best = actions[np.argmax(counts)]
        policy[best] = 1.0
    elif temperature == 1.0:
        total = counts.sum()
        if total > 0:
            for a, c in zip(actions, counts):
                policy[a] = c / total
    else:
        log_c = np.log(counts + 1e-10) / temperature
        log_c -= log_c.max()
        exp_c = np.exp(log_c)
        total = exp_c.sum()
        for a, e in zip(actions, exp_c):
            policy[a] = e / total

    return policy


def select_action(root: Node, temperature: float = 1.0) -> int:
    """Sample an action from the visit-count policy."""
    policy = get_policy(root, temperature)
    if temperature == 0:
        return int(np.argmax(policy))
    return int(np.random.choice(ACTION_SPACE, p=policy))


def root_value(root: Node) -> float:
    """MCTS value estimate of the root position (side-to-move's POV)."""
    return root.value
