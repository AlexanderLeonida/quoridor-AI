"""Strong Quoridor AI.

Search
------
Iterative-deepening negamax with:
  * alpha-beta + Principal Variation Search (PVS, a.k.a. NegaScout),
  * Zobrist-hashed transposition table with EXACT / LOWER / UPPER bounds,
  * killer-move heuristic (two slots per ply),
  * move ordering:  TT best  ->  killer moves  ->  pawn moves that advance
    toward the goal  ->  walls ordered by how much they lengthen the
    opponent's shortest path (computed once per node).

Wall pruning
------------
Only wall anchors adjacent to a cell on *either* player's current shortest
path are considered -- the standard practical pruning for Quoridor. This
keeps the effective branching factor small enough for deep search.

Evaluation (from the side-to-move's perspective)
------------------------------------------------
    f = (opp_path - my_path) * W_PATH
      + (my_walls - opp_walls) * W_WALL
      + (my_mobility - opp_mobility) * W_MOBILITY
      + (my_advance - opp_advance) * W_ADV
      + tempo                                       (+1 side-to-move bonus)

Terminal states score +/- (WIN_SCORE - ply) so shorter wins beat longer ones.

Public API (unchanged):  `find_best_move(board, max_depth, time_limit)`
"""

from __future__ import annotations

import random
import time
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from .board import (
    BOARD_SIZE,
    Board,
    INITIAL_WALLS,
    MOVE_PAWN,
    Move,
    WALL_GRID,
    WALL_H,
    WALL_V,
)

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
INF = 10 ** 9
WIN_SCORE = 10 ** 7

W_PATH = 100
W_WALL = 6
W_MOBILITY = 2
W_ADV = 1

# TT bound flags
EXACT = 0
LOWER = 1
UPPER = 2

MAX_PLY = 64

# ----------------------------------------------------------------------
# Zobrist hashing
# ----------------------------------------------------------------------
_rng = random.Random(0xC0FFEE)
_Z_PAWN = [
    [_rng.getrandbits(64) for _ in range(BOARD_SIZE * BOARD_SIZE)]
    for _ in range(2)
]
_Z_WALL_H = [
    [_rng.getrandbits(64) for _ in range(WALL_GRID)]
    for _ in range(WALL_GRID)
]
_Z_WALL_V = [
    [_rng.getrandbits(64) for _ in range(WALL_GRID)]
    for _ in range(WALL_GRID)
]
_Z_WALLS_LEFT = [
    [_rng.getrandbits(64) for _ in range(INITIAL_WALLS + 1)]
    for _ in range(2)
]
_Z_TURN = _rng.getrandbits(64)


def zobrist(b: Board) -> int:
    h = 0
    for p in (0, 1):
        r, c = b.pawns[p]
        h ^= _Z_PAWN[p][r * BOARD_SIZE + c]
        h ^= _Z_WALLS_LEFT[p][b.walls_left[p]]
    for (r, c) in b.h_walls:
        h ^= _Z_WALL_H[r][c]
    for (r, c) in b.v_walls:
        h ^= _Z_WALL_V[r][c]
    if b.turn == 1:
        h ^= _Z_TURN
    return h


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def evaluate(b: Board, player: Optional[int] = None) -> int:
    """Static evaluation.

    If ``player`` is given, return the score from that player's perspective
    (used by external callers / tests). Otherwise, return from the
    side-to-move's perspective (what the search uses internally).
    """
    me = b.turn if player is None else player
    opp = 1 - me

    w = b.winner()
    if w == me:
        return WIN_SCORE
    if w is not None:
        return -WIN_SCORE

    my_p = b.shortest_path_length(me)
    op_p = b.shortest_path_length(opp)
    if my_p is None:
        return -WIN_SCORE
    if op_p is None:
        return WIN_SCORE

    s = (op_p - my_p) * W_PATH
    s += (b.walls_left[me] - b.walls_left[opp]) * W_WALL
    s += (len(b.pawn_moves(me)) - len(b.pawn_moves(opp))) * W_MOBILITY

    my_r, _ = b.pawns[me]
    op_r, _ = b.pawns[opp]
    my_adv = (BOARD_SIZE - 1 - my_r) if me == 0 else my_r
    op_adv = (BOARD_SIZE - 1 - op_r) if opp == 0 else op_r
    s += (my_adv - op_adv) * W_ADV

    # Tempo bonus for the side to move.
    if b.turn == me:
        s += 1
    else:
        s -= 1
    return s


# ----------------------------------------------------------------------
# Shortest-path helpers (for wall pruning)
# ----------------------------------------------------------------------
def _shortest_path_cells(b: Board, player: int) -> Set[Tuple[int, int]]:
    """All cells lying on *some* shortest path from player to goal."""
    start = b.pawns[player]
    goal_r = b.goal_row(player)
    dist: Dict[Tuple[int, int], int] = {start: 0}
    q: deque = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                continue
            if (nr, nc) in dist:
                continue
            if b.is_blocked(r, c, nr, nc):
                continue
            dist[(nr, nc)] = dist[(r, c)] + 1
            q.append((nr, nc))
    goal_cells = [cell for cell in dist if cell[0] == goal_r]
    if not goal_cells:
        return set()
    best = min(dist[g] for g in goal_cells)
    on_path: Set[Tuple[int, int]] = set()
    stack = [g for g in goal_cells if dist[g] == best]
    while stack:
        cell = stack.pop()
        if cell in on_path:
            continue
        on_path.add(cell)
        r, c = cell
        d = dist[cell]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (nr, nc) not in dist or dist[(nr, nc)] != d - 1:
                continue
            if b.is_blocked(r, c, nr, nc):
                continue
            stack.append((nr, nc))
    return on_path


def _candidate_wall_anchors(b: Board) -> Set[Tuple[int, int]]:
    hot = _shortest_path_cells(b, 0) | _shortest_path_cells(b, 1)
    anchors: Set[Tuple[int, int]] = set()
    for r, c in hot:
        for ar in (r - 1, r):
            for ac in (c - 1, c):
                if 0 <= ar < WALL_GRID and 0 <= ac < WALL_GRID:
                    anchors.add((ar, ac))
    return anchors


# ----------------------------------------------------------------------
# Move generation
# ----------------------------------------------------------------------
def _generate_moves(b: Board) -> List[Tuple[Move, int]]:
    """Return (move, ordering_score) pairs. Higher score searched first."""
    player = b.turn
    opp = 1 - player
    goal_r = b.goal_row(player)
    cur_r, _ = b.pawns[player]

    out: List[Tuple[Move, int]] = []

    # Pawn moves: score by how much they reduce distance to goal row.
    for (r, c) in b.pawn_moves(player):
        progress = abs(cur_r - goal_r) - abs(r - goal_r)
        out.append((Move(MOVE_PAWN, r, c), 10_000 + progress * 100))

    if b.walls_left[player] <= 0:
        return out

    cur_opp = b.shortest_path_length(opp) or 0
    cur_me = b.shortest_path_length(player) or 0
    anchors = _candidate_wall_anchors(b)

    for (r, c) in anchors:
        # Horizontal wall.
        if b._wall_h_shape_ok(r, c):
            b.h_walls.add((r, c))
            new_me = b.shortest_path_length(player)
            new_opp = b.shortest_path_length(opp)
            b.h_walls.discard((r, c))
            if new_me is not None and new_opp is not None:
                disruption = (new_opp - cur_opp) - (new_me - cur_me)
                # Favor walls that hurt opponent more than us.
                out.append((Move(WALL_H, r, c), disruption * 80))
        # Vertical wall.
        if b._wall_v_shape_ok(r, c):
            b.v_walls.add((r, c))
            new_me = b.shortest_path_length(player)
            new_opp = b.shortest_path_length(opp)
            b.v_walls.discard((r, c))
            if new_me is not None and new_opp is not None:
                disruption = (new_opp - cur_opp) - (new_me - cur_me)
                out.append((Move(WALL_V, r, c), disruption * 80))
    return out


# ----------------------------------------------------------------------
# Transposition table + killer moves
# ----------------------------------------------------------------------
class _TT:
    """Simple dict-backed transposition table: key -> (depth, flag, value, best_move)."""
    __slots__ = ("table",)

    def __init__(self) -> None:
        self.table: Dict[int, Tuple[int, int, int, Optional[Move]]] = {}

    def get(self, key: int):
        return self.table.get(key)

    def put(
        self,
        key: int,
        depth: int,
        flag: int,
        value: int,
        best_move: Optional[Move],
    ) -> None:
        existing = self.table.get(key)
        if existing is None or existing[0] <= depth:
            self.table[key] = (depth, flag, value, best_move)


class _Timeout(Exception):
    pass


# ----------------------------------------------------------------------
# Move ordering
# ----------------------------------------------------------------------
def _order_moves(
    scored: List[Tuple[Move, int]],
    tt_move: Optional[Move],
    killers: Tuple[Optional[Move], Optional[Move]],
) -> List[Move]:
    def key(item: Tuple[Move, int]) -> int:
        m, s = item
        if tt_move is not None and m == tt_move:
            return 10 ** 9
        if killers[0] is not None and m == killers[0]:
            return 10 ** 8
        if killers[1] is not None and m == killers[1]:
            return 10 ** 8 - 1
        return s

    return [m for (m, _) in sorted(scored, key=key, reverse=True)]


# ----------------------------------------------------------------------
# Negamax search with PVS
# ----------------------------------------------------------------------
def _negamax(
    b: Board,
    depth: int,
    alpha: int,
    beta: int,
    ply: int,
    deadline: Optional[float],
    tt: _TT,
    killers: List[List[Optional[Move]]],
) -> Tuple[int, Optional[Move]]:
    if deadline is not None and time.time() > deadline:
        raise _Timeout()

    # Terminal: score relative to side-to-move. Losing => negative.
    w = b.winner()
    if w is not None:
        # b.turn has no move / has already lost if the opponent reached goal.
        # winner == b.turn would mean side-to-move already won -- impossible
        # since apply() flips turn. Treat either way safely.
        if w == b.turn:
            return WIN_SCORE - ply, None
        return -(WIN_SCORE - ply), None

    if depth <= 0:
        return evaluate(b), None

    alpha_orig = alpha
    key = zobrist(b)
    entry = tt.get(key)
    tt_move: Optional[Move] = None
    if entry is not None:
        e_depth, e_flag, e_value, e_best = entry
        if e_depth >= depth:
            if e_flag == EXACT:
                return e_value, e_best
            if e_flag == LOWER and e_value > alpha:
                alpha = e_value
            elif e_flag == UPPER and e_value < beta:
                beta = e_value
            if alpha >= beta:
                return e_value, e_best
        tt_move = e_best

    scored = _generate_moves(b)
    if not scored:
        return evaluate(b), None

    k_slot = killers[ply] if ply < MAX_PLY else [None, None]
    moves = _order_moves(scored, tt_move, (k_slot[0], k_slot[1]))

    best = -INF
    best_move: Optional[Move] = moves[0]
    first = True
    for m in moves:
        child = b.apply(m)
        if first:
            val, _ = _negamax(child, depth - 1, -beta, -alpha, ply + 1, deadline, tt, killers)
            val = -val
        else:
            # Null (zero) window probe.
            val, _ = _negamax(child, depth - 1, -alpha - 1, -alpha, ply + 1, deadline, tt, killers)
            val = -val
            if alpha < val < beta:
                val, _ = _negamax(child, depth - 1, -beta, -val, ply + 1, deadline, tt, killers)
                val = -val
        first = False

        if val > best:
            best = val
            best_move = m
        if val > alpha:
            alpha = val
        if alpha >= beta:
            # Beta cutoff: record killer if this was a wall (non-captures analog).
            if m.kind != MOVE_PAWN and ply < MAX_PLY:
                if k_slot[0] != m:
                    k_slot[1] = k_slot[0]
                    k_slot[0] = m
            break

    if best <= alpha_orig:
        flag = UPPER
    elif best >= beta:
        flag = LOWER
    else:
        flag = EXACT
    tt.put(key, depth, flag, best, best_move)
    return best, best_move


# ----------------------------------------------------------------------
# Public entry: iterative-deepening time-managed root search
# ----------------------------------------------------------------------
def find_best_move(
    board: Board,
    max_depth: int = 20,
    time_limit: Optional[float] = 30.0,
) -> Optional[Move]:
    """Iterative deepening negamax+PVS under a time budget.

    The search runs depth 1, 2, 3, ... and keeps the best move found at
    the deepest *completed* iteration before ``time_limit`` seconds elapse.
    """
    deadline = time.time() + time_limit if time_limit else None
    tt = _TT()
    killers: List[List[Optional[Move]]] = [[None, None] for _ in range(MAX_PLY)]
    best: Optional[Move] = None

    for depth in range(1, max_depth + 1):
        try:
            _, mv = _negamax(board, depth, -INF, INF, 0, deadline, tt, killers)
            if mv is not None:
                best = mv
        except _Timeout:
            break

    if best is None:
        legal = board.legal_moves()
        best = legal[0] if legal else None
    return best
