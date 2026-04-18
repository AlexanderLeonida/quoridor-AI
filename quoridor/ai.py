"""Quoridor AI: alpha-beta minimax with shortest-path evaluation.

The evaluation is driven by the classic Quoridor heuristic:
    score = (opponent_path - my_path) * W_PATH
          + (my_walls - opp_walls)    * W_WALL
          + (my_advance - opp_advance) * W_ADV

Wall moves are limited to "useful" placements (walls adjacent to either
player's current shortest path) to keep the branching factor tractable;
this is a standard practical pruning for Quoridor engines.
"""

from __future__ import annotations

import time
from collections import deque
from typing import List, Optional, Set, Tuple

from .board import (
    BOARD_SIZE,
    WALL_GRID,
    Board,
    MOVE_PAWN,
    Move,
    WALL_H,
    WALL_V,
)

INF = 10**9
WIN_SCORE = 10**7

W_PATH = 100
W_WALL = 3
W_ADV = 1


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------

def evaluate(board: Board, player: int) -> int:
    """Static evaluation from `player`'s perspective."""
    winner = board.winner()
    if winner == player:
        return WIN_SCORE
    if winner is not None:
        return -WIN_SCORE

    my_path = board.shortest_path_length(player)
    opp_path = board.shortest_path_length(1 - player)
    if my_path is None:
        return -WIN_SCORE
    if opp_path is None:
        return WIN_SCORE

    # Raw path advantage.
    score = (opp_path - my_path) * W_PATH
    # Walls in reserve are valuable.
    score += (board.walls_left[player] - board.walls_left[1 - player]) * W_WALL
    # Mild bias toward the goal row (tie-break when paths are equal).
    my_r, _ = board.pawns[player]
    opp_r, _ = board.pawns[1 - player]
    my_adv = (BOARD_SIZE - 1 - my_r) if player == 0 else my_r
    opp_adv = (BOARD_SIZE - 1 - opp_r) if (1 - player) == 0 else opp_r
    score += (my_adv - opp_adv) * W_ADV

    # Side-to-move tempo bonus: if it's `player`'s turn and paths are tied,
    # `player` is slightly ahead.
    if board.turn == player:
        score += 1
    else:
        score -= 1
    return score


# -------------------------------------------------------------------
# Move generation with pruning
# -------------------------------------------------------------------

def _shortest_path_cells(board: Board, player: int) -> Set[Tuple[int, int]]:
    """Return all cells on *any* shortest path from player to goal."""
    start = board.pawns[player]
    goal_r = board.goal_row(player)
    # BFS distance from start.
    dist_from_start = {start: 0}
    q: deque = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                continue
            if (nr, nc) in dist_from_start:
                continue
            if board.is_blocked(r, c, nr, nc):
                continue
            dist_from_start[(nr, nc)] = dist_from_start[(r, c)] + 1
            q.append((nr, nc))

    goal_cells = [
        (r, c) for (r, c) in dist_from_start if r == goal_r
    ]
    if not goal_cells:
        return set()
    best = min(dist_from_start[g] for g in goal_cells)
    # Backtrack on predecessors: a cell is on a shortest path iff there is
    # an unblocked neighbor whose distance is one less.
    on_path: Set[Tuple[int, int]] = set()
    targets = [g for g in goal_cells if dist_from_start[g] == best]
    stack = list(targets)
    while stack:
        r, c = stack.pop()
        if (r, c) in on_path:
            continue
        on_path.add((r, c))
        d = dist_from_start[(r, c)]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (nr, nc) not in dist_from_start:
                continue
            if dist_from_start[(nr, nc)] != d - 1:
                continue
            if board.is_blocked(r, c, nr, nc):
                continue
            stack.append((nr, nc))
    return on_path


def _candidate_wall_anchors(board: Board) -> Set[Tuple[int, int]]:
    """Anchors worth considering: those adjacent to a shortest-path cell."""
    hot = _shortest_path_cells(board, 0) | _shortest_path_cells(board, 1)
    anchors: Set[Tuple[int, int]] = set()
    for r, c in hot:
        for ar in (r - 1, r):
            for ac in (c - 1, c):
                if 0 <= ar < WALL_GRID and 0 <= ac < WALL_GRID:
                    anchors.add((ar, ac))
    return anchors


def _pruned_legal_moves(board: Board) -> List[Move]:
    player = board.turn
    moves: List[Move] = [
        Move(MOVE_PAWN, r, c) for (r, c) in board.pawn_moves(player)
    ]
    if board.walls_left[player] > 0:
        for (r, c) in _candidate_wall_anchors(board):
            if board._wall_h_shape_ok(r, c):
                board.h_walls.add((r, c))
                ok = board._paths_still_exist()
                board.h_walls.discard((r, c))
                if ok:
                    moves.append(Move(WALL_H, r, c))
            if board._wall_v_shape_ok(r, c):
                board.v_walls.add((r, c))
                ok = board._paths_still_exist()
                board.v_walls.discard((r, c))
                if ok:
                    moves.append(Move(WALL_V, r, c))
    return moves


# -------------------------------------------------------------------
# Move ordering
# -------------------------------------------------------------------

def _order_moves(board: Board, moves: List[Move], root_player: int) -> List[Move]:
    """Order moves by quick heuristic: big path swings first."""
    player = board.turn
    goal_r = board.goal_row(player)
    cur_r, _ = board.pawns[player]

    def key(m: Move) -> int:
        if m.kind == MOVE_PAWN:
            # Prefer pawn moves that get closer to goal.
            return -(abs(cur_r - goal_r) - abs(m.r - goal_r)) * 100
        # Light ordering for walls — the search will sort out the rest.
        return 50

    return sorted(moves, key=key)


# -------------------------------------------------------------------
# Alpha-beta with iterative deepening and a time budget
# -------------------------------------------------------------------

class _Timeout(Exception):
    pass


def _alpha_beta(
    board: Board,
    depth: int,
    alpha: int,
    beta: int,
    root_player: int,
    deadline: Optional[float],
) -> Tuple[int, Optional[Move]]:
    if deadline is not None and time.time() > deadline:
        raise _Timeout()

    winner = board.winner()
    if winner is not None or depth == 0:
        return evaluate(board, root_player), None

    moves = _pruned_legal_moves(board)
    if not moves:
        return evaluate(board, root_player), None
    moves = _order_moves(board, moves, root_player)

    best_move: Optional[Move] = moves[0]
    if board.turn == root_player:
        value = -INF
        for m in moves:
            child = board.apply(m)
            v, _ = _alpha_beta(child, depth - 1, alpha, beta, root_player, deadline)
            if v > value:
                value = v
                best_move = m
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break
        return value, best_move
    else:
        value = INF
        for m in moves:
            child = board.apply(m)
            v, _ = _alpha_beta(child, depth - 1, alpha, beta, root_player, deadline)
            if v < value:
                value = v
                best_move = m
            if value < beta:
                beta = value
            if alpha >= beta:
                break
        return value, best_move


def find_best_move(
    board: Board,
    max_depth: int = 3,
    time_limit: Optional[float] = 5.0,
) -> Optional[Move]:
    """Iterative-deepening alpha-beta search. Returns the best move found."""
    root_player = board.turn
    deadline = time.time() + time_limit if time_limit else None
    best: Optional[Move] = None
    for depth in range(1, max_depth + 1):
        try:
            _, move = _alpha_beta(board, depth, -INF, INF, root_player, deadline)
            if move is not None:
                best = move
        except _Timeout:
            break
    if best is None:
        # Fall back to any legal move (shouldn't happen unless game is over).
        legal = board.legal_moves()
        best = legal[0] if legal else None
    return best
