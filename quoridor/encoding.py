"""State / move / action encoding for the neural network.

This module defines the *contract* between the database and the network.
Any bug in this file silently corrupts training, so it is the single most
important thing to unit-test.

Action space  (size = ACTION_SPACE = 209)
-----------------------------------------
    [  0,  81)   pawn moves          index = r*9 + c
    [ 81, 145)   horizontal walls    index = 81 + r*8 + c
    [145, 209)   vertical walls      index = 145 + r*8 + c
    (r, c are *canonical* internal coordinates -- see below.)

Canonical view
--------------
The network always sees the position from the side-to-move's perspective:
"I am player 0, I start near row 0, my goal is row 8".
If the real side-to-move is P2, the board is row-flipped (and pawns
swapped) before encoding. `canonical_view` returns a tuple describing the
canonicalized position plus a `flipped` flag that callers pass through to
`move_to_action` / `action_to_move` so moves are mapped into the same
canonical frame.

State tensor shape: (NUM_PLANES, 9, 9), dtype float32.
Planes:
    0: my pawn position      (one-hot)
    1: opponent pawn         (one-hot)
    2: horizontal walls      (1 at anchor cells; last row/col always 0)
    3: vertical walls        (1 at anchor cells; last row/col always 0)
    4: my walls-left / 10    (broadcast)
    5: opp walls-left / 10   (broadcast)
    6: all-ones bias         (broadcast)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .board import (
    BOARD_SIZE,
    INITIAL_WALLS,
    MOVE_PAWN,
    Board,
    Move,
    WALL_GRID,
    WALL_H,
    WALL_V,
)

ACTION_PAWN_BASE = 0
ACTION_H_BASE = BOARD_SIZE * BOARD_SIZE                         # 81
ACTION_V_BASE = ACTION_H_BASE + WALL_GRID * WALL_GRID           # 145
ACTION_SPACE = ACTION_V_BASE + WALL_GRID * WALL_GRID            # 209

NUM_PLANES = 7


def _flip_cell_r(r: int) -> int:
    return BOARD_SIZE - 1 - r


def _flip_wall_r(r: int) -> int:
    return WALL_GRID - 1 - r


def canonical_view(board: Board):
    """Return the position from the side-to-move's POV.

    Returns a tuple:
        (me_pawn, opp_pawn, h_walls, v_walls, my_walls_left,
         opp_walls_left, flipped)
    Where `flipped` is True iff the real side-to-move is P2 (rows were
    mirrored).
    """
    if board.turn == 0:
        return (
            board.pawns[0],
            board.pawns[1],
            set(board.h_walls),
            set(board.v_walls),
            board.walls_left[0],
            board.walls_left[1],
            False,
        )
    me = (_flip_cell_r(board.pawns[1][0]), board.pawns[1][1])
    opp = (_flip_cell_r(board.pawns[0][0]), board.pawns[0][1])
    hw = {(_flip_wall_r(r), c) for (r, c) in board.h_walls}
    vw = {(_flip_wall_r(r), c) for (r, c) in board.v_walls}
    return me, opp, hw, vw, board.walls_left[1], board.walls_left[0], True


def encode_state(board: Board) -> np.ndarray:
    """Encode `board` as a (NUM_PLANES, 9, 9) float32 tensor (canonical view)."""
    me, opp, hw, vw, my_left, opp_left, _flipped = canonical_view(board)
    planes = np.zeros((NUM_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    planes[0, me[0], me[1]] = 1.0
    planes[1, opp[0], opp[1]] = 1.0
    for r, c in hw:
        planes[2, r, c] = 1.0
    for r, c in vw:
        planes[3, r, c] = 1.0
    planes[4].fill(my_left / INITIAL_WALLS)
    planes[5].fill(opp_left / INITIAL_WALLS)
    planes[6].fill(1.0)
    return planes


def move_to_action(move: Move, flipped: bool) -> int:
    """Map a Move (real coordinates) to a canonical action index."""
    if move.kind == MOVE_PAWN:
        r = _flip_cell_r(move.r) if flipped else move.r
        return ACTION_PAWN_BASE + r * BOARD_SIZE + move.c
    r = _flip_wall_r(move.r) if flipped else move.r
    if move.kind == WALL_H:
        return ACTION_H_BASE + r * WALL_GRID + move.c
    return ACTION_V_BASE + r * WALL_GRID + move.c


def action_to_move(idx: int, flipped: bool) -> Move:
    """Map a canonical action index back to a Move with real coordinates."""
    if idx < ACTION_H_BASE:
        r, c = divmod(idx - ACTION_PAWN_BASE, BOARD_SIZE)
        if flipped:
            r = _flip_cell_r(r)
        return Move(MOVE_PAWN, r, c)
    if idx < ACTION_V_BASE:
        r, c = divmod(idx - ACTION_H_BASE, WALL_GRID)
        if flipped:
            r = _flip_wall_r(r)
        return Move(WALL_H, r, c)
    r, c = divmod(idx - ACTION_V_BASE, WALL_GRID)
    if flipped:
        r = _flip_wall_r(r)
    return Move(WALL_V, r, c)


def legal_action_mask(board: Board) -> np.ndarray:
    """Boolean mask of length ACTION_SPACE: True iff the action is legal."""
    _, _, _, _, _, _, flipped = canonical_view(board)
    mask = np.zeros(ACTION_SPACE, dtype=bool)
    for m in board.legal_moves():
        mask[move_to_action(m, flipped)] = True
    return mask


def value_target(winner, side_to_move: int) -> float:
    """Game-outcome target z in {-1, 0, +1} from side_to_move's POV."""
    if winner is None:
        return 0.0
    return 1.0 if winner == side_to_move else -1.0


# -------------------------------------------------------------------
# Policy (de)serialisation — used to store MCTS visit distributions
# -------------------------------------------------------------------

def serialize_policy(policy: np.ndarray) -> bytes:
    """Compact binary encoding of a (ACTION_SPACE,) float32 array."""
    import io

    buf = io.BytesIO()
    np.save(buf, policy.astype(np.float32))
    return buf.getvalue()


def deserialize_policy(blob: bytes) -> np.ndarray:
    """Inverse of :func:`serialize_policy`."""
    import io

    return np.load(io.BytesIO(blob))


# -------------------------------------------------------------------
# Column-flip symmetry (data augmentation)
# -------------------------------------------------------------------
#
# Quoridor is symmetric under reflection about the central column:
# a position and its column-mirror are game-theoretically equivalent,
# with each action's column index c mapped to (N-1-c). The value is
# unchanged; the policy permutes.

def _flip_cell_c(c: int) -> int:
    return BOARD_SIZE - 1 - c


def _flip_wall_c(c: int) -> int:
    return WALL_GRID - 1 - c


def _build_col_flip_action_perm() -> np.ndarray:
    perm = np.empty(ACTION_SPACE, dtype=np.int64)
    for idx in range(ACTION_PAWN_BASE, ACTION_H_BASE):
        r, c = divmod(idx - ACTION_PAWN_BASE, BOARD_SIZE)
        perm[idx] = ACTION_PAWN_BASE + r * BOARD_SIZE + _flip_cell_c(c)
    for idx in range(ACTION_H_BASE, ACTION_V_BASE):
        r, c = divmod(idx - ACTION_H_BASE, WALL_GRID)
        perm[idx] = ACTION_H_BASE + r * WALL_GRID + _flip_wall_c(c)
    for idx in range(ACTION_V_BASE, ACTION_SPACE):
        r, c = divmod(idx - ACTION_V_BASE, WALL_GRID)
        perm[idx] = ACTION_V_BASE + r * WALL_GRID + _flip_wall_c(c)
    return perm


COL_FLIP_PERM: np.ndarray = _build_col_flip_action_perm()


def col_flip_state(state: np.ndarray) -> np.ndarray:
    """Column-flip a (NUM_PLANES, 9, 9) canonical state tensor.

    Scalar broadcast planes (walls-left, bias) are unchanged under the
    flip but ``np.flip`` handles them correctly since every column
    holds the same value.
    """
    return np.ascontiguousarray(state[:, :, ::-1])


def col_flip_policy(policy: np.ndarray) -> np.ndarray:
    """Column-flip an action-probability vector of length ACTION_SPACE."""
    return policy[COL_FLIP_PERM]
