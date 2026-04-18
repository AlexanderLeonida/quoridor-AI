"""Quoridor board state, move representation, and rules.

Coordinate conventions
----------------------
Cells:   (row, col) with row, col in [0, 8].
         Player 0 starts at (0, 4) and wins by reaching row 8.
         Player 1 starts at (8, 4) and wins by reaching row 0.

Walls:   A wall has length 2. Anchor coordinates live on an 8x8 grid.
         Horizontal wall anchored at (r, c) sits between rows r and r+1,
         spanning columns c and c+1, blocking vertical pawn movement
         between (r, c)<->(r+1, c) and (r, c+1)<->(r+1, c+1).
         Vertical wall anchored at (r, c) sits between cols c and c+1,
         spanning rows r and r+1, blocking horizontal pawn movement
         between (r, c)<->(r, c+1) and (r+1, c)<->(r+1, c+1).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

BOARD_SIZE = 9
WALL_GRID = BOARD_SIZE - 1  # 8
INITIAL_WALLS = 10

# Move kinds
MOVE_PAWN = 0
WALL_H = 1
WALL_V = 2


@dataclass(frozen=True)
class Move:
    kind: int
    r: int
    c: int

    def __str__(self) -> str:
        prefix = {MOVE_PAWN: "M", WALL_H: "H", WALL_V: "V"}[self.kind]
        return f"{prefix}{self.r}{self.c}"


class Board:
    __slots__ = ("pawns", "walls_left", "h_walls", "v_walls", "turn")

    def __init__(
        self,
        pawns: List[Tuple[int, int]],
        walls_left: List[int],
        h_walls: Set[Tuple[int, int]],
        v_walls: Set[Tuple[int, int]],
        turn: int,
    ):
        self.pawns = pawns
        self.walls_left = walls_left
        self.h_walls = h_walls
        self.v_walls = v_walls
        self.turn = turn

    @classmethod
    def initial(cls) -> "Board":
        return cls(
            pawns=[(0, 4), (8, 4)],
            walls_left=[INITIAL_WALLS, INITIAL_WALLS],
            h_walls=set(),
            v_walls=set(),
            turn=0,
        )

    def clone(self) -> "Board":
        return Board(
            pawns=list(self.pawns),
            walls_left=list(self.walls_left),
            h_walls=set(self.h_walls),
            v_walls=set(self.v_walls),
            turn=self.turn,
        )

    # --- Goal / winner -------------------------------------------------

    def goal_row(self, player: int) -> int:
        return 8 if player == 0 else 0

    def winner(self) -> Optional[int]:
        if self.pawns[0][0] == 8:
            return 0
        if self.pawns[1][0] == 0:
            return 1
        return None

    # --- Wall blocking -------------------------------------------------

    def is_blocked(self, r1: int, c1: int, r2: int, c2: int) -> bool:
        """Whether an orthogonal step from (r1, c1) to (r2, c2) is blocked."""
        if r1 == r2:
            cmin = c1 if c1 < c2 else c2
            # Vertical wall anchored at (r1-1, cmin) or (r1, cmin) blocks.
            if (r1, cmin) in self.v_walls:
                return True
            if (r1 - 1, cmin) in self.v_walls:
                return True
            return False
        else:
            rmin = r1 if r1 < r2 else r2
            if (rmin, c1) in self.h_walls:
                return True
            if (rmin, c1 - 1) in self.h_walls:
                return True
            return False

    # --- Pawn moves ----------------------------------------------------

    def pawn_moves(self, player: int) -> List[Tuple[int, int]]:
        r, c = self.pawns[player]
        opp = self.pawns[1 - player]
        out: List[Tuple[int, int]] = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                continue
            if self.is_blocked(r, c, nr, nc):
                continue
            if (nr, nc) == opp:
                # Attempt to jump over
                jr, jc = nr + dr, nc + dc
                straight_ok = (
                    0 <= jr < BOARD_SIZE
                    and 0 <= jc < BOARD_SIZE
                    and not self.is_blocked(nr, nc, jr, jc)
                )
                if straight_ok:
                    out.append((jr, jc))
                else:
                    # Side-step diagonally around the opponent
                    for ddr, ddc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        if (ddr, ddc) == (dr, dc) or (ddr, ddc) == (-dr, -dc):
                            continue
                        sr, sc = nr + ddr, nc + ddc
                        if not (0 <= sr < BOARD_SIZE and 0 <= sc < BOARD_SIZE):
                            continue
                        if self.is_blocked(nr, nc, sr, sc):
                            continue
                        out.append((sr, sc))
                continue
            out.append((nr, nc))
        return out

    # --- Shortest path (BFS) ------------------------------------------

    def shortest_path_length(self, player: int) -> Optional[int]:
        start = self.pawns[player]
        goal_r = self.goal_row(player)
        if start[0] == goal_r:
            return 0
        visited = {start}
        queue: deque = deque()
        queue.append((start[0], start[1], 0))
        while queue:
            r, c, d = queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                    continue
                if (nr, nc) in visited:
                    continue
                if self.is_blocked(r, c, nr, nc):
                    continue
                if nr == goal_r:
                    return d + 1
                visited.add((nr, nc))
                queue.append((nr, nc, d + 1))
        return None

    # --- Wall placement ------------------------------------------------

    def _wall_h_shape_ok(self, r: int, c: int) -> bool:
        if not (0 <= r < WALL_GRID and 0 <= c < WALL_GRID):
            return False
        if (r, c) in self.h_walls:
            return False
        if (r, c - 1) in self.h_walls:
            return False
        if (r, c + 1) in self.h_walls:
            return False
        if (r, c) in self.v_walls:
            return False
        return True

    def _wall_v_shape_ok(self, r: int, c: int) -> bool:
        if not (0 <= r < WALL_GRID and 0 <= c < WALL_GRID):
            return False
        if (r, c) in self.v_walls:
            return False
        if (r - 1, c) in self.v_walls:
            return False
        if (r + 1, c) in self.v_walls:
            return False
        if (r, c) in self.h_walls:
            return False
        return True

    def _paths_still_exist(self) -> bool:
        return (
            self.shortest_path_length(0) is not None
            and self.shortest_path_length(1) is not None
        )

    def legal_moves(self) -> List[Move]:
        player = self.turn
        moves: List[Move] = [
            Move(MOVE_PAWN, r, c) for (r, c) in self.pawn_moves(player)
        ]
        if self.walls_left[player] > 0:
            for r in range(WALL_GRID):
                for c in range(WALL_GRID):
                    if self._wall_h_shape_ok(r, c):
                        self.h_walls.add((r, c))
                        ok = self._paths_still_exist()
                        self.h_walls.discard((r, c))
                        if ok:
                            moves.append(Move(WALL_H, r, c))
                    if self._wall_v_shape_ok(r, c):
                        self.v_walls.add((r, c))
                        ok = self._paths_still_exist()
                        self.v_walls.discard((r, c))
                        if ok:
                            moves.append(Move(WALL_V, r, c))
        return moves

    def is_legal(self, move: Move) -> bool:
        if move.kind == MOVE_PAWN:
            return (move.r, move.c) in self.pawn_moves(self.turn)
        if self.walls_left[self.turn] <= 0:
            return False
        if move.kind == WALL_H:
            if not self._wall_h_shape_ok(move.r, move.c):
                return False
            self.h_walls.add((move.r, move.c))
            ok = self._paths_still_exist()
            self.h_walls.discard((move.r, move.c))
            return ok
        if move.kind == WALL_V:
            if not self._wall_v_shape_ok(move.r, move.c):
                return False
            self.v_walls.add((move.r, move.c))
            ok = self._paths_still_exist()
            self.v_walls.discard((move.r, move.c))
            return ok
        return False

    def apply(self, move: Move) -> "Board":
        new = self.clone()
        p = new.turn
        if move.kind == MOVE_PAWN:
            new.pawns[p] = (move.r, move.c)
        elif move.kind == WALL_H:
            new.h_walls.add((move.r, move.c))
            new.walls_left[p] -= 1
        elif move.kind == WALL_V:
            new.v_walls.add((move.r, move.c))
            new.walls_left[p] -= 1
        new.turn = 1 - p
        return new

