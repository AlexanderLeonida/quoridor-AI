"""SQLite-backed store of played Quoridor games.

Storage strategy
----------------
We store the *move list* per game (not full position tensors). States are
re-materialized on demand by replaying moves from the initial position.
Result: ~40x smaller storage than storing tensors, and it remains
correct regardless of how we later change the encoding.

Schema
------
    games   one row per game, with metadata (sources, time limits, winner)
    moves   ordered move list per game

Public API
----------
    GameDB(path=None)                 open / create the DB
    db.save_game(...)                 persist a finished (or unfinished) game
    db.count_games()                  how many games are stored
    db.iter_training_samples()        yield (board, move, z) for training
    db.load_moves(game_id)            list of Move objects
"""

from __future__ import annotations

import os
import sqlite3
from typing import Iterator, List, Optional, Sequence, Tuple

from .board import Board, Move

_SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at     TEXT    NOT NULL DEFAULT (datetime('now')),
    finished_at    TEXT,
    winner         INTEGER,
    num_plies      INTEGER NOT NULL DEFAULT 0,
    p1_source      TEXT    NOT NULL,
    p2_source      TEXT    NOT NULL,
    p1_time_limit  REAL,
    p2_time_limit  REAL,
    model_version  TEXT,
    notes          TEXT,
    final_path_gap INTEGER
);

CREATE TABLE IF NOT EXISTS moves (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id       INTEGER NOT NULL REFERENCES games(id) ON DELETE CASCADE,
    ply           INTEGER NOT NULL,
    side          INTEGER NOT NULL,
    move_kind     INTEGER NOT NULL,
    move_r        INTEGER NOT NULL,
    move_c        INTEGER NOT NULL,
    elapsed_ms    INTEGER,
    policy_blob   BLOB,
    UNIQUE (game_id, ply)
);

CREATE INDEX IF NOT EXISTS idx_moves_game ON moves(game_id);
CREATE INDEX IF NOT EXISTS idx_games_winner ON games(winner);
"""

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "quoridor.db",
)


def compute_final_path_gap(
    moves: Sequence[Move], winner: Optional[int]
) -> Optional[int]:
    """Replay ``moves`` from the initial position and return the loser's
    shortest path to their goal at game end.

    Higher = more decisive win.  Returns None for draws/unknown winners
    or if replay fails (e.g., illegal move in stored data).
    """
    if winner is None:
        return None
    board = Board.initial()
    try:
        for m in moves:
            board = board.apply(m)
    except Exception:
        return None
    return board.shortest_path_length(1 - winner)


class GameDB:
    def __init__(self, path: Optional[str] = None):
        self.path = path or DEFAULT_DB_PATH
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        # Migration: add final_path_gap column on pre-existing DBs.
        try:
            self._conn.execute("ALTER TABLE games ADD COLUMN final_path_gap INTEGER")
        except sqlite3.OperationalError:
            pass  # column already exists
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def save_game(
        self,
        moves: Sequence[Move],
        winner: Optional[int],
        p1_source: str,
        p2_source: str,
        p1_time_limit: Optional[float] = None,
        p2_time_limit: Optional[float] = None,
        model_version: Optional[str] = None,
        notes: Optional[str] = None,
        elapsed_ms: Optional[Sequence[Optional[int]]] = None,
        policies: Optional[Sequence[Optional[bytes]]] = None,
        final_path_gap: Optional[int] = None,
    ) -> int:
        """Persist a game. Returns the new game's id.

        ``final_path_gap`` is the loser's shortest-path-to-goal at game
        end (higher = more decisive win).  None for draws/unfinished or
        when not computed.  If ``moves`` and ``winner`` are provided and
        ``final_path_gap`` is not, computes it by replay.
        """
        if final_path_gap is None and winner is not None:
            final_path_gap = compute_final_path_gap(moves, winner)
        cur = self._conn.cursor()
        cur.execute(
            """INSERT INTO games
               (finished_at, winner, num_plies,
                p1_source, p2_source, p1_time_limit, p2_time_limit,
                model_version, notes, final_path_gap)
               VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                winner,
                len(moves),
                p1_source,
                p2_source,
                p1_time_limit,
                p2_time_limit,
                model_version,
                notes,
                final_path_gap,
            ),
        )
        game_id = cur.lastrowid
        rows = []
        for ply, m in enumerate(moves):
            side = ply % 2
            em = elapsed_ms[ply] if elapsed_ms is not None else None
            pol = policies[ply] if policies is not None else None
            rows.append(
                (game_id, ply, side, m.kind, m.r, m.c, em, pol)
            )
        if rows:
            cur.executemany(
                """INSERT INTO moves
                   (game_id, ply, side, move_kind, move_r, move_c,
                    elapsed_ms, policy_blob)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
        self._conn.commit()
        return game_id

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def count_games(self, finished_only: bool = False) -> int:
        q = "SELECT COUNT(*) FROM games"
        if finished_only:
            q += " WHERE winner IS NOT NULL"
        return self._conn.execute(q).fetchone()[0]

    def count_positions(self, finished_only: bool = True) -> int:
        q = "SELECT COALESCE(SUM(num_plies), 0) FROM games"
        if finished_only:
            q += " WHERE winner IS NOT NULL"
        return self._conn.execute(q).fetchone()[0]

    def iter_games(self, finished_only: bool = False):
        # Column order is part of the API; existing callers index by
        # position.  New columns must be appended at the end.
        # row[0]=id, [1]=created_at, [2]=finished_at, [3]=winner,
        # [4]=num_plies, [5]=p1_source, [6]=p2_source, [7]=model_version,
        # [8]=final_path_gap.
        q = ("SELECT id, created_at, finished_at, winner, num_plies, "
             "p1_source, p2_source, model_version, final_path_gap "
             "FROM games")
        if finished_only:
            q += " WHERE winner IS NOT NULL"
        q += " ORDER BY id"
        yield from self._conn.execute(q).fetchall()

    def load_moves(self, game_id: int) -> List[Move]:
        cur = self._conn.execute(
            "SELECT move_kind, move_r, move_c FROM moves "
            "WHERE game_id = ? ORDER BY ply",
            (game_id,),
        )
        return [Move(k, r, c) for (k, r, c) in cur.fetchall()]

    def load_policy_blobs(self, game_id: int) -> List[Optional[bytes]]:
        """Return the raw policy_blob for each ply (None when absent)."""
        cur = self._conn.execute(
            "SELECT policy_blob FROM moves "
            "WHERE game_id = ? ORDER BY ply",
            (game_id,),
        )
        return [row[0] for row in cur.fetchall()]

    def iter_training_samples(
        self,
        include_unfinished: bool = False,
    ) -> Iterator[Tuple[Board, Move, float]]:
        """Yield `(board, move_taken, z)` for every ply across all games.

        `z` is +1 if the side-to-move won, -1 if they lost, 0 if draw /
        unfinished. Replay is performed in Python; expect roughly
        (avg_ply_count * game_count) yields total.
        """
        for row in self.iter_games(finished_only=not include_unfinished):
            game_id = row[0]
            winner = row[3]
            moves = self.load_moves(game_id)
            b = Board.initial()
            for m in moves:
                side = b.turn
                if winner is None:
                    z = 0.0
                else:
                    z = 1.0 if winner == side else -1.0
                yield b, m, z
                b = b.apply(m)
