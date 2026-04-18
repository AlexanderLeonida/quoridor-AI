"""Convenience wrapper to record games into the GameDB.

Usage
-----
    rec = GameRecorder(p1_source="human", p2_source="alphabeta",
                       p1_time_limit=None, p2_time_limit=30.0)
    rec.start()
    while not game_over:
        m = ... # figure out next move
        rec.record(m)
        board = board.apply(m)
    rec.finish(winner=board.winner())    # winner may be None if aborted
"""

from __future__ import annotations

import time
from typing import List, Optional

from .board import Move
from .database import GameDB


class GameRecorder:
    def __init__(
        self,
        p1_source: str,
        p2_source: str,
        p1_time_limit: Optional[float] = None,
        p2_time_limit: Optional[float] = None,
        model_version: Optional[str] = None,
        notes: Optional[str] = None,
        db: Optional[GameDB] = None,
    ):
        self._db = db if db is not None else GameDB()
        self._own_db = db is None
        self.p1_source = p1_source
        self.p2_source = p2_source
        self.p1_time_limit = p1_time_limit
        self.p2_time_limit = p2_time_limit
        self.model_version = model_version
        self.notes = notes
        self.moves: List[Move] = []
        self.elapsed_ms: List[Optional[int]] = []
        self._last_t: Optional[float] = None
        self._finished = False

    def start(self) -> None:
        self._last_t = time.perf_counter()

    def record(self, move: Move) -> None:
        now = time.perf_counter()
        if self._last_t is None:
            self.elapsed_ms.append(None)
        else:
            self.elapsed_ms.append(int((now - self._last_t) * 1000))
        self._last_t = now
        self.moves.append(move)

    def finish(self, winner: Optional[int]) -> Optional[int]:
        """Persist the game. Returns the new game id, or None if nothing to save."""
        if self._finished:
            return None
        self._finished = True
        gid = None
        if self.moves:
            gid = self._db.save_game(
                self.moves,
                winner=winner,
                p1_source=self.p1_source,
                p2_source=self.p2_source,
                p1_time_limit=self.p1_time_limit,
                p2_time_limit=self.p2_time_limit,
                model_version=self.model_version,
                notes=self.notes,
                elapsed_ms=self.elapsed_ms,
            )
        if self._own_db:
            self._db.close()
        return gid
