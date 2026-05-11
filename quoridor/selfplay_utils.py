"""Helpers shared between the self-play loop and downstream evaluation tools.

These functions used to live in ``selfplay.py`` but are imported by
``tournament.py`` and ``bench_matrix.py`` (and now ``eval/`` scripts
after the repo reorganisation), so they belong in the core package
rather than at the top of an experiment driver.
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

from .board import Board, Move


def randomise_opening(board: Board, num_random: int) -> Tuple[Board, List[Tuple[Board, Move]]]:
    """Play *num_random* uniformly random legal moves from *board*.

    Returns the resulting board and the list of (Board, Move) pairs
    played (needed so the recorder can capture the opening moves too).
    """
    history: List[Tuple[Board, Move]] = []
    for _ in range(num_random):
        legal = board.legal_moves()
        if not legal or board.winner() is not None:
            break
        move = random.choice(legal)
        history.append((board, move))
        board = board.apply(move)
    return board, history


def adjudicate_winner(final_board: Board, min_gap: int = 2) -> Optional[int]:
    """Resolve a max-moves timeout to a decisive winner by path length.

    Returns the winner (0 or 1) if the path-length gap is >= ``min_gap``,
    else None (unambiguous draw).  This converts the bulk of 'stall
    draws' into decisive training signal, which is what the value head
    needs to learn meaningful positional evaluation.
    """
    d0 = final_board.shortest_path_length(0)
    d1 = final_board.shortest_path_length(1)
    if d0 is None and d1 is None:
        return None
    if d0 is None:
        return 1
    if d1 is None:
        return 0
    if d0 + min_gap <= d1:
        return 0
    if d1 + min_gap <= d0:
        return 1
    return None
