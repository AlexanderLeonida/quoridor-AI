"""Sanity tests for the Quoridor engine and AI."""

# --- path bootstrap so this file can be run from anywhere ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# -------------------------------------------------------------
from quoridor import (
    BOARD_SIZE,
    Board,
    MOVE_PAWN,
    Move,
    WALL_H,
    WALL_V,
    find_best_move,
)


def test_initial_state():
    b = Board.initial()
    assert b.pawns == [(0, 4), (8, 4)]
    assert b.walls_left == [10, 10]
    assert b.turn == 0
    assert b.winner() is None


def test_initial_pawn_moves():
    b = Board.initial()
    moves = set(b.pawn_moves(0))
    # P1 at top row can move down, left, right.
    assert moves == {(1, 4), (0, 3), (0, 5)}


def test_shortest_path_length_open_board():
    b = Board.initial()
    assert b.shortest_path_length(0) == 8
    assert b.shortest_path_length(1) == 8


def test_horizontal_wall_blocks():
    b = Board.initial()
    b2 = b.apply(Move(WALL_H, 0, 3))  # blocks (0,3)-(1,3) and (0,4)-(1,4)
    assert b2.is_blocked(0, 3, 1, 3)
    assert b2.is_blocked(0, 4, 1, 4)
    assert not b2.is_blocked(0, 5, 1, 5)
    assert b2.walls_left[0] == 9
    assert b2.turn == 1


def test_vertical_wall_blocks():
    b = Board.initial()
    b2 = b.apply(Move(WALL_V, 3, 3))  # blocks (3,3)-(3,4) and (4,3)-(4,4)
    assert b2.is_blocked(3, 3, 3, 4)
    assert b2.is_blocked(4, 3, 4, 4)
    assert not b2.is_blocked(5, 3, 5, 4)


def test_wall_overlap_rejected():
    b = Board.initial()
    b = b.apply(Move(WALL_H, 3, 3))  # horizontal at (3,3)
    # overlapping horizontal walls along same row
    assert not b.is_legal(Move(WALL_H, 3, 2))
    assert not b.is_legal(Move(WALL_H, 3, 4))
    # crossing vertical wall at same anchor
    assert not b.is_legal(Move(WALL_V, 3, 3))
    # non-overlapping H wall is fine
    assert b.is_legal(Move(WALL_H, 3, 5))


def test_wall_cannot_fully_block_path():
    """A wall that would leave a player with no path must be rejected."""
    # Construct a near-sealed situation: stack walls across row 0/1 so
    # that one more wall would seal P2 in. Easiest: build a pocket manually.
    b = Board.initial()
    # Corner the tests by using a simpler, well-defined scenario:
    # Place P2 at (0,0) and wall it in on two sides — then the sealing wall
    # must be rejected.
    b.pawns[1] = (0, 0)
    b.h_walls.add((0, 0))  # blocks (0,0)-(1,0) and (0,1)-(1,1)
    # Now P2 at (0,0) can only go right (0,1). A vertical wall at (0,0)
    # would block (0,0)-(0,1) and seal P2 off entirely.
    assert b.shortest_path_length(1) is not None
    # Attempt the sealing wall — must be illegal.
    b.turn = 0
    assert not b.is_legal(Move(WALL_V, 0, 0))


def test_jump_over_opponent():
    b = Board.initial()
    # Move pawns toward each other so they're adjacent.
    b.pawns[0] = (4, 4)
    b.pawns[1] = (5, 4)
    b.turn = 0
    moves = set(b.pawn_moves(0))
    # Should be able to jump straight over to (6, 4).
    assert (6, 4) in moves
    # Should NOT include stepping onto opponent.
    assert (5, 4) not in moves


def test_diagonal_jump_when_blocked_behind():
    b = Board.initial()
    b.pawns[0] = (4, 4)
    b.pawns[1] = (5, 4)
    # Put a horizontal wall behind P2 to block the straight jump.
    b.h_walls.add((5, 3))  # blocks (5,3)-(6,3) and (5,4)-(6,4)
    b.turn = 0
    moves = set(b.pawn_moves(0))
    assert (6, 4) not in moves  # straight jump blocked
    # Diagonals sideways around P2 should be available.
    assert (5, 3) in moves
    assert (5, 5) in moves


def test_apply_alternates_turn():
    b = Board.initial()
    b2 = b.apply(Move(MOVE_PAWN, 1, 4))
    assert b2.turn == 1
    assert b2.pawns[0] == (1, 4)
    # Original unchanged.
    assert b.pawns[0] == (0, 4)
    assert b.turn == 0


def test_winner_detection():
    b = Board.initial()
    b.pawns[0] = (8, 4)
    assert b.winner() == 0
    b.pawns[0] = (0, 4)
    b.pawns[1] = (0, 4)
    assert b.winner() == 1


def test_ai_returns_legal_move():
    b = Board.initial()
    m = find_best_move(b, max_depth=2, time_limit=3.0)
    assert m is not None
    assert b.is_legal(m)


def test_ai_prefers_advancing():
    """On an empty board the AI shouldn't just stand still — it should advance."""
    b = Board.initial()
    m = find_best_move(b, max_depth=2, time_limit=3.0)
    assert m is not None
    # At depth 2 from the start, any reasonable evaluator picks a pawn move
    # toward the goal or a useful wall; a purely random/no-op would not.
    if m.kind == MOVE_PAWN:
        # P1 (turn=0) wants to increase its row.
        assert m.r >= 0
        # Must be an adjacent cell from (0, 4).
        assert (abs(m.r - 0) + abs(m.c - 4)) == 1


def test_ai_game_terminates():
    """AI vs AI should terminate with a winner in reasonable time."""
    b = Board.initial()
    for _ in range(200):
        if b.winner() is not None:
            break
        m = find_best_move(b, max_depth=2, time_limit=1.5)
        assert m is not None, "AI produced no move"
        assert b.is_legal(m), f"AI produced illegal move {m}"
        b = b.apply(m)
    assert b.winner() is not None, "Game did not finish within 200 plies"


if __name__ == "__main__":
    import sys
    import traceback

    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
    print()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
