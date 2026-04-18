"""Tests for the ML-facing pieces: encoding round-trips and the game DB.

These tests do not require PyTorch. Only numpy is needed, and numpy is
already a transitive dependency of torch so practically always present.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from quoridor import (
    BOARD_SIZE,
    Board,
    GameDB,
    MOVE_PAWN,
    Move,
    WALL_H,
    WALL_V,
)
from quoridor.encoding import (
    ACTION_H_BASE,
    ACTION_SPACE,
    ACTION_V_BASE,
    NUM_PLANES,
    action_to_move,
    canonical_view,
    deserialize_policy,
    encode_state,
    legal_action_mask,
    move_to_action,
    serialize_policy,
    value_target,
)


# -------------------------------------------------------------------
# Encoding
# -------------------------------------------------------------------

def test_action_space_layout():
    # Pawn cells: 81. Walls: 64 h + 64 v. Total 209.
    assert ACTION_H_BASE == 81
    assert ACTION_V_BASE == 145
    assert ACTION_SPACE == 209


def test_encode_state_shape_and_planes_p1():
    b = Board.initial()
    s = encode_state(b)
    assert s.shape == (NUM_PLANES, BOARD_SIZE, BOARD_SIZE)
    assert s.dtype == np.float32
    # P1 (side-to-move) at e1 = (0, 4), opp at e9 = (8, 4).
    assert s[0, 0, 4] == 1.0
    assert s[1, 8, 4] == 1.0
    assert s[0].sum() == 1.0
    assert s[1].sum() == 1.0
    # Wall planes empty on a fresh board.
    assert s[2].sum() == 0.0
    assert s[3].sum() == 0.0
    # Walls-left normalised to 1.0 (10/10).
    assert np.allclose(s[4], 1.0)
    assert np.allclose(s[5], 1.0)
    assert np.allclose(s[6], 1.0)


def test_encode_state_flips_for_p2():
    # After P1 moves, it's P2's turn. The canonical view should show P2
    # near row 0 and P1 near row 8.
    b = Board.initial().apply(Move(MOVE_PAWN, 1, 4))
    assert b.turn == 1
    s = encode_state(b)
    # "me" is P2, originally at (8, 4) -> canonical (0, 4).
    assert s[0, 0, 4] == 1.0
    # "opp" is P1, now at (1, 4) -> canonical (7, 4).
    assert s[1, 7, 4] == 1.0


def test_pawn_action_round_trip_both_sides():
    b = Board.initial()
    _, _, _, _, _, _, flipped_p1 = canonical_view(b)
    assert flipped_p1 is False
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            m = Move(MOVE_PAWN, r, c)
            idx = move_to_action(m, flipped_p1)
            back = action_to_move(idx, flipped_p1)
            assert back == m

    b2 = b.apply(Move(MOVE_PAWN, 1, 4))  # now P2 to move
    _, _, _, _, _, _, flipped_p2 = canonical_view(b2)
    assert flipped_p2 is True
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            m = Move(MOVE_PAWN, r, c)
            idx = move_to_action(m, flipped_p2)
            back = action_to_move(idx, flipped_p2)
            assert back == m


def test_wall_action_round_trip_both_sides():
    b = Board.initial()
    _, _, _, _, _, _, flipped_p1 = canonical_view(b)
    b2 = b.apply(Move(MOVE_PAWN, 1, 4))
    _, _, _, _, _, _, flipped_p2 = canonical_view(b2)

    for kind in (WALL_H, WALL_V):
        for r in range(8):
            for c in range(8):
                m = Move(kind, r, c)
                for flipped in (flipped_p1, flipped_p2):
                    idx = move_to_action(m, flipped)
                    assert 0 <= idx < ACTION_SPACE
                    back = action_to_move(idx, flipped)
                    assert back == m


def test_legal_action_mask_matches_legal_moves():
    b = Board.initial()
    mask = legal_action_mask(b)
    assert mask.dtype == np.bool_
    assert mask.shape == (ACTION_SPACE,)
    assert mask.sum() == len(b.legal_moves())


def test_value_target_mapping():
    # side_to_move won
    assert value_target(0, 0) == 1.0
    assert value_target(1, 1) == 1.0
    # side_to_move lost
    assert value_target(0, 1) == -1.0
    assert value_target(1, 0) == -1.0
    # unfinished
    assert value_target(None, 0) == 0.0


# -------------------------------------------------------------------
# Database round-trip
# -------------------------------------------------------------------

def _tmp_db() -> str:
    fd, path = tempfile.mkstemp(prefix="quoridor_test_", suffix=".db")
    os.close(fd)
    return path


def test_db_save_and_load_moves():
    path = _tmp_db()
    try:
        with GameDB(path) as db:
            moves = [
                Move(MOVE_PAWN, 1, 4),   # P1
                Move(MOVE_PAWN, 7, 4),   # P2
                Move(WALL_H, 3, 3),      # P1 places wall
                Move(MOVE_PAWN, 6, 4),   # P2
            ]
            gid = db.save_game(
                moves,
                winner=None,
                p1_source="test",
                p2_source="test",
                elapsed_ms=[10, 12, 14, 16],
            )
            assert isinstance(gid, int)
            assert db.count_games() == 1
            assert db.count_positions(finished_only=False) == 4

            loaded = db.load_moves(gid)
            assert loaded == moves
    finally:
        os.remove(path)


def test_db_iter_training_samples_side_and_z():
    path = _tmp_db()
    try:
        with GameDB(path) as db:
            # Fabricate a trivial 2-ply "game" and mark P1 as winner.
            moves = [Move(MOVE_PAWN, 1, 4), Move(MOVE_PAWN, 7, 4)]
            db.save_game(
                moves,
                winner=0,
                p1_source="test",
                p2_source="test",
            )
            samples = list(db.iter_training_samples())
            assert len(samples) == 2

            b0, m0, z0 = samples[0]
            b1, m1, z1 = samples[1]
            # First ply: it was P1's turn, and P1 won.
            assert b0.turn == 0
            assert m0 == moves[0]
            assert z0 == 1.0
            # Second ply: it was P2's turn, and P2 lost.
            assert b1.turn == 1
            assert m1 == moves[1]
            assert z1 == -1.0
    finally:
        os.remove(path)


def test_db_unfinished_game_has_zero_z():
    path = _tmp_db()
    try:
        with GameDB(path) as db:
            db.save_game(
                [Move(MOVE_PAWN, 1, 4)],
                winner=None,
                p1_source="test",
                p2_source="test",
            )
            # finished_only default -> unfinished games are skipped.
            assert list(db.iter_training_samples()) == []
            # But include_unfinished surfaces them with z=0.
            samples = list(db.iter_training_samples(include_unfinished=True))
            assert len(samples) == 1
            _, _, z = samples[0]
            assert z == 0.0
    finally:
        os.remove(path)


# -------------------------------------------------------------------
# Policy serialisation round-trip
# -------------------------------------------------------------------

def test_serialize_deserialize_policy_round_trip():
    rng = np.random.RandomState(42)
    policy = rng.dirichlet(np.ones(ACTION_SPACE)).astype(np.float32)
    blob = serialize_policy(policy)
    assert isinstance(blob, bytes)
    recovered = deserialize_policy(blob)
    assert recovered.shape == (ACTION_SPACE,)
    np.testing.assert_allclose(recovered, policy, atol=1e-7)


def test_db_policy_blob_round_trip():
    """Save a game with policy blobs and read them back."""
    path = _tmp_db()
    try:
        rng = np.random.RandomState(7)
        pol0 = rng.dirichlet(np.ones(ACTION_SPACE)).astype(np.float32)
        pol1 = rng.dirichlet(np.ones(ACTION_SPACE)).astype(np.float32)
        moves = [Move(MOVE_PAWN, 1, 4), Move(MOVE_PAWN, 7, 4)]
        blobs = [serialize_policy(pol0), serialize_policy(pol1)]
        with GameDB(path) as db:
            gid = db.save_game(
                moves,
                winner=0,
                p1_source="selfplay_nn",
                p2_source="selfplay_nn",
                policies=blobs,
            )
            loaded_blobs = db.load_policy_blobs(gid)
            assert len(loaded_blobs) == 2
            np.testing.assert_allclose(
                deserialize_policy(loaded_blobs[0]), pol0, atol=1e-7,
            )
            np.testing.assert_allclose(
                deserialize_policy(loaded_blobs[1]), pol1, atol=1e-7,
            )
    finally:
        os.remove(path)
