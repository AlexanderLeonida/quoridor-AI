"""Build a verified training set from positions in human-vs-NN games.

For every position in the user's wins where the bot was on turn, run
depth-N alpha-beta to determine ground-truth best move.  Save (state,
one-hot-policy on AB's move, value=0) triples as training data.

Optionally synthesizes nearby positions:
  - Move opponent pawn by ±1 in legal directions
  - Toggle one wall on the periphery
For each synthesized position, AB target is computed too — giving a
much larger dataset than just the original ~120 bot-turn positions.

Time: depth-8 AB at 1-2s per position × 1000 positions / 2 workers
      ≈ 8-15 min.  Runs alongside r6 with low contention if r6 uses
      6 workers (leaving 2 free).

Output: a .npz file with arrays (states, policy_targets, value_targets,
weights) — ready to be loaded by a future targeted-training script.
"""
from __future__ import annotations

import argparse
import os
import pickle
import random
import sqlite3
import time
from typing import List, Optional, Tuple

import numpy as np

from quoridor import Board, GameDB
from quoridor.ai import find_best_move
from quoridor.board import (
    BOARD_SIZE, MOVE_PAWN, Move, WALL_GRID, WALL_H, WALL_V,
)
from quoridor.encoding import (
    ACTION_SPACE, action_to_move, canonical_view, encode_state, move_to_action,
)


# -------------------------------------------------------------
# Mining real positions from human wins
# -------------------------------------------------------------
def mine_real_positions(db: GameDB, exclude_game_ids: List[int]) -> List[Tuple[Board, int]]:
    """Returns (board, bot_side) for every bot-turn position in human wins."""
    out = []
    rows = list(db.iter_games(finished_only=False))
    for r in rows:
        gid, _created, _finished, winner, n_plies, p1s, p2s = r[:7]
        mv = r[7] if len(r) > 7 else None
        if not (mv and mv.startswith("gui-nn")):
            continue
        if (p1s == "human" and p2s == "neural_nn"):
            bot_side = 1
        elif (p1s == "neural_nn" and p2s == "human"):
            bot_side = 0
        else:
            continue
        if winner != (1 - bot_side):
            continue   # human win only
        if gid in exclude_game_ids:
            continue
        moves = db.load_moves(gid)
        board = Board.initial()
        for mv_obj in moves:
            if board.turn == bot_side:
                # Mine when bot is tied or losing the race - that's where
                # defensive understanding matters most.
                d_bot = board.shortest_path_length(bot_side) or 0
                d_human = board.shortest_path_length(1 - bot_side) or 0
                if d_human <= d_bot + 2:   # tied or up to 2 squares behind
                    out.append((board, bot_side))
            board = board.apply(mv_obj)
    return out


# -------------------------------------------------------------
# Permutations of a position
# -------------------------------------------------------------
def permute_position(board: Board, n: int = 5, seed: int = 0) -> List[Board]:
    """Generate near-by positions by tweaking pawn locations / walls."""
    rng = random.Random(seed + hash(str(board.pawns) + str(sorted(board.h_walls))))
    out = []
    for _ in range(n * 3):  # over-sample, filter to legal
        if len(out) >= n:
            break
        new_pawns = list(board.pawns)
        # 50/50: move opponent pawn or shift a wall
        if rng.random() < 0.5:
            # Move opponent pawn 1 square in random direction
            opp = 1 - board.turn
            dr, dc = rng.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
            nr, nc = new_pawns[opp][0] + dr, new_pawns[opp][1] + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and (nr, nc) != new_pawns[board.turn]:
                new_pawns[opp] = (nr, nc)
                try:
                    cand = Board(
                        pawns=list(new_pawns),
                        walls_left=list(board.walls_left),
                        h_walls=set(board.h_walls),
                        v_walls=set(board.v_walls),
                        turn=board.turn,
                    )
                    if (cand.shortest_path_length(0) is not None
                        and cand.shortest_path_length(1) is not None):
                        out.append(cand)
                except Exception:
                    pass
        else:
            # Toggle one wall: add a random wall not currently placed
            r_wall = rng.randint(0, WALL_GRID - 1)
            c_wall = rng.randint(0, WALL_GRID - 1)
            kind = rng.choice([WALL_H, WALL_V])
            if (r_wall, c_wall) in board.h_walls or (r_wall, c_wall) in board.v_walls:
                continue
            new_h = set(board.h_walls)
            new_v = set(board.v_walls)
            if kind == WALL_H:
                new_h.add((r_wall, c_wall))
            else:
                new_v.add((r_wall, c_wall))
            try:
                cand = Board(
                    pawns=list(board.pawns),
                    walls_left=list(board.walls_left),
                    h_walls=new_h,
                    v_walls=new_v,
                    turn=board.turn,
                )
                # The added wall must not block either player from reaching goal
                if (cand.shortest_path_length(0) is not None
                    and cand.shortest_path_length(1) is not None):
                    out.append(cand)
            except Exception:
                pass
    return out


# -------------------------------------------------------------
# AB ground truth + dataset assembly
# -------------------------------------------------------------
def position_to_training_example(
    board: Board, ab_depth: int, ab_time: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Run AB on the position, return (state, one-hot-policy, value=0)."""
    mv = find_best_move(board, max_depth=ab_depth, time_limit=ab_time)
    if mv is None:
        return None
    _, _, _, _, _, _, flipped = canonical_view(board)
    action_idx = move_to_action(mv, flipped)
    state = encode_state(board)
    policy = np.zeros(ACTION_SPACE, dtype=np.float32)
    policy[action_idx] = 1.0
    return state, policy, 0.0


# Module-level worker function (must be pickleable for multiprocessing.Pool)
_AB_DEPTH = 8
_AB_TIME = 2.0


def _worker_init(ab_depth: int, ab_time: float):
    global _AB_DEPTH, _AB_TIME
    _AB_DEPTH = ab_depth
    _AB_TIME = ab_time
    import torch
    torch.set_num_threads(1)


def _worker_run(arg):
    board, src = arg
    try:
        ex = position_to_training_example(board, _AB_DEPTH, _AB_TIME)
        return src, ex
    except Exception:
        return src, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/quoridor_v3.db")
    p.add_argument("--exclude-games", type=int, nargs="*", default=[9083],
                   help="Game IDs to exclude (e.g., the buggy human-walls one)")
    p.add_argument("--ab-depth", type=int, default=8)
    p.add_argument("--ab-time", type=float, default=2.0)
    p.add_argument("--permutations-per-position", type=int, default=8)
    p.add_argument("--workers", type=int, default=2,
                   help="Parallel AB jobs (low to coexist with r6)")
    p.add_argument("--out", default="data/human_training_set.npz")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    db = GameDB(args.db)
    print(f"Mining real positions from {args.db} (excluding {args.exclude_games})...")
    real_positions = mine_real_positions(db, args.exclude_games)
    print(f"  {len(real_positions)} real bot-turn positions")

    # Generate synthetic positions
    print(f"Generating {args.permutations_per_position} permutations per real position...")
    all_jobs: List[Tuple[Board, str]] = []  # (board, source_label)
    for i, (board, _bot_side) in enumerate(real_positions):
        all_jobs.append((board, f"real_{i}"))
        perms = permute_position(board, n=args.permutations_per_position, seed=args.seed + i)
        for j, perm in enumerate(perms):
            all_jobs.append((perm, f"perm_{i}_{j}"))
    print(f"  total positions to process: {len(all_jobs)}")

    print(f"\nRunning depth-{args.ab_depth} alpha-beta on {len(all_jobs)} positions "
          f"({args.workers} workers, ~{args.ab_time}s each)...")
    t0 = time.perf_counter()

    # Use process pool with module-level worker
    import multiprocessing as mp
    ctx = mp.get_context("spawn")

    states_l = []
    policies_l = []
    values_l = []
    weights_l = []
    sources_l = []
    n_done = 0
    n_real_examples = 0

    with ctx.Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(args.ab_depth, args.ab_time),
    ) as pool:
        for src, ex in pool.imap_unordered(_worker_run, all_jobs, chunksize=2):
            n_done += 1
            if ex is not None:
                state, policy, value = ex
                states_l.append(state)
                policies_l.append(policy)
                values_l.append(value)
                # Real positions weighted higher than permutations
                w = 3.0 if src.startswith("real_") else 1.0
                weights_l.append(w)
                sources_l.append(src)
                if src.startswith("real_"):
                    n_real_examples += 1
            if n_done % 20 == 0:
                rate = n_done / (time.perf_counter() - t0)
                eta = (len(all_jobs) - n_done) / max(rate, 1e-3)
                print(f"  {n_done}/{len(all_jobs)} done  ({rate:.1f}/s, eta {eta:.0f}s)")

    elapsed = time.perf_counter() - t0
    print(f"\nFinished {len(states_l)} examples ({n_real_examples} real, "
          f"{len(states_l) - n_real_examples} permuted) in {elapsed:.0f}s")

    # Save
    states = np.stack(states_l)
    policies = np.stack(policies_l)
    values = np.array(values_l, dtype=np.float32)
    weights = np.array(weights_l, dtype=np.float32)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        states=states, policies=policies, values=values, weights=weights,
    )
    # Wall-target stats
    n_wall_target = 0
    for pol in policies_l:
        idx = int(np.argmax(pol))
        # Reverse-decode: action_idx -> kind
        # Pawn moves are 0..80 (9*9 - well, see encoding); walls beyond.
        # Easiest: re-encode and check.
    # Actually just count by re-decoding sample
    from quoridor.encoding import action_to_move
    wall_targets = 0
    pawn_targets = 0
    for pol, src in zip(policies_l, sources_l):
        idx = int(np.argmax(pol))
        mv = action_to_move(idx, False)
        if mv.kind == MOVE_PAWN:
            pawn_targets += 1
        else:
            wall_targets += 1
    print(f"  AB picked walls in {wall_targets}/{len(policies_l)} positions "
          f"({wall_targets/len(policies_l)*100:.0f}%); pawns in {pawn_targets}")
    print(f"Saved {args.out}  shape: states={states.shape}, "
          f"policies={policies.shape}, values={values.shape}")


if __name__ == "__main__":
    main()
