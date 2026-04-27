"""AlphaZero self-play training pipeline for Quoridor.

Orchestrates the full training loop used by DeepMind's AlphaZero and
Meta's follow-on work:

    1. **Self-play** — generate games using MCTS guided by the current
       neural network.  Visit-count distributions are stored as soft
       policy targets.
    2. **Training** — update the network on recent self-play data with
       cross-entropy (soft targets) + MSE (value) + L2 regularisation.
    3. **Evaluation / gating** — pit the newly trained network against
       the current best.  Promote only if score > threshold.
    4. **Repeat.**

Key improvements over vanilla AlphaZero for Quoridor:

    - **Draw penalty**: draws receive z = -draw_penalty (default -0.1)
      instead of 0, preventing the "safe stalling" equilibrium.
    - **Progress-aware draw values**: when a game ends in a draw, the
      shortest-path difference at the final position is used to give
      partial credit to the side that was closer to winning.
    - **Opening randomisation**: the first N plies are played randomly
      to break mirror symmetry and increase data diversity.
    - **Evaluation with noise**: eval games use randomised openings
      and slight temperature to produce decisive results.

Usage
-----
    python3 selfplay.py --iterations 100 --games-per-iter 50 \\
                        --simulations 400 --checkpoint-dir checkpoints/

    python3 selfplay.py --resume checkpoints/best.pt --iterations 50

    # Quick smoke-test:
    python3 selfplay.py --iterations 2 --games-per-iter 4 \\
                        --simulations 50 --eval-games 4 --epochs 2
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from quoridor import Board, GameDB
from quoridor.encoding import (
    ACTION_SPACE,
    COL_FLIP_PERM,
    action_to_move,
    canonical_view,
    col_flip_policy,
    col_flip_state,
    deserialize_policy,
    encode_state,
    move_to_action,
    serialize_policy,
)

# numpy alias used under augmentation — indexing a numpy array with
# another numpy array avoids the torch.tensor overhead we'd otherwise pay.
COL_FLIP_PERM_NP = COL_FLIP_PERM
from quoridor.mcts import (
    EvalCache,
    MCTSConfig,
    get_policy,
    root_value,
    search,
    select_action,
)
from quoridor.net import (
    best_available_device,
    build_net,
    load_checkpoint,
    save_checkpoint,
)


# ======================================================================
# Opening randomisation
# ======================================================================

def _randomise_opening(board: Board, num_random: int) -> Tuple[Board, List]:
    """Play *num_random* uniformly random legal moves from *board*.

    Returns the resulting board and the list of (Board, Move) pairs
    played (needed so the recorder can capture the opening moves too).
    """
    history: List[Tuple[Board, "Move"]] = []
    for _ in range(num_random):
        legal = board.legal_moves()
        if not legal or board.winner() is not None:
            break
        move = random.choice(legal)
        history.append((board, move))
        board = board.apply(move)
    return board, history


# ======================================================================
# Draw value computation and adjudication
# ======================================================================

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


def _draw_z(
    final_board: Board,
    side: int,
    draw_penalty: float,
    *,
    game_length: Optional[int] = None,
    max_moves: Optional[int] = None,
    stall_weight: float = 0.4,
    progress_weight: float = 0.5,
) -> float:
    """Compute the value target for a drawn game from *side*'s POV.

    Base: ``-draw_penalty`` (both sides penalised for drawing).

    Stall scaling (when ``game_length`` and ``max_moves`` are given):
        effective_penalty = draw_penalty + stall_weight * (plies / max_moves)

    So a 30-ply premature draw is penalised less than a 90-ply
    max-cutoff stall.  This teaches the net that letting the game run
    to the cutoff is strictly worse than an earlier decisive stall.

    Progress bonus: the side whose shortest path is shorter at the
    final position gets partial credit — a less negative ``z``.
    """
    effective_penalty = draw_penalty
    if game_length is not None and max_moves is not None and max_moves > 0:
        stall_factor = min(game_length / max_moves, 1.0)
        effective_penalty += stall_weight * stall_factor

    d0 = final_board.shortest_path_length(0)
    d1 = final_board.shortest_path_length(1)
    # Positive when P0 is closer to winning.
    raw_progress = (d1 - d0) / 6.0
    progress = math.tanh(raw_progress)  # squash to (-1, 1)
    # From P0's POV the bonus is +progress; from P1's POV it's -progress.
    bonus = progress if side == 0 else -progress
    return float(np.clip(-effective_penalty + bonus * progress_weight, -1.0, 1.0))


# ======================================================================
# Self-play game generation
# ======================================================================

def play_game_vs_alphabeta(
    net,
    config: MCTSConfig,
    device,
    *,
    ab_depth: int = 4,
    ab_time: float = 1.5,
    nn_side: int = 0,
    temp_threshold: int = 15,
    max_moves: int = 120,
    opening_random: int = 0,
    adjudicate_gap: int = 1,
):
    """Play one game NN-vs-alphabeta, returning training data only for
    NN moves (we want to learn from MCTS visits, not alphabeta moves).

    The NN's perspective drives the policy/value targets.  Alphabeta
    plays moves at its turn but those positions are *also* recorded —
    with the NN's MCTS policy at that position, so the net learns to
    answer alphabeta-style threats correctly.

    Same return shape as ``play_game`` so the rest of the pipeline
    (DB writes, training) stays identical.
    """
    from quoridor.ai import find_best_move  # local import; ai.py is heavy

    board = Board.initial()
    if opening_random > 0:
        board, _opening = _randomise_opening(board, opening_random)

    boards: List[Board] = []
    policies: List[np.ndarray] = []
    actions: List[int] = []
    move_num = 0
    cache = EvalCache()

    while board.winner() is None and move_num < max_moves:
        if board.turn == nn_side:
            # NN move — full MCTS, record everything.
            root = search(board, net, config, device, add_noise=True, cache=cache)
            temp = 1.0 if move_num < temp_threshold else 0.0
            policy = get_policy(root, temp)
            action = select_action(root, temp)
            boards.append(board)
            policies.append(policy)
            actions.append(action)
            _, _, _, _, _, _, flipped = canonical_view(board)
            board = board.apply(action_to_move(action, flipped))
        else:
            # Alphabeta move.  Run a one-shot MCTS too to record the
            # net's policy at this position — gives the net training
            # signal about how to *respond* to alphabeta-style play.
            root = search(board, net, config, device, add_noise=False, cache=cache)
            policy = get_policy(root, temperature=1.0)
            mv = find_best_move(board, max_depth=ab_depth, time_limit=ab_time)
            _, _, _, _, _, _, flipped = canonical_view(board)
            ab_action = move_to_action(mv, flipped)
            boards.append(board)
            policies.append(policy)
            actions.append(ab_action)
            board = board.apply(mv)
        move_num += 1

    winner = board.winner()
    if winner is None and adjudicate_gap > 0:
        winner = adjudicate_winner(board, min_gap=adjudicate_gap)
    return boards, policies, actions, winner, board


def play_game(
    net,
    config: MCTSConfig,
    device,
    *,
    temp_threshold: int = 15,
    max_moves: int = 120,
    opening_random: int = 0,
    use_cache: bool = True,
    adjudicate_gap: int = 2,
) -> Tuple[List[Board], List[np.ndarray], List[int], Optional[int], Board]:
    """Play one self-play game using MCTS.

    Returns (boards, policies, actions, winner, final_board).
    ``boards`` / ``policies`` / ``actions`` start after the random opening.
    ``final_board`` is needed for progress-aware draw values.

    If the game reaches ``max_moves`` without a winner, the outcome is
    adjudicated by shortest-path gap: the side with the shorter path
    wins iff the gap is >= ``adjudicate_gap``.  Set to 0 to disable.
    """
    board = Board.initial()

    # Random opening — not recorded as MCTS training data because
    # uniform-random policies are noise, not signal.
    if opening_random > 0:
        board, _opening = _randomise_opening(board, opening_random)

    boards: List[Board] = []
    policies: List[np.ndarray] = []
    actions: List[int] = []
    move_num = 0

    # One NN-eval cache per game so transpositions within the tree and
    # across successive moves reuse forwards. Safe: Zobrist key fully
    # identifies the state.
    cache = EvalCache() if use_cache else None

    # Tree reuse: keep the subtree under the chosen action as the next
    # search's root.  Saves all the NN evaluations already invested
    # under that node — typically 30–50% of simulations are inherited.
    next_root: Optional["Node"] = None  # noqa: F821  (Node imported via mcts)

    while board.winner() is None and move_num < max_moves:
        root = search(
            board, net, config, device,
            add_noise=True, cache=cache, reuse_root=next_root,
        )

        temp = 1.0 if move_num < temp_threshold else 0.0
        policy = get_policy(root, temp)
        action = select_action(root, temp)

        boards.append(board)
        policies.append(policy)
        actions.append(action)

        # Preserve the chosen child's subtree for the next move.
        # If the child wasn't expanded (shouldn't happen when action is
        # sampled from root.children, but be defensive), fall back to a
        # fresh search next move.
        next_root = root.children.get(action)
        if next_root is not None and not next_root.expanded:
            next_root = None

        _, _, _, _, _, _, flipped = canonical_view(board)
        move = action_to_move(action, flipped)
        board = board.apply(move)
        move_num += 1

    winner = board.winner()
    if winner is None and adjudicate_gap > 0:
        winner = adjudicate_winner(board, min_gap=adjudicate_gap)

    return boards, policies, actions, winner, board


def save_game_to_db(
    db: GameDB,
    boards: List[Board],
    policies: List[np.ndarray],
    actions: List[int],
    winner: Optional[int],
    model_version: str,
) -> int:
    """Persist a self-play game into the database."""
    moves = []
    blobs = []
    for board, policy, action in zip(boards, policies, actions):
        _, _, _, _, _, _, flipped = canonical_view(board)
        move = action_to_move(action, flipped)
        moves.append(move)
        blobs.append(serialize_policy(policy))

    return db.save_game(
        moves,
        winner=winner,
        p1_source="selfplay_nn",
        p2_source="selfplay_nn",
        model_version=model_version,
        notes="mcts_selfplay",
        policies=blobs,
    )


def generate_games(
    net,
    config: MCTSConfig,
    device,
    db: GameDB,
    num_games: int,
    model_version: str,
    *,
    temp_threshold: int = 15,
    max_moves: int = 120,
    opening_random: int = 4,
    adjudicate_gap: int = 2,
) -> Dict[str, int]:
    """Generate *num_games* self-play games and save them to *db*."""
    stats: Dict[str, int] = {
        "games": 0, "p1_wins": 0, "p2_wins": 0,
        "draws": 0, "total_moves": 0,
    }
    net.eval()

    for i in range(num_games):
        t0 = time.perf_counter()
        boards, policies, actions, winner, _final = play_game(
            net, config, device,
            temp_threshold=temp_threshold,
            max_moves=max_moves,
            opening_random=opening_random,
            adjudicate_gap=adjudicate_gap,
        )
        gid = save_game_to_db(db, boards, policies, actions, winner, model_version)
        dt = time.perf_counter() - t0

        stats["games"] += 1
        stats["total_moves"] += len(actions)
        if winner == 0:
            stats["p1_wins"] += 1
        elif winner == 1:
            stats["p2_wins"] += 1
        else:
            stats["draws"] += 1

        outcome = "P1" if winner == 0 else ("P2" if winner == 1 else "draw")
        avg = stats["total_moves"] / stats["games"]
        print(
            f"  game {i+1:>3}/{num_games}  id={gid:<5}  "
            f"plies={len(actions):<4}  winner={outcome:<5}  "
            f"{dt:.1f}s  avg_len={avg:.0f}"
        )

    return stats


# ======================================================================
# Parallel self-play (multiprocessing)
# ======================================================================
#
# Self-play is the dominant cost in the pipeline (100 games * ~60s).
# Games are fully independent, so parallelising across CPU worker
# processes is a safe pure-throughput win — same data, same learning
# curve, just generated faster.
#
# Design:
#   - Main process saves the current best net to a checkpoint file.
#   - Each worker is initialised once with that checkpoint, loading the
#     net to CPU (the net is tiny; MPS is not worth the cross-process
#     sharing complexity and small-batch overhead).
#   - Workers play games independently and return the compact (moves,
#     policy_blobs, winner, n_plies, elapsed) tuple.
#   - Main process does all DB writes (SQLite + multiprocess = pain),
#     so this avoids any concurrency concerns on the DB side.
#   - Progress is printed as games stream back via imap_unordered.

_WORKER_NET = None
_WORKER_CONFIG: Optional[MCTSConfig] = None
_WORKER_PLAY_KWARGS: Optional[Dict] = None
_WORKER_DEVICE = None


def _worker_init(
    ckpt_path: str,
    mcts_kwargs: Dict,
    play_kwargs: Dict,
    seed_base: int,
) -> None:
    """Pool initializer: load the net once per worker process."""
    global _WORKER_NET, _WORKER_CONFIG, _WORKER_PLAY_KWARGS, _WORKER_DEVICE
    import os as _os

    import torch  # noqa: WPS433

    # Prevent thread oversubscription when N workers run concurrently.
    torch.set_num_threads(1)

    # Per-worker seeding so parallel games aren't correlated.
    pid = _os.getpid()
    seed = (seed_base + pid) & 0x7FFFFFFF
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    net, _ = load_checkpoint(ckpt_path, map_location="cpu")
    net.to("cpu")
    net.eval()
    _WORKER_NET = net
    _WORKER_CONFIG = MCTSConfig(**mcts_kwargs)
    _WORKER_PLAY_KWARGS = play_kwargs
    _WORKER_DEVICE = torch.device("cpu")


def _worker_play_one(idx: int):
    """Play one self-play or NN-vs-alphabeta game.

    The fraction of games that are NN-vs-AB is set by
    ``ab_mix_frac`` in ``_WORKER_PLAY_KWARGS``; the index ``idx`` is
    used as a deterministic selector so workers don't all pick the
    same kind of game.
    """
    import torch  # noqa: WPS433

    assert _WORKER_NET is not None
    assert _WORKER_CONFIG is not None
    assert _WORKER_PLAY_KWARGS is not None

    t0 = time.perf_counter()
    play_kwargs = dict(_WORKER_PLAY_KWARGS)
    ab_mix_frac = play_kwargs.pop("ab_mix_frac", 0.0)
    ab_depth = play_kwargs.pop("ab_depth", 4)
    ab_time = play_kwargs.pop("ab_time", 1.5)

    is_ab_game = ab_mix_frac > 0.0 and (
        # Hash the index so 0.2 frac → roughly every 5th game is ab.
        ((idx * 2654435761) % 1000) / 1000.0 < ab_mix_frac
    )

    if is_ab_game:
        # Alternate which side the NN plays so the net learns both
        # offence and defence vs alphabeta.
        nn_side = idx % 2
        boards, policies, actions, winner, _final = play_game_vs_alphabeta(
            _WORKER_NET, _WORKER_CONFIG, _WORKER_DEVICE,
            ab_depth=ab_depth, ab_time=ab_time, nn_side=nn_side,
            temp_threshold=play_kwargs.get("temp_threshold", 15),
            max_moves=play_kwargs.get("max_moves", 120),
            opening_random=play_kwargs.get("opening_random", 0),
            adjudicate_gap=play_kwargs.get("adjudicate_gap", 1),
        )
        kind = "ab"
    else:
        boards, policies, actions, winner, _final = play_game(
            _WORKER_NET,
            _WORKER_CONFIG,
            _WORKER_DEVICE,
            **play_kwargs,
        )
        kind = "selfplay"

    moves = []
    blobs = []
    for board, policy, action in zip(boards, policies, actions):
        _, _, _, _, _, _, flipped = canonical_view(board)
        moves.append(action_to_move(action, flipped))
        blobs.append(serialize_policy(policy))
    return moves, blobs, winner, len(actions), time.perf_counter() - t0, kind


def generate_games_parallel(
    best_net,
    config: MCTSConfig,
    db: GameDB,
    num_games: int,
    model_version: str,
    checkpoint_dir: str,
    *,
    temp_threshold: int,
    max_moves: int,
    opening_random: int,
    num_workers: int,
    adjudicate_gap: int = 2,
    ab_mix_frac: float = 0.0,
    ab_depth: int = 4,
    ab_time: float = 1.5,
) -> Dict[str, int]:
    """Parallel self-play across *num_workers* CPU processes.

    A fraction ``ab_mix_frac`` of games are NN-vs-alphabeta (depth
    ``ab_depth``, ``ab_time`` seconds budget) — this introduces a
    fundamentally different opponent than the net itself, breaking
    the self-imitation loop that drives drift.
    """
    import multiprocessing as mp
    from dataclasses import asdict

    # Snapshot the current net to disk for workers to load.
    os.makedirs(checkpoint_dir, exist_ok=True)
    snap_path = os.path.join(checkpoint_dir, "_worker_net.pt")
    save_checkpoint(best_net, snap_path, iteration=-1, note="worker_snapshot")

    play_kwargs = {
        "temp_threshold": temp_threshold,
        "max_moves": max_moves,
        "opening_random": opening_random,
        "adjudicate_gap": adjudicate_gap,
        "ab_mix_frac": ab_mix_frac,
        "ab_depth": ab_depth,
        "ab_time": ab_time,
    }
    seed_base = random.randint(0, 2**30)
    initargs = (snap_path, asdict(config), play_kwargs, seed_base)

    ctx = mp.get_context("spawn")
    stats: Dict[str, int] = {
        "games": 0, "p1_wins": 0, "p2_wins": 0,
        "draws": 0, "total_moves": 0, "ab_games": 0,
    }

    print(f"  (parallel self-play on {num_workers} CPU workers)")
    with ctx.Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=initargs,
    ) as pool:
        for moves, blobs, winner, n_plies, elapsed, kind in pool.imap_unordered(
            _worker_play_one, range(num_games)
        ):
            # Keep both sources as 'selfplay_nn' so the training
            # loader picks the games up; distinguish via ``notes``.
            note = "nn_vs_alphabeta" if kind == "ab" else "mcts_selfplay"
            gid = db.save_game(
                moves,
                winner=winner,
                p1_source="selfplay_nn",
                p2_source="selfplay_nn",
                model_version=model_version,
                notes=note,
                policies=blobs,
            )
            stats["games"] += 1
            stats["total_moves"] += n_plies
            if kind == "ab":
                stats["ab_games"] = stats.get("ab_games", 0) + 1
            if winner == 0:
                stats["p1_wins"] += 1
            elif winner == 1:
                stats["p2_wins"] += 1
            else:
                stats["draws"] += 1
            outcome = "P1" if winner == 0 else ("P2" if winner == 1 else "draw")
            avg = stats["total_moves"] / stats["games"]
            tag = " [AB]" if kind == "ab" else ""
            print(
                f"  game {stats['games']:>3}/{num_games}  id={gid:<5}  "
                f"plies={n_plies:<4}  winner={outcome:<5}  "
                f"{elapsed:.1f}s  avg_len={avg:.0f}{tag}"
            )

    # Snapshot is regenerated every iteration; leave it in place so the
    # most recent weights remain inspectable, but it's not essential.
    return stats


# ======================================================================
# Training on self-play data
# ======================================================================

def train_on_recent_games(
    net,
    db: GameDB,
    device,
    *,
    max_games: int = 1000,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    val_frac: float = 0.05,
    draw_penalty: float = 0.1,
    max_moves: int = 90,
    policy_temp: float = 1.0,
    value_weight: float = 1.0,
    augment: bool = True,
    warmup_frac: float = 0.05,
    min_version_iter: Optional[int] = None,
    extra_examples: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    aux_value_weight: float = 0.0,
) -> Tuple:
    """Train *net* on the most recent games from *db*.

    Returns (net, metrics_dict).

    *policy_temp* sharpens soft policy targets when < 1.0:
    ``target = counts ** (1/temp) / sum``.  Default 1.0 = no change.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    # --- materialise dataset (game-level train/val split) ---
    # Only use self-play games (exclude human/alphabeta games that may be
    # in the same DB).  We keep draws (winner=None) from self-play since
    # max-moves cutoffs are legitimate training data with progress-aware
    # value targets.  The source filter already excludes aborted human
    # games whose winner=None would inject noisy labels.
    def _accept_version(mv) -> bool:
        # model_version is a string column (row[9]).  Tournament-
        # champion games (prefix "tourney-") are always kept since
        # those are deliberately high-quality supervision.  For self-
        # play versions tagged "selfplay-vNN", optionally filter to
        # games from the current-best iteration or newer so a chain of
        # rejected candidates' self-play data doesn't pull the net
        # back toward their (by-definition-weaker) distribution.
        if not isinstance(mv, str):
            return True
        if mv.startswith("tourney-"):
            return True
        if min_version_iter is None or not mv.startswith("selfplay-v"):
            return True
        try:
            n = int(mv[len("selfplay-v"):])
        except ValueError:
            return True
        return n >= min_version_iter

    all_games = [
        row for row in db.iter_games(finished_only=False)
        if row[5] == "selfplay_nn" and row[6] == "selfplay_nn"
        and _accept_version(row[7])
    ]
    all_games = all_games[-max_games:]
    if not all_games:
        print("  No games to train on.")
        return net, {}

    # Split at the GAME level to avoid position leakage.
    game_idx = np.arange(len(all_games))
    np.random.shuffle(game_idx)
    n_val_games = max(1, int(len(all_games) * val_frac)) if (
        val_frac > 0 and len(all_games) > 10
    ) else 0
    val_game_set = set(game_idx[:n_val_games].tolist())

    def _load_games(game_rows):
        states_l: List[np.ndarray] = []
        pols_l: List[np.ndarray] = []
        vals_l: List[float] = []
        weights_l: List[float] = []
        for row in game_rows:
            game_id = row[0]
            winner = row[3]
            # iter_games returns: (id, created_at, finished_at, winner,
            # num_plies, p1_source, p2_source, model_version) — 8 cols.
            mv_str = row[7] if len(row) > 7 else None
            moves = db.load_moves(game_id)
            blobs = db.load_policy_blobs(game_id)
            # Replay to find final board.
            board = Board.initial()
            boards_in_game: List[Board] = []
            for move in moves:
                boards_in_game.append(board)
                board = board.apply(move)
            final_board = board
            # Per-position weight: inversely proportional to game length
            # so that shorter decisive games aren't drowned by long draws.
            # Upweight decisive games 4× so the signal isn't dominated by
            # draw positions.  Tournament-champion games (model_version
            # prefix "tourney-") get an additional 2× boost — these are
            # the highest-quality supervision we have, since by
            # construction the player who generated them outranked the
            # rest of the pool by Elo.  This is a soft form of
            # hard-example mining: positions where a known-strong net
            # made a winning choice are over-represented in the gradient.
            game_len = max(len(moves), 1)
            decisive_mult = 4.0 if winner is not None else 1.0
            tourney_mult = 2.0 if (
                isinstance(mv_str, str) and mv_str.startswith("tourney-")
            ) else 1.0
            w = decisive_mult * tourney_mult / game_len
            for brd, move, blob in zip(boards_in_game, moves, blobs):
                states_l.append(encode_state(brd))
                if blob is not None:
                    pol = deserialize_policy(blob)
                    if policy_temp < 1.0:
                        pol = pol ** (1.0 / policy_temp)
                        pol_sum = pol.sum()
                        if pol_sum > 0:
                            pol /= pol_sum
                    pols_l.append(pol)
                else:
                    _, _, _, _, _, _, flipped = canonical_view(brd)
                    act = move_to_action(move, flipped)
                    onehot = np.zeros(ACTION_SPACE, dtype=np.float32)
                    onehot[act] = 1.0
                    pols_l.append(onehot)
                if winner is not None:
                    z = 1.0 if winner == brd.turn else -1.0
                else:
                    z = _draw_z(
                        final_board, brd.turn, draw_penalty,
                        game_length=len(moves), max_moves=max_moves,
                    )
                # Auxiliary value: blend in tanh-normalised shortest-
                # path differential from side-to-move's POV.  Provides
                # dense supervision for the value head independent of
                # the (often noisy) game outcome — especially useful
                # mid-game when the outcome is many plies away.
                if aux_value_weight > 0.0:
                    me = brd.turn
                    opp = 1 - me
                    d_me = brd.shortest_path_length(me)
                    d_opp = brd.shortest_path_length(opp)
                    if d_me is not None and d_opp is not None:
                        path_signal = math.tanh((d_opp - d_me) / 6.0)
                        z = (1.0 - aux_value_weight) * z + aux_value_weight * path_signal
                        z = float(np.clip(z, -1.0, 1.0))
                vals_l.append(z)
                weights_l.append(w)
        return states_l, pols_l, vals_l, weights_l

    train_games = [all_games[i] for i in range(len(all_games)) if i not in val_game_set]
    val_games = [all_games[i] for i in val_game_set]

    tr_s, tr_p, tr_v, tr_w = _load_games(train_games)
    n = len(tr_s)
    if n == 0:
        print("  No training positions.")
        return net, {}

    n_draws = sum(1 for r in all_games if r[3] is None)
    n_decisive = len(all_games) - n_draws
    print(
        f"  {n + sum(len(db.load_moves(g[0])) for g in val_games):,} positions "
        f"from {len(all_games)} game(s) ({n_decisive} decisive, {n_draws} draws)"
    )

    tr_s_np = np.stack(tr_s)
    tr_p_np = np.stack(tr_p)
    tr_v_np = np.array(tr_v, dtype=np.float32)
    tr_w_np = np.array(tr_w, dtype=np.float32)

    # Append in-memory hard examples (mined from the rejected
    # candidates' games at the last tournament revert).  These are
    # (state, policy) pairs from the champion's MCTS at positions
    # where the candidate diverged.  Value targets are 0 (we know the
    # policy but not the position's true value).  Weight is 1.0
    # absolute — roughly ~8× the typical normalised self-play weight,
    # enough for the gradient to notice without drowning everything
    # else.
    if extra_examples:
        ex_s = np.stack([s for s, _ in extra_examples])
        ex_p = np.stack([p for _, p in extra_examples])
        ex_v = np.zeros(len(extra_examples), dtype=np.float32)
        ex_w = np.full(len(extra_examples), 1.0, dtype=np.float32)
        tr_s_np = np.concatenate([tr_s_np, ex_s], axis=0)
        tr_p_np = np.concatenate([tr_p_np, ex_p], axis=0)
        tr_v_np = np.concatenate([tr_v_np, ex_v], axis=0)
        tr_w_np = np.concatenate([tr_w_np, ex_w], axis=0)
        n = tr_s_np.shape[0]
        print(f"  +{len(extra_examples)} hard-example positions appended")

    # Column-flip augmentation: Quoridor is symmetric about the central
    # column, so (state, policy) can be mirrored and value is unchanged.
    # Doubling the effective dataset is essentially free and typically
    # halves the overfitting gap on small windows.
    if augment:
        flipped_s = np.ascontiguousarray(tr_s_np[:, :, :, ::-1])
        flipped_p = tr_p_np[:, COL_FLIP_PERM_NP]
        tr_s_np = np.concatenate([tr_s_np, flipped_s], axis=0)
        tr_p_np = np.concatenate([tr_p_np, flipped_p], axis=0)
        tr_v_np = np.concatenate([tr_v_np, tr_v_np], axis=0)
        tr_w_np = np.concatenate([tr_w_np, tr_w_np], axis=0)
        n = tr_s_np.shape[0]

    states = torch.from_numpy(tr_s_np)
    pols = torch.from_numpy(tr_p_np)
    vals = torch.from_numpy(tr_v_np)
    sample_weights = torch.from_numpy(tr_w_np)
    # Normalise weights so they average to 1.
    sample_weights = sample_weights / sample_weights.mean()

    val_loader = None
    if val_games:
        vs, vp, vv, _ = _load_games(val_games)
        if vs:
            val_ds = TensorDataset(
                torch.from_numpy(np.stack(vs)),
                torch.from_numpy(np.stack(vp)),
                torch.from_numpy(np.array(vv, dtype=np.float32)),
            )
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    train_ds = TensorDataset(states, pols, vals, sample_weights)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # --- optimiser ---
    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = epochs * len(train_loader)
    warmup_steps = max(1, int(total_steps * warmup_frac))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)

    # --- training loop ---
    # Track best-val weights so the candidate we ship is the best snapshot
    # we saw during training, not the (typically overfit) end-of-training
    # one.  Without this, val loss tends to rise monotonically after
    # epoch 1 and we ship a degraded net to the gating match.
    best_val_loss: Optional[float] = None
    best_val_state: Optional[dict] = None
    for epoch in range(1, epochs + 1):
        net.train()
        tl = tp = tv = 0.0
        tn = 0
        for xb, pb, vb, wb in train_loader:
            xb = xb.to(device)
            pb = pb.to(device)
            vb = vb.to(device)
            wb = wb.to(device)
            p_logits, v_pred = net(xb)
            log_p = F.log_softmax(p_logits, dim=1)
            # Per-sample weighted losses.
            loss_p_per = -(pb * log_p).sum(dim=1)
            loss_v_per = (v_pred - vb) ** 2
            loss_p = (loss_p_per * wb).mean()
            loss_v = (loss_v_per * wb).mean()
            loss = loss_p + value_weight * loss_v
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            opt.step()
            scheduler.step()
            bs = xb.size(0)
            tl += loss.item() * bs
            tp += loss_p.item() * bs
            tv += loss_v.item() * bs
            tn += bs

        line = (
            f"    epoch {epoch}/{epochs}  "
            f"loss={tl/tn:.4f} (p={tp/tn:.4f} v={tv/tn:.4f})"
        )

        if val_loader:
            net.eval()
            vl = vp2 = vv = 0.0
            vn = 0
            with torch.no_grad():
                for xb, pb, vb in val_loader:
                    xb, pb, vb = xb.to(device), pb.to(device), vb.to(device)
                    p_logits, v_pred = net(xb)
                    log_p = F.log_softmax(p_logits, dim=1)
                    lp = -(pb * log_p).sum(dim=1).mean()
                    lv = F.mse_loss(v_pred, vb)
                    bs = xb.size(0)
                    vl += (lp.item() + lv.item()) * bs
                    vp2 += lp.item() * bs
                    vv += lv.item() * bs
                    vn += bs
            line += f"  val={vl/vn:.4f} (p={vp2/vn:.4f} v={vv/vn:.4f})"
            cur_val = vl / vn
            if best_val_loss is None or cur_val < best_val_loss:
                best_val_loss = cur_val
                best_val_state = {
                    k: v.detach().cpu().clone()
                    for k, v in net.state_dict().items()
                }
                line += "  [best]"

        print(line)

    # Restore the best-val weights (if validation was enabled) so the
    # gating match sees the best snapshot rather than the final one.
    if best_val_state is not None:
        net.load_state_dict(best_val_state)

    metrics = {"train_loss": tl / tn, "policy_loss": tp / tn, "value_loss": tv / tn}
    if best_val_loss is not None:
        metrics["best_val_loss"] = best_val_loss
    return net, metrics


# ======================================================================
# Evaluation (gating)
# ======================================================================

def evaluate_nets(
    net_new,
    net_old,
    device,
    *,
    num_games: int = 20,
    simulations: int = 200,
    opening_random: int = 4,
    max_moves: int = 120,
    eval_temp: float = 0.0,
    eval_temp_moves: int = 0,
    adjudicate_gap: int = 1,
) -> float:
    """Play *num_games* between two networks, return *net_new*'s score.

    Score: win = 1, draw = 0.5, loss = 0 (normalised to [0, 1]).
    Colors alternate each game.  Evaluation uses **no root noise** and
    **greedy** (temperature = 0) play after ``eval_temp_moves`` so that
    the result reflects true model strength, not MCTS randomness. For the
    first ``eval_temp_moves`` plies, a small ``eval_temp`` samples the
    visit counts — this breaks deterministic mirror lines between two
    near-identical nets so decisive games emerge.
    """
    eval_cfg = MCTSConfig(
        num_simulations=simulations,
        dirichlet_epsilon=0.0,  # NO exploration noise in eval
    )
    net_new.eval()
    net_old.eval()

    score = 0.0
    n_wins = n_losses = n_draws = 0
    for g in range(num_games):
        if g % 2 == 0:
            nets = {0: net_new, 1: net_old}
            new_side = 0
        else:
            nets = {0: net_old, 1: net_new}
            new_side = 1

        board = Board.initial()
        if opening_random > 0:
            board, _ = _randomise_opening(board, opening_random)

        # Per-side caches so hits are net-specific (two nets playing).
        caches = {0: EvalCache(), 1: EvalCache()}
        move_count = 0
        while board.winner() is None and move_count < max_moves:
            cur_net = nets[board.turn]
            root = search(
                board, cur_net, eval_cfg, device,
                add_noise=False, cache=caches[board.turn],
            )
            temp = eval_temp if move_count < eval_temp_moves else 0.0
            action = select_action(root, temperature=temp)
            _, _, _, _, _, _, flipped = canonical_view(board)
            move = action_to_move(action, flipped)
            board = board.apply(move)
            move_count += 1

        winner = board.winner()
        if winner is None and adjudicate_gap > 0:
            winner = adjudicate_winner(board, min_gap=adjudicate_gap)
        if winner is not None and winner == new_side:
            score += 1.0
            tag = "W"
            n_wins += 1
        elif winner is not None and winner != new_side:
            tag = "L"
            n_losses += 1
        else:
            score += 0.5
            tag = "D"
            n_draws += 1

        done = g + 1
        pct = score / done
        print(
            f"    eval game {done}/{num_games}  "
            f"new={'P1' if new_side==0 else 'P2'}  "
            f"moves={move_count}  {tag}  "
            f"({pct:.0%} W{n_wins}/L{n_losses}/D{n_draws})"
        )

    return {"score": score / num_games, "wins": n_wins,
            "losses": n_losses, "draws": n_draws}


# ======================================================================
# Parallel evaluation
# ======================================================================
#
# At --eval-games 50 the serial evaluator dominates each iteration's
# wall-clock.  Games between the two fixed nets are independent, so
# parallelising is safe (identical outcome distribution) and scales
# near-linearly with workers.

_EVAL_NEW_NET = None
_EVAL_OLD_NET = None
_EVAL_CFG: Optional[MCTSConfig] = None
_EVAL_KWARGS: Optional[Dict] = None
_EVAL_DEVICE = None


def _eval_worker_init(
    new_ckpt: str,
    old_ckpt: str,
    mcts_kwargs: Dict,
    eval_kwargs: Dict,
    seed_base: int,
) -> None:
    """Pool initializer: load both nets once per worker process."""
    global _EVAL_NEW_NET, _EVAL_OLD_NET, _EVAL_CFG, _EVAL_KWARGS, _EVAL_DEVICE
    import os as _os

    import torch  # noqa: WPS433

    torch.set_num_threads(1)

    pid = _os.getpid()
    seed = (seed_base + pid) & 0x7FFFFFFF
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    new_net, _ = load_checkpoint(new_ckpt, map_location="cpu")
    new_net.to("cpu")
    new_net.eval()
    old_net, _ = load_checkpoint(old_ckpt, map_location="cpu")
    old_net.to("cpu")
    old_net.eval()

    _EVAL_NEW_NET = new_net
    _EVAL_OLD_NET = old_net
    _EVAL_CFG = MCTSConfig(**mcts_kwargs)
    _EVAL_KWARGS = eval_kwargs
    _EVAL_DEVICE = torch.device("cpu")


def _eval_worker_play_one(g: int) -> Tuple[int, float, str, int, int]:
    """Play one eval game from global state.

    Returns (game_idx, score_for_new, tag, move_count, new_side).
    """
    import torch  # noqa: WPS433

    assert _EVAL_NEW_NET is not None and _EVAL_OLD_NET is not None
    assert _EVAL_CFG is not None and _EVAL_KWARGS is not None

    # Alternate colors deterministically by game index — same contract
    # as the serial evaluator so the distribution of (new_side) matches.
    if g % 2 == 0:
        nets = {0: _EVAL_NEW_NET, 1: _EVAL_OLD_NET}
        new_side = 0
    else:
        nets = {0: _EVAL_OLD_NET, 1: _EVAL_NEW_NET}
        new_side = 1

    opening_random = _EVAL_KWARGS["opening_random"]
    max_moves = _EVAL_KWARGS["max_moves"]
    eval_temp = _EVAL_KWARGS.get("eval_temp", 0.0)
    eval_temp_moves = _EVAL_KWARGS.get("eval_temp_moves", 0)
    adjudicate_gap = _EVAL_KWARGS.get("adjudicate_gap", 1)

    board = Board.initial()
    if opening_random > 0:
        board, _ = _randomise_opening(board, opening_random)

    # Per-side caches: each net gets its own memoization (cache hits
    # are only meaningful within one net's turn set).
    caches = {0: EvalCache(), 1: EvalCache()}
    move_count = 0
    while board.winner() is None and move_count < max_moves:
        cur_net = nets[board.turn]
        root = search(
            board, cur_net, _EVAL_CFG, _EVAL_DEVICE,
            add_noise=False, cache=caches[board.turn],
        )
        temp = eval_temp if move_count < eval_temp_moves else 0.0
        action = select_action(root, temperature=temp)
        _, _, _, _, _, _, flipped = canonical_view(board)
        move = action_to_move(action, flipped)
        board = board.apply(move)
        move_count += 1

    winner = board.winner()
    if winner is None and adjudicate_gap > 0:
        winner = adjudicate_winner(board, min_gap=adjudicate_gap)
    if winner is not None and winner == new_side:
        return g, 1.0, "W", move_count, new_side
    if winner is not None and winner != new_side:
        return g, 0.0, "L", move_count, new_side
    return g, 0.5, "D", move_count, new_side


def evaluate_nets_parallel(
    net_new,
    net_old,
    checkpoint_dir: str,
    *,
    num_games: int,
    simulations: int,
    opening_random: int,
    max_moves: int,
    num_workers: int,
    eval_temp: float = 0.0,
    eval_temp_moves: int = 0,
    adjudicate_gap: int = 1,
) -> float:
    """Parallel net-vs-net evaluation.  Returns *net_new*'s score in [0, 1]."""
    import multiprocessing as mp
    from dataclasses import asdict

    os.makedirs(checkpoint_dir, exist_ok=True)
    new_ckpt = os.path.join(checkpoint_dir, "_eval_new.pt")
    old_ckpt = os.path.join(checkpoint_dir, "_eval_old.pt")
    save_checkpoint(net_new, new_ckpt, iteration=-1, note="eval_new")
    save_checkpoint(net_old, old_ckpt, iteration=-1, note="eval_old")

    eval_cfg = MCTSConfig(
        num_simulations=simulations,
        dirichlet_epsilon=0.0,
    )
    eval_kwargs = {
        "opening_random": opening_random,
        "max_moves": max_moves,
        "eval_temp": eval_temp,
        "eval_temp_moves": eval_temp_moves,
        "adjudicate_gap": adjudicate_gap,
    }
    seed_base = random.randint(0, 2**30)
    initargs = (
        new_ckpt, old_ckpt, asdict(eval_cfg), eval_kwargs, seed_base,
    )

    ctx = mp.get_context("spawn")
    score = 0.0
    n_wins = n_losses = n_draws = 0
    done = 0
    print(f"    (parallel evaluation on {num_workers} CPU workers)")
    with ctx.Pool(
        processes=num_workers,
        initializer=_eval_worker_init,
        initargs=initargs,
    ) as pool:
        for g, s, tag, move_count, new_side in pool.imap_unordered(
            _eval_worker_play_one, range(num_games)
        ):
            score += s
            done += 1
            if tag == "W":
                n_wins += 1
            elif tag == "L":
                n_losses += 1
            else:
                n_draws += 1
            pct = score / done
            print(
                f"    eval game {done}/{num_games}  "
                f"(idx {g})  new={'P1' if new_side==0 else 'P2'}  "
                f"moves={move_count}  {tag}  "
                f"({pct:.0%} W{n_wins}/L{n_losses}/D{n_draws})"
            )
    return {"score": score / num_games, "wins": n_wins,
            "losses": n_losses, "draws": n_draws}


# ======================================================================
# Confidence interval for evaluation
# ======================================================================

def wilson_ci(wins: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score 95% confidence interval for a proportion.

    *wins* can be fractional (draws count as 0.5) so we treat it as the
    number of "successes" in *n* Bernoulli trials.
    """
    if n == 0:
        return 0.0, 1.0
    p_hat = wins / n
    denom = 1 + z * z / n
    centre = p_hat + z * z / (2 * n)
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n)
    lo = (centre - spread) / denom
    hi = (centre + spread) / denom
    return max(0.0, lo), min(1.0, hi)


# ======================================================================
# Elo tracking
# ======================================================================

class EloTracker:
    """Simple Elo rating tracker persisted to a JSON file.

    Each model version gets a rating.  After each evaluation match the
    ratings of both participants are updated with the standard Elo
    formula (K=32).  The initial rating is 1000.
    """

    def __init__(self, path: str, k: float = 32.0):
        self.path = path
        self.k = k
        self.ratings: Dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                self.ratings = json.load(f)

    def _save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self.ratings, f, indent=2)

    def _ensure(self, name: str) -> None:
        if name not in self.ratings:
            self.ratings[name] = 1000.0

    def update(self, player_a: str, player_b: str, score_a: float) -> Tuple[float, float]:
        """Update ratings after a match.  *score_a* is in [0, 1].

        Returns (new_rating_a, new_rating_b).
        """
        self._ensure(player_a)
        self._ensure(player_b)
        ra, rb = self.ratings[player_a], self.ratings[player_b]
        ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        eb = 1.0 - ea
        self.ratings[player_a] = ra + self.k * (score_a - ea)
        self.ratings[player_b] = rb + self.k * ((1.0 - score_a) - eb)
        self._save()
        return self.ratings[player_a], self.ratings[player_b]

    def get(self, name: str) -> float:
        self._ensure(name)
        return self.ratings[name]

    def summary(self, n: int = 10) -> str:
        """Return a formatted string of the top-N rated models."""
        if not self.ratings:
            return "  (no ratings yet)"
        ranked = sorted(self.ratings.items(), key=lambda x: -x[1])
        lines = []
        for i, (name, elo) in enumerate(ranked[:n], 1):
            lines.append(f"  {i:>2}. {name:<25s} {elo:7.1f}")
        return "\n".join(lines)


# ======================================================================
# Periodic calibration tournament
# ======================================================================
#
# The per-iteration gating match updates Elo from one head-to-head only,
# so a chain of "just-barely-better" promotions can drift the best net
# downward over many iterations without any single gate match catching
# it.  The fix: every N iterations, play a round-robin among the last K
# promoted nets + a few historical anchors, compute globally-consistent
# Elos (Bradley–Terry MLE), and revert ``best_net`` to the actual
# champion.  Winning games are folded into the training DB so the next
# training pass distils from the champion.

def _run_calibration_tournament(
    args,
    pool: List[Tuple[str, str]],
    anchor_label: str,
):
    """Invoke tournament.py in a subprocess.  Returns (ratings, champion_label).

    ``pool`` is a list of (label, ckpt_path).  We shell out rather than
    call in-process so the pool's multiprocessing doesn't nest badly
    inside the self-play pool.
    """
    import subprocess

    out_path = os.path.join(args.checkpoint_dir, "_tournament_current.json")
    cmd = [
        "python3", "-u", "tournament.py",
        "--games", str(args.tournament_games),
        "--sims", str(args.tournament_sims),
        "--workers", str(max(1, args.workers // 2)),  # share CPU politely
        "--opening-random", str(args.opening_random),
        "--max-moves", str(args.max_moves),
        "--adjudicate-gap", str(max(1, args.adjudicate_gap)),
        "--anchor", anchor_label,
        "--out", out_path,
        "--save-to-db", args.db or "data/quoridor.db",
        "--save-champion-only",
    ]
    for label, path in pool:
        cmd += ["--ckpt", f"{path}:{label}"]
    print(f"  (tournament: {len(pool)} players, "
          f"{args.tournament_games} games/pair, sims={args.tournament_sims})")
    subprocess.run(cmd, check=True)
    import json as _json
    with open(out_path) as f:
        data = _json.load(f)
    ratings = data["ratings"]
    champion = max(ratings, key=ratings.get)
    return ratings, champion


# ----------------------------------------------------------------------
# Hard-example mining on revert
# ----------------------------------------------------------------------

def _mine_hard_examples(
    champion_net,
    db: GameDB,
    device,
    *,
    rejected_versions: List[str],
    sims: int,
    max_positions: int = 1500,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """After a tournament revert, find positions where the rejected
    candidates' self-play moves diverged from what the champion would
    have played.  Returns a list of ``(state_tensor, policy_tensor)``
    pairs — concentrated lessons on "what you should have done
    instead."

    The returned examples are passed directly to the next training
    pass as ``extra_examples`` (in-memory) — we deliberately do *not*
    persist them to the games DB because the DB schema replays moves
    from the initial board, and our examples come from arbitrary mid-
    game positions across different games that can't be reconstructed
    from a flat move list.
    """
    if not rejected_versions:
        return []

    cfg = MCTSConfig(num_simulations=sims, dirichlet_epsilon=0.0)
    cache = EvalCache()
    scanned = 0

    placeholders = ",".join("?" for _ in rejected_versions)
    cur = db._conn.execute(
        f"SELECT id FROM games "
        f"WHERE p1_source='selfplay_nn' AND p2_source='selfplay_nn' "
        f"AND model_version IN ({placeholders}) "
        f"ORDER BY id DESC LIMIT 200",
        rejected_versions,
    )
    game_ids = [row[0] for row in cur.fetchall()]

    examples: List[Tuple[np.ndarray, np.ndarray]] = []
    for game_id in game_ids:
        moves = db.load_moves(game_id)
        if not moves:
            continue
        board = Board.initial()
        for played_move in moves:
            if scanned >= max_positions:
                break
            scanned += 1
            root = search(
                board, champion_net, cfg, device,
                add_noise=False, cache=cache,
            )
            # Soft policy: visit-count distribution, useful as
            # supervision even where the loser happened to agree on
            # the top move (the priors elsewhere still differ).
            soft_policy = get_policy(root, temperature=1.0)
            top_action = int(np.argmax(soft_policy))

            _, _, _, _, _, _, flipped = canonical_view(board)
            played_action = move_to_action(played_move, flipped)
            if top_action != played_action:
                examples.append((
                    encode_state(board),
                    soft_policy.astype(np.float32),
                ))

            board = board.apply(played_move)
        if scanned >= max_positions:
            break
    return examples


# ======================================================================
# Pipeline orchestration
# ======================================================================

def run_pipeline(args) -> None:
    import torch
    from datetime import datetime as _dt

    device = (
        torch.device(args.device) if args.device else best_available_device()
    )
    print(f"Device:            {device}")
    print(f"Workers:           {args.workers}")
    print(f"Simulations/move:  {args.simulations}")
    print(f"Games/iteration:   {args.games_per_iter}")
    print(f"Training epochs:   {args.epochs}")
    print(f"Eval games:        {args.eval_games}")
    print(f"Training window:   {args.window} games")
    print(f"Max game length:   {args.max_moves}")
    print(f"Opening random:    {args.opening_random}")
    print(f"Draw penalty:      {args.draw_penalty}")
    print()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ------------------------------------------------------------------
    # Persistent metrics CSV — one row per iteration, never overwritten.
    # Lets analysis scripts reconstruct training history even if
    # logs/train.log is rotated.  Schema is column-stable; new columns
    # are appended at the end.
    # ------------------------------------------------------------------
    metrics_csv = os.path.join("logs", "metrics.csv")
    os.makedirs("logs", exist_ok=True)
    csv_columns = [
        "timestamp", "global_iter", "version",
        "sp_p1_wins", "sp_p2_wins", "sp_draws", "sp_avg_plies",
        "ab_games", "sims_used",
        "train_loss", "policy_loss", "value_loss", "best_val_loss",
        "aux_value_weight", "lr_used", "epochs_used",
        "eval_score", "eval_w", "eval_l", "eval_d",
        "promoted", "reverted_to",
    ]
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w") as f:
            f.write(",".join(csv_columns) + "\n")

    def _append_metrics_row(row: Dict) -> None:
        with open(metrics_csv, "a") as f:
            vals = [str(row.get(c, "")) for c in csv_columns]
            # Quote any value containing a comma or quote.
            cleaned = [
                f'"{v.replace(chr(34), chr(34)*2)}"'
                if ("," in v or '"' in v or "\n" in v) else v
                for v in vals
            ]
            f.write(",".join(cleaned) + "\n")

    # --- network (auto-resume from best.pt if no explicit --resume) ---
    resume_path = args.resume
    if resume_path is None:
        auto_best = os.path.join(args.checkpoint_dir, "best.pt")
        if os.path.exists(auto_best):
            resume_path = auto_best
            print(f"Auto-resuming from {auto_best}  (use --resume to override)")

    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        best_net, meta = load_checkpoint(resume_path, map_location=str(device))
        start_iter = meta.get("iteration", 0)
        # best_iteration tracks when the best net was actually promoted,
        # not the latest iteration number (which advances even on rejection).
        best_iteration = meta.get("best_iteration", start_iter)
        print(f"  meta: {meta}")
    else:
        print("Starting with a randomly initialised network.")
        best_net = build_net()
        start_iter = 0
        best_iteration = 0
    best_net.to(device)
    best_net.eval()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    mcts_cfg = MCTSConfig(
        num_simulations=args.simulations,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=0.25,
        c_init=args.c_init,
        max_moves=args.max_moves,
    )

    db = GameDB(args.db)
    elo = EloTracker(os.path.join(args.checkpoint_dir, "elo.json"))
    best_version = f"selfplay-v{best_iteration}" if best_iteration > 0 else "init"

    # Rolling history of recently promoted checkpoint paths (for the
    # calibration tournament).  We accumulate newest-first and truncate
    # to args.tournament_pool_size.  Paths point at
    # ``promoted_{version}.pt`` (stable, never-overwritten) rather than
    # ``iter_{NNNN}.pt`` (can be clobbered when a later run reuses the
    # same global iteration number).
    promoted_history: List[Tuple[str, str]] = []
    # Seed with the currently-best checkpoint.  Prefer the stable
    # promoted file if it exists; fall back to best.pt (which is a copy
    # of the promoted weights anyway).
    stable_candidates = [
        os.path.join(args.checkpoint_dir, f"promoted_{best_version}.pt"),
        os.path.join(args.checkpoint_dir, "best.pt"),
    ]
    for cand in stable_candidates:
        if os.path.exists(cand):
            promoted_history.append((best_version, cand))
            break

    # Hard examples mined at the last revert.  Consumed by the next
    # training pass and cleared.  We deliberately don't persist these
    # across training calls (the lesson is most relevant immediately
    # after a revert; later iterations use champion-quality self-play
    # data instead).
    pending_hard_examples: List[Tuple[np.ndarray, np.ndarray]] = []

    try:
        for it in range(1, args.iterations + 1):
            global_it = start_iter + it
            version = f"selfplay-v{global_it}"
            print(f"\n{'='*60}")
            print(f"  Iteration {it}/{args.iterations}  (global {global_it})")
            print(f"{'='*60}")

            # --- 1. Self-play ---
            print("\n[1/3] Self-play")
            if args.workers > 1:
                stats = generate_games_parallel(
                    best_net, mcts_cfg, db,
                    num_games=args.games_per_iter,
                    model_version=version,
                    checkpoint_dir=args.checkpoint_dir,
                    temp_threshold=args.temp_threshold,
                    max_moves=args.max_moves,
                    opening_random=args.opening_random,
                    num_workers=args.workers,
                    adjudicate_gap=args.adjudicate_gap,
                    ab_mix_frac=args.ab_mix_frac,
                    ab_depth=args.ab_depth,
                    ab_time=args.ab_time,
                )
            else:
                stats = generate_games(
                    best_net, mcts_cfg, device, db,
                    num_games=args.games_per_iter,
                    model_version=version,
                    temp_threshold=args.temp_threshold,
                    max_moves=args.max_moves,
                    opening_random=args.opening_random,
                    adjudicate_gap=args.adjudicate_gap,
                )
            total = db.count_games()
            total_pos = db.count_positions(finished_only=False)
            print(
                f"  P1 wins: {stats['p1_wins']}  P2 wins: {stats['p2_wins']}  "
                f"draws: {stats['draws']}"
            )
            print(f"  DB total: {total} games, {total_pos:,} positions")

            # --- 2. Training ---
            print("\n[2/3] Training")
            min_ver = best_iteration if args.train_from_best_version else None

            def _train_candidate(lr_use, wd_use):
                cand = copy.deepcopy(best_net)
                cand, m = train_on_recent_games(
                    cand, db, device,
                    max_games=args.window,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=lr_use,
                    weight_decay=wd_use,
                    draw_penalty=args.draw_penalty,
                    max_moves=args.max_moves,
                    policy_temp=args.policy_temp,
                    value_weight=args.value_weight,
                    min_version_iter=min_ver,
                    extra_examples=pending_hard_examples or None,
                    aux_value_weight=args.aux_value_weight,
                )
                return cand, m

            candidate, metrics = _train_candidate(args.lr, args.weight_decay)

            # Lightweight PBT: every N iterations, also train a second
            # candidate with mutated hparams.  Compare by val loss,
            # keep the better one as *the* candidate.  This explores
            # local hparam neighborhoods without a full PBT pool.
            if (
                args.pbt_mutate_every > 0
                and it % args.pbt_mutate_every == 0
            ):
                # Mutation: ±50% lr, ±2× weight_decay (geometrically).
                mutate_lr = args.lr * (random.choice([0.5, 1.5, 2.0]))
                mutate_wd = args.weight_decay * (random.choice([0.5, 2.0]))
                print(f"\n[PBT] Spawning sibling candidate "
                      f"(lr={mutate_lr:.2g}, wd={mutate_wd:.2g})")
                sibling, sib_metrics = _train_candidate(mutate_lr, mutate_wd)
                cand_v = metrics.get("best_val_loss",
                                     metrics.get("train_loss", 1e9))
                sib_v = sib_metrics.get("best_val_loss",
                                        sib_metrics.get("train_loss", 1e9))
                if sib_v < cand_v:
                    print(f"[PBT] Sibling wins on val loss ({sib_v:.4f} "
                          f"< {cand_v:.4f}); using its weights.")
                    candidate = sibling
                    metrics = sib_metrics
                    metrics["pbt_mutated"] = 1
                    metrics["pbt_lr"] = mutate_lr
                    metrics["pbt_wd"] = mutate_wd
                else:
                    print(f"[PBT] Original wins ({cand_v:.4f} <= "
                          f"{sib_v:.4f}); keeping standard candidate.")

            # Hard examples are consumed once — the next training pass
            # uses fresh champion-quality self-play data instead.
            pending_hard_examples = []

            # --- 3. Evaluation ---
            if args.eval_games > 0:
                print("\n[3/3] Evaluation")
                eval_sims = args.simulations
                if args.workers > 1:
                    eval_result = evaluate_nets_parallel(
                        candidate, best_net,
                        checkpoint_dir=args.checkpoint_dir,
                        num_games=args.eval_games,
                        simulations=eval_sims,
                        opening_random=args.opening_random,
                        max_moves=args.max_moves,
                        num_workers=args.workers,
                        eval_temp=args.eval_temp,
                        eval_temp_moves=args.eval_temp_moves,
                        adjudicate_gap=args.adjudicate_gap,
                    )
                else:
                    eval_result = evaluate_nets(
                        candidate, best_net, device,
                        num_games=args.eval_games,
                        simulations=eval_sims,
                        opening_random=args.opening_random,
                        max_moves=args.max_moves,
                        eval_temp=args.eval_temp,
                        eval_temp_moves=args.eval_temp_moves,
                        adjudicate_gap=args.adjudicate_gap,
                    )
                score = eval_result["score"]
                e_w = eval_result["wins"]
                e_l = eval_result["losses"]
                e_d = eval_result["draws"]
                ci_lo, ci_hi = wilson_ci(
                    eval_result["wins"] + eval_result["draws"] * 0.5,
                    args.eval_games,
                )
                print(
                    f"  New net score: {score:.1%}  "
                    f"(W{e_w}/L{e_l}/D{e_d})  "
                    f"95% CI: [{ci_lo:.1%}, {ci_hi:.1%}]"
                )

                # Elo update: candidate vs current best.
                elo_new, elo_old = elo.update(version, best_version, score)
                print(
                    f"  Elo: {version}={elo_new:.0f}  "
                    f"{best_version}={elo_old:.0f}"
                )

                if score > args.gate_threshold:
                    print("  >>> Promoted new network!")
                    best_net = candidate
                    best_version = version
                    best_iteration = global_it
                    # Persistent promoted checkpoint: never overwritten by
                    # later iterations (iter_{NNNN}.pt uses global_it which
                    # can collide across runs). This is the source of truth
                    # for rollbacks and tournament anchors.
                    promoted_path = os.path.join(
                        args.checkpoint_dir, f"promoted_{version}.pt",
                    )
                    save_checkpoint(
                        best_net, promoted_path,
                        iteration=global_it,
                        best_iteration=best_iteration,
                        **metrics,
                    )
                    print(f"  Saved stable promoted snapshot: {promoted_path}")
                    promoted_history.insert(0, (version, promoted_path))
                    promoted_history[:] = promoted_history[:args.tournament_pool_size]
                else:
                    print("  --- Keeping previous network.")
            else:
                # No gating; always accept.
                best_net = candidate
                best_iteration = global_it

            # Save checkpoint.
            ckpt_path = os.path.join(
                args.checkpoint_dir, f"iter_{global_it:04d}.pt"
            )
            best_path = os.path.join(args.checkpoint_dir, "best.pt")
            save_checkpoint(
                best_net, ckpt_path,
                iteration=global_it,
                best_iteration=best_iteration,
                **metrics,
            )
            save_checkpoint(
                best_net, best_path,
                iteration=global_it,
                best_iteration=best_iteration,
                **metrics,
            )
            print(f"  Saved {ckpt_path}")

            # Print Elo leaderboard.
            print(f"\n  Elo ratings (top 10):\n{elo.summary(10)}")

            # Persist a row to logs/metrics.csv so the analysis suite
            # has stable history independent of train.log rotation.
            avg_plies = (
                stats["total_moves"] / stats["games"] if stats["games"] else 0
            )
            csv_row = {
                "timestamp": _dt.now().isoformat(timespec="seconds"),
                "global_iter": global_it,
                "version": version,
                "sp_p1_wins": stats["p1_wins"],
                "sp_p2_wins": stats["p2_wins"],
                "sp_draws": stats["draws"],
                "sp_avg_plies": f"{avg_plies:.2f}",
                "ab_games": stats.get("ab_games", 0),
                "sims_used": args.simulations,
                "train_loss": f"{metrics.get('train_loss', 0):.4f}",
                "policy_loss": f"{metrics.get('policy_loss', 0):.4f}",
                "value_loss": f"{metrics.get('value_loss', 0):.4f}",
                "best_val_loss": (
                    f"{metrics['best_val_loss']:.4f}"
                    if "best_val_loss" in metrics else ""
                ),
                "aux_value_weight": args.aux_value_weight,
                "lr_used": args.lr,
                "epochs_used": args.epochs,
                "eval_score": (
                    f"{eval_result['score']:.4f}"
                    if args.eval_games > 0 else ""
                ),
                "eval_w": eval_result["wins"] if args.eval_games > 0 else "",
                "eval_l": eval_result["losses"] if args.eval_games > 0 else "",
                "eval_d": eval_result["draws"] if args.eval_games > 0 else "",
                "promoted": "1" if (
                    args.eval_games > 0 and score > args.gate_threshold
                ) else ("1" if args.eval_games == 0 else "0"),
                "reverted_to": "",
            }
            _append_metrics_row(csv_row)

            # --- 4. Periodic calibration tournament (revert-to-champion) ---
            # Fire when the pool (promoted history + anchors) will have
            # at least two distinct checkpoints — we only need 2+ players
            # to compute any meaningful pairwise Elo.
            _anchor_count = sum(
                1 for p in (args.tournament_anchors or []) if os.path.exists(p)
            )
            if (
                args.tournament_every > 0
                and it % args.tournament_every == 0
                and (len(promoted_history) + _anchor_count) >= 2
            ):
                print(f"\n[4/4] Calibration tournament (every "
                      f"{args.tournament_every} iterations)")
                # Build pool: promoted history (deduped) + user anchors.
                seen = set()
                pool: List[Tuple[str, str]] = []
                for label, path in promoted_history:
                    if label not in seen and os.path.exists(path):
                        pool.append((label, path))
                        seen.add(label)
                for anchor_path in (args.tournament_anchors or []):
                    lbl = os.path.splitext(os.path.basename(anchor_path))[0]
                    if lbl in seen or not os.path.exists(anchor_path):
                        continue
                    pool.append((lbl, anchor_path))
                    seen.add(lbl)
                if len(pool) < 2:
                    print("  (not enough players, skipping)")
                else:
                    anchor_lbl = pool[-1][0]  # anchor last = oldest / baseline
                    try:
                        ratings, champion = _run_calibration_tournament(
                            args, pool, anchor_lbl,
                        )
                    except Exception as e:
                        print(f"  tournament failed: {e}")
                        ratings, champion = {}, None

                    if ratings:
                        ranked = sorted(ratings.items(), key=lambda kv: -kv[1])
                        print("  Calibrated Elo (tournament):")
                        for i, (label, r) in enumerate(ranked, 1):
                            marker = " ←" if label == best_version else ""
                            print(f"    {i:>2}. {label:<25s} {r:7.1f}{marker}")

                        # Only revert if the champion is meaningfully
                        # ahead of the current best.  With small game
                        # counts per pair (--tournament-games 4), a
                        # tied 2-2 record produces equal Elos and
                        # max() then picks an arbitrary key — that
                        # would otherwise cause spurious reverts (e.g.
                        # losing the distilled +366-Elo gain to dict
                        # ordering noise).
                        REVERT_GAP_ELO = 25.0
                        cur_elo = ratings.get(best_version, ratings.get(champion))
                        champ_elo = ratings.get(champion, cur_elo)
                        gap = champ_elo - cur_elo
                        if (
                            champion and champion != best_version
                            and gap > REVERT_GAP_ELO
                        ):
                            # Find champion checkpoint and reload weights.
                            champ_path = dict(pool).get(champion)
                            if champ_path and os.path.exists(champ_path):
                                print(f"  >>> REVERTING best_net: "
                                      f"{best_version} -> {champion}  "
                                      f"(Elo gap {gap:+.1f})")
                                # Identify rejected candidates: every
                                # promoted version since the champion
                                # turned out to be a wrong turn.  Their
                                # self-play games are scanned for hard
                                # examples vs the champion.
                                rejected = [
                                    label for label, _ in promoted_history
                                    if label != champion
                                ]
                                champ_net, _ = load_checkpoint(
                                    champ_path, map_location=str(device),
                                )
                                champ_net.to(device)
                                champ_net.eval()
                                # Hard-example mining BEFORE swapping
                                # best_net so the champion is what we
                                # use to score positions.  Mined
                                # examples are kept in-memory and
                                # passed to the next training pass.
                                if args.hard_example_mining and rejected:
                                    print("  Mining hard examples from "
                                          f"{rejected[:3]}...")
                                    new_examples = _mine_hard_examples(
                                        champ_net, db, device,
                                        rejected_versions=rejected,
                                        sims=args.tournament_sims,
                                        max_positions=args.hard_example_positions,
                                    )
                                    pending_hard_examples.extend(new_examples)
                                    print(f"  Mined {len(new_examples)} "
                                          f"hard examples; will be folded "
                                          f"into next training pass.")
                                best_net = champ_net
                                best_version = champion
                                # Re-save best.pt as the champion.
                                save_checkpoint(
                                    best_net,
                                    os.path.join(args.checkpoint_dir, "best.pt"),
                                    iteration=global_it,
                                    best_iteration=best_iteration,
                                    reverted_to=champion,
                                )
                                # Append a metrics row marking the
                                # revert event so analysis can show it.
                                _append_metrics_row({
                                    "timestamp": _dt.now().isoformat(
                                        timespec="seconds"
                                    ),
                                    "global_iter": global_it,
                                    "version": champion,
                                    "promoted": "0",
                                    "reverted_to": champion,
                                })
                        else:
                            if champion and champion != best_version:
                                print(f"  Tournament champion is {champion} "
                                      f"but Elo gap {gap:+.1f} <= "
                                      f"{REVERT_GAP_ELO} threshold; "
                                      "treating as a tie and keeping "
                                      f"current {best_version}.")
                            else:
                                print("  Current best is the tournament "
                                      "champion. No revert.")

    finally:
        db.close()

    print("\nPipeline complete.")


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- pipeline ---
    p.add_argument("--iterations", type=int, default=100,
                   help="Number of generate-train-evaluate cycles (default 100).")
    p.add_argument("--resume", type=str, default=None,
                   help="Checkpoint to resume from.")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                   help="Directory for checkpoints (default: checkpoints/).")
    p.add_argument("--db", type=str, default=None,
                   help="Path to games DB (default: data/quoridor.db).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None,
                   help="Force device: cpu / cuda / mps.")

    # --- self-play ---
    g = p.add_argument_group("self-play")
    g.add_argument("--games-per-iter", type=int, default=100,
                   help="Self-play games per iteration (default 100).")
    g.add_argument("--simulations", type=int, default=200,
                   help="MCTS simulations per move (default 200). "
                        "Lower early on for more exploration diversity.")
    g.add_argument("--max-moves", type=int, default=80,
                   help="Max plies per game before draw (default 80). "
                        "Shorter forces decisive play.")
    g.add_argument("--adjudicate-gap", type=int, default=2,
                   help="At max-moves, declare winner by shortest-path gap "
                        "if >= this value (default 2). 0 disables.")
    g.add_argument("--temp-threshold", type=int, default=20,
                   help="Use proportional sampling for first N full moves, "
                        "then greedy (default 20).")
    g.add_argument("--opening-random", type=int, default=12,
                   help="Play N random moves at the start to break symmetry "
                        "(default 12).")
    g.add_argument("--workers", type=int, default=1,
                   help="Parallel self-play worker processes (default 1). "
                        "Each worker plays games on CPU; net snapshot is "
                        "shared via a checkpoint file. Games are "
                        "independent so this does not affect the learning "
                        "curve — pure throughput win. Try 4–8 on a "
                        "multi-core Mac.")
    g.add_argument("--dirichlet-alpha", type=float, default=0.3,
                   help="Dirichlet noise alpha (default 0.3).")
    g.add_argument("--c-init", type=float, default=1.25,
                   help="PUCT c_init (default 1.25).")
    g.add_argument("--ab-mix-frac", type=float, default=0.0,
                   help="Fraction of self-play games played NN-vs-alphabeta "
                        "instead of self-play. Breaks the self-imitation "
                        "loop (where the net only learns from itself) by "
                        "exposing it to a fundamentally different opponent. "
                        "0.0 disables (default); 0.2-0.3 recommended.")
    g.add_argument("--ab-depth", type=int, default=4,
                   help="Alpha-beta search depth for ab-mix games (default 4).")
    g.add_argument("--ab-time", type=float, default=1.5,
                   help="Per-move time budget (s) for the alpha-beta opponent "
                        "(default 1.5).")

    # --- training ---
    g = p.add_argument_group("training")
    g.add_argument("--epochs", type=int, default=10,
                   help="Training epochs per iteration (default 10).")
    g.add_argument("--batch-size", type=int, default=256)
    g.add_argument("--lr", type=float, default=2e-3,
                   help="Learning rate (default 2e-3).")
    g.add_argument("--weight-decay", type=float, default=1e-4)
    g.add_argument("--window", type=int, default=5000,
                   help="Train on the N most recent games (default 5000). "
                        "Larger windows = more position diversity = less "
                        "overfitting, at the cost of slightly slower epochs.")
    g.add_argument("--draw-penalty", type=float, default=0.5,
                   help="Base value penalty for draws (default 0.5). "
                        "Higher values push the net to play for wins.")
    g.add_argument("--policy-temp", type=float, default=0.7,
                   help="Temperature for sharpening MCTS policy targets "
                        "during training (default 0.7). <1 = sharper targets, "
                        "1.0 = no sharpening.")
    g.add_argument("--value-weight", type=float, default=1.0,
                   help="Multiplier on value-head loss during training "
                        "(default 1.0). Lower values (0.1–0.3) protect a "
                        "well-calibrated value head — e.g. after distillation "
                        "— from being smashed by blunt outcome targets.")
    g.add_argument("--train-from-best-version", action="store_true",
                   help="Filter training data to only include games from "
                        "self-play version >= current best_iteration, plus "
                        "tournament-champion games. Stops chains of rejected-"
                        "candidate self-play from pulling the net back toward "
                        "a weaker distribution. Highly recommended once a "
                        "strong best_iteration is established.")
    g.add_argument("--aux-value-weight", type=float, default=0.0,
                   help="Blend tanh(path_diff/6) into the value target with "
                        "this weight (0=outcome only, 1=path-diff only, "
                        "0.3-0.5 recommended). Provides dense supervision "
                        "for the value head independent of game outcome — "
                        "addresses the 'value head can't learn from sparse "
                        "outcome labels' problem.")
    g.add_argument("--pbt-mutate-every", type=int, default=0,
                   help="Every N iterations, train a sibling candidate with "
                        "mutated lr/weight_decay and keep whichever has the "
                        "lower val loss. Lightweight population-based "
                        "training. 0 disables (default).")

    # --- evaluation ---
    g = p.add_argument_group("evaluation")
    g.add_argument("--eval-games", type=int, default=200,
                   help="Games for net-vs-net evaluation (0 to disable "
                        "gating; default 200). At N=200 the standard error "
                        "is ~3.5%%, giving reliable gating signal.")
    g.add_argument("--gate-threshold", type=float, default=0.52,
                   help="Min score to promote the new net (default 0.52). "
                        "Score: win=1, draw=0.5, loss=0.")
    g.add_argument("--eval-temp", type=float, default=0.5,
                   help="Temperature for sampling eval moves before "
                        "--eval-temp-moves (default 0.5). Breaks deterministic "
                        "mirror lines between near-identical nets so games "
                        "become decisive; 0.0 restores pure greedy eval.")
    g.add_argument("--eval-temp-moves", type=int, default=10,
                   help="First N plies of eval sampled at --eval-temp; "
                        "greedy afterwards (default 10). Larger values add "
                        "more diversity at the cost of signal fidelity.")

    # --- calibration tournament ---
    g = p.add_argument_group("calibration tournament")
    g.add_argument("--tournament-every", type=int, default=0,
                   help="Every N iterations, run a round-robin among recent "
                        "promoted nets + anchors; revert best_net to the Elo "
                        "champion. Saves champion games to the DB as extra "
                        "training data. 0 disables (default).")
    g.add_argument("--tournament-games", type=int, default=4,
                   help="Games per pair in each calibration tournament "
                        "(default 4). Kept modest since tournament cost scales "
                        "as O(pool^2 * games).")
    g.add_argument("--tournament-sims", type=int, default=200,
                   help="MCTS simulations per move in tournament games.")
    g.add_argument("--tournament-pool-size", type=int, default=4,
                   help="Max recent promoted nets to carry into the next "
                        "tournament (default 4).")
    g.add_argument("--tournament-anchors", action="append", default=None,
                   help="Path to a checkpoint that should appear in every "
                        "tournament (repeat). Typical use: an old historical "
                        "champion like iter_0036.pt to detect drift.")
    g.add_argument("--hard-example-mining", action="store_true",
                   help="On revert, scan rejected candidates' self-play "
                        "games and save positions where the champion's MCTS "
                        "top move differs from the played move. These get "
                        "high weight in the next training pass, directly "
                        "addressing 'why did we lose to the champion'.")
    g.add_argument("--hard-example-positions", type=int, default=1500,
                   help="Max positions scanned for hard-example mining "
                        "(default 1500). Higher = more signal, more compute.")

    args = p.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
