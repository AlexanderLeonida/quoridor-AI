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
    action_to_move,
    canonical_view,
    deserialize_policy,
    encode_state,
    move_to_action,
    serialize_policy,
)
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
# Draw value computation
# ======================================================================

def _draw_z(
    final_board: Board,
    side: int,
    draw_penalty: float,
    *,
    game_length: Optional[int] = None,
    max_moves: Optional[int] = None,
    stall_weight: float = 0.4,
    progress_weight: float = 0.3,
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
    raw_progress = (d1 - d0) / 10.0
    progress = math.tanh(raw_progress)  # squash to (-1, 1)
    # From P0's POV the bonus is +progress; from P1's POV it's -progress.
    bonus = progress if side == 0 else -progress
    return float(np.clip(-effective_penalty + bonus * progress_weight, -1.0, 1.0))


# ======================================================================
# Self-play game generation
# ======================================================================

def play_game(
    net,
    config: MCTSConfig,
    device,
    *,
    temp_threshold: int = 15,
    max_moves: int = 120,
    opening_random: int = 0,
    use_cache: bool = True,
) -> Tuple[List[Board], List[np.ndarray], List[int], Optional[int], Board]:
    """Play one self-play game using MCTS.

    Returns (boards, policies, actions, winner, final_board).
    ``boards`` / ``policies`` / ``actions`` start after the random opening.
    ``final_board`` is needed for progress-aware draw values.
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

    return boards, policies, actions, board.winner(), board


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


def _worker_play_one(_idx: int):
    """Play one self-play game and return a picklable result tuple."""
    import torch  # noqa: WPS433

    assert _WORKER_NET is not None
    assert _WORKER_CONFIG is not None
    assert _WORKER_PLAY_KWARGS is not None

    t0 = time.perf_counter()
    boards, policies, actions, winner, _final = play_game(
        _WORKER_NET,
        _WORKER_CONFIG,
        _WORKER_DEVICE,
        **_WORKER_PLAY_KWARGS,
    )
    # Convert to (moves, blobs) since the Move objects and bytes are
    # cheap to pickle; raw Boards/tensors are not.
    moves = []
    blobs = []
    for board, policy, action in zip(boards, policies, actions):
        _, _, _, _, _, _, flipped = canonical_view(board)
        moves.append(action_to_move(action, flipped))
        blobs.append(serialize_policy(policy))
    return moves, blobs, winner, len(actions), time.perf_counter() - t0


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
) -> Dict[str, int]:
    """Parallel self-play across *num_workers* CPU processes."""
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
    }
    seed_base = random.randint(0, 2**30)
    initargs = (snap_path, asdict(config), play_kwargs, seed_base)

    ctx = mp.get_context("spawn")
    stats: Dict[str, int] = {
        "games": 0, "p1_wins": 0, "p2_wins": 0,
        "draws": 0, "total_moves": 0,
    }

    print(f"  (parallel self-play on {num_workers} CPU workers)")
    with ctx.Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=initargs,
    ) as pool:
        for moves, blobs, winner, n_plies, elapsed in pool.imap_unordered(
            _worker_play_one, range(num_games)
        ):
            gid = db.save_game(
                moves,
                winner=winner,
                p1_source="selfplay_nn",
                p2_source="selfplay_nn",
                model_version=model_version,
                notes="mcts_selfplay",
                policies=blobs,
            )
            stats["games"] += 1
            stats["total_moves"] += n_plies
            if winner == 0:
                stats["p1_wins"] += 1
            elif winner == 1:
                stats["p2_wins"] += 1
            else:
                stats["draws"] += 1
            outcome = "P1" if winner == 0 else ("P2" if winner == 1 else "draw")
            avg = stats["total_moves"] / stats["games"]
            print(
                f"  game {stats['games']:>3}/{num_games}  id={gid:<5}  "
                f"plies={n_plies:<4}  winner={outcome:<5}  "
                f"{elapsed:.1f}s  avg_len={avg:.0f}"
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
    all_games = [
        row for row in db.iter_games(finished_only=False)
        if row[5] == "selfplay_nn" and row[6] == "selfplay_nn"
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
            # Additionally upweight decisive games 2× so that the training
            # signal isn't dominated by draw positions (draws tend to be
            # long, and their value targets are uncertain by construction).
            game_len = max(len(moves), 1)
            decisive_mult = 2.0 if winner is not None else 1.0
            w = decisive_mult / game_len
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

    states = torch.from_numpy(np.stack(tr_s))
    pols = torch.from_numpy(np.stack(tr_p))
    vals = torch.from_numpy(np.array(tr_v, dtype=np.float32))
    sample_weights = torch.from_numpy(np.array(tr_w, dtype=np.float32))
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs * len(train_loader),
    )

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
            loss = loss_p + loss_v
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
) -> float:
    """Play *num_games* between two networks, return *net_new*'s score.

    Score: win = 1, draw = 0.5, loss = 0 (normalised to [0, 1]).
    Colors alternate each game.  Evaluation uses **no root noise** and
    **greedy** (temperature = 0) play so that the result reflects true
    model strength, not MCTS randomness.  Randomised openings still
    break symmetry.
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
            action = select_action(root, temperature=0.0)  # greedy
            _, _, _, _, _, _, flipped = canonical_view(board)
            move = action_to_move(action, flipped)
            board = board.apply(move)
            move_count += 1

        winner = board.winner()
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
        action = select_action(root, temperature=0.0)
        _, _, _, _, _, _, flipped = canonical_view(board)
        move = action_to_move(action, flipped)
        board = board.apply(move)
        move_count += 1

    winner = board.winner()
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
# Pipeline orchestration
# ======================================================================

def run_pipeline(args) -> None:
    import torch

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
                )
            else:
                stats = generate_games(
                    best_net, mcts_cfg, device, db,
                    num_games=args.games_per_iter,
                    model_version=version,
                    temp_threshold=args.temp_threshold,
                    max_moves=args.max_moves,
                    opening_random=args.opening_random,
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
            candidate = copy.deepcopy(best_net)
            candidate, metrics = train_on_recent_games(
                candidate, db, device,
                max_games=args.window,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                draw_penalty=args.draw_penalty,
                max_moves=args.max_moves,
                policy_temp=args.policy_temp,
            )

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
                    )
                else:
                    eval_result = evaluate_nets(
                        candidate, best_net, device,
                        num_games=args.eval_games,
                        simulations=eval_sims,
                        opening_random=args.opening_random,
                        max_moves=args.max_moves,
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
    g.add_argument("--max-moves", type=int, default=90,
                   help="Max plies per game before draw (default 90). "
                        "Shorter forces decisive play.")
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
    g.add_argument("--draw-penalty", type=float, default=0.3,
                   help="Base value penalty for draws (default 0.3). "
                        "Higher values push the net to play for wins.")
    g.add_argument("--policy-temp", type=float, default=0.7,
                   help="Temperature for sharpening MCTS policy targets "
                        "during training (default 0.7). <1 = sharper targets, "
                        "1.0 = no sharpening.")

    # --- evaluation ---
    g = p.add_argument_group("evaluation")
    g.add_argument("--eval-games", type=int, default=200,
                   help="Games for net-vs-net evaluation (0 to disable "
                        "gating; default 200). At N=200 the standard error "
                        "is ~3.5%%, giving reliable gating signal.")
    g.add_argument("--gate-threshold", type=float, default=0.55,
                   help="Min score to promote the new net (default 0.55). "
                        "Score: win=1, draw=0.5, loss=0.")

    args = p.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
