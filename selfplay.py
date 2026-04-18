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
    progress_weight: float = 0.3,
) -> float:
    """Compute the value target for a drawn game from *side*'s POV.

    Base: -draw_penalty (both sides are slightly penalised for drawing).
    Bonus: shortest-path advantage at the final position gives partial
    credit — the side closer to its goal gets a less negative z.
    """
    d0 = final_board.shortest_path_length(0)
    d1 = final_board.shortest_path_length(1)
    # Positive when P0 is closer to winning.
    raw_progress = (d1 - d0) / 10.0
    progress = math.tanh(raw_progress)  # squash to (-1, 1)
    # From P0's POV the bonus is +progress; from P1's POV it's -progress.
    bonus = progress if side == 0 else -progress
    return float(np.clip(-draw_penalty + bonus * progress_weight, -1.0, 1.0))


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

    while board.winner() is None and move_num < max_moves:
        root = search(board, net, config, device, add_noise=True)

        temp = 1.0 if move_num < temp_threshold else 0.0
        policy = get_policy(root, temp)
        action = select_action(root, temp)

        boards.append(board)
        policies.append(policy)
        actions.append(action)

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
) -> Tuple:
    """Train *net* on the most recent games from *db*.

    Returns (net, metrics_dict).
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    # --- materialise dataset ---
    games = list(db.iter_games(finished_only=False))
    games = games[-max_games:]
    if not games:
        print("  No games to train on.")
        return net, {}

    states_list: List[np.ndarray] = []
    policies_list: List[np.ndarray] = []
    values_list: List[float] = []

    for row in games:
        game_id = row[0]
        winner = row[3]
        moves = db.load_moves(game_id)
        blobs = db.load_policy_blobs(game_id)

        # Replay to find the final board (for progress-aware draw z).
        board = Board.initial()
        boards_in_game: List[Board] = []
        for move in moves:
            boards_in_game.append(board)
            board = board.apply(move)
        final_board = board

        for idx, (brd, move, blob) in enumerate(zip(boards_in_game, moves, blobs)):
            states_list.append(encode_state(brd))
            if blob is not None:
                policies_list.append(deserialize_policy(blob))
            else:
                _, _, _, _, _, _, flipped = canonical_view(brd)
                act = move_to_action(move, flipped)
                onehot = np.zeros(ACTION_SPACE, dtype=np.float32)
                onehot[act] = 1.0
                policies_list.append(onehot)

            if winner is not None:
                z = 1.0 if winner == brd.turn else -1.0
            else:
                z = _draw_z(final_board, brd.turn, draw_penalty)
            values_list.append(z)

    n = len(states_list)
    if n == 0:
        print("  No training positions.")
        return net, {}

    n_draws = sum(1 for r in games if r[3] is None)
    n_decisive = len(games) - n_draws
    print(
        f"  {n:,} positions from {len(games)} game(s) "
        f"({n_decisive} decisive, {n_draws} draws)"
    )

    states = torch.from_numpy(np.stack(states_list))
    pols = torch.from_numpy(np.stack(policies_list))
    vals = torch.from_numpy(np.array(values_list, dtype=np.float32))

    # --- split ---
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_val = max(1, int(n * val_frac)) if val_frac > 0 and n > 20 else 0

    def loader(sel, shuf):
        ds = TensorDataset(states[sel], pols[sel], vals[sel])
        return DataLoader(ds, batch_size=batch_size, shuffle=shuf, drop_last=False)

    train_loader = loader(idx[n_val:], True)
    val_loader = loader(idx[:n_val], False) if n_val else None

    # --- optimiser ---
    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs * len(train_loader),
    )

    # --- training loop ---
    for epoch in range(1, epochs + 1):
        net.train()
        tl = tp = tv = 0.0
        tn = 0
        for xb, pb, vb in train_loader:
            xb, pb, vb = xb.to(device), pb.to(device), vb.to(device)
            p_logits, v_pred = net(xb)
            log_p = F.log_softmax(p_logits, dim=1)
            loss_p = -(pb * log_p).sum(dim=1).mean()
            loss_v = F.mse_loss(v_pred, vb)
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

        print(line)

    metrics = {"train_loss": tl / tn, "policy_loss": tp / tn, "value_loss": tv / tn}
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
    Colors alternate each game.  Eval games use randomised openings
    and a small temperature (0.1) to break deterministic mirror play.
    """
    eval_cfg = MCTSConfig(
        num_simulations=simulations,
        dirichlet_alpha=0.15,       # lighter noise than self-play
        dirichlet_epsilon=0.15,
    )
    net_new.eval()
    net_old.eval()

    score = 0.0
    for g in range(num_games):
        if g % 2 == 0:
            nets = {0: net_new, 1: net_old}
            new_side = 0
        else:
            nets = {0: net_old, 1: net_new}
            new_side = 1

        board = Board.initial()
        # Randomised opening — same sequence for both sides in this game.
        if opening_random > 0:
            board, _ = _randomise_opening(board, opening_random)

        move_count = 0
        while board.winner() is None and move_count < max_moves:
            cur_net = nets[board.turn]
            root = search(board, cur_net, eval_cfg, device, add_noise=True)
            # Small temperature to avoid fully deterministic play.
            action = select_action(root, temperature=0.1)
            _, _, _, _, _, _, flipped = canonical_view(board)
            move = action_to_move(action, flipped)
            board = board.apply(move)
            move_count += 1

        winner = board.winner()
        if winner is not None and winner == new_side:
            score += 1.0
            tag = "W"
        elif winner is not None and winner != new_side:
            tag = "L"
        else:
            score += 0.5  # draw = half point
            tag = "D"

        print(
            f"    eval game {g+1}/{num_games}  "
            f"new={'P1' if new_side==0 else 'P2'}  "
            f"moves={move_count}  {tag}"
        )

    return score / num_games


# ======================================================================
# Pipeline orchestration
# ======================================================================

def run_pipeline(args) -> None:
    import torch

    device = (
        torch.device(args.device) if args.device else best_available_device()
    )
    print(f"Device:            {device}")
    print(f"Simulations/move:  {args.simulations}")
    print(f"Games/iteration:   {args.games_per_iter}")
    print(f"Training epochs:   {args.epochs}")
    print(f"Eval games:        {args.eval_games}")
    print(f"Max game length:   {args.max_moves}")
    print(f"Opening random:    {args.opening_random}")
    print(f"Draw penalty:      {args.draw_penalty}")
    print()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # --- network ---
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        best_net, meta = load_checkpoint(args.resume, map_location=str(device))
        start_iter = meta.get("iteration", 0)
        print(f"  meta: {meta}")
    else:
        print("Starting with a randomly initialised network.")
        best_net = build_net()
        start_iter = 0
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

    try:
        for it in range(1, args.iterations + 1):
            global_it = start_iter + it
            version = f"selfplay-v{global_it}"
            print(f"\n{'='*60}")
            print(f"  Iteration {it}/{args.iterations}  (global {global_it})")
            print(f"{'='*60}")

            # --- 1. Self-play ---
            print("\n[1/3] Self-play")
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
            )

            # --- 3. Evaluation ---
            if args.eval_games > 0:
                print("\n[3/3] Evaluation")
                score = evaluate_nets(
                    candidate, best_net, device,
                    num_games=args.eval_games,
                    simulations=max(args.simulations // 4, 50),
                    opening_random=args.opening_random,
                    max_moves=args.max_moves,
                )
                print(f"  New net score: {score:.1%}")

                if score > args.gate_threshold:
                    print("  >>> Promoted new network!")
                    best_net = candidate
                else:
                    print("  --- Keeping previous network.")
            else:
                # No gating; always accept.
                best_net = candidate

            # Save checkpoint.
            ckpt_path = os.path.join(
                args.checkpoint_dir, f"iter_{global_it:04d}.pt"
            )
            best_path = os.path.join(args.checkpoint_dir, "best.pt")
            save_checkpoint(
                best_net, ckpt_path,
                iteration=global_it,
                **metrics,
            )
            save_checkpoint(
                best_net, best_path,
                iteration=global_it,
                **metrics,
            )
            print(f"  Saved {ckpt_path}")

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
    g.add_argument("--games-per-iter", type=int, default=50,
                   help="Self-play games per iteration (default 50).")
    g.add_argument("--simulations", type=int, default=400,
                   help="MCTS simulations per move (default 400).")
    g.add_argument("--max-moves", type=int, default=120,
                   help="Max plies per game before draw (default 120).")
    g.add_argument("--temp-threshold", type=int, default=15,
                   help="Use proportional sampling for first N full moves, "
                        "then greedy (default 15).")
    g.add_argument("--opening-random", type=int, default=4,
                   help="Play N random moves at the start to break symmetry "
                        "(default 4).")
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
    g.add_argument("--window", type=int, default=1000,
                   help="Train on the N most recent games (default 1000).")
    g.add_argument("--draw-penalty", type=float, default=0.1,
                   help="Base value penalty for draws (default 0.1). "
                        "Progress-aware bonus is layered on top.")

    # --- evaluation ---
    g = p.add_argument_group("evaluation")
    g.add_argument("--eval-games", type=int, default=20,
                   help="Games for net-vs-net evaluation (0 to disable gating).")
    g.add_argument("--gate-threshold", type=float, default=0.55,
                   help="Min score to promote the new net (default 0.55). "
                        "Score: win=1, draw=0.5, loss=0.")

    args = p.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
