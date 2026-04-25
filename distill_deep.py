"""Distill a stronger search-based teacher into the network.

Two teacher modes:

    --teacher mcts  → Run extreme-sim MCTS (e.g., 5000 sims) on each
                      sampled position, use the visit-count distribution
                      as the policy target.  This makes the *current
                      net* its own teacher but at much deeper search
                      than the training-time MCTS produces.

    --teacher ab    → Run alpha-beta search at high depth on each
                      sampled position, one-hot encode the chosen move
                      as the policy target.  Brings in supervision from
                      a fundamentally different evaluator.

The student is updated to match the teacher's policy + a value target
that is either derived from the search or zeroed out (let the value
head learn from regular self-play).

Usage
-----
    python3 distill_deep.py --teacher mcts --teacher-sims 4000 \\
        --student checkpoints/best.pt --positions 5000 \\
        --out checkpoints/best_distilled.pt --workers 4

    python3 distill_deep.py --teacher ab --ab-depth 8 --ab-time 5 \\
        --student checkpoints/best.pt --positions 2000 \\
        --out checkpoints/best_ab_distilled.pt --workers 4

After running, gate-test the result:
    python3 bench.py --ckpt checkpoints/best_ab_distilled.pt \\
        --vs-other checkpoints/best.pt --n-games 30 --sims 200
"""
from __future__ import annotations

import argparse
import os
import random
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from quoridor import Board, GameDB
from quoridor.encoding import (
    ACTION_SPACE, action_to_move, canonical_view, encode_state, move_to_action,
    serialize_policy,
)
from quoridor.mcts import EvalCache, MCTSConfig, get_policy, search
from quoridor.net import best_available_device, load_checkpoint, save_checkpoint


# ---------------------------------------------------------------------
# Position sampling (same idea as distill.py)
# ---------------------------------------------------------------------
def sample_positions(db: GameDB, n_target: int, seed: int = 0) -> List[Board]:
    rng = random.Random(seed)
    games = [row for row in db.iter_games(finished_only=False)
             if row[5] == "selfplay_nn" and row[6] == "selfplay_nn"]
    rng.shuffle(games)
    boards: List[Board] = []
    for row in games:
        moves = db.load_moves(row[0])
        board = Board.initial()
        for move in moves:
            boards.append(board)
            if len(boards) >= n_target:
                return boards
            board = board.apply(move)
        boards.append(board)
        if len(boards) >= n_target:
            return boards
    return boards


# ---------------------------------------------------------------------
# Teacher: extreme-sim MCTS
# ---------------------------------------------------------------------
def _mcts_teacher_one(args_tuple):
    """Pool worker — run high-sim MCTS on one board, return its policy."""
    board, sims = args_tuple
    # Worker-side: net is loaded in initargs.
    cfg = MCTSConfig(num_simulations=sims, dirichlet_epsilon=0.0)
    cache = EvalCache()
    root = search(board, _TEACHER_NET, cfg, _TEACHER_DEVICE,
                  add_noise=False, cache=cache)
    pol = get_policy(root, temperature=1.0).astype(np.float32)
    # Value target: root's MCTS value estimate.
    return encode_state(board), pol, root.value


# ---------------------------------------------------------------------
# Teacher: alpha-beta
# ---------------------------------------------------------------------
def _ab_teacher_one(args_tuple):
    from quoridor.ai import find_best_move
    board, depth, time_limit = args_tuple
    mv = find_best_move(board, max_depth=depth, time_limit=time_limit)
    _, _, _, _, _, _, flipped = canonical_view(board)
    a = move_to_action(mv, flipped)
    onehot = np.zeros(ACTION_SPACE, dtype=np.float32)
    onehot[a] = 1.0
    # No value signal from alphabeta directly; use 0 (let value head
    # train on outcome data elsewhere).
    return encode_state(board), onehot, 0.0


# Pool-global teacher state (set by initializer).
_TEACHER_NET = None
_TEACHER_DEVICE = None


def _mcts_init(ckpt_path: str):
    global _TEACHER_NET, _TEACHER_DEVICE
    torch.set_num_threads(1)
    net, _ = load_checkpoint(ckpt_path, map_location="cpu")
    net.to("cpu")
    net.eval()
    _TEACHER_NET = net
    _TEACHER_DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------
# Student training
# ---------------------------------------------------------------------
def distill_student(student, examples, device, *, epochs, batch_size, lr,
                    weight_decay, val_frac=0.05):
    states = torch.from_numpy(np.stack([e[0] for e in examples]))
    pols = torch.from_numpy(np.stack([e[1] for e in examples]))
    vals = torch.from_numpy(np.array([e[2] for e in examples], dtype=np.float32))

    n = states.size(0)
    n_val = max(1, int(n * val_frac)) if val_frac > 0 else 0
    perm = torch.randperm(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    tr_loader = DataLoader(
        TensorDataset(states[tr_idx], pols[tr_idx], vals[tr_idx]),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(states[val_idx], pols[val_idx], vals[val_idx]),
        batch_size=batch_size, shuffle=False,
    ) if n_val > 0 else None

    student.to(device)
    opt = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(tr_loader))

    best_val = None
    best_state = None
    for epoch in range(1, epochs + 1):
        student.train()
        tl = tp = tv = 0.0
        tn = 0
        for xb, pb, vb in tr_loader:
            xb, pb, vb = xb.to(device), pb.to(device), vb.to(device)
            p_logits, v_pred = student(xb)
            log_p = F.log_softmax(p_logits, dim=1)
            loss_p = -(pb * log_p).sum(dim=1).mean()
            loss_v = F.mse_loss(v_pred, vb)
            loss = loss_p + loss_v
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step(); sched.step()
            bs = xb.size(0)
            tl += loss.item() * bs; tp += loss_p.item() * bs
            tv += loss_v.item() * bs; tn += bs
        line = (f"  epoch {epoch}/{epochs}  "
                f"loss={tl/tn:.4f} (p={tp/tn:.4f} v={tv/tn:.4f})")
        if val_loader:
            student.eval()
            vl = 0.0; vn = 0
            with torch.no_grad():
                for xb, pb, vb in val_loader:
                    xb, pb, vb = xb.to(device), pb.to(device), vb.to(device)
                    p_logits, v_pred = student(xb)
                    log_p = F.log_softmax(p_logits, dim=1)
                    lp = -(pb * log_p).sum(dim=1).mean()
                    lv = F.mse_loss(v_pred, vb)
                    bs = xb.size(0)
                    vl += (lp.item() + lv.item()) * bs; vn += bs
            cur = vl / vn
            line += f"  val={cur:.4f}"
            if best_val is None or cur < best_val:
                best_val = cur
                best_state = {k: v.detach().cpu().clone()
                              for k, v in student.state_dict().items()}
                line += "  [best]"
        print(line)
    if best_state is not None:
        student.load_state_dict(best_state)
    return student, best_val if best_val is not None else tl / tn


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher", choices=["mcts", "ab"], required=True)
    p.add_argument("--student", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--db", default="data/quoridor_v3.db")
    p.add_argument("--positions", type=int, default=2000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    # MCTS teacher options
    p.add_argument("--teacher-sims", type=int, default=4000,
                   help="MCTS simulations for the teacher (mcts mode).")
    # Alphabeta teacher options
    p.add_argument("--ab-depth", type=int, default=6)
    p.add_argument("--ab-time", type=float, default=3.0)
    # Student training options
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    print(f"Sampling {args.positions} positions from {args.db}...")
    db = GameDB(args.db)
    boards = sample_positions(db, args.positions, seed=args.seed)
    print(f"  got {len(boards)} positions")

    print(f"Generating teacher targets ({args.teacher}, "
          f"{args.workers} workers)...")
    t0 = time.perf_counter()
    if args.teacher == "mcts":
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers,
                      initializer=_mcts_init,
                      initargs=(args.student,)) as pool:
            jobs = [(b, args.teacher_sims) for b in boards]
            examples = list(pool.imap_unordered(_mcts_teacher_one, jobs,
                                                chunksize=4))
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            jobs = [(b, args.ab_depth, args.ab_time) for b in boards]
            examples = list(pool.imap_unordered(_ab_teacher_one, jobs,
                                                chunksize=4))
    elapsed = time.perf_counter() - t0
    print(f"  {len(examples)} teacher targets in {elapsed:.0f}s "
          f"({elapsed/len(examples):.2f}s/pos)")

    device = best_available_device()
    print(f"Distilling into student on {device}...")
    student, student_meta = load_checkpoint(args.student, map_location=str(device))
    student, best_val = distill_student(
        student, examples, device,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay,
    )

    save_iter = student_meta.get("iteration", 0)
    save_best = student_meta.get("best_iteration", save_iter)
    save_checkpoint(
        student, args.out,
        iteration=save_iter,
        best_iteration=save_best,
        deep_distilled_from=args.teacher,
        deep_distill_positions=len(examples),
        deep_distill_val_loss=float(best_val) if best_val is not None else 0.0,
    )
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
