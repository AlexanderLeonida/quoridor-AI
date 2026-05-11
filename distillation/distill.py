"""Distill a teacher network into a student with different architecture.

Use case: we have iter_0074.pt (6x64) which is our strongest model by Elo,
but our current best.pt is a fresh-start 10x128 net that trails it. Rather
than throw away v74's knowledge, we transfer it: run positions through v74
to get (policy, value) targets, then train the 10x128 to match those
targets. The student inherits v74's strength, and future self-play builds
on top.

Usage:
    python3 distill.py \\
        --teacher checkpoints/iter_0074.pt \\
        --student checkpoints/best.pt \\
        --db data/quoridor_v2.db \\
        --out checkpoints/best.pt \\
        --positions 80000 --epochs 4
"""
from __future__ import annotations

import argparse
import random
import time
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# --- path bootstrap so this file can be run from anywhere ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# -------------------------------------------------------------
from quoridor import Board
from quoridor.database import GameDB
from quoridor.encoding import encode_state
from quoridor.net import load_checkpoint, save_checkpoint, best_available_device


def sample_positions(db: GameDB, n_target: int, seed: int = 0) -> List[Board]:
    """Collect up to n_target unique positions from self-play games in db.

    Iterates games newest-first, replays each game, and snapshots every
    position. Stops once n_target positions are collected.
    """
    rng = random.Random(seed)
    games = [row for row in db.iter_games(finished_only=False)
             if row[5] == "selfplay_nn" and row[6] == "selfplay_nn"]
    rng.shuffle(games)

    boards: List[Board] = []
    for row in games:
        game_id = row[0]
        moves = db.load_moves(game_id)
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


@torch.no_grad()
def teacher_targets(teacher, states: torch.Tensor, device, batch_size: int = 512):
    """Forward states through teacher, return (soft_policy, value) targets."""
    teacher.eval()
    policies = []
    values = []
    for i in range(0, states.size(0), batch_size):
        xb = states[i:i + batch_size].to(device)
        p_logits, v = teacher(xb)
        policies.append(F.softmax(p_logits, dim=1).cpu())
        values.append(v.cpu())
    return torch.cat(policies, dim=0), torch.cat(values, dim=0)


def distill(
    teacher,
    student,
    states: torch.Tensor,
    pol_targets: torch.Tensor,
    val_targets: torch.Tensor,
    device,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    val_frac: float = 0.05,
):
    n = states.size(0)
    n_val = max(1, int(n * val_frac)) if val_frac > 0 else 0
    perm = torch.randperm(n)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    tr_ds = TensorDataset(states[tr_idx], pol_targets[tr_idx], val_targets[tr_idx])
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    val_loader = None
    if n_val > 0:
        val_ds = TensorDataset(states[val_idx], pol_targets[val_idx], val_targets[val_idx])
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    student.to(device)
    opt = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs * len(tr_loader),
    )

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
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            opt.step()
            scheduler.step()
            bs = xb.size(0)
            tl += loss.item() * bs
            tp += loss_p.item() * bs
            tv += loss_v.item() * bs
            tn += bs

        line = (
            f"  epoch {epoch}/{epochs}  "
            f"loss={tl/tn:.4f} (p={tp/tn:.4f} v={tv/tn:.4f})"
        )

        if val_loader is not None:
            student.eval()
            vl = vp2 = vv = 0.0
            vn = 0
            with torch.no_grad():
                for xb, pb, vb in val_loader:
                    xb, pb, vb = xb.to(device), pb.to(device), vb.to(device)
                    p_logits, v_pred = student(xb)
                    log_p = F.log_softmax(p_logits, dim=1)
                    lp = -(pb * log_p).sum(dim=1).mean()
                    lv = F.mse_loss(v_pred, vb)
                    bs = xb.size(0)
                    vl += (lp.item() + lv.item()) * bs
                    vp2 += lp.item() * bs
                    vv += lv.item() * bs
                    vn += bs
            cur = vl / vn
            line += f"  val={cur:.4f} (p={vp2/vn:.4f} v={vv/vn:.4f})"
            if best_val is None or cur < best_val:
                best_val = cur
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in student.state_dict().items()
                }
                line += "  [best]"
        print(line)

    if best_state is not None:
        student.load_state_dict(best_state)
    return student, {"train_loss": tl / tn, "policy_loss": tp / tn,
                     "value_loss": tv / tn,
                     "best_val_loss": best_val if best_val is not None else tl / tn}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True)
    p.add_argument("--student", required=True)
    p.add_argument("--db", default="data/quoridor_v2.db")
    p.add_argument("--out", required=True)
    p.add_argument("--positions", type=int, default=80000)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = best_available_device()
    print(f"Device: {device}")

    teacher, teacher_meta = load_checkpoint(args.teacher, map_location=str(device))
    teacher.to(device)
    print(f"Teacher {args.teacher}  config={teacher.config}  meta={teacher_meta}")

    student, student_meta = load_checkpoint(args.student, map_location=str(device))
    student.to(device)
    print(f"Student {args.student}  config={student.config}  meta={student_meta}")

    print(f"Sampling up to {args.positions} positions from {args.db}...")
    t0 = time.perf_counter()
    db = GameDB(args.db)
    boards = sample_positions(db, args.positions, seed=args.seed)
    print(f"  got {len(boards)} positions  ({time.perf_counter()-t0:.1f}s)")

    print("Encoding states...")
    t0 = time.perf_counter()
    states_np = np.stack([encode_state(b) for b in boards])
    states = torch.from_numpy(states_np)
    print(f"  encoded {states.size(0)} x {tuple(states.shape[1:])}  ({time.perf_counter()-t0:.1f}s)")

    print("Generating teacher targets...")
    t0 = time.perf_counter()
    pol_targets, val_targets = teacher_targets(teacher, states, device)
    print(f"  pol={tuple(pol_targets.shape)} val={tuple(val_targets.shape)}  ({time.perf_counter()-t0:.1f}s)")
    print(f"  teacher value mean={val_targets.mean().item():+.3f} std={val_targets.std().item():.3f}")

    print(f"\nDistilling for {args.epochs} epoch(s), batch={args.batch_size}, lr={args.lr}...")
    student, metrics = distill(
        teacher, student, states, pol_targets, val_targets, device,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay,
    )
    print(f"Final metrics: {metrics}")

    # Preserve iteration/best_iteration from student's prior meta so
    # self-play bookkeeping continues coherently.
    save_iter = student_meta.get("iteration", 0)
    save_best = student_meta.get("best_iteration", save_iter)
    save_checkpoint(
        student, args.out,
        iteration=save_iter,
        best_iteration=save_best,
        distilled_from=args.teacher,
        **metrics,
    )
    print(f"Saved distilled student to {args.out}")


if __name__ == "__main__":
    main()
