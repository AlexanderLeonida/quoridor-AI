"""Architecture widening + dual-source distillation.

When iterative training and depth-N AB distillation both saturate, the
binding constraint is the network's *capacity*.  This script:

1.  Builds a wider net (e.g., 14×192 or 20×256) from scratch.
2.  Generates teacher targets from two sources combined:
    - The current 10×128 best.pt (its policy/value on sampled positions).
      Captures everything the net has learned so far.
    - Optionally fresh depth-8 alphabeta targets on the same positions.
      Adds search-deep ground truth on top of the net's learned patterns.
3.  Distills both signals into the wider net with rehearsal+KL-reg
    (mitigations from PROCESS.md §35).
4.  Saves the wider net for benching against the current best.

Usage
-----
    python3 widen_distill.py \\
        --student-blocks 14 --student-filters 192 \\
        --teacher-net checkpoints/best.pt \\
        --use-ab --ab-depth 8 --ab-time 5 \\
        --positions 4000 \\
        --out checkpoints/best_widened_14x192.pt

If `--use-ab` is omitted, only the existing-net teacher is used (pure
capacity-expansion distillation, no new search knowledge — useful as a
sanity check).
"""
from __future__ import annotations

import argparse
import copy as _copy
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
    action_to_move, canonical_view, deserialize_policy, encode_state, move_to_action,
)
from quoridor.mcts import EvalCache, MCTSConfig, get_policy, search
from quoridor.net import best_available_device, build_net, load_checkpoint, save_checkpoint


# ---------------------------------------------------------------------
# Position sampling (reuse logic from distill_deep.py)
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
# Net teacher (pool worker)
# ---------------------------------------------------------------------
_NET_TEACHER = None
_NET_DEVICE = None


def _net_teacher_init(ckpt_path: str):
    global _NET_TEACHER, _NET_DEVICE
    torch.set_num_threads(1)
    net, _ = load_checkpoint(ckpt_path, map_location="cpu")
    net.to("cpu"); net.eval()
    _NET_TEACHER = net
    _NET_DEVICE = torch.device("cpu")


def _net_teacher_one(args_tuple):
    """Forward through teacher net, return (state, soft_policy, value)."""
    board, sims = args_tuple
    cfg = MCTSConfig(num_simulations=sims, dirichlet_epsilon=0.0)
    cache = EvalCache()
    root = search(board, _NET_TEACHER, cfg, _NET_DEVICE,
                  add_noise=False, cache=cache)
    pol = get_policy(root, temperature=1.0).astype(np.float32)
    return encode_state(board), pol, float(root.value)


# ---------------------------------------------------------------------
# Alphabeta teacher (pool worker)
# ---------------------------------------------------------------------
def _ab_teacher_one(args_tuple):
    from quoridor.ai import find_best_move
    from quoridor.encoding import ACTION_SPACE
    board, depth, time_limit = args_tuple
    mv = find_best_move(board, max_depth=depth, time_limit=time_limit)
    _, _, _, _, _, _, flipped = canonical_view(board)
    a = move_to_action(mv, flipped)
    onehot = np.zeros(ACTION_SPACE, dtype=np.float32)
    onehot[a] = 1.0
    return encode_state(board), onehot, 0.0


# ---------------------------------------------------------------------
# Student training (mirrors distill_deep.distill_student)
# ---------------------------------------------------------------------
def distill_into_wider(
    student, examples, device, *,
    epochs, batch_size, lr, weight_decay,
    val_frac=0.05, reference=None, reg_lambda=0.0,
):
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
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs * len(tr_loader),
    )

    best_val = None
    best_state = None
    for epoch in range(1, epochs + 1):
        student.train()
        tl = tp = tv = tk = 0.0
        tn = 0
        for xb, pb, vb in tr_loader:
            xb, pb, vb = xb.to(device), pb.to(device), vb.to(device)
            p_logits, v_pred = student(xb)
            log_p = F.log_softmax(p_logits, dim=1)
            loss_p = -(pb * log_p).sum(dim=1).mean()
            loss_v = F.mse_loss(v_pred, vb)
            loss = loss_p + loss_v
            kl_val = 0.0
            if reference is not None and reg_lambda > 0:
                with torch.no_grad():
                    ref_logits, _ = reference(xb)
                    ref_log_p = F.log_softmax(ref_logits, dim=1)
                    ref_p = ref_log_p.exp()
                kl = (ref_p * (ref_log_p - log_p)).sum(dim=1).mean()
                loss = loss + reg_lambda * kl
                kl_val = float(kl.item())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step(); sched.step()
            bs = xb.size(0)
            tl += loss.item() * bs; tp += loss_p.item() * bs
            tv += loss_v.item() * bs; tk += kl_val * bs; tn += bs
        line = (f"  epoch {epoch}/{epochs}  "
                f"loss={tl/tn:.4f} (p={tp/tn:.4f} v={tv/tn:.4f}"
                f"{f' kl={tk/tn:.4f}' if reference is not None else ''})")
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
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--student-blocks", type=int, default=14)
    p.add_argument("--student-filters", type=int, default=192)
    p.add_argument("--teacher-net", default="checkpoints/best.pt",
                   help="Existing net to inherit knowledge from "
                        "(typically the current best).")
    p.add_argument("--db", default="data/quoridor_v3.db")
    p.add_argument("--positions", type=int, default=4000)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)

    # Net-teacher options
    p.add_argument("--teacher-sims", type=int, default=400,
                   help="MCTS sims when running the existing net as teacher "
                        "(more sims = stronger targets).")

    # Optional AB co-teacher
    p.add_argument("--use-ab", action="store_true",
                   help="In addition to the net-teacher, also generate "
                        "depth-8 AB targets on the same positions and mix "
                        "them in.  Adds ~40 min for AB but gives the "
                        "wider net both 'what current net knows' AND "
                        "'what deeper search would do'.")
    p.add_argument("--ab-depth", type=int, default=8)
    p.add_argument("--ab-time", type=float, default=5.0)
    p.add_argument("--ab-mix-frac", type=float, default=0.5,
                   help="If --use-ab, fraction of positions where AB "
                        "targets *replace* net targets (rest use net "
                        "targets only).  0.5 means half each.")

    # Student training
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=2e-4)
    p.add_argument("--reg-lambda", type=float, default=0.0,
                   help="If non-zero, add KL(teacher_net || student) "
                        "regularisation during training.  For widening, "
                        "the student starts blank and we *want* it to "
                        "match the teacher closely, so default 0 (no "
                        "regularisation pulling student away from teacher).")

    p.add_argument("--out", required=True)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    print(f"Building wider student: {args.student_blocks}×{args.student_filters}")
    student = build_net(blocks=args.student_blocks,
                        filters=args.student_filters)
    n_params = sum(t.numel() for t in student.state_dict().values())
    print(f"  {n_params/1e6:.2f}M parameters")

    print(f"\nSampling {args.positions} positions from {args.db}...")
    db = GameDB(args.db)
    boards = sample_positions(db, args.positions, seed=args.seed)
    print(f"  got {len(boards)} positions")

    # --- Net-teacher targets ---
    print(f"\nGenerating net-teacher targets ({args.teacher_net}, "
          f"sims={args.teacher_sims}, {args.workers} workers)...")
    t0 = time.perf_counter()
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers,
                  initializer=_net_teacher_init,
                  initargs=(args.teacher_net,)) as pool:
        jobs = [(b, args.teacher_sims) for b in boards]
        net_examples = list(pool.imap_unordered(_net_teacher_one, jobs,
                                                chunksize=4))
    print(f"  {len(net_examples)} net-teacher targets in "
          f"{time.perf_counter()-t0:.0f}s")

    # --- Optional AB co-teacher ---
    examples = net_examples
    if args.use_ab:
        n_ab = int(len(boards) * args.ab_mix_frac)
        ab_boards = random.sample(boards, n_ab)
        print(f"\nGenerating AB-teacher targets on {n_ab} positions "
              f"(depth={args.ab_depth}, time={args.ab_time}s)...")
        t0 = time.perf_counter()
        with ctx.Pool(processes=args.workers) as pool:
            jobs = [(b, args.ab_depth, args.ab_time) for b in ab_boards]
            ab_examples = list(pool.imap_unordered(_ab_teacher_one, jobs,
                                                   chunksize=4))
        print(f"  {len(ab_examples)} AB-teacher targets in "
              f"{time.perf_counter()-t0:.0f}s")
        # Concat — wider net sees both signals
        examples = examples + ab_examples
        print(f"\nCombined training set: {len(examples)} (state, policy, value) triples")

    # --- Student training ---
    device = best_available_device()
    print(f"\nDistilling into wider net on {device}...")
    student, best_val = distill_into_wider(
        student, examples, device,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay,
        reference=None, reg_lambda=args.reg_lambda,
    )

    save_checkpoint(
        student, args.out,
        iteration=0,
        best_iteration=0,
        widened_from=args.teacher_net,
        widened_blocks=args.student_blocks,
        widened_filters=args.student_filters,
        ab_co_teacher=args.use_ab,
        widen_distill_val_loss=float(best_val) if best_val else 0.0,
    )
    print(f"\nSaved {args.out}")
    print(f"  config: {args.student_blocks}×{args.student_filters} "
          f"({n_params/1e6:.2f}M params)")


if __name__ == "__main__":
    main()
