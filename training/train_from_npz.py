"""Targeted training using a pre-built .npz dataset.

Trains the current net on the 1200-position human-derived dataset
(built by build_human_training_set.py) plus rehearsal data from
self-play, with KL anchor to the pre-training reference.

Usage:
    python3 train_from_npz.py \\
        --in checkpoints/best.pt \\
        --out checkpoints/best_human_v2.pt \\
        --npz data/human_training_set.npz \\
        --rehearsal-frac 0.6 --epochs 6 --reg-lambda 1.0
"""
from __future__ import annotations

import argparse
import copy as _copy
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# --- path bootstrap so this file can be run from anywhere ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# -------------------------------------------------------------
from quoridor import Board, GameDB
from quoridor.encoding import (
    ACTION_SPACE, deserialize_policy, encode_state,
)
from quoridor.net import best_available_device, load_checkpoint, save_checkpoint


def sample_rehearsal(rehearsal_db_path: str, n_target: int, seed: int = 0):
    """Sample (state, soft_policy, value, weight) from existing self-play games."""
    import random
    rng = random.Random(seed)
    db = GameDB(rehearsal_db_path)
    games = [
        r for r in db.iter_games(finished_only=False)
        if r[5] == "selfplay_nn" and r[6] == "selfplay_nn"
    ]
    rng.shuffle(games)
    out = []
    for r in games:
        gid = r[0]
        winner = r[3]
        moves = db.load_moves(gid)
        blobs = db.load_policy_blobs(gid)
        board = Board.initial()
        for move, blob in zip(moves, blobs):
            if blob is None:
                board = board.apply(move)
                continue
            state = encode_state(board)
            pol = deserialize_policy(blob).astype(np.float32)
            if winner is None:
                z = 0.0
            else:
                z = 1.0 if winner == board.turn else -1.0
            out.append((state, pol, float(z), 1.0))
            board = board.apply(move)
            if len(out) >= n_target:
                return out
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="input", default="checkpoints/best.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--npz", default="data/human_training_set.npz")
    p.add_argument("--rehearsal-db", default="data/quoridor_v3.db")
    p.add_argument("--rehearsal-frac", type=float, default=0.6,
                   help="Fraction of training data drawn from rehearsal (default 0.6)")
    p.add_argument("--human-weight-mult", type=float, default=1.0,
                   help="Multiplier on top of the per-sample weights "
                        "stored in the .npz (real=3, permuted=1)")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--reg-lambda", type=float, default=1.0,
                   help="KL regularisation toward pre-training reference (default 1.0)")
    p.add_argument("--value-weight", type=float, default=0.3,
                   help="Loss weight on value head (default 0.3 — protects "
                        "the distilled value head)")
    args = p.parse_args()

    print(f"Loading training set from {args.npz}...")
    data = np.load(args.npz)
    h_states = data["states"]
    h_policies = data["policies"]
    h_values = data["values"]
    h_weights = data["weights"] * args.human_weight_mult
    print(f"  {h_states.shape[0]} human-derived examples "
          f"(real weights: {h_weights[h_weights > 1.5].shape[0]}, "
          f"permuted: {h_weights[h_weights <= 1.5].shape[0]})")

    n_human = h_states.shape[0]
    n_rehearsal_target = int(n_human * args.rehearsal_frac /
                             max(1e-6, 1.0 - args.rehearsal_frac))
    print(f"\nSampling {n_rehearsal_target} rehearsal examples from {args.rehearsal_db}...")
    rehearsal = sample_rehearsal(args.rehearsal_db, n_rehearsal_target)
    print(f"  got {len(rehearsal)} rehearsal triples")

    # Combine
    r_states = np.stack([e[0] for e in rehearsal])
    r_policies = np.stack([e[1] for e in rehearsal])
    r_values = np.array([e[2] for e in rehearsal], dtype=np.float32)
    r_weights = np.ones(len(rehearsal), dtype=np.float32)

    states = np.concatenate([h_states, r_states])
    policies = np.concatenate([h_policies, r_policies])
    values = np.concatenate([h_values, r_values])
    weights = np.concatenate([h_weights, r_weights])
    weights = weights / weights.mean()  # normalise

    print(f"\nFinal training set: {states.shape[0]} examples "
          f"({n_human} human × weight ~{h_weights.mean():.1f}, "
          f"{len(rehearsal)} rehearsal × weight 1.0)")

    # Load nets
    device = best_available_device()
    print(f"\nDevice: {device}")
    net, meta = load_checkpoint(args.input, map_location=str(device))
    print(f"Loaded {args.input}  meta={meta}")

    reference = _copy.deepcopy(net) if args.reg_lambda > 0 else None
    if reference:
        reference.to(device); reference.eval()
        for p_ in reference.parameters():
            p_.requires_grad = False
        print(f"  KL regularisation enabled (λ={args.reg_lambda})")

    # Tensor dataset
    states_t = torch.from_numpy(states.astype(np.float32))
    pols_t = torch.from_numpy(policies.astype(np.float32))
    vals_t = torch.from_numpy(values.astype(np.float32))
    weights_t = torch.from_numpy(weights.astype(np.float32))

    n = states_t.size(0)
    n_val = max(1, n // 20)
    perm = torch.randperm(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    tr_loader = DataLoader(
        TensorDataset(states_t[tr_idx], pols_t[tr_idx], vals_t[tr_idx], weights_t[tr_idx]),
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(states_t[val_idx], pols_t[val_idx], vals_t[val_idx]),
        batch_size=args.batch_size, shuffle=False,
    )

    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(tr_loader))

    best_val, best_state = None, None
    print()
    for epoch in range(1, args.epochs + 1):
        net.train()
        tl = tp = tv = tk = 0.0; tn = 0
        t0 = time.perf_counter()
        for xb, pb, vb, wb in tr_loader:
            xb, pb, vb, wb = (t.to(device) for t in (xb, pb, vb, wb))
            p_logits, v_pred = net(xb)
            log_p = F.log_softmax(p_logits, dim=1)
            loss_p = -(wb * (pb * log_p).sum(dim=1)).mean()
            loss_v = (wb * (v_pred - vb) ** 2).mean()
            loss = loss_p + args.value_weight * loss_v
            kl_val = 0.0
            if reference is not None:
                with torch.no_grad():
                    ref_logits, _ = reference(xb)
                    ref_log_p = F.log_softmax(ref_logits, dim=1)
                    ref_p = ref_log_p.exp()
                kl = (ref_p * (ref_log_p - log_p)).sum(dim=1).mean()
                loss = loss + args.reg_lambda * kl
                kl_val = float(kl.item())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            bs = xb.size(0)
            tl += loss.item()*bs; tp += loss_p.item()*bs
            tv += loss_v.item()*bs; tk += kl_val*bs; tn += bs
        dt = time.perf_counter() - t0
        line = (f"  epoch {epoch}/{args.epochs}  "
                f"loss={tl/tn:.4f} (p={tp/tn:.4f} v={tv/tn:.4f}"
                f"{f' kl={tk/tn:.4f}' if reference is not None else ''})  ({dt:.0f}s)")
        net.eval()
        vl = vp_ = vv_ = 0.0; vn = 0
        with torch.no_grad():
            for xb, pb, vb in val_loader:
                xb, pb, vb = (t.to(device) for t in (xb, pb, vb))
                p_logits, v_pred = net(xb)
                log_p = F.log_softmax(p_logits, dim=1)
                lp = -(pb * log_p).sum(dim=1).mean()
                lv = F.mse_loss(v_pred, vb)
                bs = xb.size(0)
                vl += (lp.item() + lv.item()) * bs
                vp_ += lp.item() * bs
                vv_ += lv.item() * bs
                vn += bs
        cur_val = vl / vn
        line += f"  val={cur_val:.4f} (p={vp_/vn:.4f} v={vv_/vn:.4f})"
        if best_val is None or cur_val < best_val:
            best_val = cur_val
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            line += " [best]"
        print(line)

    if best_state is not None:
        net.load_state_dict(best_state)
        print(f"\nRestored best-val checkpoint (val={best_val:.4f})")

    save_iter = meta.get("iteration", 0)
    save_best = meta.get("best_iteration", save_iter)
    save_checkpoint(
        net, args.out,
        iteration=save_iter,
        best_iteration=save_best,
        human_v2_trained=True,
        n_human_examples=n_human,
        rehearsal_frac=args.rehearsal_frac,
        reg_lambda=args.reg_lambda,
        value_weight=args.value_weight,
        best_val_loss=float(best_val),
    )
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
