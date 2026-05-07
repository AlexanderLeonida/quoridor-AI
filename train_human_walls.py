"""Targeted distillation: teach the NN the wall moves it missed against Alex.

Mines (state, oracle_wall_action) pairs from human-vs-NN games where the
human won and the bot rushed when a high-delay defensive wall was
available.  Trains the current best.pt to match these targets, with
rehearsal samples + KL regularisation toward the pre-training reference
to prevent catastrophic forgetting (PROCESS.md §35 mitigations).

Usage:
    python3 train_human_walls.py \\
        --in checkpoints/best.pt \\
        --out checkpoints/best_human_walls.pt \\
        --human-db data/quoridor.db \\
        --rehearsal-db data/quoridor_v3.db \\
        --epochs 8 --rehearsal-frac 0.5
"""
from __future__ import annotations

import argparse
import copy as _copy
import sqlite3
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from quoridor import Board, GameDB
from quoridor.board import MOVE_PAWN, Move
from quoridor.encoding import (
    ACTION_SPACE, action_to_move, canonical_view, deserialize_policy,
    encode_state, move_to_action,
)
from quoridor.net import best_available_device, load_checkpoint, save_checkpoint


def find_best_wall(board: Board, opp: int) -> Tuple[Optional[Move], int, int]:
    me = board.turn
    pre_opp = board.shortest_path_length(opp) or 0
    pre_me = board.shortest_path_length(me) or 0
    best, best_delay, best_my_inc = None, -1, 0
    for mv in board.legal_moves():
        if mv.kind == MOVE_PAWN:
            continue
        try:
            new_b = board.apply(mv)
        except Exception:
            continue
        new_opp = new_b.shortest_path_length(opp)
        new_me = new_b.shortest_path_length(me)
        if new_opp is None or new_me is None:
            continue
        delay = new_opp - pre_opp
        my_inc = new_me - pre_me
        if delay > best_delay or (delay == best_delay and my_inc < best_my_inc):
            best_delay, best, best_my_inc = delay, mv, my_inc
    return best, best_delay, best_my_inc


def mine_missed_wall_positions(human_db_path: str, min_delay: int = 2):
    """Find positions where bot rushed but a high-delay wall existed.

    Returns list of (state, action_idx, value) triples.  value = +1 from
    bot's POV (the wall was the move that would have helped it win).
    """
    conn = sqlite3.connect(human_db_path)
    cur = conn.execute(
        "SELECT id, winner, p1_source, p2_source, model_version FROM games "
        "WHERE (p1_source='human' AND p2_source='neural_nn') "
        "   OR (p1_source='neural_nn' AND p2_source='human') "
        "ORDER BY id"
    )
    rows = cur.fetchall()
    print(f"  Found {len(rows)} human-vs-NN games in {human_db_path}")

    db = GameDB(human_db_path)
    examples: List[Tuple[np.ndarray, int, float]] = []
    for gid, winner, p1s, p2s, mv_ver in rows:
        if p1s == "human":
            bot_side = 1
        elif p2s == "human":
            bot_side = 0
        else:
            continue
        # Only mine games the human won (bot's losing positions are the
        # informative ones).  Skip drawn / unfinished games.
        if winner != (1 - bot_side):
            continue
        moves = db.load_moves(gid)
        board = Board.initial()
        for mv in moves:
            if board.turn == bot_side and mv.kind == MOVE_PAWN:
                # Bot played pawn — was a useful wall available?
                d_bot = board.shortest_path_length(bot_side) or 0
                d_human = board.shortest_path_length(1 - bot_side) or 0
                # Only mine when bot is tied or behind — defending matters most.
                if d_human <= d_bot + 1:
                    best_wall, best_delay, my_inc = find_best_wall(
                        board, opp=1 - bot_side,
                    )
                    if (
                        best_wall is not None
                        and best_delay >= min_delay
                        and my_inc <= 1
                    ):
                        # Encode (state, oracle_wall_action) for training
                        state = encode_state(board)
                        _, _, _, _, _, _, flipped = canonical_view(board)
                        action_idx = move_to_action(best_wall, flipped)
                        # Value target: +1 means "this is a winning move"
                        examples.append((state, action_idx, +1.0))
            board = board.apply(mv)
    return examples


def sample_rehearsal(rehearsal_db_path: str, n_target: int, seed: int = 0):
    """Sample (state, soft_policy, value) from existing self-play games."""
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
            out.append((state, pol, float(z)))
            board = board.apply(move)
            if len(out) >= n_target:
                return out
    return out


def make_one_hot_policy(action_idx: int) -> np.ndarray:
    p = np.zeros(ACTION_SPACE, dtype=np.float32)
    p[action_idx] = 1.0
    return p


def train(net, examples, device, *, epochs, batch_size, lr, weight_decay,
          reference, reg_lambda):
    """Mixed-loss training with policy CE + KL anchor to reference net."""
    states = torch.from_numpy(np.stack([e[0] for e in examples]))
    pols = torch.from_numpy(np.stack([e[1] for e in examples]))
    vals = torch.from_numpy(np.array([e[2] for e in examples], dtype=np.float32))
    weights = torch.from_numpy(np.array([e[3] for e in examples], dtype=np.float32))
    weights = weights / weights.mean()

    n = states.size(0)
    n_val = max(1, n // 20)
    perm = torch.randperm(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    tr_loader = DataLoader(
        TensorDataset(states[tr_idx], pols[tr_idx], vals[tr_idx], weights[tr_idx]),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(states[val_idx], pols[val_idx], vals[val_idx]),
        batch_size=batch_size, shuffle=False,
    ) if n_val > 0 else None

    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(tr_loader))

    if reference is not None:
        reference.to(device); reference.eval()
        for p in reference.parameters():
            p.requires_grad = False

    best_val, best_state = None, None
    for epoch in range(1, epochs + 1):
        net.train()
        tl = tp = tv = tk = 0.0; tn = 0
        for xb, pb, vb, wb in tr_loader:
            xb, pb, vb, wb = (t.to(device) for t in (xb, pb, vb, wb))
            p_logits, v_pred = net(xb)
            log_p = F.log_softmax(p_logits, dim=1)
            loss_p = -(wb * (pb * log_p).sum(dim=1)).mean()
            loss_v = (wb * (v_pred - vb) ** 2).mean()
            loss = loss_p + 0.3 * loss_v   # protect value head from human-game noise
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
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            bs = xb.size(0)
            tl += loss.item()*bs; tp += loss_p.item()*bs
            tv += loss_v.item()*bs; tk += kl_val*bs; tn += bs
        line = f"  epoch {epoch}/{epochs}  loss={tl/tn:.4f} (p={tp/tn:.4f} v={tv/tn:.4f}"
        if reference is not None:
            line += f" kl={tk/tn:.4f}"
        line += ")"
        if val_loader:
            net.eval()
            vl = 0.0; vn = 0
            with torch.no_grad():
                for xb, pb, vb in val_loader:
                    xb, pb, vb = (t.to(device) for t in (xb, pb, vb))
                    p_logits, v_pred = net(xb)
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
                              for k, v in net.state_dict().items()}
                line += " [best]"
        print(line)
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, best_val


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="input", default="checkpoints/best.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--human-db", default="data/quoridor.db")
    p.add_argument("--rehearsal-db", default="data/quoridor_v3.db")
    p.add_argument("--rehearsal-frac", type=float, default=0.5,
                   help="Fraction of training data drawn from rehearsal "
                        "(rest is human-derived oracle-wall positions)")
    p.add_argument("--human-weight", type=float, default=10.0,
                   help="Per-sample weight for human-derived positions "
                        "(rehearsal positions get weight 1.0)")
    p.add_argument("--min-delay", type=int, default=2,
                   help="Minimum oracle-wall delay (squares) to count as "
                        "a missed-wall training example")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--reg-lambda", type=float, default=0.5,
                   help="KL regularisation toward pre-training reference")
    args = p.parse_args()

    print(f"\n=== Mining missed-wall positions from {args.human_db} ===")
    raw_examples = mine_missed_wall_positions(args.human_db, min_delay=args.min_delay)
    if not raw_examples:
        print("  No missed-wall positions found.  Nothing to train.")
        return
    print(f"  Mined {len(raw_examples)} missed-wall positions "
          f"(min_delay={args.min_delay}).")

    # Convert to (state, one-hot-policy, value, weight) tuples
    human_examples = [
        (state, make_one_hot_policy(action_idx), value, args.human_weight)
        for state, action_idx, value in raw_examples
    ]

    print(f"\n=== Sampling rehearsal data from {args.rehearsal_db} ===")
    n_human = len(human_examples)
    n_rehearsal = int(n_human * args.rehearsal_frac /
                      max(1e-6, 1.0 - args.rehearsal_frac))
    rehearsal_raw = sample_rehearsal(args.rehearsal_db, n_rehearsal)
    rehearsal_examples = [
        (state, pol, value, 1.0)
        for state, pol, value in rehearsal_raw
    ]
    print(f"  {len(rehearsal_examples)} rehearsal examples (target was {n_rehearsal})")

    examples = human_examples + rehearsal_examples
    print(f"\n=== Training: {len(examples)} examples "
          f"({len(human_examples)} human × weight {args.human_weight}, "
          f"{len(rehearsal_examples)} rehearsal × weight 1.0) ===")

    device = best_available_device()
    print(f"Device: {device}")
    net, meta = load_checkpoint(args.input, map_location=str(device))
    print(f"Loaded {args.input}  meta={meta}")

    reference = _copy.deepcopy(net) if args.reg_lambda > 0 else None
    if reference:
        print(f"  KL regularisation enabled (λ={args.reg_lambda})")

    net, best_val = train(
        net, examples, device,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay,
        reference=reference, reg_lambda=args.reg_lambda,
    )

    save_iter = meta.get("iteration", 0)
    save_best = meta.get("best_iteration", save_iter)
    save_checkpoint(
        net, args.out,
        iteration=save_iter,
        best_iteration=save_best,
        human_walls_trained=True,
        n_human_examples=len(human_examples),
        human_weight=args.human_weight,
        rehearsal_frac=args.rehearsal_frac,
        reg_lambda=args.reg_lambda,
        best_val_loss=float(best_val) if best_val else 0.0,
    )
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
