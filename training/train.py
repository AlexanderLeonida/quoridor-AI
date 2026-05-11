"""Supervised trainer for the Quoridor policy/value network.

Reads games from the SQLite database (see `quoridor/database.py`) and
trains the network on (state, policy_target, z) triples with:

    loss = CrossEntropy(policy_logits, policy_target)   [policy head]
         + MSE(value, z)                                [value head]

When MCTS policy distributions (policy_blob) are stored alongside
moves, they are used as soft targets.  Otherwise the target is a one-hot
on the action taken ("behavior cloning" of the alpha-beta engine).

Usage
-----
    python3 train.py --epochs 10 --batch-size 256 --lr 1e-3 \
                     --out checkpoints/v1.pt

    # resume from an existing checkpoint:
    python3 train.py --resume checkpoints/v1.pt --epochs 5
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List, Optional, Tuple

import numpy as np

# --- path bootstrap so this file can be run from anywhere ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# -------------------------------------------------------------
from quoridor import GameDB
from quoridor.encoding import (
    ACTION_SPACE,
    canonical_view,
    deserialize_policy,
    encode_state,
    move_to_action,
)
from quoridor.net import (
    best_available_device,
    build_net,
    load_checkpoint,
    save_checkpoint,
)


def _lazy_torch():
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as e:
        print("PyTorch is required. Install with: pip install -r requirements.txt")
        raise SystemExit(1) from e
    return torch, F, DataLoader, TensorDataset


def build_dataset(
    db: GameDB,
    include_unfinished: bool = False,
    max_games: Optional[int] = None,
    draw_penalty: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialize (states, policy_targets, values) arrays from the DB.

    ``policy_targets`` is (N, ACTION_SPACE) float32 — soft distributions
    when policy_blob is available, one-hot vectors otherwise.
    """
    import math
    from quoridor import Board

    states: List[np.ndarray] = []
    policies: List[np.ndarray] = []
    values: List[float] = []

    games = list(db.iter_games(finished_only=not include_unfinished))
    if max_games is not None:
        games = games[-max_games:]
    n_games = len(games)
    print(f"Reading {n_games} game(s) from {db.path} ...")

    t0 = time.perf_counter()
    soft_count = 0
    for row in games:
        game_id = row[0]
        winner = row[3]
        moves = db.load_moves(game_id)
        blobs = db.load_policy_blobs(game_id)

        # Replay to find final board (for progress-aware draw values).
        board_state = Board.initial()
        boards_in_game: List = []
        for m in moves:
            boards_in_game.append(board_state)
            board_state = board_state.apply(m)
        final_board = board_state

        board_state = Board.initial()
        for move, blob in zip(moves, blobs):
            states.append(encode_state(board_state))
            if blob is not None:
                pol = deserialize_policy(blob)
                soft_count += 1
            else:
                _, _, _, _, _, _, flipped = canonical_view(board_state)
                act = move_to_action(move, flipped)
                pol = np.zeros(ACTION_SPACE, dtype=np.float32)
                pol[act] = 1.0
            policies.append(pol)
            if winner is not None:
                z = 1.0 if winner == board_state.turn else -1.0
            else:
                # Progress-aware draw value (same logic as selfplay.py).
                d0 = final_board.shortest_path_length(0)
                d1 = final_board.shortest_path_length(1)
                progress = math.tanh((d1 - d0) / 10.0)
                bonus = progress if board_state.turn == 0 else -progress
                z = float(np.clip(-draw_penalty + bonus * 0.3, -1.0, 1.0))
            values.append(z)
            board_state = board_state.apply(move)

    if not states:
        raise SystemExit(
            "No training samples found. Play some games first "
            "(e.g. `python3 play.py --selfplay`) so the DB is populated."
        )

    s = np.stack(states).astype(np.float32)
    p = np.stack(policies).astype(np.float32)
    v = np.asarray(values, dtype=np.float32)
    dt = time.perf_counter() - t0
    print(
        f"  {len(states):,} samples ({soft_count:,} with MCTS policy) "
        f"in {dt:.1f}s"
    )
    return s, p, v


def train(
    db_path: Optional[str],
    epochs: int,
    batch_size: int,
    lr: float,
    out_path: str,
    resume: Optional[str],
    include_unfinished: bool,
    val_frac: float,
    device_override: Optional[str],
    seed: int,
    max_games: Optional[int] = None,
    draw_penalty: float = 0.1,
) -> None:
    torch, F, DataLoader, TensorDataset = _lazy_torch()

    device = (
        torch.device(device_override) if device_override else best_available_device()
    )
    print(f"Device: {device}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    with GameDB(db_path) as db:
        states, policy_targets, values = build_dataset(
            db,
            include_unfinished=include_unfinished,
            max_games=max_games,
            draw_penalty=draw_penalty,
        )

    n = states.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_val = max(1, int(n * val_frac)) if val_frac > 0 else 0
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    print(f"Train samples: {len(train_idx):,}    Val samples: {len(val_idx):,}")

    def make_loader(sel: np.ndarray, shuffle: bool):
        ds = TensorDataset(
            torch.from_numpy(states[sel]),
            torch.from_numpy(policy_targets[sel]),
            torch.from_numpy(values[sel]),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader = make_loader(val_idx, shuffle=False) if n_val else None

    if resume and os.path.exists(resume):
        print(f"Resuming from {resume}")
        net, meta = load_checkpoint(resume, map_location=str(device))
        print(f"  previous meta: {meta}")
    else:
        net = build_net()
    net.to(device)

    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

    best_val: Optional[float] = None
    for epoch in range(1, epochs + 1):
        net.train()
        t0 = time.perf_counter()
        tot_loss = tot_p = tot_v = 0.0
        tot_n = 0
        for xb, pb, vb in train_loader:
            xb = xb.to(device, non_blocking=True)
            pb = pb.to(device, non_blocking=True)
            vb = vb.to(device, non_blocking=True)
            p_logits, v_pred = net(xb)

            # Soft cross-entropy: −Σ target * log_softmax(logits)
            log_probs = F.log_softmax(p_logits, dim=1)
            loss_p = -(pb * log_probs).sum(dim=1).mean()
            loss_v = F.mse_loss(v_pred, vb)
            loss = loss_p + loss_v

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            opt.step()

            bs = xb.size(0)
            tot_loss += loss.item() * bs
            tot_p += loss_p.item() * bs
            tot_v += loss_v.item() * bs
            tot_n += bs

        tr_loss = tot_loss / tot_n
        tr_p = tot_p / tot_n
        tr_v = tot_v / tot_n
        dt = time.perf_counter() - t0

        line = (
            f"epoch {epoch:>3}/{epochs}  "
            f"loss={tr_loss:.4f} (p={tr_p:.4f} v={tr_v:.4f})  ({dt:.1f}s)"
        )

        val_loss: Optional[float] = None
        if val_loader is not None:
            net.eval()
            vl = vp = vv = 0.0
            vn = 0
            with torch.no_grad():
                for xb, pb, vb in val_loader:
                    xb = xb.to(device)
                    pb = pb.to(device)
                    vb = vb.to(device)
                    p_logits, v_pred = net(xb)
                    log_probs = F.log_softmax(p_logits, dim=1)
                    lp = -(pb * log_probs).sum(dim=1).mean()
                    lv = F.mse_loss(v_pred, vb)
                    bs = xb.size(0)
                    vl += (lp.item() + lv.item()) * bs
                    vp += lp.item() * bs
                    vv += lv.item() * bs
                    vn += bs
            val_loss = vl / vn
            line += f"  | val={val_loss:.4f} (p={vp/vn:.4f} v={vv/vn:.4f})"

        print(line)

        if val_loss is not None:
            if best_val is None or val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    net,
                    out_path,
                    epoch=epoch,
                    train_loss=tr_loss,
                    val_loss=val_loss,
                    n_train=int(len(train_idx)),
                    n_val=int(len(val_idx)),
                )
                print(f"  -> saved {out_path} (best val_loss so far)")
        else:
            save_checkpoint(
                net, out_path, epoch=epoch, train_loss=tr_loss,
                n_train=int(len(train_idx)),
            )

    print("Done.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=str, default=None,
                   help="Path to games DB (defaults to data/quoridor.db).")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", type=str, default="checkpoints/quoridor.pt",
                   help="Where to save the trained checkpoint.")
    p.add_argument("--resume", type=str, default=None,
                   help="Checkpoint to initialise weights from.")
    p.add_argument("--include-unfinished", action="store_true",
                   help="Also train on games without a recorded winner (z=0).")
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="Held-out validation fraction (0 to disable).")
    p.add_argument("--device", type=str, default=None,
                   help="Force device: 'cpu', 'cuda', or 'mps'.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-games", type=int, default=None,
                   help="Only use the N most recent games (sliding window).")
    p.add_argument("--draw-penalty", type=float, default=0.1,
                   help="Base value penalty for draws (default 0.1).")
    args = p.parse_args()

    train(
        db_path=args.db,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_path=args.out,
        resume=args.resume,
        include_unfinished=args.include_unfinished,
        val_frac=args.val_frac,
        device_override=args.device,
        seed=args.seed,
        max_games=args.max_games,
        draw_penalty=args.draw_penalty,
    )


if __name__ == "__main__":
    main()
