"""Benchmark a set of NN checkpoints against alpha-beta at several depths.

Plays N games per (checkpoint × AB-setting) combo, alternating colours,
and saves the W/L/D matrix to ``analysis/bench_matrix.json``.

The plot script ``analysis/08_nn_vs_ab.py`` reads that JSON and renders
a heatmap (rows: checkpoints, columns: AB depth/time settings).

Usage
-----
    python3 bench_matrix.py \\
        --ckpt checkpoints/best.pt:current \\
        --ckpt checkpoints/iter_0034.pt:v34 \\
        --ckpt checkpoints/iter_0019.pt:v19_distill \\
        --ckpt checkpoints/warmstart_10x128.pt:warmstart \\
        --ab "d3,t1.0" --ab "d4,t2.0" --ab "d5,t4.0" \\
        --games 12 --sims 200 --workers 4

Higher AB depth + time = stronger reference opponent.  At d6/t8s a
single game can run a minute or two — keep the player + AB matrix
modest unless you have time to burn.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

from quoridor import Board
from quoridor.ai import find_best_move
from quoridor.encoding import action_to_move, canonical_view
from quoridor.mcts import EvalCache, MCTSConfig, search, select_action
from quoridor.net import load_checkpoint
from selfplay import _randomise_opening, adjudicate_winner


# ---------------------------------------------------------------------
# One game
# ---------------------------------------------------------------------
def _play_nn_vs_ab(
    net, sims: int, ab_depth: int, ab_time: float,
    nn_side: int, opening_random: int, max_moves: int,
    adjudicate_gap: int, seed: int,
):
    random.seed(seed); np.random.seed(seed)
    cfg = MCTSConfig(num_simulations=sims, dirichlet_epsilon=0.0)
    board = Board.initial()
    if opening_random > 0:
        board, _ = _randomise_opening(board, opening_random)
    cache = EvalCache()
    moves = 0
    device = torch.device("cpu")
    while board.winner() is None and moves < max_moves:
        if board.turn == nn_side:
            root = search(board, net, cfg, device, add_noise=False, cache=cache)
            a = select_action(root, 0.0)
            _, _, _, _, _, _, flipped = canonical_view(board)
            mv = action_to_move(a, flipped)
        else:
            mv = find_best_move(board, max_depth=ab_depth, time_limit=ab_time)
        board = board.apply(mv)
        moves += 1
    winner = board.winner()
    if winner is None and adjudicate_gap > 0:
        winner = adjudicate_winner(board, min_gap=adjudicate_gap)
    return winner, moves


# ---------------------------------------------------------------------
# Worker pool — loads each checkpoint once.
# ---------------------------------------------------------------------
_BENCH_NETS: Dict[str, torch.nn.Module] = {}


def _worker_init(ckpts: List[Tuple[str, str]]):
    global _BENCH_NETS
    torch.set_num_threads(1)
    _BENCH_NETS = {}
    for label, path in ckpts:
        net, _ = load_checkpoint(path, map_location="cpu")
        net.to("cpu"); net.eval()
        _BENCH_NETS[label] = net


def _worker_play(job):
    label, sims, ab_depth, ab_time, nn_side, openrand, maxmv, adj_gap, seed = job
    winner, moves = _play_nn_vs_ab(
        _BENCH_NETS[label], sims, ab_depth, ab_time,
        nn_side, openrand, maxmv, adj_gap, seed,
    )
    return label, ab_depth, ab_time, nn_side, winner, moves


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def _parse_ab_spec(spec: str) -> Tuple[int, float]:
    """``"d4,t2.0"`` → (4, 2.0)."""
    parts = dict(p.strip() for p in [] if False) if False else {}
    parts = {}
    for tok in spec.split(","):
        tok = tok.strip()
        if tok.startswith("d"):
            parts["d"] = int(tok[1:])
        elif tok.startswith("t"):
            parts["t"] = float(tok[1:])
    if "d" not in parts or "t" not in parts:
        raise SystemExit(f"Bad --ab spec {spec!r}; want 'd4,t2.0'")
    return parts["d"], parts["t"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", action="append", required=True,
                   help="path:label (repeat).")
    p.add_argument("--ab", action="append", required=True,
                   help="AB spec 'd<depth>,t<time>' (repeat).")
    p.add_argument("--games", type=int, default=10,
                   help="Games per (ckpt × AB) combo, alternating colours.")
    p.add_argument("--sims", type=int, default=200,
                   help="MCTS sims for the NN side.")
    p.add_argument("--opening-random", type=int, default=4)
    p.add_argument("--max-moves", type=int, default=120)
    p.add_argument("--adjudicate-gap", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="analysis/bench_matrix.json")
    args = p.parse_args()

    ckpts: List[Tuple[str, str]] = []
    for spec in args.ckpt:
        if ":" in spec:
            path, label = spec.split(":", 1)
        else:
            path, label = spec, os.path.splitext(os.path.basename(spec))[0]
        ckpts.append((label, path))

    ab_specs = [_parse_ab_spec(s) for s in args.ab]
    print(f"Players: {[l for l, _ in ckpts]}")
    print(f"AB settings: {ab_specs}")

    rng = random.Random(args.seed)
    jobs: List[Tuple] = []
    for (label, _), (d, t) in itertools.product(ckpts, ab_specs):
        for k in range(args.games):
            nn_side = k % 2  # alternate
            jobs.append((
                label, args.sims, d, t, nn_side,
                args.opening_random, args.max_moves, args.adjudicate_gap,
                rng.randint(0, 2**31 - 1),
            ))
    total = len(jobs)
    print(f"Total games: {total}  ({args.games} per combo × "
          f"{len(ckpts) * len(ab_specs)} combos)")

    # Aggregate: matrix[label][ab_key] = {"wins": ..., "losses": ..., "draws": ...}
    matrix: Dict[str, Dict[str, Dict[str, int]]] = {}
    for label, _ in ckpts:
        matrix[label] = {}
        for d, t in ab_specs:
            matrix[label][f"d{d}_t{t}"] = {"wins": 0, "losses": 0, "draws": 0,
                                            "moves": []}

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    with ctx.Pool(processes=args.workers, initializer=_worker_init,
                  initargs=(ckpts,)) as pool:
        for done, (label, d, t, nn_side, winner, moves) in enumerate(
            pool.imap_unordered(_worker_play, jobs), 1,
        ):
            key = f"d{d}_t{t}"
            cell = matrix[label][key]
            cell["moves"].append(moves)
            if winner is None:
                cell["draws"] += 1
                tag = "D"
            elif winner == nn_side:
                cell["wins"] += 1
                tag = "W"
            else:
                cell["losses"] += 1
                tag = "L"
            elapsed = time.perf_counter() - t0
            eta = elapsed / done * (total - done)
            print(f"  [{done}/{total}]  {label:<15} vs {key:<10} "
                  f"nn_side={nn_side}  {moves:3}p  {tag}  "
                  f"[{elapsed:.0f}s, eta {eta:.0f}s]")

    # Summary
    print("\nFinal matrix (rows: ckpt, cols: AB setting, score = (W+0.5D)/N):")
    cols = [f"d{d}_t{t}" for d, t in ab_specs]
    header = f"  {'ckpt':<20}  " + "  ".join(f"{c:<14}" for c in cols)
    print(header)
    for label, _ in ckpts:
        cells = []
        for c in cols:
            r = matrix[label][c]
            n = r["wins"] + r["losses"] + r["draws"]
            score = (r["wins"] + 0.5 * r["draws"]) / max(1, n)
            cells.append(f"{score:.0%} ({r['wins']}-{r['losses']}-{r['draws']})")
        print(f"  {label:<20}  " + "  ".join(f"{c:<14}" for c in cells))

    # Save (move list dropped to keep JSON compact; mean kept).
    out = {
        "config": {
            "games_per_combo": args.games, "sims": args.sims,
            "opening_random": args.opening_random,
            "max_moves": args.max_moves, "adjudicate_gap": args.adjudicate_gap,
        },
        "ab_specs": [{"depth": d, "time": t} for d, t in ab_specs],
        "results": {
            label: {
                key: {
                    "wins": r["wins"], "losses": r["losses"], "draws": r["draws"],
                    "score": (r["wins"] + 0.5 * r["draws"]) /
                             max(1, r["wins"] + r["losses"] + r["draws"]),
                    "avg_moves": float(np.mean(r["moves"])) if r["moves"] else 0.0,
                }
                for key, r in d.items()
            }
            for label, d in matrix.items()
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
