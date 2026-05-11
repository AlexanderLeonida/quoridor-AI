"""Quick benchmark: best.pt vs random init, plus vs alphabeta."""
from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch

# --- path bootstrap so this file can be run from anywhere ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# -------------------------------------------------------------
from quoridor import Board
from quoridor.ai import find_best_move
from quoridor.encoding import action_to_move, canonical_view
from quoridor.mcts import EvalCache, MCTSConfig, search, select_action
from quoridor.net import build_net, load_checkpoint


def play_nn_vs_nn(net_a, net_b, sims: int, opening_random: int, max_moves: int, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    cfg = MCTSConfig(num_simulations=sims, dirichlet_epsilon=0.0)
    b = Board.initial()
    for _ in range(opening_random):
        legal = b.legal_moves()
        if not legal or b.winner() is not None:
            break
        b = b.apply(random.choice(legal))
    caches = {0: EvalCache(), 1: EvalCache()}
    nets = {0: net_a, 1: net_b}
    moves = 0
    while b.winner() is None and moves < max_moves:
        cur = nets[b.turn]
        root = search(b, cur, cfg, torch.device("cpu"), add_noise=False, cache=caches[b.turn])
        a = select_action(root, 0.0)
        _, _, _, _, _, _, flipped = canonical_view(b)
        b = b.apply(action_to_move(a, flipped))
        moves += 1
    return b.winner(), moves


def play_nn_vs_ab(net, sims: int, ab_depth: int, ab_time: float, opening_random: int, max_moves: int, seed: int, nn_side: int):
    random.seed(seed)
    np.random.seed(seed)
    cfg = MCTSConfig(num_simulations=sims, dirichlet_epsilon=0.0)
    b = Board.initial()
    for _ in range(opening_random):
        legal = b.legal_moves()
        if not legal or b.winner() is not None:
            break
        b = b.apply(random.choice(legal))
    cache = EvalCache()
    moves = 0
    while b.winner() is None and moves < max_moves:
        if b.turn == nn_side:
            root = search(b, net, cfg, torch.device("cpu"), add_noise=False, cache=cache)
            a = select_action(root, 0.0)
            _, _, _, _, _, _, flipped = canonical_view(b)
            mv = action_to_move(a, flipped)
        else:
            mv = find_best_move(b, max_depth=ab_depth, time_limit=ab_time)
        b = b.apply(mv)
        moves += 1
    return b.winner(), moves


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/best.pt")
    p.add_argument("--n-games", type=int, default=10)
    p.add_argument("--sims", type=int, default=100)
    p.add_argument("--vs-random", action="store_true")
    p.add_argument("--vs-ab", action="store_true")
    p.add_argument("--vs-other", default=None,
                   help="Path to another checkpoint to play against.")
    p.add_argument("--ab-depth", type=int, default=2)
    p.add_argument("--ab-time", type=float, default=1.0)
    p.add_argument("--opening-random", type=int, default=4)
    p.add_argument("--max-moves", type=int, default=120)
    args = p.parse_args()

    best, meta = load_checkpoint(args.ckpt, map_location="cpu")
    best.eval()
    print(f"Loaded {args.ckpt}  meta={meta}")

    if args.vs_random:
        rand_net = build_net(blocks=meta.get("config", {}).get("blocks", 10),
                              filters=meta.get("config", {}).get("filters", 128))
        rand_net.eval()
        wins = draws = losses = 0
        t0 = time.perf_counter()
        for g in range(args.n_games):
            # alternate sides
            if g % 2 == 0:
                a, b = best, rand_net
                best_side = 0
            else:
                a, b = rand_net, best
                best_side = 1
            winner, mv = play_nn_vs_nn(a, b, args.sims, args.opening_random, args.max_moves, seed=1000 + g)
            if winner is None:
                draws += 1; tag = "D"
            elif winner == best_side:
                wins += 1; tag = "W"
            else:
                losses += 1; tag = "L"
            pct = (wins + 0.5 * draws) / (g + 1)
            print(f"  vs_rand g{g+1:02}  best_side={best_side}  moves={mv:3}  {tag}  score={pct:.0%} (W{wins}/L{losses}/D{draws})")
        print(f"\n  best vs random: W{wins} L{losses} D{draws}  score={(wins+0.5*draws)/args.n_games:.1%}  ({time.perf_counter()-t0:.0f}s)\n")

    if args.vs_other:
        other, other_meta = load_checkpoint(args.vs_other, map_location="cpu")
        other.eval()
        print(f"Loaded opponent {args.vs_other}  meta={other_meta}")
        wins = draws = losses = 0
        t0 = time.perf_counter()
        for g in range(args.n_games):
            if g % 2 == 0:
                a, b = best, other
                best_side = 0
            else:
                a, b = other, best
                best_side = 1
            winner, mv = play_nn_vs_nn(a, b, args.sims, args.opening_random, args.max_moves, seed=3000 + g)
            if winner is None:
                draws += 1; tag = "D"
            elif winner == best_side:
                wins += 1; tag = "W"
            else:
                losses += 1; tag = "L"
            pct = (wins + 0.5 * draws) / (g + 1)
            print(f"  vs_other g{g+1:02}  best_side={best_side}  moves={mv:3}  {tag}  score={pct:.0%} (W{wins}/L{losses}/D{draws})")
        print(f"\n  best vs {args.vs_other}: W{wins} L{losses} D{draws}  score={(wins+0.5*draws)/args.n_games:.1%}  ({time.perf_counter()-t0:.0f}s)\n")

    if args.vs_ab:
        wins = draws = losses = 0
        t0 = time.perf_counter()
        for g in range(args.n_games):
            nn_side = g % 2
            winner, mv = play_nn_vs_ab(best, args.sims, args.ab_depth, args.ab_time, args.opening_random, args.max_moves, seed=2000 + g, nn_side=nn_side)
            if winner is None:
                draws += 1; tag = "D"
            elif winner == nn_side:
                wins += 1; tag = "W"
            else:
                losses += 1; tag = "L"
            pct = (wins + 0.5 * draws) / (g + 1)
            print(f"  vs_ab g{g+1:02}  nn_side={nn_side}  moves={mv:3}  {tag}  score={pct:.0%} (W{wins}/L{losses}/D{draws})")
        print(f"\n  nn vs alphabeta(d={args.ab_depth},t={args.ab_time}s): W{wins} L{losses} D{draws}  score={(wins+0.5*draws)/args.n_games:.1%}  ({time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()
