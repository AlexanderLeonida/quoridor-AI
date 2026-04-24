"""Round-robin tournament between checkpoints → globally-consistent Elos.

The per-iteration gating match in selfplay.py updates two players' Elos
by K=32 against each head-to-head outcome.  That is a *local* signal:
two nets that never meet cannot be compared, and a net that retired
after its last gate match (e.g. v74) is frozen at whatever Elo it had
then — even if newer, stronger nets exist in other arches.

This tool plays a round-robin between a user-specified list of
checkpoints, then fits Elo ratings by maximum likelihood (iterative
Bradley–Terry) so every rating reflects every match outcome at once.

Usage
-----
    python3 tournament.py \\
        --ckpt checkpoints/best.pt:current \\
        --ckpt checkpoints/iter_0074.pt:v74_6x64 \\
        --ckpt checkpoints/iter_0019.pt:v19_distill \\
        --ckpt checkpoints/iter_0036.pt:v36_run2peak \\
        --ckpt checkpoints/iter_0048.pt:v48_run3peak \\
        --ckpt checkpoints/warmstart_10x128.pt:warmstart \\
        --games 10 --sims 200 --workers 4 \\
        --out checkpoints/elo_tournament.json

Each unordered pair plays ``--games`` games total with alternating
colors (so each colour is played --games/2 times).  Outcomes are
aggregated into a W/L/D record and Elos are solved by gradient descent
on the Bradley–Terry log-likelihood with ``anchor`` fixed at 1000.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import time
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import torch

from quoridor import Board
from quoridor.encoding import action_to_move, canonical_view, serialize_policy
from quoridor.mcts import EvalCache, MCTSConfig, get_policy, search, select_action
from quoridor.net import load_checkpoint
from selfplay import _randomise_opening, adjudicate_winner


# ---------------------------------------------------------------------
# One game between two nets (CPU)
# ---------------------------------------------------------------------
def _play_game(
    net_a, net_b, sims: int, opening_random: int, max_moves: int,
    adjudicate_gap: int, seed: int, record_policies: bool = False,
):
    """Play one greedy-MCTS game between *net_a* (P0) and *net_b* (P1).

    If ``record_policies`` is True, the MCTS visit-count distribution at
    every move is recorded.  These are valid training targets even though
    the action was chosen greedily — visits encode what the net searched
    and found promising.
    """
    random.seed(seed)
    np.random.seed(seed)
    cfg = MCTSConfig(num_simulations=sims, dirichlet_epsilon=0.0)
    board = Board.initial()
    if opening_random > 0:
        board, _ = _randomise_opening(board, opening_random)
    caches = {0: EvalCache(), 1: EvalCache()}
    nets = {0: net_a, 1: net_b}
    move_count = 0
    device = torch.device("cpu")
    moves_played: List = []
    policy_blobs: List[bytes] = []
    while board.winner() is None and move_count < max_moves:
        cur = nets[board.turn]
        root = search(
            board, cur, cfg, device, add_noise=False, cache=caches[board.turn],
        )
        if record_policies:
            policy_blobs.append(serialize_policy(get_policy(root, 1.0)))
        action = select_action(root, 0.0)
        _, _, _, _, _, _, flipped = canonical_view(board)
        move = action_to_move(action, flipped)
        moves_played.append(move)
        board = board.apply(move)
        move_count += 1
    winner = board.winner()
    if winner is None and adjudicate_gap > 0:
        winner = adjudicate_winner(board, min_gap=adjudicate_gap)
    if record_policies:
        return winner, move_count, moves_played, policy_blobs
    return winner, move_count


# ---------------------------------------------------------------------
# Worker pool (one job = one game)
# ---------------------------------------------------------------------
_TOURNEY_NETS: Dict[str, "torch.nn.Module"] = {}
_TOURNEY_CFG: Dict = {}


def _worker_init(ckpt_paths: List[Tuple[str, str]], cfg: Dict) -> None:
    global _TOURNEY_NETS, _TOURNEY_CFG
    import torch as _torch  # noqa: F401

    torch.set_num_threads(1)
    _TOURNEY_NETS = {}
    for label, path in ckpt_paths:
        net, _ = load_checkpoint(path, map_location="cpu")
        net.to("cpu")
        net.eval()
        _TOURNEY_NETS[label] = net
    _TOURNEY_CFG = cfg


def _worker_play(job):
    a_label, b_label, seed = job
    net_a = _TOURNEY_NETS[a_label]
    net_b = _TOURNEY_NETS[b_label]
    record = _TOURNEY_CFG.get("record_policies", False)
    out = _play_game(
        net_a, net_b,
        sims=_TOURNEY_CFG["sims"],
        opening_random=_TOURNEY_CFG["opening_random"],
        max_moves=_TOURNEY_CFG["max_moves"],
        adjudicate_gap=_TOURNEY_CFG["adjudicate_gap"],
        seed=seed,
        record_policies=record,
    )
    if record:
        winner, moves, moves_played, policy_blobs = out
        return a_label, b_label, winner, moves, moves_played, policy_blobs
    winner, moves = out
    return a_label, b_label, winner, moves, None, None


# ---------------------------------------------------------------------
# Bradley–Terry MLE for Elo
# ---------------------------------------------------------------------
def compute_elos(
    matches: List[Tuple[str, str, int, int, int]],
    anchor_label: str = None,
    anchor_rating: float = 1000.0,
    iterations: int = 3000,
    lr: float = 1.0,
) -> Dict[str, float]:
    """Fit Elo ratings via gradient descent on BT log-likelihood.

    Each entry in ``matches`` is (label_a, label_b, wins_a, wins_b, draws).
    Draws count as 0.5 for each side.
    """
    players = sorted({a for a, _, *_ in matches} | {b for _, b, *_ in matches})
    R = {p: float(anchor_rating) for p in players}
    if anchor_label is None:
        anchor_label = players[0]

    # Gradient of BT log-likelihood w.r.t. R_i is:
    #   sum over j of (s_ij - E_ij) * total_ij * ln(10) / 400
    # We absorb constants and anchor R[anchor_label] after each step.
    scale = math.log(10) / 400.0
    for _ in range(iterations):
        grad = {p: 0.0 for p in players}
        for a, b, wa, wb, d in matches:
            total = wa + wb + d
            if total == 0:
                continue
            s_a = (wa + 0.5 * d) / total
            # Expected score for a
            e_a = 1.0 / (1.0 + 10.0 ** ((R[b] - R[a]) / 400.0))
            delta = (s_a - e_a) * total
            grad[a] += delta
            grad[b] -= delta
        for p in players:
            R[p] += lr * grad[p] / scale / max(1, len(players))
        # Re-anchor so ratings don't drift
        shift = anchor_rating - R[anchor_label]
        for p in players:
            R[p] += shift
    return R


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ckpt", action="append", required=True,
                   help="path:label (repeat). label defaults to basename.")
    p.add_argument("--games", type=int, default=10,
                   help="Games per unordered pair (colors alternate).")
    p.add_argument("--sims", type=int, default=200)
    p.add_argument("--opening-random", type=int, default=4)
    p.add_argument("--max-moves", type=int, default=100)
    p.add_argument("--adjudicate-gap", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--anchor", type=str, default=None,
                   help="Label to anchor at 1000 (defaults to first).")
    p.add_argument("--out", type=str, default="checkpoints/elo_tournament.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-to-db", type=str, default=None,
                   help="If set, record MCTS policies and save winning-side "
                        "games to this DB as training data (e.g. "
                        "data/quoridor_v3.db).")
    p.add_argument("--save-champion-only", action="store_true",
                   help="With --save-to-db, save only games won by the "
                        "post-hoc Elo champion (highest final rating).")
    args = p.parse_args()

    # Parse checkpoints
    ckpts: List[Tuple[str, str]] = []
    for spec in args.ckpt:
        if ":" in spec:
            path, label = spec.split(":", 1)
        else:
            path, label = spec, os.path.splitext(os.path.basename(spec))[0]
        ckpts.append((label, path))
    if len({l for l, _ in ckpts}) != len(ckpts):
        raise SystemExit("Duplicate labels in --ckpt list.")
    labels = [l for l, _ in ckpts]
    print(f"Players ({len(labels)}): {labels}")
    anchor = args.anchor or labels[0]

    # Build pair schedule with alternating colors
    jobs: List[Tuple[str, str, int]] = []
    rng = random.Random(args.seed)
    for i, j in itertools.combinations(range(len(labels)), 2):
        a, b = labels[i], labels[j]
        for k in range(args.games):
            # alternate colors: even k → a=P0, odd k → b=P0
            if k % 2 == 0:
                jobs.append((a, b, rng.randint(0, 2**31 - 1)))
            else:
                jobs.append((b, a, rng.randint(0, 2**31 - 1)))
    total_games = len(jobs)
    print(f"Total games: {total_games}  ({args.games} per pair × "
          f"{len(labels)*(len(labels)-1)//2} pairs)")

    cfg = {
        "sims": args.sims,
        "opening_random": args.opening_random,
        "max_moves": args.max_moves,
        "adjudicate_gap": args.adjudicate_gap,
        "record_policies": args.save_to_db is not None,
    }

    # Buffer of recorded games (only populated if --save-to-db is set).
    recorded: List[Dict] = []

    # W/L/D tally — key is (player, opponent) with player as "row"
    wld: Dict[Tuple[str, str], List[int]] = {}
    for a, b in itertools.combinations(labels, 2):
        wld[(a, b)] = [0, 0, 0]  # wins_a, wins_b, draws

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    with ctx.Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(ckpts, cfg),
    ) as pool:
        for done, (a_side, b_side, winner, moves, moves_played, policy_blobs) in enumerate(
            pool.imap_unordered(_worker_play, jobs), 1,
        ):
            # Canonicalise unordered key (alphabetical) for aggregation
            if a_side < b_side:
                key, swap = (a_side, b_side), False
            else:
                key, swap = (b_side, a_side), True
            wld.setdefault(key, [0, 0, 0])
            if winner is None:
                wld[key][2] += 1
                tag = "D"
            elif (winner == 0 and not swap) or (winner == 1 and swap):
                wld[key][0] += 1
                tag = f"{a_side} W"
            else:
                wld[key][1] += 1
                tag = f"{b_side} W"
            if cfg["record_policies"] and moves_played is not None:
                recorded.append({
                    "p1_label": a_side,
                    "p2_label": b_side,
                    "winner": winner,
                    "moves": moves_played,
                    "blobs": policy_blobs,
                })
            elapsed = time.perf_counter() - t0
            eta = elapsed / done * (total_games - done)
            print(f"  [{done}/{total_games}]  {a_side} vs {b_side} "
                  f"({'P1' if not swap else 'P2'}-P{2 if not swap else 1})  "
                  f"{moves:3}p  {tag:>12}  "
                  f"[{elapsed:.0f}s elapsed, eta {eta:.0f}s]")

    # Print match matrix
    print("\nMatch summary (row beats column, score %):")
    header = "                    " + " ".join(f"{l:>14}" for l in labels)
    print(header)
    for a in labels:
        row = [f"{a:<20}"]
        for b in labels:
            if a == b:
                row.append(f"{'-':>14}")
            else:
                key = (a, b) if (a, b) in wld else (b, a)
                wa, wb, d = wld[key]
                if key == (a, b):
                    total = wa + wb + d
                    score = (wa + 0.5 * d) / total if total else 0
                else:
                    total = wa + wb + d
                    score = (wb + 0.5 * d) / total if total else 0
                if total:
                    row.append(f"{score:.0%} ({wa}-{wb}-{d})"[:14].rjust(14))
                else:
                    row.append(f"{'n/a':>14}")
        print(" ".join(row))

    # Compute Elos
    match_rows = [(a, b, w[0], w[1], w[2]) for (a, b), w in wld.items() if sum(w) > 0]
    elos = compute_elos(match_rows, anchor_label=anchor, anchor_rating=1000.0)

    print("\nElo ratings (anchored at 1000 for "
          f"{anchor!r}):")
    ranked = sorted(elos.items(), key=lambda kv: -kv[1])
    for i, (label, r) in enumerate(ranked, 1):
        print(f"  {i:>2}. {label:<25s} {r:7.1f}")

    # Save results
    out = {
        "anchor": anchor,
        "config": cfg,
        "games_per_pair": args.games,
        "matches": [
            {"a": a, "b": b, "wins_a": w[0], "wins_b": w[1], "draws": w[2]}
            for (a, b), w in wld.items()
        ],
        "ratings": elos,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {args.out}")

    # Optionally save games to the self-play DB as training fodder.
    # Only winning-side games are saved: we only want to teach the net
    # moves that won, not moves that lost.  When --save-champion-only is
    # set we further restrict to games the Elo champion won — the single
    # strongest target signal available.
    if args.save_to_db and recorded:
        from quoridor import GameDB

        champion = ranked[0][0] if ranked else None
        kept = 0
        db = GameDB(args.save_to_db)
        try:
            for rec in recorded:
                if rec["winner"] is None:
                    continue  # skip draws
                winner_label = rec["p1_label"] if rec["winner"] == 0 else rec["p2_label"]
                if args.save_champion_only and winner_label != champion:
                    continue
                db.save_game(
                    rec["moves"],
                    winner=rec["winner"],
                    p1_source="selfplay_nn",   # so training loader picks it up
                    p2_source="selfplay_nn",
                    model_version=f"tourney-{winner_label}",
                    notes=f"tournament_{rec['p1_label']}_vs_{rec['p2_label']}",
                    policies=rec["blobs"],
                )
                kept += 1
        finally:
            db.close()
        print(f"Saved {kept} tournament games into {args.save_to_db}")


if __name__ == "__main__":
    main()
