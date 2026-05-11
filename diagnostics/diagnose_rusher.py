"""Diagnose why the bot loses to a "walk forward" strategy.

A forward-rusher always plays the pawn move that most reduces its
shortest-path distance to its goal row.  Never places walls.  No
search, no thinking.  Just walk.

We pit this trivial opponent against:
    1.  Pure depth-N alpha-beta (no neural net at all)
    2.  Current best.pt at the GUI sim count (neural net + MCTS)

If the rusher beats AB, the eval function is broken.
If AB beats the rusher but the net loses, the net+MCTS layer is broken.
The win-rates tell us where the bug lives.

Usage
-----
    python3 diagnose_rusher.py --n-games 30 --ab-depth 8 --ab-time 3.0 \
        --net checkpoints/best.pt --net-sims 800
"""
from __future__ import annotations

import argparse
import time
from typing import Optional, Tuple, List

import numpy as np
import torch

# --- path bootstrap so this file can be run from anywhere ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# -------------------------------------------------------------
from quoridor import Board
from quoridor.ai import find_best_move
from quoridor.board import MOVE_PAWN, Move
from quoridor.encoding import action_to_move, canonical_view, move_to_action
from quoridor.mcts import EvalCache, MCTSConfig, search, select_action
from quoridor.net import best_available_device, load_checkpoint


def rusher_move(board: Board) -> Move:
    """Pick the pawn move that most reduces shortest-path distance."""
    me = board.turn
    cur_r, _ = board.pawns[me]
    goal_r = board.goal_row(me)
    cur_dist = board.shortest_path_length(me) or 999

    best_move = None
    best_new_dist = float("inf")
    best_progress = -float("inf")
    for (nr, nc) in board.pawn_moves(me):
        # Try this pawn move and check the new shortest-path length.
        cand = board.apply(Move(MOVE_PAWN, nr, nc))
        # cand.turn is now opponent; we want OUR new path length.
        # shortest_path_length takes a player explicitly, so pass me.
        new_dist = cand.shortest_path_length(me)
        if new_dist is None:
            continue
        # Tie-break: prefer the move whose row is closer to goal_r
        # (in case shortest path is the same, e.g., side-step has same
        # path length as moving forward — pick the forward one).
        progress = abs(cur_r - goal_r) - abs(nr - goal_r)
        if new_dist < best_new_dist or (
            new_dist == best_new_dist and progress > best_progress
        ):
            best_new_dist = new_dist
            best_progress = progress
            best_move = Move(MOVE_PAWN, nr, nc)

    if best_move is None:
        # Fallback: any legal pawn move
        moves = board.pawn_moves(me)
        if moves:
            r, c = moves[0]
            return Move(MOVE_PAWN, r, c)
        # Should never reach here in a real game
        return board.legal_moves()[0]
    return best_move


def smart_rusher_move(board: Board, plies_played: int, walls_used_by_me: int) -> Move:
    """Rusher with 1-2 'human-like' early defensive walls then pure rush.

    Heuristic: on plies 2-4, if we have walls left and there's a wall that
    increases the opponent's shortest path by ≥2, place it.  Otherwise
    just rush.  Caps at 2 walls total, so race speed is preserved.
    """
    me = board.turn
    opp = 1 - me
    if walls_used_by_me < 2 and 2 <= plies_played <= 6 and board.walls_left[me] > 0:
        # Find the wall that maximally extends opponent's shortest path
        cur_opp_dist = board.shortest_path_length(opp) or 0
        best_wall = None
        best_delay = 0
        for mv in board.legal_moves():
            if mv.kind == MOVE_PAWN:
                continue
            try:
                trial = board.apply(mv)
            except Exception:
                continue
            new_opp = trial.shortest_path_length(opp)
            if new_opp is None:
                continue
            # Also need our path to remain reasonable
            new_me = trial.shortest_path_length(me)
            if new_me is None:
                continue
            delay = new_opp - cur_opp_dist
            if delay > best_delay:
                best_delay = delay
                best_wall = mv
        if best_wall is not None and best_delay >= 2:
            return best_wall
    return rusher_move(board)


def play_game_vs_rusher(
    bot_kind: str,
    bot_state: dict,
    rusher_side: int,
    max_moves: int = 200,
    rusher_kind: str = "rush",
) -> Tuple[Optional[int], int, dict]:
    """Play one game between rusher and bot.

    Returns (winner, n_plies, stats) where stats includes:
        walls_placed: number of walls the bot played
        avg_wall_delay: average increase in rusher's shortest path per wall
        useless_walls: walls that increased rusher path by 0 or 1
        useful_walls: walls that increased rusher path by ≥2
    """
    board = Board.initial()
    move_count = 0
    walls_placed = 0
    wall_delays: List[int] = []   # path-length increase caused by each bot wall
    rusher_walls_used = 0

    while board.winner() is None and move_count < max_moves:
        if board.turn == rusher_side:
            if rusher_kind == "smart":
                mv = smart_rusher_move(board, move_count, rusher_walls_used)
                if mv is not None and mv.kind != MOVE_PAWN:
                    rusher_walls_used += 1
            else:
                mv = rusher_move(board)
        else:
            if bot_kind == "ab":
                mv = find_best_move(
                    board,
                    max_depth=bot_state["ab_depth"],
                    time_limit=bot_state["ab_time"],
                )
            elif bot_kind == "nn":
                cfg = bot_state["cfg"]
                root = search(
                    board, bot_state["net"], cfg, bot_state["device"],
                    add_noise=False, cache=bot_state["cache"],
                )
                action = select_action(root, temperature=0.0)
                _, _, _, _, _, _, flipped = canonical_view(board)
                mv = action_to_move(action, flipped)
            else:
                raise ValueError(bot_kind)
            # Measure wall effectiveness: did this wall actually slow the rusher?
            if mv is not None and mv.kind != MOVE_PAWN:
                walls_placed += 1
                pre_dist = board.shortest_path_length(rusher_side) or 0
                tentative = board.apply(mv)
                post_dist = tentative.shortest_path_length(rusher_side) or 0
                wall_delays.append(post_dist - pre_dist)
        if mv is None:
            break
        board = board.apply(mv)
        move_count += 1

    winner = board.winner()
    avg_delay = (sum(wall_delays) / len(wall_delays)) if wall_delays else 0
    useless = sum(1 for d in wall_delays if d <= 1)
    useful = sum(1 for d in wall_delays if d >= 2)
    stats = {
        "walls_placed": walls_placed,
        "avg_wall_delay": avg_delay,
        "useless_walls": useless,
        "useful_walls": useful,
    }
    return winner, move_count, stats


def run_match(
    bot_kind: str,
    bot_state: dict,
    n_games: int,
    label: str,
    rusher_kind: str = "rush",
) -> dict:
    rusher_wins = 0
    bot_wins = 0
    draws = 0
    total_plies = 0
    total_walls = 0
    total_useless = 0
    total_useful = 0
    delays: List[float] = []
    print(f"\n=== {label}: {n_games} games vs rusher ===")
    t0 = time.perf_counter()
    for g in range(n_games):
        # Alternate which side rushes
        rusher_side = g % 2
        winner, plies, stats = play_game_vs_rusher(
            bot_kind, bot_state, rusher_side, rusher_kind=rusher_kind
        )
        if winner is None:
            draws += 1
            tag = "D"
        elif winner == rusher_side:
            rusher_wins += 1
            tag = "RUSHER WIN"
        else:
            bot_wins += 1
            tag = "bot win"
        total_plies += plies
        total_walls += stats["walls_placed"]
        total_useless += stats["useless_walls"]
        total_useful += stats["useful_walls"]
        if stats["walls_placed"] > 0:
            delays.append(stats["avg_wall_delay"])
        print(
            f"  game {g+1:>2}/{n_games}  rusher=P{rusher_side+1}  "
            f"plies={plies:>3}  {tag:<11}  "
            f"walls={stats['walls_placed']:>2}  "
            f"useful={stats['useful_walls']}  useless={stats['useless_walls']}  "
            f"avg_delay={stats['avg_wall_delay']:.1f}"
        )
    dt = time.perf_counter() - t0
    rate = bot_wins / n_games
    print(
        f"\n  {label}: bot {bot_wins}-{rusher_wins}-{draws}  "
        f"({rate:.0%} win-rate)  "
        f"walls={total_walls} (useful={total_useful}, useless={total_useless})  "
        f"avg_delay/wall={sum(delays)/len(delays) if delays else 0:.2f}  "
        f"({dt:.0f}s, avg_plies={total_plies/n_games:.0f})"
    )
    return {
        "label": label,
        "bot_wins": bot_wins,
        "rusher_wins": rusher_wins,
        "draws": draws,
        "total_walls": total_walls,
        "useful_walls": total_useful,
        "useless_walls": total_useless,
        "win_rate": rate,
        "avg_plies": total_plies / n_games,
        "avg_delay_per_wall": sum(delays) / len(delays) if delays else 0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-games", type=int, default=20)
    p.add_argument("--ab-depth", type=int, default=8)
    p.add_argument("--ab-time", type=float, default=3.0)
    p.add_argument("--net", type=str, default="checkpoints/best.pt")
    p.add_argument("--net-sims", type=int, default=800)
    p.add_argument("--skip-ab", action="store_true")
    p.add_argument("--skip-nn", action="store_true")
    p.add_argument("--rusher", choices=["rush", "smart"], default="rush",
                   help="rush: pure pawn-forward; smart: 1-2 defensive walls + race")
    args = p.parse_args()

    results = []

    # 1. Pure alpha-beta vs rusher
    if not args.skip_ab:
        ab_state = {"ab_depth": args.ab_depth, "ab_time": args.ab_time}
        results.append(run_match(
            "ab", ab_state, args.n_games,
            f"alpha-beta depth={args.ab_depth} time={args.ab_time}s ({args.rusher})",
            rusher_kind=args.rusher,
        ))

    # 2. Neural net + MCTS vs rusher
    if not args.skip_nn:
        device = torch.device("cpu")  # CPU plenty for one game at a time
        net, meta = load_checkpoint(args.net, map_location="cpu")
        net.to(device)
        net.eval()
        cfg = MCTSConfig(num_simulations=args.net_sims, dirichlet_epsilon=0.0)
        nn_state = {
            "net": net, "device": device, "cfg": cfg, "cache": EvalCache(),
        }
        results.append(run_match(
            "nn", nn_state, args.n_games,
            f"neural net ({args.net}) sims={args.net_sims} ({args.rusher})",
            rusher_kind=args.rusher,
        ))

    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    for r in results:
        useful_pct = (
            r["useful_walls"] / r["total_walls"] * 100
            if r["total_walls"] else 0
        )
        print(
            f"  {r['label']:<55}  "
            f"bot {r['bot_wins']:>2}-{r['rusher_wins']:>2}-{r['draws']:>2}  "
            f"({r['win_rate']:.0%})  "
            f"walls={r['total_walls']} ({useful_pct:.0f}% useful)  "
            f"avg_delay/wall={r['avg_delay_per_wall']:.2f}  "
            f"avg_plies={r['avg_plies']:.0f}"
        )
    print("\nInterpretation:")
    print("  - bot win-rate < 50% → losing to a trivial 'walk-around-walls' strategy")
    print("  - useful_walls % low → walls placed are easy to walk around (1-square detour)")
    print("  - avg_delay/wall ~1.0 → walls are wasted (rusher just sidesteps)")
    print("  - avg_delay/wall ≥ 3.0 → walls force real detours (good)")
    print("  - if AB high useful% but NN low → NN policy is the bug (aux-value teaches path-counting)")
    print("  - if both have low useful% → wall-pruning or eval is the bug")


if __name__ == "__main__":
    main()
