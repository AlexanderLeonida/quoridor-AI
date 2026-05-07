"""Detect 'drift' — moves where the bot moved backward or stayed level.

For each bot pawn move, measure:
  - Did it advance toward goal? (progress > 0)
  - Or stay level / move backward? (progress <= 0)

A productive game has the bot advancing nearly every pawn move.  A
'drifty' game has long stretches of zero/negative progress where the
bot oscillates between adjacent squares without making real progress.

Reports the longest non-productive run per game and where it started.
"""
from __future__ import annotations

import argparse
import sqlite3

from quoridor import Board, GameDB
from quoridor.board import MOVE_PAWN


def analyze_drift(db_path: str, game_id: int, bot_side: int):
    db = GameDB(db_path)
    moves = db.load_moves(game_id)
    board = Board.initial()
    print(f"\nGame {game_id}: bot=P{bot_side+1}")
    print(f"{'ply':>3}  {'mover':>5}  {'move':<8}  {'bot_path':>8}  {'progress':>8}  {'note':<20}")

    last_bot_path = None
    bot_progress_history = []
    drift_streak = 0
    max_drift_streak = 0
    drift_start_ply = None
    longest_drift_start = None

    for ply, mv in enumerate(moves):
        mover = "BOT" if board.turn == bot_side else "you"
        bot_path_before = board.shortest_path_length(bot_side) or 0
        if board.turn == bot_side and mv.kind == MOVE_PAWN:
            cur_r, _ = board.pawns[bot_side]
            goal_r = board.goal_row(bot_side)
            new_r = mv.r
            progress = abs(cur_r - goal_r) - abs(new_r - goal_r)
            bot_progress_history.append(progress)
            note = ""
            if progress > 0:
                note = f"forward +{progress}"
                if drift_streak > max_drift_streak:
                    max_drift_streak = drift_streak
                    longest_drift_start = drift_start_ply
                drift_streak = 0
                drift_start_ply = None
            elif progress == 0:
                note = "lateral (no progress)"
                if drift_streak == 0:
                    drift_start_ply = ply
                drift_streak += 1
            else:
                note = f"BACKWARD {progress}"
                if drift_streak == 0:
                    drift_start_ply = ply
                drift_streak += 1
            print(f"  {ply:>3}  {mover:>5}  {str(mv):<8}  "
                  f"{bot_path_before:>8}  {progress:>+8}  {note:<20}")
        elif board.turn == bot_side:
            print(f"  {ply:>3}  {mover:>5}  {str(mv):<8}  "
                  f"{bot_path_before:>8}  {'(wall)':>8}  ")
        board = board.apply(mv)

    if drift_streak > max_drift_streak:
        max_drift_streak = drift_streak
        longest_drift_start = drift_start_ply

    bot_pawn_moves = [p for p in bot_progress_history]
    forward = sum(1 for p in bot_pawn_moves if p > 0)
    lateral = sum(1 for p in bot_pawn_moves if p == 0)
    backward = sum(1 for p in bot_pawn_moves if p < 0)
    print(f"\n  Bot pawn moves: {len(bot_pawn_moves)} total")
    print(f"    forward: {forward}  lateral: {lateral}  backward: {backward}")
    print(f"    longest drift streak (no forward progress): {max_drift_streak} "
          f"consecutive bot pawn moves, starting around ply {longest_drift_start}")
    return forward, lateral, backward, max_drift_streak


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/quoridor.db")
    args = p.parse_args()

    conn = sqlite3.connect(args.db)
    rows = conn.execute(
        "SELECT id, winner, num_plies, p1_source, p2_source, model_version "
        "FROM games WHERE model_version LIKE 'gui-nn%' "
        "AND (p1_source='human' OR p2_source='human') "
        "ORDER BY id"
    ).fetchall()
    if not rows:
        print(f"No human-vs-NN games in {args.db}")
        return

    print(f"Found {len(rows)} human-vs-NN games")
    totals = {"forward": 0, "lateral": 0, "backward": 0}
    for r in rows:
        gid, winner, n_plies, p1s, p2s, mv = r
        bot_side = 1 if p1s == "human" else 0
        if n_plies < 5:
            continue
        f, l, b, max_streak = analyze_drift(args.db, gid, bot_side)
        totals["forward"] += f
        totals["lateral"] += l
        totals["backward"] += b

    print(f"\n{'='*65}")
    print(f"AGGREGATE across all {len(rows)} games:")
    total_pawn = sum(totals.values())
    if total_pawn > 0:
        print(f"  Forward moves:  {totals['forward']:>4}  ({totals['forward']/total_pawn*100:>4.0f}%)")
        print(f"  Lateral moves:  {totals['lateral']:>4}  ({totals['lateral']/total_pawn*100:>4.0f}%)")
        print(f"  Backward moves: {totals['backward']:>4}  ({totals['backward']/total_pawn*100:>4.0f}%)")


if __name__ == "__main__":
    main()
