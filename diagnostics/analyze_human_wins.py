"""Analyze games where the human beat the neural net.

For each move where the bot's choice was suboptimal, find what the bot
*should* have done — specifically: did it have a defensive wall available
that would force a real (≥2-square) detour?

Output:
    - Per-game summary (length, winner, bot's wall placement quality)
    - Position-by-position list of moves where the bot played a pawn move
      but a high-delay wall was available
    - Top "missed wall" positions across all games — these become training
      targets
"""
from __future__ import annotations

import argparse
import sqlite3
from typing import List, Optional, Tuple

# --- path bootstrap so this file can be run from anywhere ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# -------------------------------------------------------------
from quoridor import Board, GameDB
from quoridor.board import MOVE_PAWN, Move
from quoridor.encoding import canonical_view, move_to_action


def find_best_wall(board: Board, opp: int) -> Tuple[Optional[Move], int, int]:
    """Returns (best_wall_move, opponent_path_increase, my_path_increase)."""
    me = board.turn
    pre_opp = board.shortest_path_length(opp) or 0
    pre_me = board.shortest_path_length(me) or 0

    best = None
    best_delay = -1
    best_my_inc = 0
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
            best_delay = delay
            best = mv
            best_my_inc = my_inc
    return best, best_delay, best_my_inc


def analyze_game(db_path: str, game_id: int, bot_side: int):
    db = GameDB(db_path)
    moves = db.load_moves(game_id)

    board = Board.initial()
    print(f"\n{'='*65}")
    print(f"Game {game_id}: {len(moves)} moves, bot=P{bot_side+1}, "
          f"human=P{(1-bot_side)+1}")
    print(f"{'='*65}")
    missed_walls = []
    bot_walls_played = 0
    bot_walls_useful = 0
    for ply_num, mv in enumerate(moves):
        if board.turn == bot_side:
            d_bot = board.shortest_path_length(bot_side) or 0
            d_human = board.shortest_path_length(1 - bot_side) or 0
            best_wall, best_delay, my_inc = find_best_wall(board, opp=1 - bot_side)
            actual_was_wall = mv.kind != MOVE_PAWN
            if actual_was_wall:
                # measure quality of the wall the bot actually played
                pre_h = board.shortest_path_length(1 - bot_side) or 0
                try:
                    after = board.apply(mv)
                    post_h = after.shortest_path_length(1 - bot_side) or 0
                    actual_delay = post_h - pre_h
                except Exception:
                    actual_delay = 0
                bot_walls_played += 1
                if actual_delay >= 2:
                    bot_walls_useful += 1
                if best_wall is not None and best_delay > actual_delay + 1:
                    print(f"  ply {ply_num:>2}: bot played wall delay={actual_delay}; "
                          f"best wall would delay={best_delay} "
                          f"(human path {d_human}->{d_human+best_delay}, "
                          f"bot inc {my_inc})")
            else:
                # bot played pawn, did it skip a useful wall?
                if best_wall is not None and best_delay >= 2 and my_inc <= 1:
                    # human is close enough that this wall would actually matter?
                    if d_human <= d_bot + 1:  # bot losing race or tied
                        missed_walls.append((game_id, ply_num, board, best_wall, best_delay, d_bot, d_human))
                        print(f"  ply {ply_num:>2}: BOT PLAYED PAWN ({mv}) but had a "
                              f"+{best_delay}-delay wall available "
                              f"(human path {d_human}, bot path {d_bot})")
        board = board.apply(mv)

    print(f"\n  Summary: bot played {bot_walls_played} walls, "
          f"{bot_walls_useful} were useful (≥2 delay)")
    print(f"  Missed-wall positions (where bot rushed instead of walling): "
          f"{len(missed_walls)}")
    return missed_walls


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/quoridor.db")
    p.add_argument("--gui-only", action="store_true",
                   help="Filter to gui-nn games (vs neural net)")
    args = p.parse_args()

    conn = sqlite3.connect(args.db)
    cur = conn.execute(
        "SELECT id, winner, num_plies, p1_source, p2_source, model_version "
        "FROM games WHERE p1_source='human' OR p2_source='human' "
        "ORDER BY id DESC LIMIT 30"
    )
    rows = cur.fetchall()
    if args.gui_only:
        rows = [r for r in rows if r[5] and r[5].startswith("gui-nn")]

    print(f"Found {len(rows)} human games" +
          (" (vs NN only)" if args.gui_only else ""))
    if not rows:
        return

    all_missed = []
    for r in rows:
        gid, winner, n_plies, p1s, p2s, mv = r
        # Determine bot side
        if p1s == "human":
            bot_side = 1
        elif p2s == "human":
            bot_side = 0
        else:
            continue
        # Skip very short / aborted games
        if n_plies < 3:
            continue
        winner_str = "HUMAN" if winner == (1 - bot_side) else (
            "BOT" if winner == bot_side else "unfinished"
        )
        print(f"\n--- Game {gid}: {winner_str} won ({n_plies} plies, "
              f"version={mv}) ---")
        missed = analyze_game(args.db, gid, bot_side)
        all_missed.extend(missed)

    print(f"\n{'='*65}")
    print(f"TOTAL MISSED-WALL POSITIONS across {len(rows)} games: {len(all_missed)}")
    print(f"{'='*65}")
    if all_missed:
        print("These are positions where the bot rushed forward but should "
              "have placed a defensive wall.  They are training targets.")
        print(f"\nTop missed-wall positions by delay (bot was tied/losing race):")
        sorted_missed = sorted(all_missed, key=lambda x: -x[4])[:10]
        for gid, ply, _board, wall, delay, d_bot, d_human in sorted_missed:
            print(f"  game {gid} ply {ply}: wall would delay human by {delay} "
                  f"(human dist={d_human}, bot dist={d_bot})")


if __name__ == "__main__":
    main()
