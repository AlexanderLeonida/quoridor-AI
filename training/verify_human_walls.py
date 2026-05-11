"""Verify the human-walls-trained net now picks the oracle walls.

Reload the same 24 mined positions and compare:
    - Before-net (checkpoints/best.pt): what was the policy ranking of the oracle wall?
    - After-net (checkpoints/best_human_walls.pt): same comparison.

If the after-net consistently ranks the oracle wall at top-1 or top-3,
the targeted training worked.  If still buried at low rank, the
training didn't transfer (capacity issue or eval-bug).
"""
from __future__ import annotations

import argparse
import sqlite3

import numpy as np
import torch
import torch.nn.functional as F

from quoridor import Board, GameDB
from quoridor.board import MOVE_PAWN, Move
from quoridor.encoding import (
    action_to_move, canonical_view, encode_state, legal_action_mask,
    move_to_action,
)
from quoridor.net import load_checkpoint
from train_human_walls import find_best_wall


def policy_for_position(net, board: Board) -> np.ndarray:
    state = encode_state(board)
    with torch.no_grad():
        p_logits, _ = net(torch.from_numpy(state).unsqueeze(0))
    mask = legal_action_mask(board)
    logits = p_logits.squeeze(0).numpy()
    logits[~mask] = -1e9
    return F.softmax(torch.from_numpy(logits), dim=0).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--before", default="checkpoints/best.pt")
    p.add_argument("--after", default="checkpoints/best_human_walls.pt")
    p.add_argument("--human-db", default="data/quoridor.db")
    args = p.parse_args()

    print(f"Loading before: {args.before}")
    net_before, _ = load_checkpoint(args.before, map_location="cpu")
    net_before.eval()
    print(f"Loading after:  {args.after}")
    net_after, _ = load_checkpoint(args.after, map_location="cpu")
    net_after.eval()

    # Re-mine the same positions (must match train_human_walls)
    conn = sqlite3.connect(args.human_db)
    rows = conn.execute(
        "SELECT id, winner, p1_source, p2_source FROM games "
        "WHERE (p1_source='human' AND p2_source='neural_nn') "
        "   OR (p1_source='neural_nn' AND p2_source='human') "
        "ORDER BY id"
    ).fetchall()
    db = GameDB(args.human_db)

    n = 0
    before_top1 = before_top3 = before_top10 = 0
    after_top1 = after_top3 = after_top10 = 0
    print("\n  Position-by-position oracle-wall ranking (lower is better)")
    print(f"  {'pos':>3}  {'oracle_delay':>12}  {'before_rank':>12}  "
          f"{'after_rank':>11}  {'before_p':>10}  {'after_p':>10}")
    for gid, winner, p1s, p2s in rows:
        bot_side = 1 if p1s == "human" else 0
        if winner != (1 - bot_side):
            continue
        moves = db.load_moves(gid)
        board = Board.initial()
        for mv in moves:
            if board.turn == bot_side and mv.kind == MOVE_PAWN:
                d_bot = board.shortest_path_length(bot_side) or 0
                d_human = board.shortest_path_length(1 - bot_side) or 0
                if d_human <= d_bot + 1:
                    best_wall, best_delay, my_inc = find_best_wall(
                        board, opp=1 - bot_side
                    )
                    if (best_wall is not None and best_delay >= 2 and my_inc <= 1):
                        n += 1
                        _, _, _, _, _, _, flipped = canonical_view(board)
                        oracle_idx = move_to_action(best_wall, flipped)
                        probs_b = policy_for_position(net_before, board)
                        probs_a = policy_for_position(net_after, board)
                        rank_b = int(np.where(np.argsort(-probs_b) == oracle_idx)[0][0]) + 1
                        rank_a = int(np.where(np.argsort(-probs_a) == oracle_idx)[0][0]) + 1
                        if rank_b == 1: before_top1 += 1
                        if rank_b <= 3: before_top3 += 1
                        if rank_b <= 10: before_top10 += 1
                        if rank_a == 1: after_top1 += 1
                        if rank_a <= 3: after_top3 += 1
                        if rank_a <= 10: after_top10 += 1
                        print(f"  {n:>3}  {best_delay:>12}  {rank_b:>12}  "
                              f"{rank_a:>11}  {probs_b[oracle_idx]:>10.4f}  "
                              f"{probs_a[oracle_idx]:>10.4f}")
            board = board.apply(mv)

    print(f"\n  Summary across {n} mined positions:")
    print(f"  {'metric':<20}  {'before':>10}  {'after':>10}")
    print(f"  {'oracle in top-1':<20}  "
          f"{before_top1}/{n} ({before_top1/n*100:>3.0f}%)  "
          f"{after_top1}/{n} ({after_top1/n*100:>3.0f}%)")
    print(f"  {'oracle in top-3':<20}  "
          f"{before_top3}/{n} ({before_top3/n*100:>3.0f}%)  "
          f"{after_top3}/{n} ({after_top3/n*100:>3.0f}%)")
    print(f"  {'oracle in top-10':<20}  "
          f"{before_top10}/{n} ({before_top10/n*100:>3.0f}%)  "
          f"{after_top10}/{n} ({after_top10/n*100:>3.0f}%)")


if __name__ == "__main__":
    main()
