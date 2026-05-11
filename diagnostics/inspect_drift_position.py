"""Replay a specific game to a specific ply, then inspect the NN's
policy + value at that exact position.  Tells us whether 'drifting'
moves are policy choices or MCTS noise."""
from __future__ import annotations

import argparse
import numpy as np
import torch
import torch.nn.functional as F

# --- path bootstrap so this file can be run from anywhere ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# -------------------------------------------------------------
from quoridor import Board, GameDB
from quoridor.board import MOVE_PAWN
from quoridor.encoding import (
    action_to_move, canonical_view, encode_state, legal_action_mask,
    move_to_action,
)
from quoridor.net import load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/quoridor.db")
    p.add_argument("--game", type=int, required=True)
    p.add_argument("--ply", type=int, required=True,
                   help="Stop replay at this ply (will inspect bot's choice if it's bot turn)")
    p.add_argument("--net", default="checkpoints/best.pt")
    p.add_argument("--top-k", type=int, default=12)
    args = p.parse_args()

    db = GameDB(args.db)
    moves = db.load_moves(args.game)
    board = Board.initial()
    for i, mv in enumerate(moves):
        if i >= args.ply:
            break
        board = board.apply(mv)

    print(f"\nReplayed game {args.game} to ply {args.ply}")
    print(f"  bot path={board.shortest_path_length(0)} P0,  "
          f"path={board.shortest_path_length(1)} P1")
    print(f"  to_play=P{board.turn+1}")
    print(f"  walls left: P1={board.walls_left[0]}, P2={board.walls_left[1]}")
    print(f"  pawns: P1={board.pawns[0]}, P2={board.pawns[1]}")

    net, _ = load_checkpoint(args.net, map_location="cpu")
    net.eval()
    state = encode_state(board)
    with torch.no_grad():
        p_logits, v = net(torch.from_numpy(state).unsqueeze(0))
    mask = legal_action_mask(board)
    logits = p_logits.squeeze(0).numpy()
    logits[~mask] = -1e9
    probs = F.softmax(torch.from_numpy(logits), dim=0).numpy()
    print(f"\n  NN value (current player's POV): {float(v.item()):+.3f}")
    print(f"  Policy entropy: {-(probs[probs > 0] * np.log(probs[probs > 0])).sum():.3f} nats")
    print(f"  Max policy prob: {probs.max():.3f}  (top-1 confidence)")

    cur_r, _ = board.pawns[board.turn]
    goal_r = board.goal_row(board.turn)
    print(f"\n  Top-{args.top_k} moves from raw policy:")
    print(f"  {'rank':>4}  {'move':<14}  {'prob':>7}  {'progress':>9}  {'note':<10}")
    sorted_idx = np.argsort(-probs)[:args.top_k]
    _, _, _, _, _, _, flipped = canonical_view(board)
    for rank, idx in enumerate(sorted_idx, 1):
        if probs[idx] < 1e-6:
            break
        mv_obj = action_to_move(int(idx), flipped)
        if mv_obj.kind == MOVE_PAWN:
            new_r = mv_obj.r
            progress = abs(cur_r - goal_r) - abs(new_r - goal_r)
            kind = "pawn"
            note = "FORWARD" if progress > 0 else (
                "lateral" if progress == 0 else "BACKWARD"
            )
            print(f"  {rank:>4}  {kind}({mv_obj.r},{mv_obj.c})    "
                  f"{probs[idx]:>7.3f}  {progress:>+9}  {note}")
        else:
            kind = "wallH" if mv_obj.kind == 1 else "wallV"
            print(f"  {rank:>4}  {kind}({mv_obj.r},{mv_obj.c})  "
                  f"{probs[idx]:>7.3f}  {'(wall)':>9}  ")


if __name__ == "__main__":
    main()
