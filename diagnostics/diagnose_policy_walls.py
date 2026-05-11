"""Inspect the NN's raw policy on positions where the opponent is rushing.

We construct positions where the rusher is 2-4 plies from goal and measure:
1. What the bot's RAW POLICY (no MCTS) ranks as top moves
2. How those top moves rank wall placements vs pawn moves
3. Whether the policy's top wall (if any) is actually defensive

If the raw policy ranks pawn-forward over defensive walls, the policy is
the bug (failed distillation transfer).  If the raw policy ranks walls
high but MCTS still picks pawn-forward, the value head is the bug
(aux-value-blend teaching path-counting).
"""
from __future__ import annotations

import argparse
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# --- path bootstrap so this file can be run from anywhere ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# -------------------------------------------------------------
from quoridor import Board
from quoridor.board import MOVE_PAWN, Move
from quoridor.encoding import (
    ACTION_SPACE, action_to_move, canonical_view, encode_state,
    legal_action_mask,
)
from quoridor.net import load_checkpoint


def play_to_critical_position(rusher_side: int, n_rusher_moves: int) -> Board:
    """Play forward-rush by both sides to put the rusher near its goal.

    Both sides walk straight forward.  After n_rusher_moves moves by the
    rusher, return the board.  This positions the rusher 2-4 plies from
    its goal so we can see how the bot defends.
    """
    board = Board.initial()
    while True:
        if board.winner() is not None:
            return board
        # Both sides just walk forward (toward their goal row)
        me = board.turn
        cur_r, cur_c = board.pawns[me]
        goal_r = board.goal_row(me)
        best_move = None
        best_progress = -float("inf")
        for (nr, nc) in board.pawn_moves(me):
            progress = abs(cur_r - goal_r) - abs(nr - goal_r)
            if progress > best_progress:
                best_progress = progress
                best_move = Move(MOVE_PAWN, nr, nc)
        if best_move is None:
            break
        board = board.apply(best_move)
        # Stop after the rusher (rusher_side) has made n_rusher_moves moves
        # and we're back to the bot's turn
        rusher_moves = (
            sum(1 for _ in range(0))  # placeholder; we count via path lengths instead
        )
        rusher_dist = board.shortest_path_length(rusher_side)
        if rusher_dist is not None and rusher_dist <= 8 - n_rusher_moves and board.turn != rusher_side:
            return board
    return board


def policy_top_k(net, board: Board, k: int = 10) -> List[Tuple[int, str, float]]:
    """Return top-k actions from raw policy (no MCTS)."""
    state = encode_state(board)
    state_t = torch.from_numpy(state).unsqueeze(0)
    with torch.no_grad():
        p_logits, v = net(state_t)
    mask = legal_action_mask(board)
    logits = p_logits.squeeze(0).numpy()
    logits[~mask] = -1e9
    probs = F.softmax(torch.from_numpy(logits), dim=0).numpy()

    top_idx = np.argsort(-probs)[:k]
    out = []
    _, _, _, _, _, _, flipped = canonical_view(board)
    for idx in top_idx:
        if not mask[idx]:
            continue
        mv = action_to_move(int(idx), flipped)
        kind = "pawn" if mv.kind == MOVE_PAWN else (
            "wallH" if mv.kind == 1 else "wallV"
        )
        label = f"{kind}({mv.r},{mv.c})"
        out.append((int(idx), label, float(probs[idx])))
    return out, float(v.item())


def measure_wall_quality(board: Board, mv: Move, opp: int) -> int:
    """Return how many squares this move increases the opponent's path."""
    if mv.kind == MOVE_PAWN:
        return 0
    pre = board.shortest_path_length(opp) or 0
    try:
        new_board = board.apply(mv)
    except Exception:
        return 0
    post = new_board.shortest_path_length(opp) or 0
    return post - pre


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--net", default="checkpoints/best.pt")
    p.add_argument("--moves", type=int, default=5,
                   help="Rusher makes this many forward moves before we inspect")
    p.add_argument("--top-k", type=int, default=10)
    args = p.parse_args()

    net, meta = load_checkpoint(args.net, map_location="cpu")
    net.eval()
    print(f"Loaded {args.net}  meta={meta}")

    for rusher_side in (0, 1):
        bot_side = 1 - rusher_side
        print(f"\n{'='*60}")
        print(f"Rusher = P{rusher_side+1}  (bot = P{bot_side+1})")
        print(f"{'='*60}")
        board = play_to_critical_position(rusher_side, args.moves)
        d_rusher = board.shortest_path_length(rusher_side)
        d_bot = board.shortest_path_length(bot_side)
        print(f"Position: rusher path={d_rusher}, bot path={d_bot}, "
              f"turn=P{board.turn+1} ({'bot' if board.turn == bot_side else 'rusher'})")

        if board.turn != bot_side:
            print("(bot not on turn — running rusher's natural move first)")
            cur_r, _ = board.pawns[board.turn]
            goal_r = board.goal_row(board.turn)
            best_move, best_p = None, -float("inf")
            for (nr, nc) in board.pawn_moves(board.turn):
                pr = abs(cur_r - goal_r) - abs(nr - goal_r)
                if pr > best_p:
                    best_p = pr
                    best_move = Move(MOVE_PAWN, nr, nc)
            if best_move:
                board = board.apply(best_move)

        top_actions, value = policy_top_k(net, board, k=args.top_k)
        print(f"\nNN value estimate (bot POV): {value:+.3f}")
        print(f"Top-{args.top_k} raw policy moves:")
        n_walls = 0
        max_wall_delay = 0
        for i, (idx, label, prob) in enumerate(top_actions, 1):
            mv = action_to_move(idx, canonical_view(board)[6])
            delay = measure_wall_quality(board, mv, rusher_side)
            tag = ""
            if mv.kind != MOVE_PAWN:
                n_walls += 1
                if delay > max_wall_delay:
                    max_wall_delay = delay
                tag = f"  delay={delay}"
                if delay >= 2:
                    tag += " [USEFUL]"
                else:
                    tag += " [useless]"
            print(f"  {i:>2}. {label:<14}  p={prob:.3f}{tag}")
        print(f"\n  Walls in top-{args.top_k}: {n_walls}")
        if n_walls > 0:
            print(f"  Best wall delay in top-{args.top_k}: {max_wall_delay} squares")


if __name__ == "__main__":
    main()
