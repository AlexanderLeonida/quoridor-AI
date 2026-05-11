"""For the same critical positions, find the OBJECTIVELY BEST defensive
wall (regardless of policy) and compare it to what the NN's policy says.

If the best objective wall has high delay (≥2) but the policy ranks it
near zero, the policy is the bug — distillation didn't transfer wall
quality.  If the best objective wall is itself only 1-square delay,
the position genuinely has no good defense and the bot's behavior is
correct (just lost the race earlier).
"""
from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F
import numpy as np

# --- path bootstrap so this file can be run from anywhere ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# -------------------------------------------------------------
from quoridor import Board
from quoridor.board import MOVE_PAWN, Move
from quoridor.encoding import (
    ACTION_SPACE, action_to_move, canonical_view, encode_state,
    legal_action_mask, move_to_action,
)
from quoridor.net import load_checkpoint


def play_to_critical_position(rusher_side: int, n_rusher_moves: int) -> Board:
    board = Board.initial()
    while True:
        if board.winner() is not None:
            return board
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
        rusher_dist = board.shortest_path_length(rusher_side)
        if rusher_dist is not None and rusher_dist <= 8 - n_rusher_moves and board.turn != rusher_side:
            return board
    return board


def find_best_wall(board: Board, opp: int) -> tuple:
    """Search ALL legal walls, return the one that maximally delays opponent.

    Returns (Move, opponent_path_increase, my_path_increase).
    """
    me = board.turn
    pre_opp = board.shortest_path_length(opp) or 0
    pre_me = board.shortest_path_length(me) or 0

    best_wall = None
    best_delay = -1
    my_increase_for_best = 0

    for mv in board.legal_moves():
        if mv.kind == MOVE_PAWN:
            continue
        try:
            new_board = board.apply(mv)
        except Exception:
            continue
        new_opp = new_board.shortest_path_length(opp)
        new_me = new_board.shortest_path_length(me)
        if new_opp is None or new_me is None:
            continue
        delay = new_opp - pre_opp
        my_inc = new_me - pre_me
        # Prefer high opponent delay; tie-break by low self-cost
        if delay > best_delay or (
            delay == best_delay and my_inc < my_increase_for_best
        ):
            best_delay = delay
            best_wall = mv
            my_increase_for_best = my_inc
    return best_wall, best_delay, my_increase_for_best


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--net", default="checkpoints/best.pt")
    p.add_argument("--moves", type=int, default=5)
    args = p.parse_args()

    net, meta = load_checkpoint(args.net, map_location="cpu")
    net.eval()
    print(f"Loaded {args.net}\n")

    for rusher_side in (0, 1):
        bot_side = 1 - rusher_side
        print(f"=== Rusher = P{rusher_side+1}, bot = P{bot_side+1} ===")
        board = play_to_critical_position(rusher_side, args.moves)
        # Make sure it's bot's turn
        if board.turn != bot_side:
            cur_r, _ = board.pawns[board.turn]
            goal_r = board.goal_row(board.turn)
            best, best_p = None, -float("inf")
            for (nr, nc) in board.pawn_moves(board.turn):
                pr = abs(cur_r - goal_r) - abs(nr - goal_r)
                if pr > best_p:
                    best_p = pr
                    best = Move(MOVE_PAWN, nr, nc)
            if best:
                board = board.apply(best)

        d_rusher = board.shortest_path_length(rusher_side) or 0
        d_bot = board.shortest_path_length(bot_side) or 0
        walls_left_bot = board.walls_left[bot_side]
        print(f"  Bot to move.  rusher_path={d_rusher}, bot_path={d_bot}, "
              f"bot_walls_left={walls_left_bot}")
        if d_rusher >= d_bot:
            print(f"  (bot is winning race; defensive wall less critical)")
        else:
            print(f"  (bot is LOSING race by {d_bot - d_rusher} square(s); "
                  f"needs defensive wall)")

        # Find the objectively best wall
        best_wall, delay, my_inc = find_best_wall(board, rusher_side)
        if best_wall is None:
            print("  No legal walls available.")
            continue
        print(f"  Best objective wall: H/V({best_wall.kind})({best_wall.r},{best_wall.c})  "
              f"delays opponent by {delay} square(s), increases own path by {my_inc}")

        # Where does the policy rank this wall?
        state = encode_state(board)
        with torch.no_grad():
            p_logits, v = net(state.reshape(1, *state.shape).astype(np.float32) if False
                              else torch.from_numpy(state).unsqueeze(0))
        mask = legal_action_mask(board)
        logits = p_logits.squeeze(0).numpy()
        logits[~mask] = -1e9
        probs = F.softmax(torch.from_numpy(logits), dim=0).numpy()
        # Find ranking of best_wall
        _, _, _, _, _, _, flipped = canonical_view(board)
        best_action_idx = move_to_action(best_wall, flipped)
        best_prob = probs[best_action_idx]
        sorted_indices = np.argsort(-probs)
        rank = int(np.where(sorted_indices == best_action_idx)[0][0]) + 1
        total_legal = int(mask.sum())
        print(f"  Policy says: prob={best_prob:.4f}  rank={rank}/{total_legal} legal moves")

        # Top-3 policy moves for context
        print(f"  Top-3 policy moves:")
        for i, idx in enumerate(sorted_indices[:3], 1):
            mv = action_to_move(int(idx), flipped)
            kind = "pawn" if mv.kind == MOVE_PAWN else "wall"
            print(f"    {i}. {kind}({mv.r},{mv.c})  p={probs[idx]:.3f}")
        print()


if __name__ == "__main__":
    main()
