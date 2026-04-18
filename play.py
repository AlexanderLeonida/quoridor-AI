"""Play Quoridor from the terminal.

Notation
--------
Columns are `a`-`i` (left to right). Rows are `1`-`9`.
Player 1 (Red) starts at `e1` and must reach row 9.
Player 2 (Blue) starts at `e9` and must reach row 1.

At the prompt:
    e2          move your pawn to e2
    e5h         place a horizontal wall anchored at e5  (rows 1-8, cols a-h)
    e5v         place a vertical   wall anchored at e5
    moves       list all legal moves (in this notation)
    q           quit

Usage
-----
    python3 play.py               # prompts for side
    python3 play.py --player 1    # skip prompt; play as P1
    python3 play.py --player 2    # play as P2
    python3 play.py --selfplay    # AI vs AI
    python3 play.py --depth 4 --time 8
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Tuple

from quoridor import (
    BOARD_SIZE,
    Board,
    GameRecorder,
    MOVE_PAWN,
    Move,
    WALL_GRID,
    WALL_H,
    WALL_V,
    find_best_move,
)

# -------------------------------------------------------------------
# Notation
# -------------------------------------------------------------------

COL_LABELS = "abcdefghi"  # internal col 0..8


def format_cell(r: int, c: int) -> str:
    return f"{COL_LABELS[c]}{r + 1}"


def format_move(m: Move) -> str:
    base = format_cell(m.r, m.c)
    if m.kind == MOVE_PAWN:
        return base
    return base + ("h" if m.kind == WALL_H else "v")


def parse_cell(s: str) -> Optional[Tuple[int, int]]:
    s = s.lower()
    if len(s) < 2 or s[0] not in COL_LABELS:
        return None
    try:
        row = int(s[1:])
    except ValueError:
        return None
    if not (1 <= row <= BOARD_SIZE):
        return None
    return (row - 1, COL_LABELS.index(s[0]))


def parse_move(s: str) -> Optional[Move]:
    s = s.strip().lower().replace(" ", "")
    if not s:
        return None
    # Wall notation: cell + suffix h|v (e.g. `e5h`, `a1v`).
    if len(s) >= 3 and s[-1] in ("h", "v"):
        cell = parse_cell(s[:-1])
        if cell is None:
            return None
        r, c = cell
        if not (0 <= r < WALL_GRID and 0 <= c < WALL_GRID):
            return None
        return Move(WALL_H if s[-1] == "h" else WALL_V, r, c)
    # Pawn move.
    cell = parse_cell(s)
    if cell is None:
        return None
    return Move(MOVE_PAWN, cell[0], cell[1])


# -------------------------------------------------------------------
# Rendering
# -------------------------------------------------------------------

RESET = "\033[0m"
RED = "\033[1;31m"
BLUE = "\033[1;34m"
GRAY = "\033[90m"
BOLD = "\033[1m"


def _color_enabled(override: Optional[bool]) -> bool:
    if override is not None:
        return override
    return sys.stdout.isatty()


def _c(text: str, color: str, on: bool) -> str:
    return f"{color}{text}{RESET}" if on else text


def render_board(
    board: Board,
    human_player: Optional[int] = 1,
    color: Optional[bool] = None,
) -> str:
    """Render with `human_player` at the top of the screen (AI at the bottom).

    `human_player` is 0 (P1/Red) or 1 (P2/Blue). Pass None for self-play
    (defaults to the natural internal orientation — internal row 0 at top).
    """
    col_on = _color_enabled(color)
    flip = human_player == 1  # P2 (internal row 8) flips to appear at top

    row_order = list(range(BOARD_SIZE))
    if flip:
        row_order.reverse()

    lines: List[str] = []
    col_header = "    " + "   ".join(COL_LABELS[c] for c in range(BOARD_SIZE))
    lines.append(col_header)

    for i, r in enumerate(row_order):
        row_label = str(r + 1)
        parts: List[str] = [f"{row_label}  "]
        for c in range(BOARD_SIZE):
            if board.pawns[0] == (r, c):
                parts.append(_c(" R ", RED, col_on))
            elif board.pawns[1] == (r, c):
                parts.append(_c(" B ", BLUE, col_on))
            else:
                parts.append(" . ")
            if c < BOARD_SIZE - 1:
                if board.is_blocked(r, c, r, c + 1):
                    parts.append(_c("|", GRAY, col_on))
                else:
                    parts.append(" ")
        parts.append(f"  {row_label}")
        lines.append("".join(parts))

        if i < BOARD_SIZE - 1:
            r_next = row_order[i + 1]
            r_top = min(r, r_next)  # wall anchor row for h_walls lookup
            sep: List[str] = ["   "]
            for c in range(BOARD_SIZE):
                if board.is_blocked(r_top, c, r_top + 1, c):
                    sep.append(_c("---", GRAY, col_on))
                else:
                    sep.append("   ")
                if c < BOARD_SIZE - 1:
                    sep.append("+")
            lines.append("".join(sep))

    lines.append(col_header)
    status = (
        f"Walls:  {_c('P1(R)', RED, col_on)}={board.walls_left[0]}   "
        f"{_c('P2(B)', BLUE, col_on)}={board.walls_left[1]}    "
        f"Turn: {_c('P1(R)', RED, col_on) if board.turn == 0 else _c('P2(B)', BLUE, col_on)}"
    )
    lines.append(status)
    return "\n".join(lines)


# -------------------------------------------------------------------
# Input / game loop
# -------------------------------------------------------------------

def ask_side(color_on: bool) -> int:
    print(f"{_c('Quoridor', BOLD, color_on)}")
    print("Choose your side:")
    print(f"  1) Player 1  ({_c('Red', RED, color_on)}, moves first,  starts at e1)")
    print(f"  2) Player 2  ({_c('Blue', BLUE, color_on)}, moves second, starts at e9)")
    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if raw == "1":
            return 0
        if raw == "2":
            return 1
        print("Enter 1 or 2.")


def human_move(board: Board) -> Optional[Move]:
    """Ask the human for a move. Returns None if they want to quit."""
    while True:
        try:
            raw = input("your move > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not raw:
            continue
        if raw in ("q", "quit", "exit"):
            return None
        if raw == "moves":
            legal = board.legal_moves()
            print(f"{len(legal)} legal moves:")
            print("  " + "  ".join(format_move(m) for m in legal))
            continue
        m = parse_move(raw)
        if m is None:
            print("Could not parse. Examples: 'e2', 'e5h', 'a1v'. Also: 'moves', 'q'.")
            continue
        if not board.is_legal(m):
            print("Illegal move.")
            continue
        return m


def ai_move(board: Board, depth: int, time_limit: float) -> Optional[Move]:
    label = "P1(R)" if board.turn == 0 else "P2(B)"
    print(f"AI ({label}) thinking...")
    move = find_best_move(board, max_depth=depth, time_limit=time_limit)
    if move is None:
        print("AI has no moves; game over.")
        return None
    print(f"AI plays: {format_move(move)}")
    return move


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Quoridor vs the AI.")
    parser.add_argument("--player", type=int, choices=(1, 2),
                        help="Skip the prompt and play as P1 or P2.")
    parser.add_argument("--selfplay", action="store_true",
                        help="AI vs AI.")
    parser.add_argument("--depth", type=int, default=3,
                        help="Max search depth (default 3).")
    parser.add_argument("--time", type=float, default=5.0,
                        help="Per-move AI time budget in seconds (default 5).")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable ANSI colors.")
    parser.add_argument("--no-record", action="store_true",
                        help="Do not persist this game to the database.")
    parser.add_argument("--notes", type=str, default=None,
                        help="Optional notes to store with the game record.")
    args = parser.parse_args()

    color_on = False if args.no_color else sys.stdout.isatty()

    human_player: Optional[int]
    if args.selfplay:
        human_player = None
    elif args.player is not None:
        human_player = args.player - 1
    else:
        human_player = ask_side(color_on)

    if human_player is not None:
        side = "P1 (Red, first)" if human_player == 0 else "P2 (Blue, second)"
        print(f"You are {side}.")

    board = Board.initial()
    # For self-play rendering, show from P2 orientation (arbitrary default).
    render_side = 1 if human_player is None else human_player
    color_kw = False if args.no_color else None

    # Set up recorder unless disabled.
    recorder: Optional[GameRecorder] = None
    if not args.no_record:
        def _source(side: int) -> str:
            if human_player is None:
                return "alphabeta"
            return "human" if side == human_player else "alphabeta"

        def _time_limit(side: int) -> Optional[float]:
            if human_player is None:
                return args.time
            return None if side == human_player else args.time

        recorder = GameRecorder(
            p1_source=_source(0),
            p2_source=_source(1),
            p1_time_limit=_time_limit(0),
            p2_time_limit=_time_limit(1),
            model_version="alphabeta-v2",
            notes=args.notes,
        )
        recorder.start()

    print()
    print(render_board(board, render_side, color_kw))

    try:
        while board.winner() is None:
            if human_player is not None and board.turn == human_player:
                move = human_move(board)
            else:
                move = ai_move(board, args.depth, args.time)
            if move is None:
                # User quit or AI had no legal moves.
                break
            if recorder is not None:
                recorder.record(move)
            board = board.apply(move)
            print()
            print(render_board(board, render_side, color_kw))
    finally:
        if recorder is not None:
            gid = recorder.finish(winner=board.winner())
            if gid is not None:
                print(f"(game saved as id={gid})")

    winner = board.winner()
    if winner is None:
        print()
        print("Game ended without a winner.")
        return
    label = "P1 (Red)" if winner == 0 else "P2 (Blue)"
    print()
    print(f"*** {label} wins! ***")


if __name__ == "__main__":
    main()
