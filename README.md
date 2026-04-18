# quoridor-AI

A Quoridor engine and AI in pure Python (no dependencies).

## Layout

```
quoridor/
  board.py      rules, state, move generation, BFS
  ai.py         alpha-beta minimax, iterative deepening, wall pruning
gui.py          Tkinter GUI (mouse play, hover preview)
play.py         terminal play (human-vs-AI, AI-vs-AI)
test_quoridor.py
```

## GUI

```sh
python3 gui.py
```

- A "Choose side" dialog appears first: click **Player 1 (Red)** or **Player 2 (Blue)**.
- The board is oriented so that you are always at the top and the AI at the bottom.
- **Move:** click one of the highlighted cells. Legal targets are marked with a small blue dot.
- **Wall:** hover over the gap between two cells — horizontal gaps preview a horizontal wall, vertical gaps preview a vertical wall. Click to place. Previews are orange when legal, red when not.
- The **Difficulty** dropdown controls AI search depth and time budget (Easy / Medium / Hard). **New Game** reopens the side dialog.

## Play

```sh
python3 play.py                # prompts for P1 or P2
python3 play.py --player 1     # skip prompt, play as P1 (Red)
python3 play.py --player 2     # play as P2 (Blue)
python3 play.py --selfplay     # AI vs AI
python3 play.py --depth 4 --time 8
python3 play.py --no-color     # disable ANSI colors
```

**Board.** Columns are `a`-`i` (left to right); rows are `1`-`9`.

- Player 1 (**Red**, moves first) starts at `e1` and must reach row 9.
- Player 2 (**Blue**, moves second) starts at `e9` and must reach row 1.
- The display is oriented so that you (the human) are always at the top of the screen and the AI at the bottom.

**Move notation at the prompt:**

| Input   | Meaning                                                    |
|---------|------------------------------------------------------------|
| `e2`    | move your pawn to e2                                       |
| `e5h`   | place a horizontal wall anchored at e5 (rows 1-8, cols a-h)|
| `e5v`   | place a vertical wall anchored at e5                       |
| `moves` | list all legal moves in this notation                      |
| `q`     | quit                                                       |

## How the AI works

- **Search**: iterative-deepening **negamax with Principal Variation Search (PVS)**. Hard difficulty is given a 30 s per-move budget and a depth ceiling of 20 — iterative deepening stops at whatever depth it completes before the clock expires.
- **Transposition table**: 64-bit **Zobrist hashing** with EXACT / LOWER / UPPER bounds. The TT's best-move is used as the first move at every node (dominant ordering heuristic).
- **Killer moves**: two slots per ply for moves that produced recent beta-cutoffs.
- **Move ordering**: TT move → killers → pawn moves toward goal → walls ordered by **disruption** (how much the wall lengthens opponent's shortest path minus how much it lengthens ours).
- **Wall pruning**: only anchors adjacent to a cell on either player's current shortest path are considered. This is the standard practical pruning for Quoridor and keeps the branching factor tractable.
- **Evaluation (from the side-to-move's perspective)**:
  `(opp_path − my_path) · 100 + wall_diff · 6 + mobility_diff · 2 + advance_diff + tempo`.
- **Terminal scores** decay with ply (`WIN_SCORE − ply`) so the engine prefers shorter wins and longer losses.

### Difficulty levels (gui.py)

| Level  | Time budget | Depth ceiling |
|--------|-------------|---------------|
| Easy   | 2 s         | 3             |
| Medium | 8 s         | 20 (time-bound) |
| Hard   | 30 s        | 20 (time-bound) |

## Tests

```sh
python3 test_quoridor.py
```