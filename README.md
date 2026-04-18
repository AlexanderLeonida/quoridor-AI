# quoridor-AI

A Quoridor engine and AI in pure Python (no dependencies).

## Layout

```
quoridor/
  board.py      rules, state, move generation, BFS, rendering
  ai.py         alpha-beta minimax, iterative deepening, wall pruning
play.py         terminal play (human-vs-AI, AI-vs-AI)
test_quoridor.py
```

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
- The display is flipped so that you (the human) are always at the bottom of the screen.

**Move notation at the prompt:**

| Input   | Meaning                                                    |
|---------|------------------------------------------------------------|
| `e2`    | move your pawn to e2                                       |
| `e5h`   | place a horizontal wall anchored at e5 (rows 1-8, cols a-h)|
| `e5v`   | place a vertical wall anchored at e5                       |
| `moves` | list all legal moves in this notation                      |
| `q`     | quit                                                       |

## How the AI works

- **Search**: alpha-beta minimax with iterative deepening under a time budget.
- **Evaluation**: `(opp_path − my_path) · 100 + wall_diff · 3 + advance_diff`, where paths are BFS shortest-path lengths to each player's goal row. `+∞/−∞` for terminal states.
- **Pruning**: walls are only considered at anchors adjacent to cells on either player's current shortest path — this keeps the effective branching factor well under 20 in most positions.
- **Legality check for walls**: placing a wall requires that both players still have a path to their goal (BFS after the tentative placement).

## Tests

```sh
python3 test_quoridor.py
```