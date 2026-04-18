# quoridor-AI

A Quoridor engine and AI in pure Python (no dependencies).

## Layout

```
quoridor/
  board.py      rules, state, move generation, BFS
  ai.py         negamax + PVS, TT, killers, wall pruning
  encoding.py   state/move/action tensors for the neural net
  database.py   SQLite store of played games (for training data)
  net.py        PyTorch policy+value network (AlphaZero-style)
  recorder.py   convenience wrapper that logs a game into the DB
  mcts.py       AlphaZero MCTS (PUCT, Dirichlet noise, FPU)
gui.py          Tkinter GUI (mouse play, hover preview)
play.py         terminal play (human-vs-AI, AI-vs-AI)
train.py        supervised trainer over the games DB
selfplay.py     AlphaZero self-play pipeline (generate → train → gate)
test_quoridor.py
test_ml.py
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

## Learning from past games

Every game played via `play.py` or `gui.py` is automatically logged to a
SQLite database at `data/quoridor.db`. Pass `--no-record` to `play.py` to
disable logging for a single session.

### Database

Schema (see `quoridor/database.py`):

- `games(id, created_at, finished_at, winner, num_plies, p1_source, p2_source, p1_time_limit, p2_time_limit, model_version, notes)`
- `moves(id, game_id, ply, side, move_kind, move_r, move_c, elapsed_ms, policy_blob)`

Only move lists are stored (not tensors). States are re-materialized on
demand by replaying from the initial position, which keeps the DB ~40×
smaller and robust to encoding changes.

```python
from quoridor import GameDB
with GameDB() as db:
    print(db.count_games(), "games,", db.count_positions(), "positions")
    for board, move, z in db.iter_training_samples():
        ...   # z is +1/-1 from the side-to-move's POV
```

### Neural network

`quoridor/net.py` defines an AlphaZero-style residual conv net:

- **Input**: `(7, 9, 9)` canonical tensor — side-to-move is always P0.
  Planes: my pawn, opp pawn, h-walls, v-walls, my walls-left / 10,
  opp walls-left / 10, all-ones bias.
- **Trunk**: 3×3 conv stem → N residual blocks (default `blocks=6`, `filters=64`).
- **Policy head**: 1×1 conv → FC → 209 logits over the flat action space
  (81 pawn cells + 64 H-walls + 64 V-walls).
- **Value head**: 1×1 conv → FC(64) → FC(1) → `tanh`.

PyTorch is imported lazily, so the engine/DB/encoding still work on a
machine without `torch` installed.

### Training (supervised / behaviour cloning)

```sh
pip install -r requirements.txt
python3 train.py --epochs 10 --batch-size 256 --lr 1e-3 \
                 --out checkpoints/v1.pt
# resume from a previous checkpoint
python3 train.py --resume checkpoints/v1.pt --epochs 5
```

Training minimises `CE(policy, action_taken) + MSE(value, z)`. It picks
the best available device automatically (CUDA → MPS → CPU); override with
`--device`.

### Self-play training (AlphaZero loop)

`selfplay.py` implements the full AlphaZero training pipeline:

1. **Self-play** — generate games using MCTS guided by the current
   neural network.  MCTS visit-count distributions are stored as soft
   policy targets in the DB's `policy_blob` column.
2. **Training** — update the network on recent games with soft
   cross-entropy (policy) + MSE (value) + L2 regularisation, gradient
   clipping, and cosine LR annealing.
3. **Evaluation / gating** — pit the new network against the current
   best via greedy MCTS.  Promote only when win-rate > 55 %.
4. **Repeat.**

```sh
# Start from scratch (random network):
python3 selfplay.py --iterations 100 --games-per-iter 50 \
                    --simulations 400 --checkpoint-dir checkpoints/

# Resume from a checkpoint:
python3 selfplay.py --resume checkpoints/best.pt --iterations 50

# Quick smoke-test:
python3 selfplay.py --iterations 2 --games-per-iter 4 \
                    --simulations 50 --eval-games 4 --epochs 2
```

**MCTS details** (`quoridor/mcts.py`):

- PUCT exploration with log-scaling c\_puct (MuZero formula)
- Dirichlet noise at root (α = 0.3, ε = 0.25)
- First Play Urgency (FPU) reduction for unvisited children
- Temperature schedule: proportional sampling for the first 15 full
  moves, then greedy
- Correct negamax value propagation for the two-player zero-sum game

## Tests

```sh
python3 test_quoridor.py   # engine + alpha-beta
python3 test_ml.py         # encoding round-trips + DB round-trip
```