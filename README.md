# quoridor-AI

<p align="center">
  <a href="https://github.com/AlexanderLeonida/quoridor-AI/raw/main/media/quoridor-demo.mp4">
    <video
      src="https://github.com/AlexanderLeonida/quoridor-AI/raw/main/media/quoridor-demo.mp4"
      poster="https://github.com/AlexanderLeonida/quoridor-AI/raw/main/media/quoridor-demo-poster.jpg"
      controls
      muted
      width="720">
    </video>
  </a>
</p>

> **▶ Demo** — if the inline player above doesn't load on your viewer, click the image to open the
> [60-second walkthrough](https://github.com/AlexanderLeonida/quoridor-AI/raw/main/media/quoridor-demo.mp4)
> showing the live web showcase (self-play, before/after RL, round-robin tournament, training dashboard).

Quoridor engine, search agents and AlphaZero-style training pipeline
implemented in Python. The codebase contains:

- A classic search engine (negamax/PVS) with practical Quoridor pruning
- An AlphaZero-style MCTS + PyTorch policy/value network for self-play
- Utilities for logging games to a SQLite DB, supervised training, and
  a small Tkinter GUI + a terminal play mode.

This repository is intended for experimentation and research rather than
production deployment.

## Quick start

1. Create a virtual environment and install dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- The core engine and DB code have no mandatory heavy dependencies.
  `torch` is optional for using the neural network and training scripts;
  it is installed via `requirements.txt` if you intend to train or run
  neural-network-guided self-play.

## Layout (key files)

```
quoridor/             # core library: engine, search, MCTS, encoding, DB, net
training/             # selfplay.py, train.py, train_from_npz.py, train_human_walls.py, verify_human_walls.py
distillation/         # distill.py, distill_deep.py, widen_distill.py
eval/                 # bench.py, bench_matrix.py, tournament.py
diagnostics/          # analyze_*.py, diagnose_*.py, inspect_*.py
data_pipeline/        # build_human_training_set.py, consolidate_human_games.py
tests/                # test_quoridor.py, test_ml.py
scripts/              # run.sh, run_r6.sh, post_r6_pipeline.sh, rungame.sh
gui.py                # small Tkinter GUI for human play
play.py               # terminal-based play (human vs AI, AI vs AI)
checkpoints/          # trained model checkpoints and evaluation snapshots
data/                 # game DB (SQLite) and generated artifacts
```

## Running

- GUI:

```sh
python3 gui.py
```

- Terminal play:

```sh
python3 play.py                # interactive prompt
python3 play.py --player 1     # play as Player 1 (Red)
python3 play.py --player 2     # play as Player 2 (Blue)
python3 play.py --selfplay     # AI vs AI
python3 play.py --depth 4 --time 8
python3 play.py --no-color     # disable ANSI colors
```

Move notation examples (at the `play.py` prompt):

- `e2`    — move pawn to e2
- `e5h`   — place a horizontal wall anchored at e5
- `e5v`   — place a vertical wall anchored at e5
- `moves` — list legal moves
- `q`     — quit

Board coordinates: columns `a`–`i`, rows `1`–`9`. Player 1 (Red)
starts at `e1` and aims for row 9; Player 2 (Blue) starts at `e9`.

## Training and self-play

- Supervised training from the games DB:

```sh
python3 training/train.py --epochs 10 --batch-size 256 --lr 1e-3 \
                          --out checkpoints/v1.pt
```

- Full AlphaZero-style self-play loop:

```sh
python3 training/selfplay.py --iterations 100 --games-per-iter 50 \
                             --simulations 400 --checkpoint-dir checkpoints/
```

See `training/selfplay.py` for available CLI flags. Training scripts
auto-select the best available device (CUDA → MPS → CPU); override with
`--device` if needed.

For long pinned-flag runs, prefer the wrappers in `scripts/`:

```sh
./scripts/run.sh                 # 12-iter self-play with the anti-drift stack
./scripts/run_r6.sh              # depth-10 AB deep distillation
./scripts/post_r6_pipeline.sh    # gate → promote → targeted-train → gate
```

## Database

Games are logged to a SQLite DB (default `data/quoridor.db`) when using
`play.py` or `gui.py`. To disable recording for a single play session,
pass `--no-record` to `play.py`.

Schema highlights (see `quoridor/database.py`):

- `games`: metadata (winner, timestamps, number of plies, sources)
- `moves`: per-ply move records and optional `policy_blob` (MCTS targets)

The DB stores move lists only; states are re-materialized by replaying
moves on demand which keeps the DB compact and stable across encoder
changes.

Example usage:

```python
from quoridor import GameDB
with GameDB() as db:
    print(db.count_games(), "games")
    for board, move, z in db.iter_training_samples():
        pass
```

## How the AI works (overview)

- Search engine: iterative-deepening negamax with Principal Variation
  Search (PVS), transposition table (Zobrist hashing), killer moves and
  practical wall pruning to keep branching manageable.
- AlphaZero stack: MCTS (PUCT) with Dirichlet root noise, a residual
  conv policy/value network (`quoridor/net.py`), and a supervised +
  self-play training loop.

See in-file docstrings and the modules in `quoridor/` for implementation
details (e.g., `quoridor/ai.py`, `quoridor/mcts.py`, `quoridor/net.py`).

## Tests

Run the small test suites:

```sh
python3 tests/test_quoridor.py   # engine + alpha-beta unit checks
# tests/test_ml.py contains pytest-style functions; invoke with pytest
# or import-and-call individual tests.
```

## Checkpoints

Model checkpoints and evaluation snapshots live in the `checkpoints/`
directory. These are used by `selfplay.py` and evaluation scripts.

## Contributing

Contributions are welcome. Typical contributions include bug fixes,
improvements to search heuristics, and experiments with network
architectures or training regimes. If you change the DB schema, include
backwards-compatible migration code or notes.

## License

This repository does not include an explicit license file. Add one if you
intend to publish or share the project publicly.
