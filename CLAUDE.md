# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

```sh
# environment
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# tests (no pytest — each file is runnable directly)
python3 test_quoridor.py        # engine + alpha-beta unit checks
python3 test_ml.py              # encoding round-trips + DB round-trip
python3 -c "from test_ml import test_encode_round_trip; test_encode_round_trip()"   # single test

# play
python3 gui.py                  # Tkinter GUI (loads checkpoints/best.pt by default)
python3 play.py [--player 1|2] [--selfplay] [--depth N] [--time S] [--no-record]

# supervised training & self-play loop
python3 train.py --epochs 10 --out checkpoints/v1.pt
python3 selfplay.py --iterations 12 --games-per-iter 100 --simulations 1000 \
    --resume checkpoints/best.pt --db data/quoridor_v3.db --checkpoint-dir checkpoints

# benches & tournaments (Bradley-Terry MLE)
python3 bench.py --ckpt checkpoints/best.pt --vs-other checkpoints/iter_0074.pt --n-games 30 --sims 200
python3 tournament.py --ckpt checkpoints/best.pt:current --ckpt checkpoints/iter_0074.pt:v74 \
    --games 10 --sims 200 --workers 4 --out checkpoints/elo_tournament.json
python3 bench_matrix.py --ckpt ... --ab "d3,t1.0" --ab "d5,t4.0"

# distillation variants
python3 distill.py            --teacher ckpt --student ckpt --out ...    # net→net (arch change)
python3 distill_deep.py       --teacher {ab|mcts} --student ckpt --out ... [--ab-depth N]
python3 widen_distill.py      --student-blocks 14 --student-filters 192 --teacher-net ...
python3 train_from_npz.py     --in ckpt --out ckpt --npz data/human_training_set.npz

# end-to-end shell pipelines (recommended for long runs — they pin the right flags)
./run.sh                      # 12-iter self-play loop with anti-drift stack
./run_r6.sh                   # depth-10 AB distillation round
./post_r6_pipeline.sh         # gate→promote→targeted-train→gate again
```

All training scripts auto-select device (CUDA → MPS → CPU); override with `--device`. Self-play workers use multiprocessing — set `--workers N` (the loop snapshots weights to `checkpoints/_worker_net.pt` and pins each worker to 1 torch thread).

## Architecture

Two layers — keep them straight when editing:

1. **`quoridor/` package** is the core library: pure-Python engine and the encoding/network contract. No CLI logic here, no training orchestration.
2. **Top-level `*.py` scripts** are the experimentation surface: training pipelines, distillation variants, benches, diagnostics. They import from `quoridor` and from each other freely. Treat each one as a self-contained experiment driver.

### Critical contract: `quoridor/encoding.py`

This file is the single source of truth for **state tensors and action indices**. Touching it without round-trip tests will silently corrupt every checkpoint. Invariants:

- Input tensor shape `(7, 9, 9)`: my pawn, opp pawn, h-walls, v-walls, my walls-left/10, opp walls-left/10, ones-bias.
- Action space size `209`: `[0,81)` pawn cells, `[81,145)` h-wall anchors, `[145,209)` v-wall anchors.
- **Canonical view**: the net always sees the side-to-move as "player 0 starting near row 0". When the real turn is P2, the board is row-flipped before encoding and `flipped=True` is threaded through `move_to_action` / `action_to_move` so policy targets stay in the canonical frame.
- DB stores **move lists only**, never tensors — `GameDB.iter_training_samples()` re-materialises positions by replaying from `Board.initial()`. This decouples on-disk format from encoding evolution; do not break this.

### Two AI stacks, used together

- **Classical search** (`quoridor/ai.py`): iterative-deepening negamax + PVS, Zobrist TT (EXACT/LOWER/UPPER), killer moves, BFS-anchored wall pruning. Entry point: `find_best_move(board, depth, time_budget)`. Used as evaluator, teacher (depth-8/10), and reference opponent.
- **AlphaZero stack** (`quoridor/mcts.py` + `quoridor/net.py`): PUCT MCTS with Dirichlet root noise, FPU reduction, MuZero log-scaled `c_puct`, subtree reuse, 20k-entry eval cache. Net is a residual conv tower (default 10×128) with policy+value heads. `MCTSConfig` controls all knobs; `search()` runs simulations and `select_action()` reads visit counts.

### Self-play pipeline (`selfplay.py`)

Orchestrates: generate games (MCTS-guided, with opening randomisation + AB-mix games to break drift) → train on a sliding window with CE+MSE+L2 → gate vs current best at win-rate threshold → promote and update Elo. The "anti-drift stack" referenced in `run.sh` includes: AB-mix injection (`--ab-mix-frac`), hard-example mining, periodic PBT mutations (`--pbt-mutate-every`), tournament checkpoints with held-out anchors (`--tournament-anchors`), draw penalty + path-aware draw values, column-flip data augmentation, and adjudication of stalled games by shortest-path gap (`--adjudicate-gap`).

### Distillation patterns

Three flavours, each solving a different problem:

- `distill.py` — **architecture swap** (e.g., transfer 6×64 → 10×128 by matching teacher's policy/value on sampled DB positions).
- `distill_deep.py` — **deeper search teacher** (`--teacher ab` at depth 8–10, or `--teacher mcts` at extreme sim counts); brings in supervision the training-time MCTS can't produce on its own.
- `widen_distill.py` — **capacity expansion** (10×128 → 14×192 etc.); combines the existing net's outputs with optional fresh AB targets, with rehearsal + KL anchor to mitigate catastrophic forgetting.
- `train_from_npz.py` / `train_human_walls.py` — **targeted training** from a curated `.npz` of (state, oracle policy, value) triples, with rehearsal frac + KL regulariser λ to preserve prior capability.

Catastrophic-forgetting mitigation is consistent across all of these: rehearsal samples from the self-play DB at ~30–60% + a `λ·KL(student ‖ pre_distill_student)` regulariser (λ≈0.5–1.0).

### Checkpoint conventions

- `checkpoints/best.pt` is the live champion; pipelines back it up before any swap (`pre_*_backup.pt`).
- `iter_NNNN.pt` are per-iteration snapshots from `selfplay.py`.
- `_worker_net.pt`, `_eval_*.pt`, `_tourney_*.json` are scratch files written by the loop and the tournament tool — safe to delete between runs.
- Checkpoints carry a `config` dict (`blocks`, `filters`, `in_planes`, `action_space`) plus a `meta` dict; `load_checkpoint` reads `config` to rebuild the right architecture.

### Database

Default DB is `data/quoridor.db` (legacy) or `data/quoridor_v3.db` (current self-play). Schema in `quoridor/database.py`:
- `games`: winner, plies, p1/p2 source, model_version, notes
- `moves`: side, kind, r, c, elapsed_ms, optional `policy_blob` (serialised MCTS visit distribution)

`p1_source` / `p2_source` distinguish `human`, `nn-mcts`, `ab`, etc. — used by `analyze_human_wins.py` and `build_human_training_set.py` to mine adversarial positions.

## Reference docs in this repo

- **`PROCESS.md`** — chronological narrative of every methodology change, numbered by section (`§N`). Shell scripts cite section numbers in comments (e.g., `§38 two-round rule`). When a script's behavior is unclear, look up the cited section.
- **`Algorithms.md`** — pseudocode + exact file/line references for MCTS, PUCT, gating, distillation losses, Bradley-Terry MLE.
- **`README.md`** — user-facing quick start; less detail than PROCESS.md.
- **`analysis/`** — plotting scripts and a generated `REPORT.md` summarising experiments.

## When making changes

- **Encoding/action-space changes** require updating `test_ml.py` round-trip tests and re-running them before anything else. A silent off-by-one here invalidates every checkpoint.
- **DB schema changes** must keep replay-from-move-list working; see `GameDB.iter_training_samples`.
- **New training scripts** should follow the rehearsal + KL-anchor pattern from `distill_deep.py` / `train_from_npz.py` if they fine-tune `best.pt`, otherwise they will catastrophically forget.
- **Gating decisions** are statistical — keep Wilson CIs and per-net W/L/D breakdowns visible; headline win-rates hide draw drift.
