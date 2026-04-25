# Analysis suite

Seven analyses that visualise different facets of the project's training journey. Run individually or all together via `make_report.py`.

## Layout

- `parser.py` — shared library: `parse_log` + `summary_text` (no CLI)
- `00_summary.py` — text table per iteration (replaces the old top-level `analyze.py`)
- `01_..06_*.py` — six plot scripts; each produces a PNG under `plots/`
- `make_report.py` — runs all of them; aggregates stdout + plots into `REPORT.md`

## Scripts

| File | What it shows | Data source |
|---|---|---|
| `00_summary.py` | Tabular text summary per iteration | `logs/train.log` |
| `01_elo_history.py` | Per-iteration gating Elo across all training runs | `checkpoints/elo.json` |
| `02_calibrated_elo.py` | Tournament-calibrated Elos with bootstrap CIs (cross-architecture comparable) | `checkpoints/elo_tournament.json` |
| `03_database_stats.py` | Outcome distribution + game length per model version, across all DBs | `data/quoridor*.db` |
| `04_training_progress.py` | Loss curves, eval scores, draw rates, promotions/reverts | `logs/train.log`, `logs/archive/*.log` |
| `05_architecture_comparison.py` | 6×64 vs 10×128 parameter count and iterations covered | `checkpoints/iter_*.pt` |
| `06_activity_timeline.py` | Self-play games per day stacked by DB | `data/quoridor*.db` |

## Run

```sh
# All at once → analysis/REPORT.md + plots in analysis/plots/
python3 analysis/make_report.py

# Or individually
python3 analysis/04_training_progress.py
```

## Forward-looking data capture

`selfplay.py` now appends one row per iteration to `logs/metrics.csv` with: timestamp, global_iter, version, self-play W/L/D + avg plies, train/policy/value/best_val losses, eval score + W/L/D, promoted flag, reverted_to. This file is **never overwritten**, so even if `train.log` rotates, the per-iteration history persists across runs. Analysis scripts can read this for reliable historical metrics.

## Caveat on `01_elo_history.py`

Iteration numbers were *reused* across architectures. After distillation, the 10×128 net resumed from `iter_0074.pt` (6×64 best) and `global_iter` continued — so e.g. `selfplay-v34` in `elo.json` could correspond to either architecture depending on when it was rated. For unambiguous strength comparison, see `02_calibrated_elo.py` (run `tournament.py` to refresh).
