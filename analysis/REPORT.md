# Quoridor AI — Analysis Report

Generated: 2026-04-25T21:32:41

## Current training recipe

Each iteration runs (1) self-play, (2) training, (3) gating, (4) periodic round-robin tournament. Six knobs do most of the work. **High-sim MCTS** (sims=600) gives the net a teacher stronger than itself. **Alpha-beta mix** (20% of self-play games) breaks the self-imitation loop with a fundamentally different evaluator. **Auxiliary path-diff value blend** (α=0.4) densifies value supervision from one outcome label per game to ~30 per game. **Hard-example mining** on revert concentrates supervision on positions where the net previously made the wrong call. **Lightweight PBT** (sibling every 6 iters) explores hparam neighborhoods cheaply. **Round-robin tournament every 5 iters** with held-out anchors (warmstart, iter_0034, iter_0040) catches drift the local gating signal can't see, with revert-to-champion semantics.

See `PROCESS.md` §34 for the full launch command and a per-knob explanation of why each one helps.

---


## 00_summary

```
5 iterations parsed
Promoted: 3    Rejected: 2    Reverts: 1
Self-play: 500 games, 6 drawn (1%)
Eval: 150 games, 4 drawn (3%)

iter  global  sp(W1/W2/D)     epoch1 train  epoch1 val   eval  CI                     WLD  result
   1      12  48/51/1                1.663       1.545    57%  [39%,73%]          17/13/0  PROMOTED
   2      13  49/49/2                1.609       1.456    52%  [35%,68%]          15/14/1  kept
   3      14  52/48/0                1.640       1.470    57%  [39%,73%]          16/12/2  PROMOTED
   4      15  57/41/2                1.603       1.432    63%  [46%,78%]          19/11/0  PROMOTED
   5      16  48/51/1                1.592       1.373    45%  [29%,62%]          13/16/1  REVERTED→selfplay-v11
```

## 01_elo_history

```
Saved analysis/plots/01_elo_history.png

82 versions plotted (12 calibrated, 70 gating-only)
```

![01_elo_history](analysis/plots/01_elo_history.png)


## 02_calibrated_elo

```
Saved analysis/plots/02_calibrated_elo.png

Tournament players (8):
   1. warmstart                     1000.0
   2. v42_run3                       945.2
   3. v48_run3peak                   922.5
   4. v52_prev_best                  907.6
   5. v34_current                    877.7
   6. v40_run3start                  855.3
   7. v19_distill                    817.1
   8. v74_6x64peak                   817.1
```

![02_calibrated_elo](analysis/plots/02_calibrated_elo.png)


## 03_database_stats

```
Saved analysis/plots/03_database_stats.png

v1: 133 games, 50% draws, avg 111.9 plies, versions v1-v2

v2: 13995 games, 38% draws, avg 56.3 plies, versions v1-v83

v3: 6280 games, 9% draws, avg 48.5 plies, versions v1-v55
```

![03_database_stats](analysis/plots/03_database_stats.png)


## 04_training_progress

```
Saved analysis/plots/04_training_progress.png

5 iterations parsed, 3 promoted, 1 reverts
```

![04_training_progress](analysis/plots/04_training_progress.png)


## 05_architecture_comparison

```
Saved analysis/plots/05_architecture_comparison.png

Architectures observed:
  10×128: 4.40M params, 17.6 MB, 54 iterations (v1–v54)
  6×64 : 1.08M params, 4.3 MB, 28 iterations (v55–v82)
```

![05_architecture_comparison](analysis/plots/05_architecture_comparison.png)


## 06_activity_timeline

```
Saved analysis/plots/06_activity_timeline.png

Games per DB:
  v1: 133
  v2: 13,995
  v3: 6,307
  Active span: 2026-04-18 → 2026-04-26
```

![06_activity_timeline](analysis/plots/06_activity_timeline.png)


## 07_intervention_metrics

```
Saved analysis/plots/07_intervention_metrics.png

18 metrics rows across iterations 1–16
```

![07_intervention_metrics](analysis/plots/07_intervention_metrics.png)


## 08_nn_vs_ab

```
No analysis/bench_matrix.json; run bench_matrix.py first.
```