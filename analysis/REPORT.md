# Quoridor AI — Analysis Report

Generated: 2026-04-25T12:56:46

---


## 00_summary

```
8 iterations parsed
Promoted: 2    Rejected: 5    Reverts: 1
Self-play: 700 games, 8 drawn (1%)
Eval: 210 games, 3 drawn (1%)

iter  global  sp(W1/W2/D)     epoch1 train  epoch1 val   eval  CI                     WLD  result
   1       1  60/39/1                1.658       1.522    30%  [17%,48%]           9/21/0  kept
   2       2  56/44/0                1.686       1.646    40%  [25%,58%]          12/18/0  kept
   3       3  50/50/0                1.713       1.670    37%  [22%,55%]          11/19/0  kept
   4       4  58/42/0                1.740       1.635    47%  [30%,64%]          14/16/0  kept
   5       5  54/46/0                1.767       1.625    57%  [39%,73%]          17/13/0  REVERTED→iter_0034
   6       6  52/43/5                1.715       1.639    57%  [39%,73%]          17/13/0  PROMOTED
   7       7  44/54/2                1.586       1.441    32%  [18%,50%]           8/19/3  kept
   8       8  0/0/0                      -           -      -  -                        -
```

## 01_elo_history

```
Saved analysis/plots/01_elo_history.png

78 versions plotted (9 calibrated, 69 gating-only)
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

v3: 5350 games, 10% draws, avg 50.9 plies, versions v1-v55
```

![03_database_stats](analysis/plots/03_database_stats.png)


## 04_training_progress

```
Saved analysis/plots/04_training_progress.png

8 iterations parsed, 2 promoted, 1 reverts
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
  v3: 5,357
  Active span: 2026-04-18 → 2026-04-25
```

![06_activity_timeline](analysis/plots/06_activity_timeline.png)


## 07_intervention_metrics

```
Saved analysis/plots/07_intervention_metrics.png

8 metrics rows across iterations 1–7
```

![07_intervention_metrics](analysis/plots/07_intervention_metrics.png)


## 08_nn_vs_ab

```
No analysis/bench_matrix.json; run bench_matrix.py first.
```