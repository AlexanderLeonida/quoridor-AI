# Quoridor AI — Analysis Report

Generated: 2026-04-25T07:49:31

---


## 00_summary

```
7 iterations parsed
Promoted: 0    Rejected: 6    Reverts: 0
Self-play: 600 games, 11 drawn (2%)
Eval: 180 games, 5 drawn (3%)

iter  global  sp(W1/W2/D)     epoch1 train  epoch1 val   eval  CI                     WLD  result
   1      40  54/44/2                1.844       1.770    20%  [10%,37%]           6/24/0  kept
   2      41  49/50/1                1.855       1.805     7%  [2%,21%]            2/28/0  kept
   3      42  42/56/2                1.860       1.847    17%  [7%,34%]            4/24/2  kept
   4      43  55/42/3                1.854       1.991    28%  [15%,46%]           8/21/1  kept
   5      44  57/42/1                1.854       1.943    25%  [13%,43%]           7/22/1  kept
   6      45  50/48/2                1.857       1.764    25%  [13%,43%]           7/22/1  kept
   7      46  0/0/0                      -           -      -  -                        -
```

## 01_elo_history

```
FAILED: Command '['/opt/homebrew/opt/python@3.14/bin/python3.14', 'analysis/01_elo_history.py']' returned non-zero exit status 1.
Traceback (most recent call last):
  File "/Users/aleon1/Desktop/quoridor-AI/analysis/01_elo_history.py", line 81, in <module>
    main()
    ~~~~^^
  File "/Users/aleon1/Desktop/quoridor-AI/analysis/01_elo_history.py", line 76, in main
    arch = "6×64" if n <= arch_split else "10×128"
                          ^^^^^^^^^^
NameError: name 'arch_split' is not defined

```

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

v3: 4518 games, 12% draws, avg 53.7 plies, versions v1-v55
```

![03_database_stats](analysis/plots/03_database_stats.png)


## 04_training_progress

```
Saved analysis/plots/04_training_progress.png

7 iterations parsed, 0 promoted, 0 reverts
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
  v3: 4,520
  Active span: 2026-04-18 → 2026-04-25
```

![06_activity_timeline](analysis/plots/06_activity_timeline.png)


## 07_intervention_metrics

```
No metrics.csv yet. Run selfplay.py at least once to populate it.
```