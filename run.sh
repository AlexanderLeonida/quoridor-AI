#!/usr/bin/env bash
# Max-out the current 10x128 net before scaling architecture.
# Resumes from best.pt (r5 distilled) and runs 12 iterations of self-play
# with high sims (1000), bigger training window (5000), 2 epochs.
# Anti-drift stack from PROCESS.md §34 stays on (ab-mix, hard-example mining,
# PBT every 6, tournament every 4 with held-out anchors + distillation backups).
#
# Adjustments after first attempt failed iter 1+2 gating (§36 ceiling pattern):
#   --epochs 3 -> 2          (val rose after epoch 1 in both rejected runs)
#   --train-from-best-version (per §23: filter weaker rejected-candidate self-play)
#   --value-weight 1.0 -> 0.3 (third attempt: protect the depth-8-AB-trained
#       value head from being smashed by blunt outcome targets — selfplay.py
#       :2004-2008 explicitly prescribes 0.1–0.3 for post-distillation nets)

set -eu
cd "$(dirname "$0")"

mkdir -p logs checkpoints

LOG="logs/train.log"

python3 -u selfplay.py \
  --iterations 12 --games-per-iter 100 --simulations 1000 \
  --eval-games 30 --epochs 2 --window 5000 --workers 8 \
  --lr 3e-4 --weight-decay 5e-4 \
  --max-moves 100 --opening-random 8 --temp-threshold 15 \
  --policy-temp 0.7 --value-weight 0.3 --draw-penalty 0.5 \
  --gate-threshold 0.52 --eval-temp 0.5 --eval-temp-moves 10 \
  --adjudicate-gap 1 \
  --ab-mix-frac 0.2 --ab-depth 4 --ab-time 1.5 \
  --aux-value-weight 0.4 \
  --train-from-best-version \
  --hard-example-mining --pbt-mutate-every 6 \
  --tournament-every 4 --tournament-games 8 --tournament-sims 200 \
  --tournament-pool-size 5 \
  --tournament-anchors checkpoints/warmstart_10x128.pt \
  --tournament-anchors checkpoints/iter_0034.pt \
  --tournament-anchors checkpoints/best_ab_distilled_r2.pt \
  --tournament-anchors checkpoints/pre_ab_distill_backup.pt \
  --db data/quoridor_v3.db --checkpoint-dir checkpoints \
  --resume checkpoints/best.pt \
  2>&1 | tee -a "$LOG"
