#!/usr/bin/env bash
# Round 6 deep distill — depth-10 AB, 8000 positions.
# Step 1 of §44 plan: validate whether AB ceiling has been hit.
# r5 was depth-8 / 4000 positions / +61 Elo over r3.
# If r6 yields <100 Elo gain over r5, §38 two-round rule triggers → widen.
#
# Backup at checkpoints/pre_r6_backup.pt (= r5 weights) for rollback.

set -eu
cd "$(dirname "$0")"

mkdir -p logs checkpoints

LOG="logs/distill_deep_r6.log"

python3 -u distill_deep.py \
  --teacher ab \
  --student checkpoints/best.pt \
  --out checkpoints/best_ab_distilled_r6.pt \
  --db data/quoridor_v3.db \
  --positions 8000 \
  --workers 6 \
  --ab-depth 10 --ab-time 10 \
  --epochs 4 --batch-size 256 --lr 1e-3 \
  --rehearsal-frac 0.3 --reg-lambda 0.5 \
  --seed 0 \
  2>&1 | tee "$LOG"
