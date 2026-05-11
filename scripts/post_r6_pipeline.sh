#!/usr/bin/env bash
# Auto-pipeline that runs after r6 completes.
#
# Steps:
#   1. Wait for checkpoints/best_ab_distilled_r6.pt to appear
#   2. Bench r6 vs current best.pt (human_v2) at 200 sims, 8 games
#   3. If r6 wins (>52%): promote r6 to best.pt, save backup
#      Otherwise: keep current best.pt
#   4. Run targeted training (train_from_npz.py) on the winner
#   5. Bench the result vs the pre-training version
#   6. Promote the targeted-trained version if it wins
#
# Backups always taken before any swap.

set -eu
cd "$(dirname "$0")/.."
mkdir -p logs checkpoints

LOG="logs/post_r6_pipeline.log"

echo "=== post-r6 pipeline starting at $(date -u +%FT%TZ) ===" | tee "$LOG"

# Wait until r6 produces output
R6_PATH="checkpoints/best_ab_distilled_r6.pt"
echo "Waiting for $R6_PATH ..." | tee -a "$LOG"
until [ -f "$R6_PATH" ]; do
    sleep 60
done
echo "$R6_PATH detected at $(date -u +%FT%TZ)" | tee -a "$LOG"
ls -la "$R6_PATH" | tee -a "$LOG"

# Backup existing best.pt
cp checkpoints/best.pt checkpoints/pre_r6_bench_backup.pt
echo "Backup: pre_r6_bench_backup.pt = current best.pt" | tee -a "$LOG"

# ---------------------------------------------------------------------
# Bench r6 vs current best.pt via tournament.py (BT Elo)
# ---------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== Bench r6 vs current best.pt (8 games, 200 sims) ===" | tee -a "$LOG"
TOURNEY_OUT="checkpoints/_tourney_r6_vs_human_v2.json"
python3 -u eval/tournament.py \
    --ckpt "$R6_PATH:r6_d10" \
    --ckpt "checkpoints/best.pt:current_best" \
    --games 8 --sims 200 --workers 6 \
    --opening-random 4 --max-moves 100 \
    --adjudicate-gap 1 \
    --anchor current_best \
    --out "$TOURNEY_OUT" \
    2>&1 | tee -a "$LOG"

# Parse Elos
R6_ELO=$(python3 -c "import json; d=json.load(open('$TOURNEY_OUT')); print(d['ratings'].get('r6_d10', 1000))")
CURRENT_ELO=$(python3 -c "import json; d=json.load(open('$TOURNEY_OUT')); print(d['ratings'].get('current_best', 1000))")
echo "" | tee -a "$LOG"
echo "Bench result: r6=${R6_ELO} current=${CURRENT_ELO}" | tee -a "$LOG"

# Decide winner: if r6 is meaningfully ahead (>15 Elo), promote it
PROMOTE_R6=$(python3 -c "
r6=$R6_ELO; cur=$CURRENT_ELO
print('yes' if (r6 - cur) > 15 else 'no')
")
echo "Promote r6? $PROMOTE_R6" | tee -a "$LOG"

if [ "$PROMOTE_R6" = "yes" ]; then
    echo "Promoting r6 to best.pt" | tee -a "$LOG"
    cp "$R6_PATH" checkpoints/best.pt
    BASE_FOR_TARGETED="r6"
else
    echo "Keeping current best.pt (human_v2)" | tee -a "$LOG"
    BASE_FOR_TARGETED="human_v2"
fi

# ---------------------------------------------------------------------
# Run targeted training on top of the winner
# ---------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== Targeted training on top of best.pt (base=$BASE_FOR_TARGETED) ===" | tee -a "$LOG"
PRE_TARGETED_PATH="checkpoints/pre_targeted_${BASE_FOR_TARGETED}_backup.pt"
cp checkpoints/best.pt "$PRE_TARGETED_PATH"
echo "Backup: $PRE_TARGETED_PATH" | tee -a "$LOG"

python3 -u training/train_from_npz.py \
    --in checkpoints/best.pt \
    --out checkpoints/best_${BASE_FOR_TARGETED}_human_v3.pt \
    --npz data/human_training_set.npz \
    --rehearsal-frac 0.6 --epochs 6 --reg-lambda 1.0 --value-weight 0.3 \
    2>&1 | tee -a "$LOG"

# ---------------------------------------------------------------------
# Bench targeted-trained version vs pre-targeted
# ---------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== Bench targeted-trained vs pre-targeted (8 games, 200 sims) ===" | tee -a "$LOG"
TOURNEY_OUT2="checkpoints/_tourney_${BASE_FOR_TARGETED}_targeted.json"
python3 -u eval/tournament.py \
    --ckpt "checkpoints/best_${BASE_FOR_TARGETED}_human_v3.pt:targeted" \
    --ckpt "$PRE_TARGETED_PATH:pre_targeted" \
    --games 8 --sims 200 --workers 6 \
    --opening-random 4 --max-moves 100 \
    --adjudicate-gap 1 \
    --anchor pre_targeted \
    --out "$TOURNEY_OUT2" \
    2>&1 | tee -a "$LOG"

TARGETED_ELO=$(python3 -c "import json; d=json.load(open('$TOURNEY_OUT2')); print(d['ratings'].get('targeted', 1000))")
PRE_ELO=$(python3 -c "import json; d=json.load(open('$TOURNEY_OUT2')); print(d['ratings'].get('pre_targeted', 1000))")
echo "" | tee -a "$LOG"
echo "Bench result: targeted=${TARGETED_ELO} pre=${PRE_ELO}" | tee -a "$LOG"

PROMOTE_TARGETED=$(python3 -c "
t=$TARGETED_ELO; p=$PRE_ELO
print('yes' if (t - p) > 15 else 'no')
")
echo "Promote targeted? $PROMOTE_TARGETED" | tee -a "$LOG"

if [ "$PROMOTE_TARGETED" = "yes" ]; then
    cp "checkpoints/best_${BASE_FOR_TARGETED}_human_v3.pt" checkpoints/best.pt
    echo "Promoted targeted to best.pt" | tee -a "$LOG"
else
    echo "Keeping pre-targeted as best.pt" | tee -a "$LOG"
fi

echo "" | tee -a "$LOG"
echo "=== post-r6 pipeline finished at $(date -u +%FT%TZ) ===" | tee -a "$LOG"
echo "Final best.pt:" | tee -a "$LOG"
python3 -c "
import torch
ck = torch.load('checkpoints/best.pt', map_location='cpu', weights_only=False)
print('  meta:', ck.get('meta', {}))
" | tee -a "$LOG"
