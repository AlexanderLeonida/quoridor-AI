#!/usr/bin/env bash
# ============================================================================
#  Quoridor RL — Investor Showcase
# ============================================================================
#  Launches a multi-window visual demo of the project, showing every stage of
#  the reinforcement-learning loop simultaneously:
#
#    1. Self-Play Spectator     — current champion plays itself (this is how
#                                 training data is generated)
#    2. Generations Match-Up    — earliest iteration vs current champion
#                                 (dramatizes the strength gap learned via RL)
#    3. Round-Robin Tournament  — 4 checkpoints (early / mid / late / best)
#                                 with a live-updating leaderboard
#    4. Training Dashboard      — live loss curves, self-play outcomes,
#                                 game-length trend, and gating eval scores
#                                 streamed from logs/metrics.csv
#
#  Every viewer runs in its own background process; closing a window does not
#  affect the others.  Ctrl+C in this terminal cleans up all of them.
#
#  Usage:
#      ./scripts/showcase.sh                 # full demo (all 4 windows)
#      ./scripts/showcase.sh --no-tournament # skip the heaviest panel
#      ./scripts/showcase.sh --light         # lower MCTS sims for slow laptops
# ============================================================================

set -u
cd "$(dirname "$0")/.."

# --- Config ---------------------------------------------------------------
SIMS_SPECTATOR=150
SIMS_TOURNAMENT=80
MOVE_DELAY=0.35
TOURN_DELAY=0.10
TOURN_GAMES=2

LAUNCH_SELFPLAY=1
LAUNCH_GENERATIONS=1
LAUNCH_TOURNAMENT=1
LAUNCH_DASHBOARD=1

while [ $# -gt 0 ]; do
    case "$1" in
        --no-selfplay)     LAUNCH_SELFPLAY=0 ;;
        --no-generations)  LAUNCH_GENERATIONS=0 ;;
        --no-tournament)   LAUNCH_TOURNAMENT=0 ;;
        --no-dashboard)    LAUNCH_DASHBOARD=0 ;;
        --light)
            SIMS_SPECTATOR=60
            SIMS_TOURNAMENT=40
            MOVE_DELAY=0.20
            TOURN_DELAY=0.05
            ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
    shift
done

# --- Sanity checks --------------------------------------------------------
if [ ! -f "checkpoints/best.pt" ]; then
    echo "[showcase] missing checkpoints/best.pt — aborting." >&2
    exit 1
fi

# Pick representative checkpoints for the tournament.  These are picked at
# even spacing across the training history so the viewer sees clear progress.
pick_existing() {
    for c in "$@"; do
        if [ -f "$c" ]; then echo "$c"; return; fi
    done
}
EARLY=$(pick_existing checkpoints/iter_0001.pt  checkpoints/iter_0002.pt)
EARLY_MID=$(pick_existing checkpoints/iter_0020.pt checkpoints/iter_0019.pt checkpoints/iter_0015.pt)
MID=$(pick_existing checkpoints/iter_0040.pt checkpoints/iter_0036.pt)
LATE=$(pick_existing checkpoints/iter_0070.pt checkpoints/iter_0074.pt checkpoints/iter_0060.pt)
CHAMPION="checkpoints/best.pt"

mkdir -p logs

LOG_SELFPLAY="logs/showcase_selfplay.log"
LOG_GENERATIONS="logs/showcase_generations.log"
LOG_TOURNAMENT="logs/showcase_tournament.log"
LOG_DASHBOARD="logs/showcase_dashboard.log"

PIDS=()

cleanup() {
    echo
    echo "[showcase] shutting down child processes..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    # second pass — SIGKILL anything still alive
    sleep 0.5
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    echo "[showcase] done."
}
trap cleanup INT TERM EXIT

launch() {
    local label="$1"; shift
    local log="$1"; shift
    echo "[showcase] -> launching $label ($*)"
    ( "$@" ) >"$log" 2>&1 &
    local pid=$!
    PIDS+=("$pid")
    echo "[showcase]    pid=$pid   log=$log"
}

echo
echo "==============================================================="
echo "  Quoridor RL — Investor Showcase"
echo "==============================================================="
echo "  selfplay sims:    $SIMS_SPECTATOR"
echo "  tournament sims:  $SIMS_TOURNAMENT"
echo "  move delay:       ${MOVE_DELAY}s"
echo "  early ckpt:       ${EARLY:-(none)}"
echo "  mid ckpt:         ${MID:-(none)}"
echo "  late ckpt:        ${LATE:-(none)}"
echo "  champion ckpt:    $CHAMPION"
echo "==============================================================="
echo

# Window 1 — champion self-play (this is what creates training data)
if [ "$LAUNCH_SELFPLAY" = "1" ]; then
    launch "Self-Play Spectator" "$LOG_SELFPLAY" \
        python3 -u viz/spectator.py \
            --p1 "$CHAMPION" --p1-name "Champion (red)" \
            --p2 "$CHAMPION" --p2-name "Champion (blue)" \
            --sims "$SIMS_SPECTATOR" --move-delay "$MOVE_DELAY" \
            --title "Self-Play  ·  Training Data Generation"
    sleep 0.4
fi

# Window 2 — earliest iteration vs champion (the "before vs after" demo)
if [ "$LAUNCH_GENERATIONS" = "1" ] && [ -n "${EARLY:-}" ]; then
    launch "Generations Match-Up" "$LOG_GENERATIONS" \
        python3 -u viz/spectator.py \
            --p1 "$EARLY" --p1-name "Iter 1  (untrained)" \
            --p2 "$CHAMPION" --p2-name "Champion" \
            --sims "$SIMS_SPECTATOR" --move-delay "$MOVE_DELAY" \
            --title "Generations  ·  Before vs After RL"
    sleep 0.4
fi

# Window 3 — multi-agent round-robin tournament with live leaderboard
if [ "$LAUNCH_TOURNAMENT" = "1" ]; then
    TOURN_ARGS=()
    [ -n "${EARLY:-}" ]     && TOURN_ARGS+=( --ckpt "${EARLY}:Iter1" )
    [ -n "${EARLY_MID:-}" ] && TOURN_ARGS+=( --ckpt "${EARLY_MID}:Early" )
    [ -n "${MID:-}" ]       && TOURN_ARGS+=( --ckpt "${MID}:Mid" )
    [ -n "${LATE:-}" ]      && TOURN_ARGS+=( --ckpt "${LATE}:Late" )
    TOURN_ARGS+=( --ckpt "${CHAMPION}:Champion" )
    if [ "${#TOURN_ARGS[@]}" -ge 4 ]; then
        launch "Round-Robin Tournament" "$LOG_TOURNAMENT" \
            python3 -u viz/tournament_viz.py \
                "${TOURN_ARGS[@]}" \
                --sims "$SIMS_TOURNAMENT" \
                --games "$TOURN_GAMES" \
                --move-delay "$TOURN_DELAY"
        sleep 0.4
    else
        echo "[showcase] tournament skipped — not enough distinct checkpoints found"
    fi
fi

# Window 4 — live training metrics dashboard
if [ "$LAUNCH_DASHBOARD" = "1" ]; then
    if [ ! -f "logs/metrics.csv" ]; then
        echo "[showcase] logs/metrics.csv missing — dashboard skipped"
    else
        launch "Training Dashboard" "$LOG_DASHBOARD" \
            python3 -u viz/training_dashboard.py \
                --metrics logs/metrics.csv --interval 2.5
        sleep 0.3
    fi
fi

if [ "${#PIDS[@]}" = "0" ]; then
    echo "[showcase] no viewers launched — exiting." >&2
    exit 1
fi

echo
echo "[showcase] ${#PIDS[@]} viewer(s) running."
echo "[showcase] tail individual logs with:  tail -f logs/showcase_*.log"
echo "[showcase] press Ctrl+C to stop them all."
echo

# Wait for any child to exit, but stay alive while at least one is running.
# We deliberately do NOT use `wait` (it would block on all children); instead
# poll so Ctrl+C is responsive.
while :; do
    alive=0
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            alive=$((alive + 1))
        fi
    done
    if [ "$alive" = "0" ]; then
        echo "[showcase] all viewers exited."
        break
    fi
    sleep 1
done
