#!/usr/bin/env bash
# ============================================================================
#  Quoridor RL — Web Showcase (React + TypeScript + FastAPI)
# ============================================================================
#  Single entry point that brings up a unified web dashboard with all four
#  visualisations in one page:
#
#    • Self-Play Spectator        (champion vs champion, looping)
#    • Generations Match-Up       (iter 1 vs champion)
#    • Round-Robin Tournament     (live standings + mini board)
#    • Training Dashboard         (loss / outcomes / plies / eval, live)
#
#  Architecture: FastAPI backend with WebSocket streams + React/TS frontend.
#
#  Pipeline:
#    1. install python deps if missing  (fastapi, uvicorn, websockets)
#    2. install node deps if missing    (vite, react, recharts, ...)
#    3. build the frontend bundle
#    4. launch uvicorn serving both the static bundle and the WebSocket API
#    5. open http://localhost:8000 in the default browser
#
#  Flags:
#    --dev           use Vite dev server (HMR) on :5173 alongside the
#                    backend on :8000 — recommended while iterating on the UI
#    --no-open       don't auto-open the browser
#    --port N        backend port (default 8000)
#    --light         lower MCTS sims so the demo runs smoothly on a laptop
#    --skip-build    skip the `npm run build` step (use existing dist/)
# ============================================================================

set -u
cd "$(dirname "$0")/.."

PORT=8000
OPEN_BROWSER=1
DEV_MODE=0
SKIP_BUILD=0
LIGHT_MODE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --dev)         DEV_MODE=1 ;;
        --no-open)     OPEN_BROWSER=0 ;;
        --port)        shift; PORT="$1" ;;
        --light)       LIGHT_MODE=1 ;;
        --skip-build)  SKIP_BUILD=1 ;;
        -h|--help)     sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
    shift
done

if [ "$LIGHT_MODE" = "1" ]; then
    export SHOWCASE_SIMS_SPECTATOR=150
    export SHOWCASE_SIMS_TOURNAMENT=80
    export SHOWCASE_MOVE_DELAY=0.20
    export SHOWCASE_TOURN_DELAY=0.05
fi

if [ ! -f "checkpoints/best.pt" ]; then
    echo "[showcase-web] missing checkpoints/best.pt — aborting." >&2
    exit 1
fi

# ----------------------------------------------------------------
# 1. Python deps
# ----------------------------------------------------------------
# Prefer the project venv if it exists.
if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    . .venv/bin/activate
fi

if ! python3 -c "import fastapi, uvicorn, websockets" >/dev/null 2>&1; then
    echo "[showcase-web] installing python deps (fastapi, uvicorn, websockets)..."
    PIP_ARGS=( --quiet "fastapi>=0.110" "uvicorn[standard]>=0.27" "websockets>=12.0" )
    if python3 -m pip install "${PIP_ARGS[@]}" 2>/tmp/pip_err.log; then
        :
    elif grep -q "externally-managed-environment" /tmp/pip_err.log 2>/dev/null; then
        echo "[showcase-web]   PEP-668 protected python — retrying with --break-system-packages"
        python3 -m pip install --break-system-packages "${PIP_ARGS[@]}"
    else
        cat /tmp/pip_err.log >&2
        exit 1
    fi
fi

# ----------------------------------------------------------------
# 2. Node deps
# ----------------------------------------------------------------
if ! command -v npm >/dev/null 2>&1; then
    echo "[showcase-web] npm not found in PATH — install Node.js first." >&2
    exit 1
fi

if [ ! -d "webapp/frontend/node_modules" ]; then
    echo "[showcase-web] installing node deps (one-time)..."
    ( cd webapp/frontend && npm install --silent )
fi

# ----------------------------------------------------------------
# 3. Build frontend
# ----------------------------------------------------------------
if [ "$DEV_MODE" = "0" ] && [ "$SKIP_BUILD" = "0" ]; then
    echo "[showcase-web] building frontend bundle..."
    ( cd webapp/frontend && npm run build )
fi

mkdir -p logs
BACKEND_LOG="logs/showcase_web_backend.log"
FRONTEND_LOG="logs/showcase_web_frontend.log"
PIDS=()

cleanup() {
    echo
    echo "[showcase-web] shutting down..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    sleep 0.4
    for pid in "${PIDS[@]}"; do
        kill -9 "$pid" 2>/dev/null || true
    done
    echo "[showcase-web] done."
}
trap cleanup INT TERM EXIT

# ----------------------------------------------------------------
# 4. Launch backend (uvicorn)
# ----------------------------------------------------------------
echo "[showcase-web] starting FastAPI backend on :$PORT"
python3 -m uvicorn webapp.server:app \
    --host 0.0.0.0 --port "$PORT" \
    >"$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!
PIDS+=("$BACKEND_PID")
echo "[showcase-web]   backend pid=$BACKEND_PID   log=$BACKEND_LOG"

# Wait for backend to come up (max ~15s)
echo "[showcase-web] waiting for backend to be ready..."
for i in $(seq 1 30); do
    if curl -fsS "http://localhost:$PORT/api/streams" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo "[showcase-web] backend exited unexpectedly — see $BACKEND_LOG" >&2
        tail -n 30 "$BACKEND_LOG" >&2
        exit 1
    fi
    sleep 0.5
done

# ----------------------------------------------------------------
# 5. Optional Vite dev server (HMR)
# ----------------------------------------------------------------
if [ "$DEV_MODE" = "1" ]; then
    echo "[showcase-web] starting Vite dev server on :5173"
    ( cd webapp/frontend && npm run dev ) >"$FRONTEND_LOG" 2>&1 &
    FRONTEND_PID=$!
    PIDS+=("$FRONTEND_PID")
    echo "[showcase-web]   vite pid=$FRONTEND_PID   log=$FRONTEND_LOG"
    APP_URL="http://localhost:5173"
else
    APP_URL="http://localhost:$PORT"
fi

# ----------------------------------------------------------------
# 6. Open browser
# ----------------------------------------------------------------
if [ "$OPEN_BROWSER" = "1" ]; then
    sleep 1.2
    case "$(uname -s)" in
        Darwin) open "$APP_URL" >/dev/null 2>&1 || true ;;
        Linux)  xdg-open "$APP_URL" >/dev/null 2>&1 || true ;;
    esac
fi

echo
echo "==============================================================="
echo "  Quoridor RL Showcase is live at:  $APP_URL"
echo "==============================================================="
echo "  backend log : $BACKEND_LOG"
[ "$DEV_MODE" = "1" ] && echo "  vite log    : $FRONTEND_LOG"
echo "  Ctrl+C in this terminal to stop everything."
echo

# Block while at least one child is alive.
while :; do
    alive=0
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            alive=$((alive + 1))
        fi
    done
    if [ "$alive" = "0" ]; then
        echo "[showcase-web] all processes exited."
        break
    fi
    sleep 1
done
