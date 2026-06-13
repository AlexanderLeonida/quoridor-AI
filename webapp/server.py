"""FastAPI backend serving the Quoridor RL showcase web app.

Runs four live data streams in the background and exposes them over
WebSockets to the React frontend.  Also serves the static frontend bundle
once it has been built (``npm run build`` in ``webapp/frontend``).

Run:
    uvicorn webapp.server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# path bootstrap so `python -m uvicorn webapp.server:app` works from repo root
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from webapp.runners import (
    AgentSpec,
    MetricsRunner,
    SpectatorRunner,
    TournamentRunner,
)


runners: dict[str, Any] = {}


def _pick(*candidates: str) -> str | None:
    for c in candidates:
        if (Path(_REPO) / c).exists():
            return c
    return None


async def _start_runners() -> None:
    """Wire every panel to real checkpoints + real training metrics.

    Sources (all live, all from this project):
      * self-play stream      → checkpoints/best.pt (the trained champion)
                                playing itself with high-quality MCTS search
      * generations stream    → checkpoints/iter_0001.pt vs best.pt — the
                                deliberate "before vs after RL" demonstration;
                                Iter 1 is intentionally weak so the strength
                                gap is visible
      * tournament stream     → 5 real iterations from the actual self-play
                                history.  Iter 1 is deliberately *excluded*
                                so every tournament game is a real fight
                                between trained nets
      * metrics stream        → logs/metrics.csv produced by training/selfplay.py
    """
    champion = "checkpoints/best.pt"
    if not (Path(_REPO) / champion).exists():
        print(f"[webapp] missing {champion}; runners will not start", flush=True)
        return

    # The truly random/untrained foil for the generations panel.
    # warmstart_10x128.pt has train_loss=inf in its meta — random-init weights,
    # which is exactly what we want for a dramatic "before RL" reference.
    # iter_0001.pt is misleading here because it shares architecture with
    # best.pt and is from a later widened run, so it's actually competent.
    untrained = _pick(
        "checkpoints/warmstart_10x128.pt",
        "checkpoints/iter_0002.pt",
        "checkpoints/iter_0001.pt",
    )

    # Tournament participants: all reasonably-trained checkpoints, spaced
    # across the back half of the training run so the leaderboard reflects
    # genuine skill differences (not weak-vs-strong blowouts).
    tourn_picks = [
        ("Iter 20",   _pick("checkpoints/iter_0020.pt", "checkpoints/iter_0019.pt")),
        ("Iter 40",   _pick("checkpoints/iter_0040.pt", "checkpoints/iter_0036.pt")),
        ("Iter 60",   _pick("checkpoints/iter_0060.pt", "checkpoints/iter_0058.pt")),
        ("Iter 80",   _pick("checkpoints/iter_0082.pt", "checkpoints/iter_0080.pt",
                            "checkpoints/iter_0074.pt")),
        ("Champion",  champion),
    ]

    # Default sims tuned for "high-level play" rather than "fast turnover".
    # The 10x128 net at 400 sims on CPU produces ~2-3s/move — viewer-pacing
    # comfortable and the search is deep enough that play is genuinely strong.
    sims_sp = int(os.environ.get("SHOWCASE_SIMS_SPECTATOR", "400"))
    sims_t = int(os.environ.get("SHOWCASE_SIMS_TOURNAMENT", "200"))
    move_delay = float(os.environ.get("SHOWCASE_MOVE_DELAY", "0.25"))
    tourn_delay = float(os.environ.get("SHOWCASE_TOURN_DELAY", "0.10"))

    loop = asyncio.get_running_loop()

    runners["selfplay"] = SpectatorRunner(
        AgentSpec(champion, "Champion (red)"),
        AgentSpec(champion, "Champion (blue)"),
        sims=sims_sp, move_delay=move_delay,
        title="Champion Self-Play",
    )
    if untrained:
        runners["generations"] = SpectatorRunner(
            AgentSpec(untrained, "Untrained (random init)"),
            AgentSpec(champion, "Champion"),
            sims=sims_sp, move_delay=move_delay,
            title="Before vs After RL",
        )

    tourn_agents = [AgentSpec(p, name) for (name, p) in tourn_picks if p]
    if len(tourn_agents) >= 2:
        runners["tournament"] = TournamentRunner(
            tourn_agents, sims=sims_t, games_per_pair=2, move_delay=tourn_delay,
        )
        print(
            "[webapp] tournament agents: "
            + ", ".join(f"{a.name}={a.ckpt}" for a in tourn_agents),
            flush=True,
        )

    runners["metrics"] = MetricsRunner("logs/metrics.csv")

    for name, r in runners.items():
        print(f"[webapp] starting runner: {name}", flush=True)
        r.start(loop)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await _start_runners()
    yield
    for r in runners.values():
        r.stop()


app = FastAPI(title="Quoridor RL Showcase", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/streams")
async def list_streams() -> JSONResponse:
    return JSONResponse({"streams": list(runners.keys())})


@app.get("/api/project_stats")
async def project_stats() -> JSONResponse:
    """Headline project numbers — real data from the DB, metrics log, and
    the champion checkpoint.  Cached for 60s to avoid hammering disk."""
    import csv
    import sqlite3
    import time

    cache = project_stats._cache  # type: ignore[attr-defined]
    if cache and time.time() - cache["ts"] < 60:
        return JSONResponse(cache["data"])

    stats: dict = {}
    # 1. Games / moves from the active self-play DB
    for db_path in ("data/quoridor_v3.db", "data/quoridor.db"):
        full = Path(_REPO) / db_path
        if full.exists():
            try:
                con = sqlite3.connect(str(full))
                games = con.execute("SELECT COUNT(*) FROM games").fetchone()[0]
                moves = con.execute("SELECT COUNT(*) FROM moves").fetchone()[0]
                stats.setdefault("games_total", 0)
                stats["games_total"] += int(games)
                stats.setdefault("moves_total", 0)
                stats["moves_total"] += int(moves)
                con.close()
            except Exception:  # noqa: BLE001
                pass

    # 2. Training metrics
    metrics_path = Path(_REPO) / "logs" / "metrics.csv"
    if metrics_path.exists():
        try:
            with open(metrics_path, newline="") as f:
                rows = list(csv.DictReader(f))
            iters = [int(r["global_iter"]) for r in rows
                     if r.get("global_iter", "").strip()]
            stats["iterations_max"] = max(iters) if iters else 0
            stats["iterations_logged"] = len(rows)
            stats["promotions"] = sum(
                1 for r in rows if (r.get("promoted") or "").strip() == "1"
            )
            last = rows[-1] if rows else {}
            try:
                stats["latest_train_loss"] = float(last.get("train_loss", "nan"))
            except (TypeError, ValueError):
                pass
        except Exception:  # noqa: BLE001
            pass

    # 3. Champion architecture + parameter count
    try:
        import torch
        ckpt = torch.load(
            str(Path(_REPO) / "checkpoints" / "best.pt"),
            map_location="cpu", weights_only=False,
        )
        cfg = ckpt.get("config", {})
        stats["arch_blocks"] = int(cfg.get("blocks", 0))
        stats["arch_filters"] = int(cfg.get("filters", 0))
        n_params = sum(
            int(getattr(p, "numel", lambda: 0)())
            for p in ckpt.get("state_dict", {}).values()
        )
        stats["param_count"] = n_params
    except Exception:  # noqa: BLE001
        pass

    # 4. Inference device + how many distinct checkpoints we have to show
    try:
        ckpts = list((Path(_REPO) / "checkpoints").glob("iter_*.pt"))
        stats["checkpoints_saved"] = len(ckpts)
    except Exception:  # noqa: BLE001
        pass

    payload = {"stats": stats}
    project_stats._cache = {"ts": time.time(), "data": payload}  # type: ignore[attr-defined]
    return JSONResponse(payload)


project_stats._cache = None  # type: ignore[attr-defined]


@app.websocket("/ws/{name}")
async def ws_endpoint(websocket: WebSocket, name: str) -> None:
    if name not in runners:
        await websocket.close(code=1008, reason=f"unknown stream: {name}")
        return
    await websocket.accept()
    runner = runners[name]
    q = runner.subscribe()
    try:
        snap = runner.snapshot()
        if snap:
            await websocket.send_json(snap)
        while True:
            state = await q.get()
            await websocket.send_json(state)
    except WebSocketDisconnect:
        pass
    finally:
        runner.unsubscribe(q)


# ----- Static frontend (after `npm run build` in webapp/frontend) -----
_DIST = Path(__file__).resolve().parent / "frontend" / "dist"
if _DIST.exists():
    _ASSETS = _DIST / "assets"
    if _ASSETS.exists():
        app.mount("/assets", StaticFiles(directory=str(_ASSETS)), name="assets")

    @app.get("/")
    async def _index() -> FileResponse:
        return FileResponse(str(_DIST / "index.html"))

    @app.get("/{full_path:path}")
    async def _spa(full_path: str) -> FileResponse:
        # SPA fallback — let React handle routing.
        candidate = _DIST / full_path
        if candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(_DIST / "index.html"))
else:
    @app.get("/")
    async def _no_build() -> JSONResponse:
        return JSONResponse(
            {
                "message": (
                    "Frontend not built. Run `cd webapp/frontend && npm install "
                    "&& npm run build`, or use scripts/showcase_web.sh."
                ),
                "streams": list(runners.keys()),
            },
            status_code=503,
        )
