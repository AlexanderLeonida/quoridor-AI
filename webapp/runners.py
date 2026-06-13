"""Background runners that produce live state for the web app.

Each runner runs in its own OS thread (so PyTorch inference proceeds without
blocking the asyncio event loop) and publishes state updates to a list of
asyncio.Queue subscribers using ``loop.call_soon_threadsafe``.
"""
from __future__ import annotations

import asyncio
import csv
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional


# ---------------------------------------------------------------------
# helpers used by both runners
# ---------------------------------------------------------------------
def _board_dict(board) -> dict:
    return {
        "pawns": [list(p) for p in board.pawns],
        "h_walls": [list(w) for w in board.h_walls],
        "v_walls": [list(w) for w in board.v_walls],
        "walls_left": list(board.walls_left),
        "turn": board.turn,
        "winner": board.winner(),
    }


def _move_to_str(m) -> str:
    from quoridor import WALL_H, WALL_V
    col = "abcdefghi"[m.c]
    base = f"{col}{m.r + 1}"
    if m.kind == WALL_H:
        return base + "h"
    if m.kind == WALL_V:
        return base + "v"
    return base


@dataclass
class AgentSpec:
    ckpt: str
    name: str


# ---------------------------------------------------------------------
# base runner: pub-sub plumbing
# ---------------------------------------------------------------------
class BaseRunner:
    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue] = []
        self._lock = threading.Lock()
        self._snapshot: dict = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop = False

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        threading.Thread(target=self._safe_run, daemon=True).start()

    def stop(self) -> None:
        self._stop = True

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=16)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def snapshot(self) -> dict:
        return self._snapshot

    def _publish(self, state: dict) -> None:
        self._snapshot = state
        loop = self._loop
        if loop is None:
            return
        with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            try:
                loop.call_soon_threadsafe(self._coalesce_put, q, state)
            except RuntimeError:
                pass

    @staticmethod
    def _coalesce_put(q: asyncio.Queue, state: dict) -> None:
        # If the queue is full, drop the oldest pending item and push the
        # newest — viewers care about latest state, not history.
        if q.full():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            q.put_nowait(state)
        except asyncio.QueueFull:
            pass

    def _safe_run(self) -> None:
        try:
            self._run()
        except Exception as exc:  # noqa: BLE001
            self._publish({"error": f"{type(exc).__name__}: {exc}"})
            raise

    def _run(self) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------
# Spectator runner — AI vs AI, looping
# ---------------------------------------------------------------------
class SpectatorRunner(BaseRunner):
    def __init__(
        self,
        p1: AgentSpec,
        p2: AgentSpec,
        sims: int,
        move_delay: float,
        title: str,
    ):
        super().__init__()
        self.p1 = p1
        self.p2 = p2
        self.sims = sims
        self.move_delay = move_delay
        self.title = title
        self.score = [0, 0, 0]  # p1 wins, p2 wins, draws
        self.game_num = 1
        self.move_history: list[str] = []
        self.top_moves: list[dict] = []

    def _run(self) -> None:
        import torch
        from quoridor import Board
        from quoridor.encoding import action_to_move, canonical_view
        from quoridor.mcts import EvalCache, MCTSConfig, root_value, search, select_action
        from quoridor.net import load_checkpoint

        device = torch.device("cpu")
        nets: dict[int, Any] = {}
        caches: dict[int, EvalCache] = {}
        for side, spec in ((0, self.p1), (1, self.p2)):
            net, _ = load_checkpoint(spec.ckpt, map_location="cpu")
            net.to(device).eval()
            nets[side] = net
            caches[side] = EvalCache()

        while not self._stop:
            board = Board.initial()
            caches[0].clear()
            caches[1].clear()
            self.move_history.clear()
            last_value = 0.0
            last_move: Optional[str] = None
            self._emit(board, last_value, last_move, "new game starting...")

            move_idx = 0
            while board.winner() is None and move_idx < 200 and not self._stop:
                side = board.turn
                cfg = MCTSConfig(num_simulations=self.sims, dirichlet_epsilon=0.0)
                t0 = time.time()
                root = search(
                    board, nets[side], cfg, device,
                    add_noise=False, cache=caches[side],
                )
                action = select_action(root, temperature=0.0)
                _, _, _, _, _, _, flipped = canonical_view(board)
                move = action_to_move(action, flipped)
                dt = time.time() - t0

                # Collect the AI's top 5 considered moves (visit counts are
                # the canonical MCTS "thinking" signal — these are what the
                # search actually committed simulations to).
                top = sorted(
                    root.children.items(),
                    key=lambda kv: kv[1].visit_count, reverse=True,
                )[:5]
                total = max(1, sum(c.visit_count for _, c in top))
                self.top_moves = [
                    {
                        "notation": _move_to_str(action_to_move(a, flipped)),
                        "visits": int(child.visit_count),
                        "weight": float(child.visit_count) / total,
                        # Node.value is a @property — not a method
                        "value": float(child.value) if child.visit_count else 0.0,
                    }
                    for a, child in top
                ]

                board = board.apply(move)
                last_value = float(root_value(root))
                last_move = _move_to_str(move)
                self.move_history.append(last_move)
                move_idx += 1
                name = self.p1.name if side == 0 else self.p2.name
                self._emit(
                    board, last_value, last_move,
                    f"{name} played {last_move} in {dt:.2f}s",
                )
                remain = max(0.0, self.move_delay - dt)
                if remain > 0:
                    time.sleep(remain)

            winner = board.winner()
            if winner is None:
                self.score[2] += 1
                msg = "game ended without a winner"
            elif winner == 0:
                self.score[0] += 1
                msg = f"{self.p1.name} won game #{self.game_num}"
            else:
                self.score[1] += 1
                msg = f"{self.p2.name} won game #{self.game_num}"
            self.game_num += 1
            self._emit(board, last_value, last_move, msg)
            time.sleep(1.4)

    def _emit(self, board, value: float, last_move: Optional[str], status: str) -> None:
        self._publish({
            "title": self.title,
            "p1_name": self.p1.name,
            "p2_name": self.p2.name,
            "board": _board_dict(board),
            "value": value,
            "last_move": last_move,
            "score": list(self.score),
            "game_num": self.game_num,
            "status": status,
            "history": list(self.move_history[-24:]),
            "top_moves": list(self.top_moves),
        })


# ---------------------------------------------------------------------
# Tournament runner — round-robin with live standings
# ---------------------------------------------------------------------
class TournamentRunner(BaseRunner):
    def __init__(
        self,
        agents: list[AgentSpec],
        sims: int,
        games_per_pair: int,
        move_delay: float,
    ):
        super().__init__()
        self.agents = agents
        self.sims = sims
        self.games_per_pair = games_per_pair
        self.move_delay = move_delay
        self.stats = [
            {"name": a.name, "wins": 0, "losses": 0, "draws": 0, "rating": 1000.0}
            for a in agents
        ]
        # head_to_head[i][j] = {wins, losses, draws} from i's perspective
        n = len(agents)
        self.head_to_head = [
            [{"wins": 0, "losses": 0, "draws": 0} for _ in range(n)]
            for _ in range(n)
        ]
        self.board = None
        self.match: Optional[tuple[int, int]] = None
        self.last_message = "starting tournament..."
        self.games_completed = 0

    def _run(self) -> None:
        import torch
        from quoridor import Board
        from quoridor.encoding import action_to_move, canonical_view
        from quoridor.mcts import EvalCache, MCTSConfig, search, select_action
        from quoridor.net import load_checkpoint

        device = torch.device("cpu")
        nets: dict[str, Any] = {}
        caches: dict[str, EvalCache] = {}
        for ag in self.agents:
            net, _ = load_checkpoint(ag.ckpt, map_location="cpu")
            net.to(device).eval()
            nets[ag.ckpt] = net
            caches[ag.ckpt] = EvalCache()

        n = len(self.agents)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(pairs)

        while not self._stop:
            for (i, j) in pairs:
                for g in range(self.games_per_pair):
                    if self._stop:
                        return
                    side0_idx, side1_idx = (i, j) if g % 2 == 0 else (j, i)
                    self.match = (side0_idx, side1_idx)
                    a0 = self.agents[side0_idx]
                    a1 = self.agents[side1_idx]
                    self.last_message = (
                        f"{a0.name} (red) vs {a1.name} (blue)"
                    )

                    board = Board.initial()
                    self.board = board
                    self._emit()

                    moves = 0
                    while board.winner() is None and moves < 160 and not self._stop:
                        side = board.turn
                        ag = a0 if side == 0 else a1
                        cfg = MCTSConfig(
                            num_simulations=self.sims, dirichlet_epsilon=0.0,
                        )
                        root = search(
                            board, nets[ag.ckpt], cfg, device,
                            add_noise=False, cache=caches[ag.ckpt],
                        )
                        action = select_action(root, temperature=0.0)
                        _, _, _, _, _, _, flipped = canonical_view(board)
                        move = action_to_move(action, flipped)
                        board = board.apply(move)
                        self.board = board
                        moves += 1
                        self._emit()
                        time.sleep(self.move_delay)

                    winner = board.winner()
                    self._record(side0_idx, side1_idx, winner)
                    self.games_completed += 1
                    self._emit()
                    time.sleep(0.9)
            random.shuffle(pairs)

    def _record(self, side0_idx: int, side1_idx: int, winner: Optional[int]) -> None:
        s0 = self.stats[side0_idx]
        s1 = self.stats[side1_idx]
        h0 = self.head_to_head[side0_idx][side1_idx]
        h1 = self.head_to_head[side1_idx][side0_idx]
        if winner is None:
            s0["draws"] += 1
            s1["draws"] += 1
            h0["draws"] += 1
            h1["draws"] += 1
            score = 0.5
            self.last_message = f"{s0['name']} vs {s1['name']} — draw"
        elif winner == 0:
            s0["wins"] += 1
            s1["losses"] += 1
            h0["wins"] += 1
            h1["losses"] += 1
            score = 1.0
            self.last_message = f"{s0['name']} won vs {s1['name']}"
        else:
            s0["losses"] += 1
            s1["wins"] += 1
            h0["losses"] += 1
            h1["wins"] += 1
            score = 0.0
            self.last_message = f"{s1['name']} won vs {s0['name']}"
        K = 24.0
        ea = 1.0 / (1.0 + 10.0 ** ((s1["rating"] - s0["rating"]) / 400.0))
        s0["rating"] += K * (score - ea)
        s1["rating"] += K * ((1.0 - score) - (1.0 - ea))

    def _emit(self) -> None:
        self._publish({
            "board": _board_dict(self.board) if self.board is not None else None,
            "match": list(self.match) if self.match is not None else None,
            "agents": [{"name": a.name} for a in self.agents],
            "standings": [dict(s) for s in self.stats],
            "head_to_head": [
                [dict(cell) for cell in row] for row in self.head_to_head
            ],
            "message": self.last_message,
            "games_completed": self.games_completed,
        })


# ---------------------------------------------------------------------
# Metrics runner — polls logs/metrics.csv
# ---------------------------------------------------------------------
class MetricsRunner(BaseRunner):
    def __init__(self, path: str, interval: float = 2.5) -> None:
        super().__init__()
        self.path = path
        self.interval = interval

    def _run(self) -> None:
        last_mtime = -1.0
        while not self._stop:
            try:
                mtime = os.path.getmtime(self.path) if os.path.exists(self.path) else 0
                if mtime != last_mtime:
                    self._publish(self._read())
                    last_mtime = mtime
            except Exception:  # noqa: BLE001
                pass
            time.sleep(self.interval)

    def _read(self) -> dict:
        if not os.path.exists(self.path):
            return {"rows": []}
        rows: list[dict] = []
        with open(self.path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                row: dict = {}
                for k, v in raw.items():
                    if k is None:
                        continue
                    if v is None or v == "":
                        row[k] = None
                        continue
                    try:
                        row[k] = float(v)
                    except (TypeError, ValueError):
                        row[k] = v
                rows.append(row)
        return {"rows": rows}
