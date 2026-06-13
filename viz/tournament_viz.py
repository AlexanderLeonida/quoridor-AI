"""Live round-robin tournament visualization.

Plays a round-robin between N checkpoints in a background thread while
displaying:
  * a mini live board of the current matchup
  * a standings table with W/L/D updating after every game
  * the running Bradley-Terry rating estimate

Usage
-----
    python3 viz/tournament_viz.py \\
        --ckpt checkpoints/iter_0001.pt:Iter1 \\
        --ckpt checkpoints/iter_0040.pt:Iter40 \\
        --ckpt checkpoints/iter_0082.pt:Iter82 \\
        --ckpt checkpoints/best.pt:Champion \\
        --sims 80 --games 2 --move-delay 0.15
"""
from __future__ import annotations

import argparse
import math
import os as _os
import random
import sys as _sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from typing import Optional, Tuple

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from quoridor import BOARD_SIZE, Board, MOVE_PAWN, WALL_GRID, WALL_H, WALL_V
from quoridor.encoding import action_to_move, canonical_view
from quoridor.mcts import EvalCache, MCTSConfig, search, select_action


# --- Layout ------------------------------------------------------------
CELL = 28
WT = 6
PITCH = CELL + WT
MARGIN = 14
BOARD_PX = 9 * CELL + 8 * WT
CANVAS_SIZE = BOARD_PX + 2 * MARGIN

COLOR_BG = "#1a1a24"
COLOR_PANEL = "#252535"
COLOR_CELL = "#3a3a4d"
COLOR_GOAL_P1 = "#5a2a2a"
COLOR_GOAL_P2 = "#2a3a5a"
COLOR_WALL = "#d4a574"
COLOR_P1 = "#e74c3c"
COLOR_P2 = "#3498db"
COLOR_TEXT = "#e8e8f0"
COLOR_MUTED = "#8888a0"
COLOR_ACCENT = "#f1c40f"
COLOR_ROW_ALT = "#2a2a3a"


@dataclass
class AgentSpec:
    ckpt_path: str
    name: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    rating: float = 1000.0

    @property
    def score(self) -> float:
        return self.wins + 0.5 * self.draws

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws


class TournamentGUI:
    def __init__(
        self,
        root: tk.Tk,
        agents: list[AgentSpec],
        sims: int,
        games_per_pair: int,
        move_delay: float,
    ):
        self.root = root
        self.agents = agents
        self.sims = sims
        self.games_per_pair = games_per_pair
        self.move_delay = move_delay
        self.board = Board.initial()
        self.match_label = ""
        self.match_status = "initializing..."
        self._stop = False

        # head-to-head matrix: results[i][j] = wins-of-i-as-side-0 vs j
        self.results = [[(0, 0, 0) for _ in agents] for _ in agents]

        # Lazily loaded
        self._nets: dict[str, object] = {}
        self._caches: dict[str, EvalCache] = {}
        self._device = None

        root.title("Quoridor — Round-Robin Tournament")
        root.configure(bg=COLOR_BG)
        root.resizable(False, False)
        self._build_ui()

        threading.Thread(target=self._run_tournament, daemon=True).start()

    # ----------------------------------------------------------------
    def _build_ui(self) -> None:
        title = tk.Frame(self.root, bg=COLOR_BG)
        title.pack(fill=tk.X, padx=14, pady=(12, 4))
        tk.Label(
            title, text="Round-Robin Tournament",
            font=("Helvetica", 18, "bold"),
            bg=COLOR_BG, fg=COLOR_ACCENT,
        ).pack(side=tk.LEFT)

        body = tk.Frame(self.root, bg=COLOR_BG)
        body.pack(padx=14, pady=4)

        # Left: board canvas + match label
        left = tk.Frame(body, bg=COLOR_BG)
        left.pack(side=tk.LEFT, padx=(0, 14))

        self.match_var = tk.StringVar(value="")
        tk.Label(
            left, textvariable=self.match_var, font=("Helvetica", 12, "bold"),
            bg=COLOR_BG, fg=COLOR_TEXT,
        ).pack()

        self.canvas = tk.Canvas(
            left, width=CANVAS_SIZE, height=CANVAS_SIZE,
            bg=COLOR_BG, highlightthickness=0,
        )
        self.canvas.pack(pady=6)

        self.status_var = tk.StringVar(value="loading...")
        tk.Label(
            left, textvariable=self.status_var,
            font=("Helvetica", 10), bg=COLOR_BG, fg=COLOR_MUTED,
            wraplength=CANVAS_SIZE, justify=tk.LEFT,
        ).pack()

        # Right: standings
        right = tk.Frame(body, bg=COLOR_PANEL)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, ipadx=10, ipady=8)

        tk.Label(
            right, text="Standings", font=("Helvetica", 14, "bold"),
            bg=COLOR_PANEL, fg=COLOR_ACCENT,
        ).grid(row=0, column=0, columnspan=6, pady=(4, 8), sticky="w")

        headers = ["#", "Name", "W", "L", "D", "Rating"]
        for c, h in enumerate(headers):
            tk.Label(
                right, text=h, font=("Helvetica", 11, "bold"),
                bg=COLOR_PANEL, fg=COLOR_MUTED, padx=8,
            ).grid(row=1, column=c, sticky="w")

        # Rows: rank #, name, W, L, D, rating
        self.row_vars: list[list[tk.StringVar]] = []
        for i in range(len(self.agents)):
            row_vars = [tk.StringVar(value="") for _ in headers]
            bg = COLOR_PANEL if i % 2 == 0 else COLOR_ROW_ALT
            for c, v in enumerate(row_vars):
                font = ("Helvetica", 11, "bold" if c in (0, 1) else "normal")
                tk.Label(
                    right, textvariable=v, font=font, bg=bg, fg=COLOR_TEXT,
                    padx=8, anchor="w", width=(10 if c == 1 else 4),
                ).grid(row=2 + i, column=c, sticky="ew", pady=1)
            self.row_vars.append(row_vars)

        # Progress
        self.progress_var = tk.StringVar(value="")
        tk.Label(
            right, textvariable=self.progress_var, font=("Helvetica", 11),
            bg=COLOR_PANEL, fg=COLOR_MUTED, pady=8,
        ).grid(row=2 + len(self.agents), column=0, columnspan=6, sticky="w")

        self._refresh_standings()

    # ----------------------------------------------------------------
    def _cell_xy(self, internal_r: int, internal_c: int) -> Tuple[int, int]:
        visual_r = (BOARD_SIZE - 1) - internal_r
        return (MARGIN + internal_c * PITCH, MARGIN + visual_r * PITCH)

    def _wall_visual_r(self, r: int) -> int:
        return (WALL_GRID - 1) - r

    def _render(self) -> None:
        self.canvas.delete("all")
        b = self.board
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                x, y = self._cell_xy(r, c)
                fill = COLOR_CELL
                if r == 8:
                    fill = COLOR_GOAL_P1
                elif r == 0:
                    fill = COLOR_GOAL_P2
                self.canvas.create_rectangle(
                    x, y, x + CELL, y + CELL, fill=fill, outline="",
                )
        for (r, c) in b.h_walls:
            vr = self._wall_visual_r(r)
            x1 = MARGIN + c * PITCH
            x2 = MARGIN + (c + 2) * PITCH - WT
            y1 = MARGIN + (vr + 1) * PITCH - WT
            self.canvas.create_rectangle(
                x1, y1, x2, y1 + WT, fill=COLOR_WALL, outline="",
            )
        for (r, c) in b.v_walls:
            vr = self._wall_visual_r(r)
            x1 = MARGIN + (c + 1) * PITCH - WT
            y1 = MARGIN + vr * PITCH
            y2 = MARGIN + (vr + 2) * PITCH - WT
            self.canvas.create_rectangle(
                x1, y1, x1 + WT, y2, fill=COLOR_WALL, outline="",
            )
        for p in (0, 1):
            r, c = b.pawns[p]
            x, y = self._cell_xy(r, c)
            color = COLOR_P1 if p == 0 else COLOR_P2
            self.canvas.create_oval(
                x + 3, y + 3, x + CELL - 3, y + CELL - 3,
                fill=color, outline="white", width=2,
            )

    def _refresh_standings(self) -> None:
        ranked = sorted(
            enumerate(self.agents),
            key=lambda kv: (-kv[1].rating, -kv[1].score),
        )
        for vis_row, (_, ag) in enumerate(ranked):
            vs = self.row_vars[vis_row]
            vs[0].set(str(vis_row + 1))
            vs[1].set(ag.name)
            vs[2].set(str(ag.wins))
            vs[3].set(str(ag.losses))
            vs[4].set(str(ag.draws))
            vs[5].set(f"{ag.rating:.0f}")

    # ----------------------------------------------------------------
    # Networking / play
    # ----------------------------------------------------------------
    def _load(self, spec: AgentSpec) -> None:
        if spec.ckpt_path in self._nets:
            return
        import torch
        from quoridor.net import load_checkpoint
        net, _ = load_checkpoint(spec.ckpt_path, map_location="cpu")
        net.to(self._device)
        net.eval()
        self._nets[spec.ckpt_path] = net
        self._caches[spec.ckpt_path] = EvalCache()

    def _ensure_device(self) -> None:
        if self._device is not None:
            return
        import torch
        self._device = torch.device("cpu")

    def _pick_move(self, spec: AgentSpec):
        cfg = MCTSConfig(num_simulations=self.sims, dirichlet_epsilon=0.0)
        root = search(
            self.board, self._nets[spec.ckpt_path], cfg, self._device,
            add_noise=False, cache=self._caches[spec.ckpt_path],
        )
        action = select_action(root, temperature=0.0)
        _, _, _, _, _, _, flipped = canonical_view(self.board)
        return action_to_move(action, flipped)

    # Bradley-Terry update — simple Elo-like update for live display.
    def _elo_update(self, a: AgentSpec, b: AgentSpec, score_a: float) -> None:
        K = 24.0
        ea = 1.0 / (1.0 + 10.0 ** ((b.rating - a.rating) / 400.0))
        a.rating += K * (score_a - ea)
        b.rating += K * ((1.0 - score_a) - (1.0 - ea))

    def _play_one(self, side0: AgentSpec, side1: AgentSpec) -> Optional[int]:
        self.board = Board.initial()
        moves = 0
        while self.board.winner() is None and moves < 160:
            spec = side0 if self.board.turn == 0 else side1
            try:
                move = self._pick_move(spec)
            except Exception as exc:  # noqa: BLE001
                self.root.after(
                    0, lambda e=exc: self.status_var.set(f"err: {e}")
                )
                return None
            self.board = self.board.apply(move)
            moves += 1
            self.root.after(0, self._render)
            time.sleep(self.move_delay)
        return self.board.winner()

    def _run_tournament(self) -> None:
        try:
            self._ensure_device()
            for ag in self.agents:
                self._load(ag)
        except Exception as exc:  # noqa: BLE001
            self.root.after(0, lambda: self.status_var.set(f"load err: {exc}"))
            return

        n = len(self.agents)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(pairs)
        total_games = len(pairs) * self.games_per_pair
        done = 0

        while not self._stop:
            for (i, j) in pairs:
                for g in range(self.games_per_pair):
                    if self._stop:
                        return
                    # Alternate colors
                    side0, side1 = (i, j) if g % 2 == 0 else (j, i)
                    a0 = self.agents[side0]
                    a1 = self.agents[side1]
                    self.match_var.set(f"{a0.name}  vs  {a1.name}")
                    self.status_var.set(
                        f"{a0.name} (red) — {a1.name} (blue)"
                    )
                    winner = self._play_one(a0, a1)
                    if winner is None:
                        a0.draws += 1
                        a1.draws += 1
                        self._elo_update(a0, a1, 0.5)
                    elif winner == 0:
                        a0.wins += 1
                        a1.losses += 1
                        self._elo_update(a0, a1, 1.0)
                    else:
                        a0.losses += 1
                        a1.wins += 1
                        self._elo_update(a0, a1, 0.0)
                    done += 1
                    self.root.after(
                        0,
                        lambda d=done, w=winner, ag0=a0, ag1=a1:
                            self._on_game_end(d, total_games, w, ag0, ag1),
                    )
                    time.sleep(1.0)
            # Loop round-robin indefinitely for demo purposes.
            random.shuffle(pairs)

    def _on_game_end(self, done: int, total: int, winner, a0, a1) -> None:
        self.progress_var.set(f"games played: {done}  (rotation of {total})")
        if winner is None:
            self.status_var.set(f"draw: {a0.name} vs {a1.name}")
        else:
            who = a0.name if winner == 0 else a1.name
            self.status_var.set(f"{who} won — updating standings...")
        self._refresh_standings()

    def stop(self) -> None:
        self._stop = True


def _parse_spec(s: str) -> AgentSpec:
    if ":" in s:
        path, name = s.split(":", 1)
        return AgentSpec(path, name)
    return AgentSpec(s, _os.path.basename(s))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt", action="append", required=True,
        help="path[:label] – repeat for each agent",
    )
    ap.add_argument("--sims", type=int, default=80)
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--move-delay", type=float, default=0.12)
    args = ap.parse_args()

    agents = [_parse_spec(s) for s in args.ckpt]
    if len(agents) < 2:
        ap.error("need at least 2 --ckpt entries")

    root = tk.Tk()
    gui = TournamentGUI(root, agents, args.sims, args.games, args.move_delay)
    root.protocol("WM_DELETE_WINDOW", lambda: (gui.stop(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
