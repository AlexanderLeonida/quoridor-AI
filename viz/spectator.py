"""Live AI-vs-AI Quoridor spectator window.

A polished Tkinter view that watches two neural-net checkpoints play each
other in real time.  Designed for investor demos: large board, clear labels
showing which checkpoint plays which color, live MCTS value gauge, running
score across consecutive games.

Usage
-----
    python3 viz/spectator.py \\
        --p1 checkpoints/best.pt --p1-name "Champion" \\
        --p2 checkpoints/iter_0001.pt --p2-name "Iter 1" \\
        --sims 200 --move-delay 0.4 --title "Self-Play"
"""
from __future__ import annotations

import argparse
import os as _os
import sys as _sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from typing import Optional, Tuple

# path bootstrap so this file can be run as `python3 viz/spectator.py`
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from quoridor import BOARD_SIZE, Board, MOVE_PAWN, Move, WALL_GRID, WALL_H, WALL_V
from quoridor.encoding import action_to_move, canonical_view
from quoridor.mcts import EvalCache, MCTSConfig, root_value, search, select_action


# --- Layout ------------------------------------------------------------
CELL = 44
WT = 9
PITCH = CELL + WT
MARGIN = 26
BOARD_PX = 9 * CELL + 8 * WT
CANVAS_SIZE = BOARD_PX + 2 * MARGIN

# --- Colors ------------------------------------------------------------
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


@dataclass
class AgentSpec:
    ckpt_path: str
    name: str


class SpectatorGUI:
    def __init__(
        self,
        root: tk.Tk,
        p1: AgentSpec,
        p2: AgentSpec,
        sims: int,
        move_delay: float,
        title: str,
    ):
        self.root = root
        self.p1 = p1
        self.p2 = p2
        self.sims = sims
        self.move_delay = move_delay
        self.title = title

        self.board: Board = Board.initial()
        self.last_value: float = 0.0
        self.last_move_str: str = "—"
        self.move_history: list[str] = []
        self.score = [0, 0, 0]  # P1 wins, P2 wins, draws
        self.game_num = 1
        self._stop = False

        # Loaded lazily in the worker thread to keep startup snappy
        self._nets: dict[int, object] = {}
        self._caches: dict[int, EvalCache] = {}
        self._device = None

        root.title(f"Quoridor — {title}")
        root.configure(bg=COLOR_BG)
        root.resizable(False, False)

        self._build_ui()
        threading.Thread(target=self._play_loop, daemon=True).start()

    # ----------------------------------------------------------------
    def _build_ui(self) -> None:
        # Title strip
        title_frame = tk.Frame(self.root, bg=COLOR_BG)
        title_frame.pack(fill=tk.X, padx=14, pady=(12, 4))
        tk.Label(
            title_frame, text=self.title, font=("Helvetica", 18, "bold"),
            bg=COLOR_BG, fg=COLOR_ACCENT,
        ).pack(side=tk.LEFT)
        self.game_var = tk.StringVar(value=f"Game #{self.game_num}")
        tk.Label(
            title_frame, textvariable=self.game_var, font=("Helvetica", 12),
            bg=COLOR_BG, fg=COLOR_MUTED,
        ).pack(side=tk.RIGHT)

        # P2 (top) name plate
        self._make_nameplate(self.p2.name, COLOR_P2, "top")

        # Canvas
        self.canvas = tk.Canvas(
            self.root, width=CANVAS_SIZE, height=CANVAS_SIZE,
            bg=COLOR_BG, highlightthickness=0,
        )
        self.canvas.pack(padx=14)

        # P1 (bottom) name plate
        self._make_nameplate(self.p1.name, COLOR_P1, "bottom")

        # Info panel
        info = tk.Frame(self.root, bg=COLOR_PANEL)
        info.pack(fill=tk.X, padx=14, pady=(8, 12))

        self.score_var = tk.StringVar(value=self._score_str())
        tk.Label(
            info, textvariable=self.score_var, font=("Helvetica", 13, "bold"),
            bg=COLOR_PANEL, fg=COLOR_TEXT,
        ).pack(side=tk.TOP, pady=(8, 2))

        self.status_var = tk.StringVar(value="loading networks...")
        tk.Label(
            info, textvariable=self.status_var, font=("Helvetica", 11),
            bg=COLOR_PANEL, fg=COLOR_MUTED,
        ).pack(side=tk.TOP, pady=(0, 6))

        # Value gauge
        gauge = tk.Frame(info, bg=COLOR_PANEL)
        gauge.pack(side=tk.TOP, fill=tk.X, padx=12, pady=(0, 10))
        tk.Label(
            gauge, text="Value est.", bg=COLOR_PANEL, fg=COLOR_MUTED,
            font=("Helvetica", 10),
        ).pack(side=tk.LEFT)
        self.gauge_canvas = tk.Canvas(
            gauge, height=14, bg=COLOR_BG, highlightthickness=0,
        )
        self.gauge_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.gauge_canvas.bind("<Configure>", lambda e: self._draw_gauge())

    def _make_nameplate(self, name: str, color: str, where: str) -> None:
        plate = tk.Frame(self.root, bg=COLOR_BG)
        plate.pack(fill=tk.X, padx=14, pady=2)
        dot = tk.Canvas(plate, width=18, height=18, bg=COLOR_BG, highlightthickness=0)
        dot.pack(side=tk.LEFT)
        dot.create_oval(2, 2, 16, 16, fill=color, outline="")
        tk.Label(
            plate, text=name, font=("Helvetica", 13, "bold"),
            bg=COLOR_BG, fg=COLOR_TEXT,
        ).pack(side=tk.LEFT, padx=(6, 0))
        tag = "Player 2 (top)" if where == "top" else "Player 1 (bottom)"
        tk.Label(
            plate, text=tag, font=("Helvetica", 10),
            bg=COLOR_BG, fg=COLOR_MUTED,
        ).pack(side=tk.RIGHT)

    def _score_str(self) -> str:
        p1w, p2w, d = self.score
        return (
            f"{self.p1.name}  {p1w}   —   "
            f"draws {d}   —   "
            f"{p2w}  {self.p2.name}"
        )

    # ----------------------------------------------------------------
    # Rendering
    # ----------------------------------------------------------------
    def _cell_xy(self, internal_r: int, internal_c: int) -> Tuple[int, int]:
        # P1 (red) is at the bottom of the screen; flip rows for display.
        visual_r = (BOARD_SIZE - 1) - internal_r
        x = MARGIN + internal_c * PITCH
        y = MARGIN + visual_r * PITCH
        return x, y

    def _wall_visual_r(self, internal_anchor_r: int) -> int:
        return (WALL_GRID - 1) - internal_anchor_r

    def _draw_h_wall(self, internal_r: int, internal_c: int, color: str) -> None:
        visual_r = self._wall_visual_r(internal_r)
        x1 = MARGIN + internal_c * PITCH
        x2 = MARGIN + (internal_c + 2) * PITCH - WT
        y1 = MARGIN + (visual_r + 1) * PITCH - WT
        self.canvas.create_rectangle(x1, y1, x2, y1 + WT, fill=color, outline="")

    def _draw_v_wall(self, internal_r: int, internal_c: int, color: str) -> None:
        visual_r = self._wall_visual_r(internal_r)
        x1 = MARGIN + (internal_c + 1) * PITCH - WT
        y1 = MARGIN + visual_r * PITCH
        y2 = MARGIN + (visual_r + 2) * PITCH - WT
        self.canvas.create_rectangle(x1, y1, x1 + WT, y2, fill=color, outline="")

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
            self._draw_h_wall(r, c, COLOR_WALL)
        for (r, c) in b.v_walls:
            self._draw_v_wall(r, c, COLOR_WALL)

        for p in (0, 1):
            r, c = b.pawns[p]
            x, y = self._cell_xy(r, c)
            color = COLOR_P1 if p == 0 else COLOR_P2
            self.canvas.create_oval(
                x + 5, y + 5, x + CELL - 5, y + CELL - 5,
                fill=color, outline="white", width=2,
            )

        # walls-left indicators
        self.canvas.create_text(
            MARGIN, CANVAS_SIZE - 6,
            anchor="sw", fill=COLOR_P1,
            text=f"walls: {b.walls_left[0]}",
            font=("Helvetica", 10, "bold"),
        )
        self.canvas.create_text(
            CANVAS_SIZE - MARGIN, 14,
            anchor="ne", fill=COLOR_P2,
            text=f"walls: {b.walls_left[1]}",
            font=("Helvetica", 10, "bold"),
        )

    def _draw_gauge(self) -> None:
        gc = self.gauge_canvas
        gc.delete("all")
        w = int(gc.winfo_width())
        h = int(gc.winfo_height())
        if w <= 2 or h <= 2:
            return
        # value in [-1, 1] from the perspective of side-to-move; we instead
        # show as P1 advantage.  When it's P2's turn, flip the sign.
        v = self.last_value
        if self.board.turn == 1:
            v = -v
        mid = w // 2
        gc.create_rectangle(0, 0, w, h, fill=COLOR_BG, outline=COLOR_MUTED)
        gc.create_line(mid, 0, mid, h, fill=COLOR_MUTED)
        if v >= 0:
            x1, x2 = mid, mid + int(mid * min(1.0, v))
            gc.create_rectangle(x1, 1, x2, h - 1, fill=COLOR_P1, outline="")
        else:
            x1, x2 = mid + int(mid * max(-1.0, v)), mid
            gc.create_rectangle(x1, 1, x2, h - 1, fill=COLOR_P2, outline="")

    # ----------------------------------------------------------------
    # Game loop (runs in a worker thread)
    # ----------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._nets:
            return
        import torch
        from quoridor.net import load_checkpoint
        self._device = torch.device("cpu")
        for side, spec in ((0, self.p1), (1, self.p2)):
            net, _ = load_checkpoint(spec.ckpt_path, map_location="cpu")
            net.to(self._device)
            net.eval()
            self._nets[side] = net
            self._caches[side] = EvalCache()

    def _pick_move(self, side: int) -> Tuple[Move, float]:
        cfg = MCTSConfig(num_simulations=self.sims, dirichlet_epsilon=0.0)
        root = search(
            self.board, self._nets[side], cfg, self._device,
            add_noise=False, cache=self._caches[side],
        )
        action = select_action(root, temperature=0.0)
        _, _, _, _, _, _, flipped = canonical_view(self.board)
        return action_to_move(action, flipped), root_value(root)

    def _move_to_notation(self, m: Move) -> str:
        col = "abcdefghi"[m.c]
        base = f"{col}{m.r + 1}"
        if m.kind == WALL_H:
            return base + "h"
        if m.kind == WALL_V:
            return base + "v"
        return base

    def _play_loop(self) -> None:
        try:
            self._ensure_loaded()
        except Exception as exc:  # noqa: BLE001
            self.root.after(0, lambda: self.status_var.set(f"load error: {exc}"))
            return

        while not self._stop:
            self.board = Board.initial()
            self._caches[0].clear()
            self._caches[1].clear()
            self.move_history.clear()
            self.last_value = 0.0
            self.root.after(0, self._refresh_ui)

            move_idx = 0
            while self.board.winner() is None and move_idx < 200 and not self._stop:
                side = self.board.turn
                name = self.p1.name if side == 0 else self.p2.name
                self.root.after(
                    0, lambda n=name: self.status_var.set(f"{n} thinking...")
                )
                t0 = time.time()
                try:
                    move, v = self._pick_move(side)
                except Exception as exc:  # noqa: BLE001
                    self.root.after(0, lambda e=exc: self.status_var.set(f"err: {e}"))
                    break
                dt = time.time() - t0

                self.board = self.board.apply(move)
                self.last_value = v
                self.last_move_str = self._move_to_notation(move)
                self.move_history.append(self.last_move_str)
                move_idx += 1
                self.root.after(0, self._refresh_ui_with_move, dt, name)

                remain = max(0.0, self.move_delay - dt)
                if remain > 0:
                    time.sleep(remain)

            winner = self.board.winner()
            if winner is None:
                self.score[2] += 1
            elif winner == 0:
                self.score[0] += 1
            else:
                self.score[1] += 1
            self.game_num += 1
            self.root.after(0, self._announce_game_end, winner)
            time.sleep(1.6)

    def _refresh_ui(self) -> None:
        self.game_var.set(f"Game #{self.game_num}")
        self.score_var.set(self._score_str())
        self._render()
        self._draw_gauge()

    def _refresh_ui_with_move(self, dt: float, name: str) -> None:
        self._render()
        self._draw_gauge()
        self.status_var.set(
            f"{name} played {self.last_move_str}  ·  {dt:.2f}s  ·  "
            f"move {len(self.move_history)}"
        )

    def _announce_game_end(self, winner: Optional[int]) -> None:
        self.score_var.set(self._score_str())
        if winner is None:
            self.status_var.set("game ended without a winner — starting next...")
        else:
            who = self.p1.name if winner == 0 else self.p2.name
            self.status_var.set(f"{who} wins game {self.game_num - 1} — next up...")

    def stop(self) -> None:
        self._stop = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1", required=True, help="checkpoint for player 1 (bottom)")
    ap.add_argument("--p2", required=True, help="checkpoint for player 2 (top)")
    ap.add_argument("--p1-name", default=None)
    ap.add_argument("--p2-name", default=None)
    ap.add_argument("--sims", type=int, default=120)
    ap.add_argument("--move-delay", type=float, default=0.35)
    ap.add_argument("--title", default="AI vs AI")
    args = ap.parse_args()

    p1 = AgentSpec(args.p1, args.p1_name or _os.path.basename(args.p1))
    p2 = AgentSpec(args.p2, args.p2_name or _os.path.basename(args.p2))

    root = tk.Tk()
    gui = SpectatorGUI(root, p1, p2, args.sims, args.move_delay, args.title)
    root.protocol("WM_DELETE_WINDOW", lambda: (gui.stop(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
