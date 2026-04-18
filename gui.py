"""Tkinter GUI for Quoridor.

Launch:  python3 gui.py

Controls
--------
* Click a highlighted cell to move your pawn.
* Click in the gap between two cells to place a wall there (orientation
  is inferred from which gap you click — horizontal gap -> horizontal
  wall; vertical gap -> vertical wall). Hover anywhere to preview.
* The board is flipped so you are always at the bottom of the screen.
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox
from typing import Optional, Tuple

from quoridor import (
    BOARD_SIZE,
    Board,
    GameRecorder,
    MOVE_PAWN,
    Move,
    WALL_GRID,
    WALL_H,
    WALL_V,
    find_best_move,
)

# --- Layout constants ---------------------------------------------------
CELL = 58
WT = 12                         # wall thickness (== gap between cells)
PITCH = CELL + WT
MARGIN = 34
BOARD_PX = 9 * CELL + 8 * WT    # = 9 * PITCH - WT
CANVAS_SIZE = BOARD_PX + 2 * MARGIN

# --- Colors -------------------------------------------------------------
COLOR_BG = "#f4eedd"
COLOR_CELL = "#e8dfc8"
COLOR_CELL_HL = "#b9d9f5"
COLOR_LEGAL_DOT = "#6fa7d6"
COLOR_WALL = "#5a3413"
COLOR_PREVIEW_OK = "#c69b6d"
COLOR_PREVIEW_BAD = "#c06060"
COLOR_P1 = "#d62828"
COLOR_P2 = "#1f6fb8"
COLOR_TEXT = "#2d2418"
COLOR_LABEL = "#7a6744"
COLOR_GOAL_P1 = "#f2d0d0"       # subtle tint for P1's goal row
COLOR_GOAL_P2 = "#d0dcf2"


# =======================================================================
class QuoridorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Quoridor")
        root.configure(bg=COLOR_BG)
        root.resizable(False, False)

        self.board: Optional[Board] = None
        self.human_player: Optional[int] = None
        self.ai_busy: bool = False
        self.preview: Optional[Tuple[str, Move, bool]] = None  # (kind, move, legal?)
        self.ai_depth: int = 20
        self.ai_time: float = 30.0
        self.recorder: Optional[GameRecorder] = None

        self._build_ui()
        self.root.after(50, self._show_start_menu)

    # -------------------------------------------------------------------
    # UI construction
    # -------------------------------------------------------------------
    def _build_ui(self) -> None:
        top = tk.Frame(self.root, bg=COLOR_BG)
        top.pack(side=tk.TOP, fill=tk.X, padx=12, pady=(12, 4))

        self.status_var = tk.StringVar(value="")
        tk.Label(
            top, textvariable=self.status_var, font=("Helvetica", 14, "bold"),
            bg=COLOR_BG, fg=COLOR_TEXT,
        ).pack(side=tk.LEFT)

        self.walls_var = tk.StringVar(value="")
        tk.Label(
            top, textvariable=self.walls_var, font=("Helvetica", 11),
            bg=COLOR_BG, fg=COLOR_TEXT,
        ).pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(
            self.root, width=CANVAS_SIZE, height=CANVAS_SIZE,
            bg=COLOR_BG, highlightthickness=0,
        )
        self.canvas.pack(padx=12, pady=4)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", lambda e: self._set_preview(None))
        self.canvas.bind("<Button-1>", self._on_click)

        bot = tk.Frame(self.root, bg=COLOR_BG)
        bot.pack(side=tk.BOTTOM, fill=tk.X, padx=12, pady=(4, 12))

        tk.Button(bot, text="New Game", command=self._show_start_menu).pack(side=tk.LEFT)

        tk.Label(bot, text="Difficulty:", bg=COLOR_BG, fg=COLOR_TEXT).pack(
            side=tk.LEFT, padx=(14, 4)
        )
        self.difficulty_var = tk.StringVar(value="Hard")
        tk.OptionMenu(
            bot, self.difficulty_var, "Easy", "Medium", "Hard",
            command=self._on_difficulty_change,
        ).pack(side=tk.LEFT)

        self.msg_var = tk.StringVar(value="")
        tk.Label(
            bot, textvariable=self.msg_var, bg=COLOR_BG, fg="#b04040",
            font=("Helvetica", 11),
        ).pack(side=tk.RIGHT)

    def _on_difficulty_change(self, val: str) -> None:
        # `ai_depth` is a ceiling for iterative deepening; the real governor
        # is `ai_time` -- the search stops at whatever depth it reaches before
        # the time budget expires.
        if val == "Easy":
            self.ai_depth, self.ai_time = 3, 2.0
        elif val == "Medium":
            self.ai_depth, self.ai_time = 20, 8.0
        else:  # Hard
            self.ai_depth, self.ai_time = 20, 30.0

    # -------------------------------------------------------------------
    # Start menu / game lifecycle
    # -------------------------------------------------------------------
    def _show_start_menu(self) -> None:
        if self.ai_busy:
            return
        dlg = tk.Toplevel(self.root)
        dlg.title("Choose side")
        dlg.configure(bg=COLOR_BG)
        dlg.transient(self.root)
        dlg.resizable(False, False)
        dlg.protocol("WM_DELETE_WINDOW", lambda: None)  # force a choice

        tk.Label(
            dlg, text="Choose your side", font=("Helvetica", 16, "bold"),
            bg=COLOR_BG, fg=COLOR_TEXT,
        ).pack(pady=(18, 4), padx=40)
        tk.Label(
            dlg, text="Player 1 always moves first.", bg=COLOR_BG, fg=COLOR_TEXT,
        ).pack(pady=(0, 12))

        frame = tk.Frame(dlg, bg=COLOR_BG)
        frame.pack(padx=20, pady=(0, 18))

        def pick(player_idx: int) -> None:
            dlg.grab_release()
            dlg.destroy()
            self._start_game(player_idx)

        tk.Button(
            frame, text="Player 1  (Red)\nmoves first\nstarts at e1",
            font=("Helvetica", 12, "bold"), fg="white", bg=COLOR_P1,
            activebackground=COLOR_P1, activeforeground="white",
            width=18, height=4, command=lambda: pick(0),
        ).pack(side=tk.LEFT, padx=10)

        tk.Button(
            frame, text="Player 2  (Blue)\nmoves second\nstarts at e9",
            font=("Helvetica", 12, "bold"), fg="white", bg=COLOR_P2,
            activebackground=COLOR_P2, activeforeground="white",
            width=18, height=4, command=lambda: pick(1),
        ).pack(side=tk.LEFT, padx=10)

        # Center over root.
        self.root.update_idletasks()
        rx = self.root.winfo_rootx()
        ry = self.root.winfo_rooty()
        rw = self.root.winfo_width()
        rh = self.root.winfo_height()
        dlg.update_idletasks()
        dw = dlg.winfo_width()
        dh = dlg.winfo_height()
        dlg.geometry(f"+{rx + (rw - dw) // 2}+{ry + (rh - dh) // 2}")
        dlg.grab_set()
        dlg.focus_set()

    def _start_game(self, human_player: int) -> None:
        # If an earlier game is still in progress, save it as aborted
        # (winner=None) so we don't lose the moves played so far.
        self._finalize_recorder(winner=None)

        self.human_player = human_player
        self.board = Board.initial()
        self.ai_busy = False
        self.preview = None
        self.msg_var.set("")

        p1_source = "human" if human_player == 0 else "alphabeta"
        p2_source = "human" if human_player == 1 else "alphabeta"
        self.recorder = GameRecorder(
            p1_source=p1_source,
            p2_source=p2_source,
            p1_time_limit=None if human_player == 0 else self.ai_time,
            p2_time_limit=None if human_player == 1 else self.ai_time,
            model_version="alphabeta-v2",
            notes="gui",
        )
        self.recorder.start()

        self._render()
        # If the human is P2, the AI (P1) moves first.
        self.root.after(120, self._maybe_ai_move)

    def _finalize_recorder(self, winner: Optional[int]) -> None:
        if self.recorder is None:
            return
        try:
            self.recorder.finish(winner=winner)
        except Exception:  # noqa: BLE001
            pass
        self.recorder = None

    # -------------------------------------------------------------------
    # Coordinate mapping (internal <-> visual rows). Columns never flip.
    # The human always appears at the top of the screen; the AI at bottom.
    # -------------------------------------------------------------------
    def _v_row(self, internal_r: int) -> int:
        return (BOARD_SIZE - 1 - internal_r) if self.human_player == 1 else internal_r

    def _i_row(self, visual_r: int) -> int:
        return (BOARD_SIZE - 1 - visual_r) if self.human_player == 1 else visual_r

    def _v_wall_r(self, internal_anchor_r: int) -> int:
        # Wall anchor r in [0,7]. When flipped, a wall between internal rows r
        # and r+1 is visually between visual rows (7-r) and (8-r) = (7-r)+1,
        # so its visual anchor (smaller visual row) is 7-r.
        return (WALL_GRID - 1 - internal_anchor_r) if self.human_player == 1 else internal_anchor_r

    def _i_wall_r(self, visual_anchor_r: int) -> int:
        return (WALL_GRID - 1 - visual_anchor_r) if self.human_player == 1 else visual_anchor_r

    def _cell_xy(self, internal_r: int, internal_c: int) -> Tuple[int, int]:
        x = MARGIN + internal_c * PITCH
        y = MARGIN + self._v_row(internal_r) * PITCH
        return x, y

    # -------------------------------------------------------------------
    # Hit-test on the canvas
    # -------------------------------------------------------------------
    def _hit_test(self, x: int, y: int) -> Optional[Move]:
        nx = x - MARGIN
        ny = y - MARGIN
        if nx < 0 or ny < 0 or nx >= 9 * PITCH - WT or ny >= 9 * PITCH - WT:
            return None
        col = int(nx // PITCH)
        vrow = int(ny // PITCH)
        mod_x = nx - col * PITCH
        mod_y = ny - vrow * PITCH
        in_cell_x = mod_x < CELL
        in_cell_y = mod_y < CELL

        if in_cell_x and in_cell_y:
            return Move(MOVE_PAWN, self._i_row(vrow), col)

        if in_cell_x and not in_cell_y:
            # Horizontal gap between visual rows vrow and vrow+1.
            if vrow >= WALL_GRID:
                return None
            internal_r = self._i_wall_r(vrow)
            anchor_c = col - 1 if mod_x < CELL / 2 else col
            anchor_c = max(0, min(WALL_GRID - 1, anchor_c))
            return Move(WALL_H, internal_r, anchor_c)

        if not in_cell_x and in_cell_y:
            # Vertical gap between visual cols col and col+1.
            if col >= WALL_GRID:
                return None
            visual_anchor_r = vrow - 1 if mod_y < CELL / 2 else vrow
            visual_anchor_r = max(0, min(WALL_GRID - 1, visual_anchor_r))
            internal_r = self._i_wall_r(visual_anchor_r)
            return Move(WALL_V, internal_r, col)

        return None

    # -------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------
    def _render(self) -> None:
        self.canvas.delete("all")
        if self.board is None:
            return
        b = self.board

        # Goal-row tints for orientation.
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

        # Column labels (a..i) top and bottom.
        for c in range(BOARD_SIZE):
            lx = MARGIN + c * PITCH + CELL / 2
            label = chr(ord("a") + c)
            self.canvas.create_text(
                lx, MARGIN / 2, text=label, font=("Helvetica", 11, "bold"),
                fill=COLOR_LABEL,
            )
            self.canvas.create_text(
                lx, MARGIN + BOARD_PX + MARGIN / 2, text=label,
                font=("Helvetica", 11, "bold"), fill=COLOR_LABEL,
            )
        # Row labels (1..9) left and right.
        for r in range(BOARD_SIZE):
            ly = MARGIN + self._v_row(r) * PITCH + CELL / 2
            label = str(r + 1)
            self.canvas.create_text(
                MARGIN / 2, ly, text=label, font=("Helvetica", 11, "bold"),
                fill=COLOR_LABEL,
            )
            self.canvas.create_text(
                MARGIN + BOARD_PX + MARGIN / 2, ly, text=label,
                font=("Helvetica", 11, "bold"), fill=COLOR_LABEL,
            )

        # Legal-move dots on the human's turn.
        if self._is_human_turn():
            for (r, c) in b.pawn_moves(self.human_player):
                x, y = self._cell_xy(r, c)
                self.canvas.create_oval(
                    x + CELL / 2 - 7, y + CELL / 2 - 7,
                    x + CELL / 2 + 7, y + CELL / 2 + 7,
                    fill=COLOR_LEGAL_DOT, outline="",
                )

        # Placed walls.
        for (r, c) in b.h_walls:
            self._draw_h_wall(r, c, COLOR_WALL)
        for (r, c) in b.v_walls:
            self._draw_v_wall(r, c, COLOR_WALL)

        # Pawns.
        for p in (0, 1):
            r, c = b.pawns[p]
            x, y = self._cell_xy(r, c)
            color = COLOR_P1 if p == 0 else COLOR_P2
            self.canvas.create_oval(
                x + 7, y + 7, x + CELL - 7, y + CELL - 7,
                fill=color, outline="white", width=3,
            )
            self.canvas.create_text(
                x + CELL / 2, y + CELL / 2, text=("R" if p == 0 else "B"),
                font=("Helvetica", 18, "bold"), fill="white",
            )

        # Hover preview.
        if self.preview is not None and self._is_human_turn():
            _kind, move, legal = self.preview
            color = COLOR_PREVIEW_OK if legal else COLOR_PREVIEW_BAD
            if move.kind == MOVE_PAWN:
                x, y = self._cell_xy(move.r, move.c)
                self.canvas.create_rectangle(
                    x + 2, y + 2, x + CELL - 2, y + CELL - 2,
                    outline=color, width=3,
                )
            elif move.kind == WALL_H:
                self._draw_h_wall(move.r, move.c, color)
            elif move.kind == WALL_V:
                self._draw_v_wall(move.r, move.c, color)

        self._update_status()

    def _draw_h_wall(self, internal_r: int, internal_c: int, color: str) -> None:
        visual_r = self._v_wall_r(internal_r)
        x1 = MARGIN + internal_c * PITCH
        x2 = MARGIN + (internal_c + 2) * PITCH - WT
        y1 = MARGIN + (visual_r + 1) * PITCH - WT
        y2 = y1 + WT
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

    def _draw_v_wall(self, internal_r: int, internal_c: int, color: str) -> None:
        visual_r = self._v_wall_r(internal_r)
        x1 = MARGIN + (internal_c + 1) * PITCH - WT
        x2 = x1 + WT
        y1 = MARGIN + visual_r * PITCH
        y2 = MARGIN + (visual_r + 2) * PITCH - WT
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

    def _update_status(self) -> None:
        if self.board is None:
            return
        winner = self.board.winner()
        if winner is not None:
            label = "Player 1 (Red)" if winner == 0 else "Player 2 (Blue)"
            self.status_var.set(f"{label} wins!")
        elif self.ai_busy:
            self.status_var.set("AI thinking...")
        else:
            turn_label = "Player 1 (Red)" if self.board.turn == 0 else "Player 2 (Blue)"
            if self.human_player is not None and self.board.turn == self.human_player:
                self.status_var.set(f"Your turn — {turn_label}")
            else:
                self.status_var.set(f"{turn_label} to move")
        self.walls_var.set(
            f"Red walls: {self.board.walls_left[0]}    "
            f"Blue walls: {self.board.walls_left[1]}"
        )

    # -------------------------------------------------------------------
    # Input
    # -------------------------------------------------------------------
    def _is_human_turn(self) -> bool:
        return (
            self.board is not None
            and self.board.winner() is None
            and self.human_player is not None
            and self.board.turn == self.human_player
            and not self.ai_busy
        )

    def _on_motion(self, event: tk.Event) -> None:
        if not self._is_human_turn():
            self._set_preview(None)
            return
        move = self._hit_test(event.x, event.y)
        if move is None:
            self._set_preview(None)
            return
        legal = self.board.is_legal(move)
        # Skip previewing illegal pawn moves — they're just dead space.
        if move.kind == MOVE_PAWN and not legal:
            self._set_preview(None)
            return
        self._set_preview(("move", move, legal))

    def _set_preview(self, p: Optional[Tuple[str, Move, bool]]) -> None:
        if p == self.preview:
            return
        self.preview = p
        self._render()

    def _on_click(self, event: tk.Event) -> None:
        if not self._is_human_turn():
            return
        move = self._hit_test(event.x, event.y)
        if move is None:
            return
        if not self.board.is_legal(move):
            self.msg_var.set("Illegal move.")
            return
        self.msg_var.set("")
        self._apply_move(move)

    # -------------------------------------------------------------------
    # Applying moves & AI
    # -------------------------------------------------------------------
    def _apply_move(self, move: Move) -> None:
        if self.recorder is not None:
            self.recorder.record(move)
        self.board = self.board.apply(move)
        self.preview = None
        self._render()
        if self.board.winner() is not None:
            self._end_game()
            return
        self.root.after(120, self._maybe_ai_move)

    def _maybe_ai_move(self) -> None:
        if self.board is None or self.board.winner() is not None:
            return
        if self.human_player is not None and self.board.turn == self.human_player:
            return
        self._ai_move()

    def _ai_move(self) -> None:
        self.ai_busy = True
        self.msg_var.set("")
        self._update_status()
        self.root.update_idletasks()

        board_copy = self.board.clone()
        depth = self.ai_depth
        time_limit = self.ai_time

        def worker() -> None:
            try:
                mv = find_best_move(board_copy, max_depth=depth, time_limit=time_limit)
                self.root.after(0, lambda: self._ai_done(mv))
            except Exception as exc:  # noqa: BLE001
                err = str(exc)
                self.root.after(0, lambda: self._ai_error(err))

        threading.Thread(target=worker, daemon=True).start()

    def _ai_done(self, move: Optional[Move]) -> None:
        self.ai_busy = False
        if move is None:
            self._update_status()
            return
        self._apply_move(move)

    def _ai_error(self, err: str) -> None:
        self.ai_busy = False
        self._update_status()
        messagebox.showerror("AI error", err)

    def _end_game(self) -> None:
        self._update_status()
        winner = self.board.winner()
        self._finalize_recorder(winner=winner)
        label = "Player 1 (Red)" if winner == 0 else "Player 2 (Blue)"
        messagebox.showinfo("Game over", f"{label} wins!")


# =======================================================================
def main() -> None:
    root = tk.Tk()
    QuoridorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
