"""Live training dashboard.

Reads `logs/metrics.csv` (written by training/selfplay.py) and animates four
panels:

  1. Loss curves (total / policy / value)
  2. Self-play P1/P2/draw winrates per iteration
  3. Average plies (game length) per iteration
  4. Eval score across gating matches with promotion markers

Usage
-----
    python3 viz/training_dashboard.py [--metrics logs/metrics.csv] [--interval 2.0]
"""
from __future__ import annotations

import argparse
import csv
import os
import os as _os
import sys as _sys
from typing import List

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def _read_metrics(path: str) -> dict:
    cols: dict[str, List[float]] = {}
    promotions: List[int] = []
    if not os.path.exists(path):
        return {"iter": [], "data": cols, "promotions": promotions}

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k is None:
                    continue
                cols.setdefault(k, []).append(v)
    iters = [int(x) for x in cols.get("global_iter", [])]

    def _floats(key: str) -> List[float]:
        out = []
        for v in cols.get(key, []):
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(float("nan"))
        return out

    promoted = cols.get("promoted", [])
    for i, p in enumerate(promoted):
        try:
            if int(p) == 1 and i < len(iters):
                promotions.append(iters[i])
        except (TypeError, ValueError):
            pass

    return {
        "iter": iters,
        "train_loss": _floats("train_loss"),
        "policy_loss": _floats("policy_loss"),
        "value_loss": _floats("value_loss"),
        "p1_wins": _floats("sp_p1_wins"),
        "p2_wins": _floats("sp_p2_wins"),
        "draws": _floats("sp_draws"),
        "plies": _floats("sp_avg_plies"),
        "eval_score": _floats("eval_score"),
        "promotions": promotions,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="logs/metrics.csv")
    ap.add_argument("--interval", type=float, default=2.5,
                    help="seconds between refreshes")
    args = ap.parse_args()

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.canvas.manager.set_window_title("Quoridor — Live Training Dashboard")
    fig.patch.set_facecolor("#1a1a24")
    for ax in axes.flat:
        ax.set_facecolor("#252535")
        for spine in ax.spines.values():
            spine.set_color("#444466")
        ax.tick_params(colors="#aaaaaa")
        ax.xaxis.label.set_color("#cccccc")
        ax.yaxis.label.set_color("#cccccc")
        ax.title.set_color("#f1c40f")

    ax_loss, ax_winrate = axes[0]
    ax_plies, ax_eval = axes[1]

    fig.suptitle(
        "Self-Play Training — live",
        color="#f1c40f", fontsize=16, fontweight="bold",
    )

    def _draw(_frame: int) -> None:
        m = _read_metrics(args.metrics)
        iters = m["iter"]
        if not iters:
            return

        ax_loss.clear()
        ax_loss.set_title("Training Loss")
        ax_loss.set_xlabel("iteration")
        ax_loss.set_ylabel("loss")
        ax_loss.set_facecolor("#252535")
        ax_loss.plot(iters, m["train_loss"], label="total", color="#f1c40f", lw=2)
        ax_loss.plot(iters, m["policy_loss"], label="policy", color="#3498db", lw=1.5)
        ax_loss.plot(iters, m["value_loss"], label="value", color="#e74c3c", lw=1.5)
        ax_loss.legend(loc="upper right", facecolor="#1a1a24", edgecolor="#444466",
                       labelcolor="#cccccc")
        ax_loss.grid(alpha=0.18)

        ax_winrate.clear()
        ax_winrate.set_title("Self-Play Outcomes / Iteration")
        ax_winrate.set_xlabel("iteration")
        ax_winrate.set_ylabel("games")
        ax_winrate.set_facecolor("#252535")
        ax_winrate.stackplot(
            iters, m["p1_wins"], m["draws"], m["p2_wins"],
            labels=("P1 wins", "draws", "P2 wins"),
            colors=("#e74c3c", "#888888", "#3498db"),
            alpha=0.85,
        )
        ax_winrate.legend(loc="upper right", facecolor="#1a1a24",
                          edgecolor="#444466", labelcolor="#cccccc")
        ax_winrate.grid(alpha=0.18)

        ax_plies.clear()
        ax_plies.set_title("Avg Game Length (plies)")
        ax_plies.set_xlabel("iteration")
        ax_plies.set_ylabel("plies")
        ax_plies.set_facecolor("#252535")
        ax_plies.plot(iters, m["plies"], color="#1abc9c", lw=2, marker="o", ms=3)
        ax_plies.grid(alpha=0.18)

        ax_eval.clear()
        ax_eval.set_title("Gating Eval Score  (●  = promoted)")
        ax_eval.set_xlabel("iteration")
        ax_eval.set_ylabel("eval score")
        ax_eval.set_facecolor("#252535")
        ax_eval.axhline(0.5, color="#666688", lw=1, ls="--", alpha=0.6)
        ax_eval.plot(iters, m["eval_score"], color="#f39c12", lw=2)
        # promotion markers
        for it in m["promotions"]:
            try:
                idx = iters.index(it)
                ax_eval.scatter(
                    [it], [m["eval_score"][idx]],
                    s=85, color="#2ecc71", edgecolor="white", lw=1.5, zorder=5,
                )
            except (ValueError, IndexError):
                pass
        ax_eval.set_ylim(0, 1)
        ax_eval.grid(alpha=0.18)

        fig.tight_layout(rect=[0, 0, 1, 0.96])

    _draw(0)
    anim = FuncAnimation(
        fig, _draw, interval=int(args.interval * 1000),
        cache_frame_data=False,
    )
    # keep `anim` alive in case the GC tries to drop it
    fig._anim = anim  # noqa: SLF001
    plt.show()


if __name__ == "__main__":
    main()
