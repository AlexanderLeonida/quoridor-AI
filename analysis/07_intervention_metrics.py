"""Plot the effect of training interventions over time.

Reads ``logs/metrics.csv`` (per-iteration row, written by selfplay.py).
Shows how the various knobs (LR, sims, ab-mix, aux-value-weight,
PBT mutations) tracked across iterations and whether they correlate
with eval-score swings.

Saved to ``analysis/plots/07_intervention_metrics.png``.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List

import matplotlib.pyplot as plt


def _load_csv(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _f(v, default=None):
    try:
        return float(v) if v not in ("", None) else default
    except (ValueError, TypeError):
        return default


def main() -> None:
    rows = _load_csv("logs/metrics.csv")
    if not rows:
        print("  No metrics.csv yet. Run selfplay.py at least once "
              "to populate it.")
        return

    its = [int(r["global_iter"]) for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    # --- Knob settings per iteration ---
    ax = axes[0]
    sims = [_f(r.get("sims_used")) for r in rows]
    ab_games = [_f(r.get("ab_games"), 0) for r in rows]
    aux_w = [_f(r.get("aux_value_weight"), 0) for r in rows]
    if any(s is not None for s in sims):
        ax.plot(its, sims, "-o", markersize=3, label="MCTS sims")
    ax.set_ylabel("MCTS sims")
    ax.set_title("Self-play settings over time")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(its, ab_games, "-x", color="orange", markersize=4,
             label="ab-mix games / iter")
    ax2.plot(its, [a * 100 for a in aux_w], "-s", color="green", markersize=3,
             label="aux_value_weight (×100)")
    ax2.set_ylabel("ab games  /  aux weight ×100")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    # --- Loss curves ---
    ax = axes[1]
    train_l = [_f(r.get("train_loss")) for r in rows]
    val_l = [_f(r.get("best_val_loss")) for r in rows]
    if any(t is not None for t in train_l):
        ax.plot(its, train_l, "-o", markersize=3, label="train loss")
    if any(v is not None for v in val_l):
        ax.plot(its, val_l, "-x", markersize=4, label="best val loss")
    ax.set_ylabel("loss")
    ax.set_title("Training loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # --- Eval score with promote/revert markers ---
    ax = axes[2]
    eval_score = [_f(r.get("eval_score")) for r in rows]
    promoted = [r.get("promoted") == "1" for r in rows]
    reverted = [r.get("reverted_to", "") for r in rows]
    valid = [(i, s, p, rev) for i, s, p, rev in zip(its, eval_score, promoted, reverted)
             if s is not None]
    if valid:
        ax.plot([v[0] for v in valid], [v[1] for v in valid], "-",
                color="grey", alpha=0.4, lw=1)
        for i, s, p, rev in valid:
            if rev:
                ax.plot(i, s, "v", color="purple", markersize=8)
            elif p:
                ax.plot(i, s, "o", color="green", markersize=6)
            else:
                ax.plot(i, s, "x", color="red", markersize=6)
    ax.axhline(0.52, ls="--", c="g", lw=1, label="promote threshold")
    ax.set_ylim(0, 1)
    ax.set_xlabel("global iteration")
    ax.set_ylabel("eval score")
    ax.set_title("Gating eval (green=promoted, red=rejected, "
                 "purple=tournament revert)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Intervention effects: settings vs outcomes "
                 f"({len(rows)} rows from logs/metrics.csv)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = "analysis/plots/07_intervention_metrics.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")
    print(f"\n{len(rows)} metrics rows across iterations "
          f"{min(its)}–{max(its)}")


if __name__ == "__main__":
    main()
