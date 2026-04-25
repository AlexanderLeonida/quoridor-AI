"""Plot tournament-calibrated Elos with bootstrap confidence intervals.

Uses ``checkpoints/elo_tournament.json`` (produced by ``tournament.py``).
Shows each player's point estimate and 95% CI bar — visually the
contrast with the gating Elo (see ``01_elo_history.py``) tells the
story of how stale per-iteration ratings can be.

Saved to ``analysis/plots/calibrated_elo.png``.
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt


def main() -> None:
    path = "checkpoints/elo_tournament.json"
    if not os.path.exists(path):
        print(f"  No tournament file at {path}; run tournament.py first.")
        return
    with open(path) as f:
        data = json.load(f)

    ratings = data["ratings"]
    cis = data.get("ratings_ci", {})
    anchor = data.get("anchor", "?")

    ranked = sorted(ratings.items(), key=lambda kv: -kv[1])
    labels = [k for k, _ in ranked]
    points = [v for _, v in ranked]
    if cis:
        lows = [cis[k]["lo_95"] for k in labels]
        highs = [cis[k]["hi_95"] for k in labels]
        err = [[p - lo for p, lo in zip(points, lows)],
               [hi - p for p, hi in zip(points, highs)]]
    else:
        err = None

    fig, ax = plt.subplots(figsize=(11, 6))
    y = list(range(len(labels)))
    ax.barh(y, points, xerr=err, color="#5b8def",
            edgecolor="black", linewidth=0.4,
            error_kw={"ecolor": "#444", "capsize": 3})
    ax.axvline(1000, ls="--", c="grey", lw=1, label="anchor=1000")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(f"Tournament Elo (anchored on '{anchor}', "
                  f"95% CI from 200 bootstrap resamples)")
    ax.set_title("Calibrated round-robin Elo with bootstrap CIs\n"
                 "(tight CIs = clear ranking, wide CIs = noisy)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    out = "analysis/plots/02_calibrated_elo.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")
    print(f"\nTournament players ({len(labels)}):")
    for i, (label, r) in enumerate(ranked, 1):
        ci = cis.get(label, {})
        if ci:
            print(f"  {i:>2}. {label:<28s} {r:7.1f}   "
                  f"[{ci['lo_95']:7.1f}, {ci['hi_95']:7.1f}]")
        else:
            print(f"  {i:>2}. {label:<28s} {r:7.1f}")


if __name__ == "__main__":
    main()
