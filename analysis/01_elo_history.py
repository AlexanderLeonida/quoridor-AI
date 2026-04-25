"""Plot the gating-Elo history across all training iterations.

Reads ``checkpoints/elo.json`` and produces a chart of every version's
final Elo, with annotations for:
    - 6×64 era vs 10×128 era
    - The distillation point (boundary between the two architectures)
    - Tournament-calibrated highlights (currently the strongest)

Saved to ``analysis/plots/elo_history.png``.
"""
from __future__ import annotations

import json
import os
import re

import matplotlib.pyplot as plt


def _parse_version(label: str):
    """Return integer iteration number for 'selfplay-vNN' labels, else None."""
    m = re.match(r"selfplay-v(\d+)", label)
    return int(m.group(1)) if m else None


def main() -> None:
    with open("checkpoints/elo.json") as f:
        elos = json.load(f)

    # Map label → iteration number (only selfplay-vNN entries have one).
    versioned = []
    for label, rating in elos.items():
        n = _parse_version(label)
        if n is not None:
            versioned.append((n, label, rating))
    versioned.sort()

    iters = [n for n, _, _ in versioned]
    ratings = [r for _, _, r in versioned]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(iters, ratings, color="#5b8def", width=0.7,
           edgecolor="black", linewidth=0.4)
    ax.axhline(1000, ls="--", c="grey", lw=1, label="rating start (1000)")
    # Note: iteration numbers were reused across architectures.  After
    # distillation, the 10×128 net resumed from the 6×64 best
    # (iter_0074.pt), and global_iter continued from there.  The Elo
    # entries below v25 may correspond to early 6×64 runs OR to the
    # post-distill 10×128 era — disambiguating is impossible from
    # elo.json alone.  See ``02_calibrated_elo.py`` for cross-arch
    # comparable numbers.

    # Annotate notable peaks
    if "init" in elos:
        ax.annotate(f"init: {elos['init']:.0f}", xy=(0.5, elos["init"]),
                    xytext=(2, elos["init"] + 5), color="black", fontsize=8)
    top3 = sorted(versioned, key=lambda x: -x[2])[:3]
    for n, label, r in top3:
        ax.annotate(f"v{n}={r:.0f}", xy=(n, r), xytext=(n, r + 7),
                    ha="center", fontsize=8, weight="bold")

    ax.set_xlabel("Self-play iteration (vN)")
    ax.set_ylabel("Gating Elo")
    ax.set_title("Gating-Elo history across training iterations\n"
                 "(noisy: per-iteration K=32 updates against immediate predecessor only)")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = "analysis/plots/01_elo_history.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")
    print(f"\nTop 5 by gating Elo:")
    for n, label, r in sorted(versioned, key=lambda x: -x[2])[:5]:
        arch = "6×64" if n <= arch_split else "10×128"
        print(f"  v{n:<3} {arch:<7} {r:7.1f}")


if __name__ == "__main__":
    main()
