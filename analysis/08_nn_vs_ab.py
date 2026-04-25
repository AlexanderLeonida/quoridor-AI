"""Plot NN-checkpoints vs alpha-beta strength matrix.

Reads ``analysis/bench_matrix.json`` (produced by ``bench_matrix.py``)
and renders:
    - A score heatmap (rows = checkpoints, columns = AB depth/time
      settings); cell colour = score (W+0.5D)/N
    - Per-checkpoint score curves vs AB depth — visualises how each
      net's win rate degrades as alpha-beta gets deeper

Saved to ``analysis/plots/08_nn_vs_ab.png``.
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    path = "analysis/bench_matrix.json"
    if not os.path.exists(path):
        print(f"  No {path}; run bench_matrix.py first.")
        return
    with open(path) as f:
        data = json.load(f)

    results = data["results"]
    ab_specs = data["ab_specs"]
    cols = [f"d{a['depth']}_t{a['time']}" for a in ab_specs]
    col_labels = [f"d{a['depth']}\nt{a['time']}s" for a in ab_specs]
    rows = list(results.keys())
    if not rows or not cols:
        print("  Empty results.")
        return

    # Build score matrix
    M = np.array([
        [results[r][c]["score"] for c in cols]
        for r in rows
    ])

    fig, axes = plt.subplots(1, 2, figsize=(15, max(4, 1.0 * len(rows))))

    # --- Heatmap ---
    ax = axes[0]
    im = ax.imshow(M, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xlabel("Alpha-beta setting")
    ax.set_title("NN vs alpha-beta (score = W + 0.5·D / N)")
    for i, row in enumerate(rows):
        for j, c in enumerate(cols):
            cell = results[row][c]
            n = cell["wins"] + cell["losses"] + cell["draws"]
            txt = f"{M[i, j]:.0%}\n{cell['wins']}-{cell['losses']}-{cell['draws']}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="black" if 0.3 < M[i, j] < 0.7 else "white")
    fig.colorbar(im, ax=ax, label="score")

    # --- Score curves vs AB depth ---
    ax = axes[1]
    depths = [a["depth"] for a in ab_specs]
    for i, row in enumerate(rows):
        ax.plot(depths, M[i], "-o", label=row, markersize=5)
    ax.axhline(0.5, ls="--", c="grey", lw=1)
    ax.set_xlabel("Alpha-beta search depth")
    ax.set_ylabel("NN score")
    ax.set_ylim(0, 1)
    ax.set_title("Score curves: how each net handles deeper search")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"NN strength benchmark vs alpha-beta "
        f"(N={data['config']['games_per_combo']} games per cell, "
        f"sims={data['config']['sims']})", fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = "analysis/plots/08_nn_vs_ab.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")
    print(f"\nMatrix:")
    print(f"  {'checkpoint':<22}  " + "  ".join(f"{c:<14}" for c in col_labels[:5]).replace("\n", " "))
    for row in rows:
        cells = []
        for c in cols:
            r = results[row][c]
            cells.append(f"{r['score']:.0%} ({r['wins']}-{r['losses']}-{r['draws']})")
        print(f"  {row:<22}  " + "  ".join(f"{c:<14}" for c in cells))


if __name__ == "__main__":
    main()
