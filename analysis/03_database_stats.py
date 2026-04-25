"""Database stats per model_version: game length, win distribution, draw rate.

Pulls from all three game DBs (``data/quoridor.db``, ``quoridor_v2.db``,
``quoridor_v3.db``) and produces:
    - Game-length boxplot per version (showing how the engine learned
      to convert short, decisive games over time)
    - Stacked bar of P1 wins / P2 wins / draws per version
    - Cumulative game count over time, by DB

Saved as three PNGs under ``analysis/plots/``.
"""
from __future__ import annotations

import os
import re
import sqlite3

import matplotlib.pyplot as plt
import numpy as np


DBS = [
    ("v1", "data/quoridor.db"),
    ("v2", "data/quoridor_v2.db"),
    ("v3", "data/quoridor_v3.db"),
]


def _version_num(label: str):
    m = re.match(r"selfplay-v(\d+)", str(label))
    return int(m.group(1)) if m else None


def _per_version(db_path: str):
    """Return list of (version_num, n_games, avg_plies, p1_wins, p2_wins, draws,
    plies_p25, plies_p50, plies_p75)."""
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT model_version, num_plies, winner FROM games "
            "WHERE p1_source='selfplay_nn' AND p2_source='selfplay_nn'"
        ).fetchall()
    finally:
        conn.close()
    by_v: dict = {}
    for mv, plies, winner in rows:
        v = _version_num(mv)
        if v is None:
            continue
        d = by_v.setdefault(v, {"plies": [], "p1": 0, "p2": 0, "draws": 0})
        if plies is not None:
            d["plies"].append(plies)
        if winner == 0:
            d["p1"] += 1
        elif winner == 1:
            d["p2"] += 1
        else:
            d["draws"] += 1
    out = []
    for v, d in sorted(by_v.items()):
        plies = np.array(d["plies"]) if d["plies"] else np.array([0])
        out.append((
            v, len(d["plies"]), float(plies.mean()),
            d["p1"], d["p2"], d["draws"],
            float(np.percentile(plies, 25)),
            float(np.percentile(plies, 50)),
            float(np.percentile(plies, 75)),
        ))
    return out


def _outcome_stack(per_version, ax, title: str):
    if not per_version:
        ax.text(0.5, 0.5, "(empty)", ha="center", va="center")
        ax.set_title(title)
        return
    vs = [r[0] for r in per_version]
    p1 = np.array([r[3] for r in per_version])
    p2 = np.array([r[4] for r in per_version])
    dr = np.array([r[5] for r in per_version])
    total = (p1 + p2 + dr).clip(min=1)
    p1_pct = p1 / total * 100
    p2_pct = p2 / total * 100
    dr_pct = dr / total * 100

    ax.bar(vs, p1_pct, color="#d24e4e", label="P1 wins")
    ax.bar(vs, p2_pct, bottom=p1_pct, color="#5b8def", label="P2 wins")
    ax.bar(vs, dr_pct, bottom=p1_pct + p2_pct, color="#999", label="draws")
    ax.set_ylim(0, 100)
    ax.set_xlabel("model version (vN)")
    ax.set_ylabel("% of games")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def _length_boxplot(per_version, ax, title: str):
    """Plot p25/median/p75 per version as min-max bands."""
    if not per_version:
        ax.text(0.5, 0.5, "(empty)", ha="center", va="center")
        ax.set_title(title)
        return
    vs = [r[0] for r in per_version]
    p25 = [r[6] for r in per_version]
    p50 = [r[7] for r in per_version]
    p75 = [r[8] for r in per_version]
    ax.fill_between(vs, p25, p75, alpha=0.3, color="#5b8def",
                    label="IQR (p25–p75)")
    ax.plot(vs, p50, "-o", color="#1a4dab", markersize=3,
            label="median plies")
    ax.set_xlabel("model version (vN)")
    ax.set_ylabel("plies per game")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def main() -> None:
    fig, axes = plt.subplots(2, len(DBS), figsize=(6 * len(DBS), 9))

    for col, (label, path) in enumerate(DBS):
        per_v = _per_version(path)
        n = sum(r[1] for r in per_v)
        title_suffix = f"{label} ({n} self-play games)"
        _outcome_stack(per_v, axes[0][col], f"Outcome distribution — {title_suffix}")
        _length_boxplot(per_v, axes[1][col], f"Game length — {title_suffix}")

    fig.suptitle("Self-play game statistics across DBs and versions",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = "analysis/plots/03_database_stats.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")

    # Also print a per-DB summary
    for label, path in DBS:
        per_v = _per_version(path)
        if not per_v:
            continue
        total_games = sum(r[1] for r in per_v)
        total_draws = sum(r[5] for r in per_v)
        avg_len = np.average([r[2] for r in per_v],
                             weights=[r[1] for r in per_v])
        first_v = min(r[0] for r in per_v)
        last_v = max(r[0] for r in per_v)
        print(f"\n{label}: {total_games} games, "
              f"{total_draws/max(1,total_games):.0%} draws, "
              f"avg {avg_len:.1f} plies, "
              f"versions v{first_v}-v{last_v}")


if __name__ == "__main__":
    main()
