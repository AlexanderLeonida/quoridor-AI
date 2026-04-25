"""Plot activity over time: games played per day across all DBs.

Useful for showing the cadence of training — bursts when scripts ran,
gaps when not. Also shows when the architecture upgrade happened.

Saved to ``analysis/plots/06_activity_timeline.png``.
"""
from __future__ import annotations

import os
import sqlite3
from collections import defaultdict
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


DBS = [
    ("v1", "data/quoridor.db", "#5b8def"),
    ("v2", "data/quoridor_v2.db", "#5bbf6e"),
    ("v3", "data/quoridor_v3.db", "#e07b39"),
]


def _per_day(db_path: str):
    if not os.path.exists(db_path):
        return {}
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT created_at FROM games "
            "WHERE p1_source='selfplay_nn' AND p2_source='selfplay_nn'"
        ).fetchall()
    finally:
        conn.close()
    by_day: dict = defaultdict(int)
    for (ts,) in rows:
        try:
            d = datetime.fromisoformat(ts).date()
            by_day[d] += 1
        except Exception:
            pass
    return dict(by_day)


def main() -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    all_days: set = set()
    series = []

    for label, path, color in DBS:
        per = _per_day(path)
        if not per:
            continue
        all_days.update(per.keys())
        days = sorted(per.keys())
        counts = [per[d] for d in days]
        ax.bar(days, counts, color=color, alpha=0.75,
               label=f"{label} ({sum(counts)} games)",
               edgecolor="black", linewidth=0.3, width=0.8)
        series.append((label, sum(counts)))

    if not all_days:
        print("  No data.")
        return

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_ylabel("self-play games per day")
    ax.set_title("Training activity timeline (self-play games per day, "
                 "stacked by DB)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    out = "analysis/plots/06_activity_timeline.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")

    print(f"\nGames per DB:")
    for label, n in series:
        print(f"  {label}: {n:,}")
    if all_days:
        print(f"  Active span: {min(all_days)} → {max(all_days)}")


if __name__ == "__main__":
    main()
