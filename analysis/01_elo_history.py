"""Strength-over-time chart spanning the entire project history.

X axis: real timestamp (when the model was active, from the DB).
Y axis: Elo (calibrated where available, gating Elo otherwise).

Every model version that appears in any DB gets a dot. Tournament-
calibrated ratings are highlighted with a star (those are the "real"
strength numbers — see PROCESS.md §22 for why gating Elos are noisy).
A running-best line tracks the highest gating Elo over time, plus a
chained-from-metrics-csv estimate for the latest run.

Saved to ``analysis/plots/01_elo_history.png``.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


# ----- data loaders ---------------------------------------------------

def _version_first_seen() -> Dict[str, datetime]:
    """Map model_version → earliest created_at across all DBs."""
    out: Dict[str, datetime] = {}
    for db in ["data/quoridor.db", "data/quoridor_v2.db", "data/quoridor_v3.db"]:
        if not os.path.exists(db):
            continue
        try:
            conn = sqlite3.connect(db)
            for mv, ts in conn.execute(
                "SELECT model_version, MIN(created_at) FROM games "
                "WHERE p1_source='selfplay_nn' AND model_version IS NOT NULL "
                "GROUP BY model_version"
            ):
                if not mv or not ts:
                    continue
                try:
                    dt = datetime.fromisoformat(ts)
                except Exception:
                    continue
                if mv not in out or dt < out[mv]:
                    out[mv] = dt
            conn.close()
        except Exception:
            pass
    return out


def _load_gating_elos() -> Dict[str, float]:
    if not os.path.exists("checkpoints/elo.json"):
        return {}
    with open("checkpoints/elo.json") as f:
        return json.load(f)


def _load_calibrated_elos() -> Dict[str, float]:
    """Pull every tournament-calibrated Elo we have (latest tournament wins)."""
    out: Dict[str, float] = {}
    for p in [
        "checkpoints/elo_tournament.json",
        "checkpoints/_tournament_current.json",
    ] + sorted(glob.glob("checkpoints/elo_tournament_*.json")):
        if not os.path.exists(p):
            continue
        try:
            with open(p) as f:
                data = json.load(f)
            for label, elo in data.get("ratings", {}).items():
                # Strip descriptive suffix: 'v34_run2peak' → 'selfplay-v34'.
                m = re.match(r"v(\d+)", label)
                key = f"selfplay-v{m.group(1)}" if m else label
                out[key] = float(elo)
        except Exception:
            pass
    return out


def _load_metrics_csv(path="logs/metrics.csv") -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


# ----- chained-Elo trajectory from metrics.csv ------------------------

def _gate_score_to_elo_delta(score: float) -> float:
    score = max(0.001, min(0.999, score))
    return max(-400.0, min(400.0, 400.0 * math.log10(score / (1 - score))))


def _chained_trajectory(metrics_rows, seed_elo):
    """Walk metrics.csv in order and chain Elos through promotions."""
    out = []
    cur = seed_elo
    for row in metrics_rows:
        try:
            ts = datetime.fromisoformat(row["timestamp"])
        except Exception:
            continue
        rev = row.get("reverted_to") or ""
        if rev:
            out.append((ts, cur, "revert"))
            continue
        try:
            score = float(row.get("eval_score") or "")
        except (TypeError, ValueError):
            continue
        delta = _gate_score_to_elo_delta(score)
        attempted = cur + delta
        if row.get("promoted") == "1":
            cur = attempted
            out.append((ts, cur, "promoted"))
        else:
            out.append((ts, attempted, "rejected"))
    return out


# ----- plot -----------------------------------------------------------

def main() -> None:
    first_seen = _version_first_seen()
    gating = _load_gating_elos()
    calibrated = _load_calibrated_elos()
    metrics = _load_metrics_csv()

    # Override timestamps for versions in metrics.csv — this is the
    # current run's authoritative log, with timestamps that disambiguate
    # the version-label-reuse issue (same "selfplay-vN" label can exist
    # in multiple DBs from different sessions; first_seen would map it
    # to the *oldest* DB, hiding the current run on the x-axis).
    metrics_ts: Dict[str, datetime] = {}
    for r in metrics:
        v = r.get("version")
        if not v:
            continue
        try:
            ts = datetime.fromisoformat(r["timestamp"])
        except Exception:
            continue
        # Latest occurrence of this version in metrics.csv wins.
        metrics_ts[v] = ts

    # Combine: every model version we know about from either elo.json
    # or any DB, with whichever Elo we have.  Prefer metrics.csv
    # timestamps when available so the current run's points land on the
    # current date instead of being dragged to an old DB's date.
    all_labels = set(gating) | set(calibrated) | set(first_seen) | set(metrics_ts)
    points: List[Tuple[datetime, str, float, bool]] = []  # (ts, label, elo, calibrated?)
    for label in all_labels:
        ts = metrics_ts.get(label) or first_seen.get(label)
        if ts is None:
            continue  # no timestamp → can't place on x-axis
        if label in calibrated:
            points.append((ts, label, calibrated[label], True))
        elif label in gating:
            points.append((ts, label, gating[label], False))
    points.sort()

    fig, ax = plt.subplots(figsize=(14, 7))

    # --- All historical model versions as scatter dots ---
    if points:
        gx = [p[0] for p in points if not p[3]]
        gy = [p[2] for p in points if not p[3]]
        cx = [p[0] for p in points if p[3]]
        cy = [p[2] for p in points if p[3]]
        if gx:
            ax.scatter(gx, gy, s=22, c="#5b8def", alpha=0.55,
                       edgecolor="white", linewidth=0.4,
                       label=f"gating Elo ({len(gx)} versions)")
        if cx:
            ax.scatter(cx, cy, s=110, c="orange", marker="*",
                       edgecolor="black", linewidth=0.5, zorder=5,
                       label=f"tournament-calibrated ({len(cx)} versions)")

    # --- Running best gating Elo line (all-time, for context) ---
    if points:
        run_best = []
        cur_max = -math.inf
        for ts, _, elo, _ in points:
            cur_max = max(cur_max, elo)
            run_best.append((ts, cur_max))
        ax.plot([t for t, _ in run_best], [e for _, e in run_best],
                "-", color="#1a4dab", lw=1.5, alpha=0.7,
                label="running max Elo (all-time)")

    # --- Chained-Elo trajectory from metrics.csv (latest run) ---
    if metrics:
        seed = calibrated.get("warmstart_10x128", 1000.0)
        traj = _chained_trajectory(metrics, seed_elo=seed)
        if traj:
            tx = [t for t, _, _ in traj]
            ty = [e for _, e, _ in traj]
            ax.plot(tx, ty, "-", color="#d2691e", lw=2,
                    label="chained Elo (latest run)")
            for ts, e, kind in traj:
                if kind == "promoted":
                    ax.plot(ts, e, "o", color="green", markersize=8,
                            markeredgecolor="black", zorder=6)
                elif kind == "revert":
                    ax.plot(ts, e, "v", color="purple", markersize=10,
                            markeredgecolor="black", zorder=6)

    # Annotate the top-Elo points (most useful labels)
    if points:
        # Top 3 calibrated + top 3 gating
        top_cal = sorted([p for p in points if p[3]],
                         key=lambda x: -x[2])[:3]
        top_gate = sorted([p for p in points if not p[3]],
                          key=lambda x: -x[2])[:3]
        for ts, label, elo, _ in top_cal + top_gate:
            ax.annotate(f"{label}={elo:.0f}",
                        xy=(ts, elo), xytext=(5, 5),
                        textcoords="offset points", fontsize=8)

    ax.axhline(1000, ls="--", c="grey", lw=1, alpha=0.5)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate()
    ax.set_xlabel("time")
    ax.set_ylabel("Elo")
    ax.set_title(f"Model strength over time — {len(points)} model versions, "
                 f"{sum(1 for p in points if p[3])} tournament-calibrated, "
                 f"latest run chained from gate scores")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = "analysis/plots/01_elo_history.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")
    print(f"\n{len(points)} versions plotted "
          f"({sum(1 for p in points if p[3])} calibrated, "
          f"{sum(1 for p in points if not p[3])} gating-only)")


if __name__ == "__main__":
    main()
