"""Run all analyses and produce a single REPORT.md summary.

Usage:
    python3 analysis/make_report.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime


SCRIPTS = [
    "analysis/00_summary.py",
    "analysis/01_elo_history.py",
    "analysis/02_calibrated_elo.py",
    "analysis/03_database_stats.py",
    "analysis/04_training_progress.py",
    "analysis/05_architecture_comparison.py",
    "analysis/06_activity_timeline.py",
    "analysis/07_intervention_metrics.py",
    "analysis/08_nn_vs_ab.py",
]


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    os.makedirs("analysis/plots", exist_ok=True)

    log_lines = []
    log_lines.append(f"# Quoridor AI — Analysis Report")
    log_lines.append(f"\nGenerated: {datetime.now().isoformat(timespec='seconds')}")
    log_lines.append("")
    log_lines.append("## Current training recipe")
    log_lines.append("")
    log_lines.append(
        "Each iteration runs (1) self-play, (2) training, (3) gating, "
        "(4) periodic round-robin tournament. Six knobs do most of the "
        "work. **High-sim MCTS** (sims=600) gives the net a teacher "
        "stronger than itself. **Alpha-beta mix** (20% of self-play "
        "games) breaks the self-imitation loop with a fundamentally "
        "different evaluator. **Auxiliary path-diff value blend** "
        "(α=0.4) densifies value supervision from one outcome label "
        "per game to ~30 per game. **Hard-example mining** on revert "
        "concentrates supervision on positions where the net previously "
        "made the wrong call. **Lightweight PBT** (sibling every 6 "
        "iters) explores hparam neighborhoods cheaply. **Round-robin "
        "tournament every 5 iters** with held-out anchors "
        "(warmstart, iter_0034, iter_0040) catches drift the local "
        "gating signal can't see, with revert-to-champion semantics."
    )
    log_lines.append("")
    log_lines.append(
        "See `PROCESS.md` §34 for the full launch command and a "
        "per-knob explanation of why each one helps."
    )
    log_lines.append("")
    log_lines.append("---\n")

    for script in SCRIPTS:
        title = os.path.basename(script).replace(".py", "")
        log_lines.append(f"\n## {title}\n")
        try:
            out = subprocess.run(
                [sys.executable, script],
                capture_output=True, text=True, check=True,
            )
            log_lines.append("```")
            log_lines.append(out.stdout.strip() or "(no stdout)")
            log_lines.append("```")
            png = f"analysis/plots/{title}.png"
            if os.path.exists(png):
                log_lines.append(f"\n![{title}]({png})\n")
        except subprocess.CalledProcessError as e:
            log_lines.append("```")
            log_lines.append(f"FAILED: {e}")
            log_lines.append(e.stderr or "")
            log_lines.append("```")

    out_path = "analysis/REPORT.md"
    with open(out_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
