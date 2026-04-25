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
]


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    os.makedirs("analysis/plots", exist_ok=True)

    log_lines = []
    log_lines.append(f"# Quoridor AI — Analysis Report")
    log_lines.append(f"\nGenerated: {datetime.now().isoformat(timespec='seconds')}")
    log_lines.append(f"\n---\n")

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
