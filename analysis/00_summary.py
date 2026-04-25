"""Print a tabular text summary of training progress.

Reads ``logs/train.log`` (and any ``logs/archive/*.log``) and prints a
per-iteration table: self-play W/L/D, epoch-1 train/val loss, eval
score with CI, promote/revert flag.

No plot — see ``04_training_progress.py`` for the visual version.

Usage:
    python3 analysis/00_summary.py
"""
from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parser import parse_log, summary_text


def main() -> None:
    log_paths = sorted(glob.glob("logs/archive/*.log"))
    if os.path.exists("logs/train.log"):
        log_paths.append("logs/train.log")
    if not log_paths:
        print("No logs found.")
        return

    all_recs = []
    for p in log_paths:
        all_recs.extend(parse_log(p))
    print(summary_text(all_recs))


if __name__ == "__main__":
    main()
