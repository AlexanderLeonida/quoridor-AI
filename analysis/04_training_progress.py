"""Plot training-loop progress: loss curves, eval scores, draw rates.

Reads ``logs/train.log`` (current) and ``logs/archive/*.log`` if any.
Produces a 4-panel chart: epoch-1 train/val loss per iteration,
gating eval scores with CI, self-play vs eval draw rates, cumulative
promotions and reverts.

Saved to ``analysis/plots/04_training_progress.png``.
"""
from __future__ import annotations

import glob
import os

import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parser import parse_log


def main() -> None:
    log_paths = sorted(glob.glob("logs/archive/*.log"))
    if os.path.exists("logs/train.log"):
        log_paths.append("logs/train.log")

    all_recs = []
    for p in log_paths:
        all_recs.extend(parse_log(p))

    if not all_recs:
        print("  No log records found.")
        return

    # Treat global_iter as the x-axis so we can stitch multiple runs.
    its = [r.global_iter for r in all_recs]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Loss curves
    ax = axes[0][0]
    e1_train = [(r.global_iter, r.epochs[0]["train"]) for r in all_recs if r.epochs]
    e1_val = [(r.global_iter, r.epochs[0].get("val")) for r in all_recs
              if r.epochs and "val" in r.epochs[0]]
    if e1_train:
        ax.plot([x for x, _ in e1_train], [y for _, y in e1_train],
                "-o", markersize=3, label="train (epoch 1)")
    if e1_val:
        ax.plot([x for x, _ in e1_val], [y for _, y in e1_val],
                "-x", markersize=4, label="val (epoch 1)")
    ax.set_xlabel("global iteration")
    ax.set_ylabel("loss")
    ax.set_title("Epoch-1 loss per iteration\n"
                 "(gap between curves = overfit; closing gap = good)")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Eval scores with CI
    ax = axes[0][1]
    valid = [(r.global_iter, r.eval_score, r.eval_ci_lo, r.eval_ci_hi,
              r.promoted, r.reverted_to)
             for r in all_recs if r.eval_score is not None]
    if valid:
        gx = [v[0] for v in valid]
        es = [v[1] for v in valid]
        cilo = [v[2] for v in valid]
        cihi = [v[3] for v in valid]
        ax.fill_between(gx, cilo, cihi, alpha=0.2, color="#5b8def",
                        label="95% CI")
        # Color points by outcome
        for (g, s, lo, hi, prom, rev) in valid:
            if rev:
                ax.plot(g, s, "v", color="purple", markersize=8,
                        label="REVERT" if "REVERT" not in [t.get_label() for t in ax.lines] else None)
            elif prom:
                ax.plot(g, s, "o", color="green", markersize=6,
                        label="PROMOTED" if "PROMOTED" not in [t.get_label() for t in ax.lines] else None)
            else:
                ax.plot(g, s, "x", color="red", markersize=6,
                        label="rejected" if "rejected" not in [t.get_label() for t in ax.lines] else None)
    ax.axhline(0.52, ls="--", c="g", lw=1, label="promote threshold")
    ax.axhline(0.50, ls=":", c="grey", lw=1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("global iteration")
    ax.set_ylabel("eval score (vs prev best)")
    ax.set_title("Gating eval scores per iteration")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    # 3. Draw rates
    ax = axes[1][0]
    sp = []
    ev = []
    for r in all_recs:
        sp_total = r.sp_p1_wins + r.sp_p2_wins + r.sp_draws
        if sp_total:
            sp.append((r.global_iter, r.sp_draws / sp_total))
        if r.eval_score is not None:
            ev_total = r.eval_w + r.eval_l + r.eval_d
            if ev_total:
                ev.append((r.global_iter, r.eval_d / ev_total))
    if sp:
        ax.plot([x for x, _ in sp], [y for _, y in sp], "-o", markersize=3,
                label="self-play draws %")
    if ev:
        ax.plot([x for x, _ in ev], [y for _, y in ev], "-x", markersize=4,
                label="eval draws %")
    ax.set_ylim(0, 1)
    ax.set_xlabel("global iteration")
    ax.set_ylabel("fraction drawn")
    ax.set_title("Draw rate over iterations\n"
                 "(adjudication + max-moves tightening should drive this down)")
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. Cumulative promotions / reverts
    ax = axes[1][1]
    cum_p = []
    cum_r = []
    p = r_ = 0
    for rec in all_recs:
        if rec.promoted:
            p += 1
        if rec.reverted_to:
            r_ += 1
        cum_p.append((rec.global_iter, p))
        cum_r.append((rec.global_iter, r_))
    ax.plot([x for x, _ in cum_p], [y for _, y in cum_p], "-",
            color="green", label="cumulative promotions")
    ax.plot([x for x, _ in cum_r], [y for _, y in cum_r], "-",
            color="purple", label="cumulative reverts")
    ax.set_xlabel("global iteration")
    ax.set_ylabel("count")
    ax.set_title("Promotions and tournament reverts")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(f"Training-loop progress ({len(all_recs)} iterations across "
                 f"{len(log_paths)} log file(s))", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = "analysis/plots/04_training_progress.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")
    print(f"\n{len(all_recs)} iterations parsed, "
          f"{sum(1 for r in all_recs if r.promoted)} promoted, "
          f"{sum(1 for r in all_recs if r.reverted_to)} reverts")


if __name__ == "__main__":
    main()
