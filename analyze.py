"""Parse logs/train.log and report training-progress metrics.

Outputs a text summary plus an optional matplotlib plot of:
    - per-iteration eval score (with 0.52 promote threshold and 0.50)
    - train/val loss per epoch
    - draw rate in self-play and in eval
    - cumulative promotions

No external dependencies beyond stdlib + matplotlib (only if --plot).

Usage:
    python3 analyze.py                # summary to stdout
    python3 analyze.py --plot out.png # also save a plot
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class IterRecord:
    iteration: int
    global_iter: int
    sp_p1_wins: int = 0
    sp_p2_wins: int = 0
    sp_draws: int = 0
    epochs: List[dict] = field(default_factory=list)  # {train, val, p, v}
    eval_score: Optional[float] = None
    eval_w: int = 0
    eval_l: int = 0
    eval_d: int = 0
    eval_ci_lo: Optional[float] = None
    eval_ci_hi: Optional[float] = None
    promoted: bool = False
    reverted_to: Optional[str] = None


_ITER_RE = re.compile(r"Iteration (\d+)/\d+\s+\(global (\d+)\)")
_SP_RE = re.compile(r"P1 wins: (\d+)\s+P2 wins: (\d+)\s+draws: (\d+)")
_EPOCH_RE = re.compile(
    r"epoch (\d+)/\d+\s+loss=([\d.]+)\s+\(p=([\d.]+) v=([\d.]+)\)"
    r"(?:\s+val=([\d.]+)\s+\(p=([\d.]+) v=([\d.]+)\))?"
)
_EVAL_RE = re.compile(
    r"New net score:\s+([\d.]+)%\s+\(W(\d+)/L(\d+)/D(\d+)\)"
    r"\s+95% CI: \[([\d.]+)%, ([\d.]+)%\]"
)
_PROMOTED_RE = re.compile(r">>> Promoted")
_REVERT_RE = re.compile(r">>> REVERTING best_net: (\S+) -> (\S+)")


def parse_log(path: str) -> List[IterRecord]:
    records: List[IterRecord] = []
    cur: Optional[IterRecord] = None
    with open(path) as f:
        for line in f:
            m = _ITER_RE.search(line)
            if m:
                if cur is not None:
                    records.append(cur)
                cur = IterRecord(iteration=int(m.group(1)),
                                 global_iter=int(m.group(2)))
                continue
            if cur is None:
                continue
            m = _SP_RE.search(line)
            if m:
                cur.sp_p1_wins = int(m.group(1))
                cur.sp_p2_wins = int(m.group(2))
                cur.sp_draws = int(m.group(3))
                continue
            m = _EPOCH_RE.search(line)
            if m:
                ep = {
                    "epoch": int(m.group(1)),
                    "train": float(m.group(2)),
                    "p": float(m.group(3)),
                    "v": float(m.group(4)),
                }
                if m.group(5):
                    ep.update(val=float(m.group(5)),
                              val_p=float(m.group(6)),
                              val_v=float(m.group(7)))
                cur.epochs.append(ep)
                continue
            m = _EVAL_RE.search(line)
            if m:
                cur.eval_score = float(m.group(1)) / 100.0
                cur.eval_w = int(m.group(2))
                cur.eval_l = int(m.group(3))
                cur.eval_d = int(m.group(4))
                cur.eval_ci_lo = float(m.group(5)) / 100.0
                cur.eval_ci_hi = float(m.group(6)) / 100.0
                continue
            if _PROMOTED_RE.search(line):
                cur.promoted = True
                continue
            m = _REVERT_RE.search(line)
            if m:
                cur.reverted_to = m.group(2)
    if cur is not None:
        records.append(cur)
    return records


def summary(recs: List[IterRecord]) -> str:
    lines: List[str] = []
    lines.append(f"{len(recs)} iterations parsed")
    if not recs:
        return "\n".join(lines)

    n_promoted = sum(1 for r in recs if r.promoted)
    n_rejected = sum(1 for r in recs
                     if r.eval_score is not None and not r.promoted)
    n_reverts = sum(1 for r in recs if r.reverted_to)
    lines.append(f"Promoted: {n_promoted}    Rejected: {n_rejected}    "
                 f"Reverts: {n_reverts}")

    sp_total = sum(r.sp_p1_wins + r.sp_p2_wins + r.sp_draws for r in recs)
    sp_draws = sum(r.sp_draws for r in recs)
    if sp_total:
        lines.append(f"Self-play: {sp_total} games, {sp_draws} drawn "
                     f"({sp_draws/sp_total:.0%})")
    eval_total = sum(r.eval_w + r.eval_l + r.eval_d for r in recs
                     if r.eval_score is not None)
    eval_draws = sum(r.eval_d for r in recs if r.eval_score is not None)
    if eval_total:
        lines.append(f"Eval: {eval_total} games, {eval_draws} drawn "
                     f"({eval_draws/eval_total:.0%})")

    lines.append("")
    lines.append(f"{'iter':>4}  {'global':>6}  {'sp(W1/W2/D)':<14}  "
                 f"{'epoch1 train':>12}  {'epoch1 val':>10}  "
                 f"{'eval':>5}  {'CI':<14}  {'WLD':>10}  result")
    for r in recs:
        sp = f"{r.sp_p1_wins}/{r.sp_p2_wins}/{r.sp_draws}"
        e1 = r.epochs[0] if r.epochs else None
        ep_train = f"{e1['train']:.3f}" if e1 else "-"
        ep_val = f"{e1.get('val', 0):.3f}" if (e1 and 'val' in e1) else "-"
        if r.eval_score is None:
            ev = "-"
            ci = "-"
            wld = "-"
        else:
            ev = f"{r.eval_score:.0%}"
            ci = f"[{r.eval_ci_lo:.0%},{r.eval_ci_hi:.0%}]"
            wld = f"{r.eval_w}/{r.eval_l}/{r.eval_d}"
        result = ""
        if r.reverted_to:
            result = f"REVERTED→{r.reverted_to}"
        elif r.promoted:
            result = "PROMOTED"
        elif r.eval_score is not None:
            result = "kept"
        lines.append(
            f"{r.iteration:>4}  {r.global_iter:>6}  {sp:<14}  "
            f"{ep_train:>12}  {ep_val:>10}  {ev:>5}  {ci:<14}  "
            f"{wld:>10}  {result}"
        )
    return "\n".join(lines)


def plot(recs: List[IterRecord], out_path: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    its = [r.global_iter for r in recs]

    # Eval scores with CI
    ax = axes[0][0]
    es = [r.eval_score for r in recs]
    cilo = [r.eval_ci_lo for r in recs]
    cihi = [r.eval_ci_hi for r in recs]
    valid = [(g, s, lo, hi) for g, s, lo, hi in zip(its, es, cilo, cihi) if s is not None]
    if valid:
        gx = [v[0] for v in valid]
        ax.plot(gx, [v[1] for v in valid], "-o", label="eval score")
        ax.fill_between(gx, [v[2] for v in valid], [v[3] for v in valid],
                        alpha=0.2, label="95% CI")
    ax.axhline(0.52, ls="--", c="g", label="promote threshold")
    ax.axhline(0.50, ls=":", c="grey", label="50%")
    ax.set_ylim(0, 1)
    ax.set_title("Eval score per iteration")
    ax.set_xlabel("global iteration")
    ax.legend()

    # Train/val loss (epoch 1)
    ax = axes[0][1]
    e1_train = [r.epochs[0]["train"] if r.epochs else None for r in recs]
    e1_val = [r.epochs[0].get("val") if r.epochs else None for r in recs]
    valid_t = [(g, t) for g, t in zip(its, e1_train) if t is not None]
    valid_v = [(g, v) for g, v in zip(its, e1_val) if v is not None]
    if valid_t:
        ax.plot([v[0] for v in valid_t], [v[1] for v in valid_t], "-o", label="train (e1)")
    if valid_v:
        ax.plot([v[0] for v in valid_v], [v[1] for v in valid_v], "-x", label="val (e1)")
    ax.set_title("Epoch-1 loss per iteration")
    ax.set_xlabel("global iteration")
    ax.legend()

    # Draw rates
    ax = axes[1][0]
    sp_dr = [(r.sp_draws / max(1, r.sp_p1_wins + r.sp_p2_wins + r.sp_draws))
             for r in recs]
    ev_dr = [(r.eval_d / max(1, r.eval_w + r.eval_l + r.eval_d))
             if r.eval_score is not None else None for r in recs]
    ax.plot(its, sp_dr, "-o", label="self-play draw %")
    valid_e = [(g, d) for g, d in zip(its, ev_dr) if d is not None]
    if valid_e:
        ax.plot([v[0] for v in valid_e], [v[1] for v in valid_e], "-x", label="eval draw %")
    ax.set_title("Draw rate per iteration")
    ax.set_xlabel("global iteration")
    ax.set_ylim(0, 1)
    ax.legend()

    # Cumulative promotions / reverts
    ax = axes[1][1]
    cum_promoted = []
    cum_rev = []
    p = r_ = 0
    for rec in recs:
        if rec.promoted:
            p += 1
        if rec.reverted_to:
            r_ += 1
        cum_promoted.append(p)
        cum_rev.append(r_)
    ax.plot(its, cum_promoted, "-o", label="cumulative promotions")
    ax.plot(its, cum_rev, "-x", label="cumulative reverts")
    ax.set_title("Promotions and reverts")
    ax.set_xlabel("global iteration")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved plot: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", default="logs/train.log")
    p.add_argument("--plot", default=None,
                   help="If set, save a 4-panel matplotlib plot to this path.")
    args = p.parse_args()

    recs = parse_log(args.log)
    print(summary(recs))
    if args.plot:
        plot(recs, args.plot)


if __name__ == "__main__":
    main()
