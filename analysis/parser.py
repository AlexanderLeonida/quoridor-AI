"""Parse training logs into structured records.

Library-only — no CLI. Used by every plot script under ``analysis/`` so
log-parsing logic lives in exactly one place.

The parser is forgiving: it skips unparseable lines and missing fields,
so partial / in-progress logs still yield as much structured data as is
present.
"""
from __future__ import annotations

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
    epochs: List[dict] = field(default_factory=list)  # {epoch, train, p, v, val?, val_p?, val_v?}
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
    """Parse one training log file into a list of IterRecord."""
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


def summary_text(recs: List[IterRecord]) -> str:
    """Return a tabular text summary of parsed records."""
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
