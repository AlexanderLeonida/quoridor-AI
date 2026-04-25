"""Compare the two network architectures used in this project.

Reads every ``checkpoints/iter_*.pt`` file, groups by config, and
reports parameters, file size, and a parameter-count chart. Distillation
moved us from 6 blocks × 64 filters → 10 blocks × 128 filters; this
visualises the capacity jump.

Saved to ``analysis/plots/05_architecture_comparison.png``.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import torch


def _count_params(state_dict: dict) -> int:
    return sum(t.numel() for t in state_dict.values())


def main() -> None:
    by_arch: dict = {}
    for f in sorted(os.listdir("checkpoints")):
        if not (f.startswith("iter_") and f.endswith(".pt")):
            continue
        path = "checkpoints/" + f
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        cfg = ck.get("config", {})
        key = (cfg.get("blocks"), cfg.get("filters"))
        params = _count_params(ck["state_dict"])
        size_mb = os.path.getsize(path) / 1e6
        d = by_arch.setdefault(key, {
            "params": params, "size_mb": size_mb,
            "iters": [], "first_iter": None,
        })
        n = int(f[5:9])
        d["iters"].append(n)
        if d["first_iter"] is None or n < d["first_iter"]:
            d["first_iter"] = n

    if not by_arch:
        print("  No checkpoints found.")
        return

    archs = sorted(by_arch.keys())
    labels = [f"{b}×{fi}" for b, fi in archs]
    params = [by_arch[k]["params"] / 1e6 for k in archs]
    counts = [len(by_arch[k]["iters"]) for k in archs]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Param count
    ax = axes[0]
    bars = ax.bar(labels, params, color=["#5b8def", "#e07b39"][:len(labels)],
                  edgecolor="black", linewidth=0.5)
    for bar, p in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width() / 2, p + 0.05,
                f"{p:.2f}M", ha="center", fontsize=10)
    ax.set_ylabel("trainable parameters (millions)")
    ax.set_title("Network capacity by architecture")
    ax.grid(axis="y", alpha=0.3)

    # Iteration counts
    ax = axes[1]
    ax.bar(labels, counts, color=["#5b8def", "#e07b39"][:len(labels)],
           edgecolor="black", linewidth=0.5)
    for i, (lbl, c) in enumerate(zip(labels, counts)):
        first = by_arch[archs[i]]["first_iter"]
        last = max(by_arch[archs[i]]["iters"])
        ax.text(i, c + 0.5, f"v{first}–v{last}", ha="center", fontsize=9)
    ax.set_ylabel("checkpoints saved")
    ax.set_title("Training iterations per architecture")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Architecture comparison: 6×64 → 10×128 distillation", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = "analysis/plots/05_architecture_comparison.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")
    print(f"\nArchitectures observed:")
    for (b, fi), d in by_arch.items():
        print(f"  {b}×{fi:<3}: {d['params']/1e6:.2f}M params, "
              f"{d['size_mb']:.1f} MB, "
              f"{len(d['iters'])} iterations "
              f"(v{d['first_iter']}–v{max(d['iters'])})")


if __name__ == "__main__":
    main()
