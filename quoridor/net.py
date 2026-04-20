"""PyTorch neural network for Quoridor (AlphaZero-style).

Architecture
------------
    Input: (B, 7, 9, 9) canonical state tensor (see encoding.py).
    Trunk: Conv3x3 stem -> N residual blocks (BN + ReLU + Conv3x3 x2).
    Policy head: Conv1x1 -> BN -> ReLU -> FC -> 209 logits.
    Value head:  Conv1x1 -> BN -> ReLU -> FC(64) -> ReLU -> FC(1) -> tanh.

Defaults (10 residual blocks, 128 filters) give the network enough
capacity to learn wall-placement tactics and long-range path planning
on the 9x9 board while remaining trainable on Apple Silicon / mid-range
GPUs.

PyTorch is imported *lazily*: the rest of the package (engine, DB,
encoding) works on a machine with no torch installed.
"""

from __future__ import annotations

from typing import Optional, Tuple


def _lazy_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for the neural network. "
            "Install with: pip install -r requirements.txt"
        ) from e
    return torch, nn, F


def build_net(blocks: int = 10, filters: int = 128):
    torch, nn, F = _lazy_torch()
    from .encoding import ACTION_SPACE, BOARD_SIZE, NUM_PLANES

    class ResBlock(nn.Module):
        def __init__(self, ch: int):
            super().__init__()
            self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.b1 = nn.BatchNorm2d(ch)
            self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.b2 = nn.BatchNorm2d(ch)

        def forward(self, x):
            r = F.relu(self.b1(self.c1(x)))
            r = self.b2(self.c2(r))
            return F.relu(x + r)

    class QuoridorNet(nn.Module):
        config = {
            "blocks": blocks,
            "filters": filters,
            "in_planes": NUM_PLANES,
            "action_space": ACTION_SPACE,
        }

        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(NUM_PLANES, filters, 3, padding=1, bias=False),
                nn.BatchNorm2d(filters),
                nn.ReLU(inplace=True),
            )
            self.tower = nn.Sequential(*[ResBlock(filters) for _ in range(blocks)])
            # Policy head
            p_ch = max(32, filters // 2)
            self.p_conv = nn.Sequential(
                nn.Conv2d(filters, p_ch, 1, bias=False),
                nn.BatchNorm2d(p_ch),
                nn.ReLU(inplace=True),
            )
            self.p_fc = nn.Linear(p_ch * BOARD_SIZE * BOARD_SIZE, ACTION_SPACE)
            # Value head
            v_ch = max(16, filters // 4)
            self.v_conv = nn.Sequential(
                nn.Conv2d(filters, v_ch, 1, bias=False),
                nn.BatchNorm2d(v_ch),
                nn.ReLU(inplace=True),
            )
            v_hidden = max(64, filters)
            self.v_fc1 = nn.Linear(v_ch * BOARD_SIZE * BOARD_SIZE, v_hidden)
            self.v_fc2 = nn.Linear(v_hidden, 1)

        def forward(self, x):
            x = self.stem(x)
            x = self.tower(x)
            p_logits = self.p_fc(self.p_conv(x).flatten(1))
            v = F.relu(self.v_fc1(self.v_conv(x).flatten(1)))
            v = torch.tanh(self.v_fc2(v)).squeeze(-1)
            return p_logits, v

    return QuoridorNet()


def save_checkpoint(net, path: str, **meta) -> None:
    torch, _, _ = _lazy_torch()
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "state_dict": net.state_dict(),
        "config": getattr(net, "config", {}),
        "meta": meta,
    }
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: Optional[str] = None):
    torch, _, _ = _lazy_torch()
    ckpt = torch.load(path, map_location=map_location or "cpu")
    cfg = ckpt.get("config", {})
    net = build_net(
        blocks=cfg.get("blocks", 10),
        filters=cfg.get("filters", 128),
    )
    net.load_state_dict(ckpt["state_dict"])
    return net, ckpt.get("meta", {})


def best_available_device():
    torch, _, _ = _lazy_torch()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
