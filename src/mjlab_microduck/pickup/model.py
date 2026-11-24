"""Tiny temporal-CNN pick-up classifier + the pause/resume state machine.

Input: the last ``WINDOW`` ticks of the 63-D feature vector (features.py),
oldest first, shape (B, WINDOW, 63). Output: logit of p(held). The feature
normalizer is a buffer of the module, so the exported ONNX takes raw features.
~25k parameters, ~0.3 MMAC per call — negligible on the Radxa's A55 cores.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mjlab_microduck.pickup.features import FEAT_DIM

WINDOW = 50  # 1 s at 50 Hz


class PickupNet(nn.Module):
    def __init__(self, feat_dim: int = FEAT_DIM, window: int = WINDOW, ch: int = 32, hidden: int = 32,
                 keep: list[int] | None = None):
        super().__init__()
        # Channels the network actually reads, selected INSIDE the graph so the input contract stays
        # 63-D. Dropping a feature here (rather than masking it to zero) also drops its compute:
        # the first convolution is most of the cost, and it scales with its input width.
        keep = list(range(feat_dim)) if keep is None else list(keep)
        self.register_buffer("keep", torch.tensor(keep, dtype=torch.long), persistent=False)  # from net_kwargs
        self.keep_list = keep
        self.register_buffer("mean", torch.zeros(feat_dim))
        self.register_buffer("std", torch.ones(feat_dim))
        # 0 = feature ignored (e.g. servo current, whose sim2real is doubtful);
        # the input contract stays 63-D either way
        self.register_buffer("mask", torch.ones(feat_dim))
        self.conv = nn.Sequential(
            nn.Conv1d(len(keep), ch, 5, stride=2), nn.ELU(),
            nn.Conv1d(ch, ch, 5, stride=2), nn.ELU(),
            nn.Conv1d(ch, ch, 5, stride=2), nn.ELU(),
        )
        with torch.no_grad():
            n = self.conv(torch.zeros(1, len(keep), window)).numel()
        self.head = nn.Sequential(nn.Linear(n, hidden), nn.ELU(), nn.Linear(hidden, 1))
        self.window = window

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, W, F) → (B,)
        x = ((x - self.mean) / self.std * self.mask)[..., self.keep]
        x = self.conv(x.transpose(1, 2))
        return self.head(x.flatten(1)).squeeze(-1)


def export_onnx(net: PickupNet, path: str) -> None:
    """Exports features[1, W, 63] → p_held[1] (sigmoid applied)."""

    class _Prob(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.n = n

        def forward(self, x):
            return torch.sigmoid(self.n(x))

    net = net.cpu().eval()
    dummy = torch.zeros(1, net.window, FEAT_DIM)
    torch.onnx.export(_Prob(net), dummy, path, input_names=["features"], output_names=["p_held"],
                      opset_version=17, dynamo=False)


@dataclass
class StateMachineCfg:
    p_pause: float = 0.8      # POLICY → PAUSED when p > p_pause for n_pause ticks
    n_pause: int = 5          # 100 ms
    # PAUSED → POLICY when p < p_resume for n_resume ticks. Resume speed is what
    # keeps a set-down robot on its feet (the pause pose tips within ~0.2-0.5 s):
    # closed-loop sim, 0.2×10 ticks → resume p50 0.30 s, 11.7 % tipped >40°;
    # 0.35×4 ticks → 0.18 s, 2.9 % tipped (false mid-carry resumes 0 → 1.5 / h).
    p_resume: float = 0.35
    n_resume: int = 4         # 80 ms
    min_pause_s: float = 0.3  # no instant flip-flop (a put-down robot tips within ~1 s: keep it short)


class PauseStateMachine:
    """Batched hysteresis over p(held). ``paused`` is the runtime's pause flag."""

    def __init__(self, num_envs: int, device, cfg: StateMachineCfg | None = None, dt: float = 0.02):
        self.cfg = cfg or StateMachineCfg()
        self.dt = dt
        self.paused = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.cnt = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.t_paused = torch.zeros(num_envs, device=device)

    def reset(self, env_ids):
        self.paused[env_ids] = False
        self.cnt[env_ids] = 0
        self.t_paused[env_ids] = 0.0

    def update(self, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (just_paused, just_resumed) masks."""
        c = self.cfg
        self.t_paused = torch.where(self.paused, self.t_paused + self.dt, torch.zeros_like(self.t_paused))
        hit = torch.where(self.paused, (p < c.p_resume) & (self.t_paused >= c.min_pause_s), p > c.p_pause)
        self.cnt = torch.where(hit, self.cnt + 1, torch.zeros_like(self.cnt))
        need = torch.where(self.paused, torch.full_like(self.cnt, c.n_resume), torch.full_like(self.cnt, c.n_pause))
        flip = self.cnt >= need
        just_paused = flip & ~self.paused
        just_resumed = flip & self.paused
        self.paused = self.paused ^ flip
        self.cnt[flip] = 0
        self.t_paused[flip] = 0.0
        return just_paused, just_resumed
