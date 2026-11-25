"""Turn-first command curriculum — the steering fix's mechanism (2026-09-16).

The defect this exists for. The walking policy produced by this recipe **cannot be
steered**: sweeping the yaw command from -34 to +34 deg/s moves its measured yaw
rate by under 2 %, with the wrong sign at a standstill, while
`track_angular_velocity` (weight 2.0, params byte-identical to the official
recipe) reads **0.119 of 1.0** — the policy abandons ~23 % of total reward rather
than turn. The official policy tracks the same command fine (+18.5 deg/s for a
+34 deg/s command). So the turn channel is not under-priced; it is unreachable
from the forward-shuffle basin the policy settled in.

These tests pin the three things that make the curriculum safe and correct:
  * it is OFF by default and env-switchable (controlled A/B),
  * the first stage asks for turning ONLY — but with a small NON-ZERO linear
    range, because a command input that never goes non-zero has dead weights
    forever (AGENTS.md),
  * it writes the LIVE CommandManager term cfg, since writes to `env.cfg` are
    silent no-ops.
"""

from __future__ import annotations

import importlib

import pytest
import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    ENABLE_TURN_FIRST_CURRICULUM,
    TURN_FIRST_STAGE1_ITERS,
    TURN_FIRST_STAGE2_ITERS,
    TURN_FIRST_TINY_LIN,
    make_microduck_velocity_env_cfg,
)

FINAL_LIN_X = (-0.40, 0.40)


class _Ranges:
    def __init__(self):
        self.lin_vel_x = (0.0, 0.0)
        self.lin_vel_y = (0.0, 0.0)
        self.ang_vel_z = (0.0, 0.0)


class _TermCfg:
    def __init__(self):
        self.ranges = _Ranges()


class _Term:
    def __init__(self):
        self.cfg = _TermCfg()


class _CmdMgr:
    def __init__(self):
        self.term = _Term()

    def get_term(self, _name):
        return self.term


class _Env:
    def __init__(self, steps: int = 0):
        self.command_manager = _CmdMgr()
        self.common_step_counter = steps


def _stages():
    return [
        {"step": 0, "lin_vel_x": (-TURN_FIRST_TINY_LIN, TURN_FIRST_TINY_LIN),
         "lin_vel_y": (-TURN_FIRST_TINY_LIN, TURN_FIRST_TINY_LIN),
         "ang_vel_z": (-1.0, 1.0)},
        {"step": TURN_FIRST_STAGE1_ITERS * 24, "lin_vel_x": (-0.20, 0.20),
         "lin_vel_y": (-0.15, 0.15), "ang_vel_z": (-1.0, 1.0)},
        {"step": TURN_FIRST_STAGE2_ITERS * 24, "lin_vel_x": FINAL_LIN_X,
         "lin_vel_y": (-0.30, 0.30), "ang_vel_z": (-1.0, 1.0)},
    ]


def _apply(steps: int) -> _Env:
    env = _Env(steps)
    microduck_mdp.velocity_command_range_curriculum(
        env, None, command_name="twist", stages=_stages())
    return env


# --------------------------------------------------------------------------- #
# cfg wiring
# --------------------------------------------------------------------------- #
def test_arm_is_off_by_default():
    assert ENABLE_TURN_FIRST_CURRICULUM is False, "must be a deliberate A/B arm"
    cfg = make_microduck_velocity_env_cfg()
    assert "turn_first" not in cfg.curriculum
    assert cfg.commands["twist"].ranges.lin_vel_x == FINAL_LIN_X


def test_arm_registers_the_curriculum_when_enabled(monkeypatch):
    monkeypatch.setenv("MICRODUCK_TURN_FIRST", "1")
    mod = importlib.import_module("mjlab_microduck.tasks.microduck_velocity_env_cfg")
    importlib.reload(mod)
    try:
        assert mod.ENABLE_TURN_FIRST_CURRICULUM is True
        cfg = mod.make_microduck_velocity_env_cfg()
        c = cfg.curriculum["turn_first"]
        stages = c.params["stages"]
        assert len(stages) == 3
        assert stages[0]["step"] == 0
        assert stages[-1]["step"] == TURN_FIRST_STAGE2_ITERS * mod.NUM_STEPS_PER_ENV
        # the LAST stage must reproduce the task's real command ranges, otherwise
        # the curriculum would silently leave the task narrower than it looks
        assert tuple(stages[-1]["lin_vel_x"]) == tuple(cfg.commands["twist"].ranges.lin_vel_x)
    finally:
        monkeypatch.delenv("MICRODUCK_TURN_FIRST", raising=False)
        importlib.reload(mod)


def test_first_stage_asks_for_turning_only_but_not_with_a_dead_input():
    """lin must be small — but NOT exactly zero (dead input neurons, AGENTS.md)."""
    s0 = _stages()[0]
    assert 0.0 < s0["lin_vel_x"][1] <= 0.1, s0["lin_vel_x"]
    assert s0["lin_vel_x"][0] == -s0["lin_vel_x"][1]
    assert s0["ang_vel_z"] == (-1.0, 1.0), "the full yaw range must be available at once"


# --------------------------------------------------------------------------- #
# the ramp
# --------------------------------------------------------------------------- #
def test_stage_selection_follows_common_step_counter():
    assert _apply(0).command_manager.term.cfg.ranges.lin_vel_x[1] == TURN_FIRST_TINY_LIN
    mid = _apply(TURN_FIRST_STAGE1_ITERS * 24)
    assert mid.command_manager.term.cfg.ranges.lin_vel_x[1] == pytest.approx(0.20)
    late = _apply(TURN_FIRST_STAGE2_ITERS * 24)
    assert late.command_manager.term.cfg.ranges.lin_vel_x == FINAL_LIN_X


def test_it_writes_the_LIVE_term_cfg():
    """env.cfg writes are silent no-ops; the manager re-reads term.cfg.ranges."""
    env = _apply(0)
    assert env.command_manager.term.cfg.ranges.lin_vel_x[1] == TURN_FIRST_TINY_LIN


def test_it_returns_a_scalar_the_curriculum_manager_can_item():
    env = _Env(0)
    out = microduck_mdp.velocity_command_range_curriculum(
        env, None, command_name="twist", stages=_stages())
    assert isinstance(out, torch.Tensor)
    assert out.numel() == 1
    assert float(out.item()) == pytest.approx(TURN_FIRST_TINY_LIN)


def test_yaw_range_is_never_narrowed():
    """The whole point is to make turning ASKED FOR, never to ask for less of it."""
    for steps in (0, TURN_FIRST_STAGE1_ITERS * 24, TURN_FIRST_STAGE2_ITERS * 24):
        assert _apply(steps).command_manager.term.cfg.ranges.ang_vel_z == (-1.0, 1.0)
