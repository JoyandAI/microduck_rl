"""`MICRODUCK_CMD_LIN_MAX` — cap the commanded linear speed, and NOTHING else.

WHY THIS SWITCH EXISTS (2026-09-20). The frozen-ck7000 speed sweep
(`logs/probe_speed/speed_vs_flight.tsv`) found a PHASE TRANSITION between 0.10 and
0.20 m/s: per-step yaw impulse x93, foot clearance x12, and 20 s net yaw 2 -> 114
deg; below 0.10 m/s the policy does not move at all. Every arm we have trained so
far is commanded up to 0.4 m/s (`command.ranges.lin_vel_x`), i.e. far above that
transition, and every arm wanders 78-571 deg. Capping the command is the cheapest
untested single variable for "is our gait only straight when it is slow?", which is
what the user asked on 2026-09-18 ("我们可以放慢速度，是不是因为速度太快导致的").

The switch must be provably single-variable: it may change the sampled command
range and nothing else in the recipe. These tests lock that, and lock that the
DEFAULT keeps the official alignment intact (the official-alignment suite compares
the whole resolved cfg against the verbatim upstream copy).
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import os

import pytest

CFG_MODULE = "mjlab_microduck.tasks.microduck_velocity_env_cfg"
CAP_ENV = "MICRODUCK_CMD_LIN_MAX"

# The official-alignment values: every recipe deviation OFF, so the only thing
# under test here is the command range.
OFFICIAL_VALUES = {
    "MICRODUCK_STANCE_HYGIENE": "0",
    "MICRODUCK_STANCE_HIP_ROLL": "0",
    "MICRODUCK_PER_JOINT_POSTURE": "0",
    "MICRODUCK_AIR_TIME_GATE": "0",
    "MICRODUCK_HEADING_HOLD": "0",
    "MICRODUCK_HEAD_STABILITY": "0",
    "MICRODUCK_YAW_HOLD_ZERO_CMD": "0",
    "MICRODUCK_ABDUCTION": "0",
    "MICRODUCK_YAW_DC": "0",
    "MICRODUCK_YAW_SYMMETRY": "0",
    "MICRODUCK_TURN_FIRST": "0",
    "MICRODUCK_SYMMETRY": "0",
    "MICRODUCK_HIP_ROLL_STD": "0.05",
    "MICRODUCK_POSE_ZERO_CMD_PROB": "0.0",
}


@contextlib.contextmanager
def _recipe(env: dict[str, str]):
    """Reload the cfg module with ``env`` applied, then restore the default state."""
    mod = importlib.import_module(CFG_MODULE)
    saved = {k: os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            os.environ[k] = v
        yield importlib.reload(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(mod)


def _cfg(extra: dict[str, str] | None = None):
    env = dict(OFFICIAL_VALUES)
    env.update(extra or {})
    with _recipe(env) as mod:
        return mod.make_microduck_velocity_env_cfg()


def _diff_leaves(x, y, path="cfg", depth=0, out=None):
    """Recursively collect differing leaves between two cfg dataclasses."""
    if out is None:
        out = []
    if depth > 8 or "spec_fn" in path:
        return out
    if type(x) is not type(y):
        out.append((path, f"TYPE {type(x).__name__} vs {type(y).__name__}"))
        return out
    if dataclasses.is_dataclass(x):
        for f in dataclasses.fields(x):
            _diff_leaves(getattr(x, f.name), getattr(y, f.name), f"{path}.{f.name}", depth + 1, out)
    elif isinstance(x, (tuple, list)):
        if len(x) != len(y):
            out.append((path, f"LEN {len(x)} vs {len(y)}"))
        else:
            for i, (a, b) in enumerate(zip(x, y)):
                _diff_leaves(a, b, f"{path}[{i}]", depth + 1, out)
    elif isinstance(x, dict):
        if set(x) != set(y):
            out.append((path, f"KEYS {sorted(set(x) ^ set(y))}"))
        else:
            for k in x:
                _diff_leaves(x[k], y[k], f"{path}[{k!r}]", depth + 1, out)
    elif x != y:
        out.append((path, f"{x!r} vs {y!r}"))
    return out


def test_cap_changes_nothing_else_in_the_recipe():
    """The whole point is a ONE-variable arm: everything else must be identical.

    Both cfgs are built from the SAME module instance (the switch is read inside
    the factory, not at import time), so reward/obs/event function objects are
    shared and a leaf diff cannot produce false positives from a reload.
    """
    with _recipe(OFFICIAL_VALUES) as mod:
        os.environ.pop(CAP_ENV, None)
        base = mod.make_microduck_velocity_env_cfg()
        os.environ[CAP_ENV] = "0.15"
        capped = mod.make_microduck_velocity_env_cfg()
        os.environ.pop(CAP_ENV, None)
    diffs = _diff_leaves(base, capped)
    assert diffs, "the cap did not change anything at all — the switch is dead"
    for path, _ in diffs:
        assert "commands" in path and "lin_vel" in path, diffs
    # leaf paths carry the tuple index, e.g. ...ranges.lin_vel_x[0]
    changed = {p.split(".")[-1].split("[")[0] for p, _ in diffs}
    assert changed == {"lin_vel_x", "lin_vel_y"}, diffs


def test_default_is_uncapped_and_official():
    cfg = _cfg()
    r = cfg.commands["twist"].ranges
    assert r.lin_vel_x == (-0.4, 0.4)
    assert r.lin_vel_y == (-0.3, 0.3)
    assert r.ang_vel_z == (-1.0, 1.0)


def test_cap_scales_linear_ranges_only():
    cfg = _cfg({CAP_ENV: "0.15"})
    r = cfg.commands["twist"].ranges
    assert r.lin_vel_x == (-0.15, 0.15)
    # y scales by the SAME ratio (0.15 / 0.4 = 0.375) so the sampled direction
    # distribution keeps its shape — otherwise the cap would quietly also change
    # how much sideways command the policy sees.
    assert r.lin_vel_y == pytest.approx((-0.1125, 0.1125))
    assert r.ang_vel_z == (-1.0, 1.0), "the yaw channel must not be touched"


def test_zero_or_garbage_means_uncapped():
    for raw in ("0", "0.0", ""):
        cfg = _cfg({CAP_ENV: raw})
        assert cfg.commands["twist"].ranges.lin_vel_x == (-0.4, 0.4), raw
