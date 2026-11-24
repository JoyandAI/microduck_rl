"""`MICRODUCK_ZERO_CMD_TERMS` — the anti-marching trio on the OFFICIAL recipe.

WHY (2026-09-21). The unified model merges every direction that was measured to
help. The one thing the winning family (official recipe + head + flight) got wrong
was STANDING: at zero command it lifted its feet 14.7 mm at 2.15 Hz — marching in
place — while its velocity-tracking reward was perfectly satisfied, because
marching does not move the body. The 12-deviation family never did that (0.34 mm at
the SAME walking speed, 0.115 m/s), and the difference is these three terms:

  * `stillness_zero_cmd`  (`stillness_at_zero_command`) — monotonically decreasing
    in motion, so there is no threshold to sit just under (AGENTS.md's recommended
    shape);
  * `no_stepping`         (`no_stepping_penalty`) — its docstring is literally
    "Discourages marching in place when the robot should stand still";
  * `yaw_hold_zero_cmd`   (`yaw_hold_at_zero_command`) — holds the heading at zero
    command.

All three are gated on `cmd_speed < 0.01`, so they pay ZERO while walking: this
switch cannot damage the gait, by construction rather than by hope. What it adds
over the full `MICRODUCK_STANCE_HYGIENE` bundle is what it does NOT add — the four
posture/actuator taxes (stance_hip_yaw, action_over_limit, neck_motion,
standing_leg_vel) that belong to the family we abandoned.

These tests lock (a) the off-by-default equality with official, (b) that turning it
on adds EXACTLY the trio and its ramp, and (c) the zero-command gating that makes
the "no effect on walking" claim true.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import os
import re

import pytest

CFG_MODULE = "mjlab_microduck.tasks.microduck_velocity_env_cfg"
SWITCH = "MICRODUCK_ZERO_CMD_TERMS"

TRIO = {"stillness_zero_cmd", "no_stepping", "yaw_hold_zero_cmd"}

# official-alignment values (every deviation off) + the two levers the unified
# model keeps: the head block and the flight penalty.
BASE = {
    "MICRODUCK_STANCE_HYGIENE": "0",
    "MICRODUCK_STANCE_HIP_ROLL": "0",
    "MICRODUCK_PER_JOINT_POSTURE": "0",
    "MICRODUCK_AIR_TIME_GATE": "0",
    "MICRODUCK_HEADING_HOLD": "0",
    # 三件套的第三项（yaw_hold_zero_cmd）另有自己的开关，默认 OFF 时它的爬坡
    # 权重是 0.0（= 死项）。统一模型要它，所以这里显式打开。
    "MICRODUCK_YAW_HOLD_ZERO_CMD": "1",
    "MICRODUCK_ABDUCTION": "0",
    "MICRODUCK_YAW_DC": "0",
    "MICRODUCK_YAW_SYMMETRY": "0",
    "MICRODUCK_TURN_FIRST": "0",
    "MICRODUCK_SYMMETRY": "0",
    "MICRODUCK_HIP_ROLL_STD": "0.05",
    "MICRODUCK_POSE_ZERO_CMD_PROB": "0.0",
    "MICRODUCK_HEAD_STABILITY": "1",
    "MICRODUCK_FLIGHT_PENALTY": "1",
}


@contextlib.contextmanager
def _cfg_builder(extra: dict[str, str] | None = None):
    """Yield a zero-arg factory that rebuilds the cfg with `extra` applied."""
    env = dict(BASE)
    env.update(extra or {})
    mod = importlib.import_module(CFG_MODULE)
    saved = {k: os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            os.environ[k] = v
        importlib.reload(mod)
        yield mod.make_microduck_velocity_env_cfg
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(mod)


def _diff_leaves(x, y, path="cfg", depth=0, out=None):
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


def test_default_is_off():
    with _cfg_builder() as make:
        cfg = make()
    assert TRIO.isdisjoint(cfg.rewards), sorted(TRIO & set(cfg.rewards))
    assert "zero_cmd_tax_weight" not in cfg.curriculum
    assert "yaw_hold_zero_cmd" not in cfg.rewards  # BASE turns its own flag on, the switch is what registers it


def _snap(cfg) -> dict:
    """JSON-safe snapshot: callables become their __name__, so two cfgs built from
    DIFFERENT module reloads can be compared without false positives."""
    import json

    def enc(o):
        if callable(o):
            return f"<{getattr(o, '__name__', type(o).__name__)}>"
        if dataclasses.is_dataclass(o):
            return {f.name: enc(getattr(o, f.name)) for f in dataclasses.fields(o)}
        if isinstance(o, dict):
            return {str(k): enc(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [enc(v) for v in o]
        return o

    d = enc(cfg)
    d.get("scene", {}).get("terrain", {}).pop("spec_fn", None)   # fresh lambda per build
    d.pop("metrics", None)                                       # diagnostics only
    return json.loads(json.dumps(d, default=str))


def test_switch_adds_exactly_the_trio_and_its_ramp():
    with _cfg_builder() as make:          # noqa: F841  (BASE only, switch off)
        base = make()
    with _cfg_builder({SWITCH: "1"}) as make:
        capped = make()
    assert set(capped.rewards) - set(base.rewards) == TRIO, sorted(set(capped.rewards) ^ set(base.rewards))
    assert not set(base.rewards) - set(capped.rewards), "the switch removed a reward term"
    assert "zero_cmd_tax_weight" in capped.curriculum and "zero_cmd_tax_weight" not in base.curriculum
    # nothing outside `rewards`/`curriculum` may change
    a, b = _snap(base), _snap(capped)
    for section in sorted(set(a) | set(b)):
        if section in ("rewards", "curriculum"):
            continue
        assert a.get(section) == b.get(section), f"{section} changed: {a.get(section)} vs {b.get(section)}"
    # and inside `rewards` only the trio differs (weights start at 0 and are ramped)
    changed = {k for k in set(a["rewards"]) | set(b["rewards"]) if a["rewards"].get(k) != b["rewards"].get(k)}
    assert changed == TRIO, changed
    # the ramp must cover exactly the trio, or the weights stay 0 and the terms are dead
    names = [s["reward_name"] for s in capped.curriculum["zero_cmd_tax_weight"].params["stages"]]
    assert set(names) == TRIO, names


def test_the_trio_only_pays_at_zero_command():
    """"No effect on walking" is a construction claim: every term must be gated on
    a near-zero command, so a walking env pays nothing."""
    with _cfg_builder({SWITCH: "1"}) as make:
        cfg = make()
    for name in sorted(TRIO):
        term = cfg.rewards[name]
        thr = (term.params or {}).get("command_threshold")
        assert thr is not None, f"{name} is not command-gated: {term.params}"
        assert thr <= 0.01, f"{name} gate {thr} is reachable while walking"


def test_sanctioned_weights_are_the_documented_ones():
    with _cfg_builder({SWITCH: "1"}) as make:
        cfg = make()
    w = {s["reward_name"]: s["weight"] for s in cfg.curriculum["zero_cmd_tax_weight"].params["stages"]}
    assert w["stillness_zero_cmd"] == pytest.approx(0.8)
    assert w["no_stepping"] == pytest.approx(-0.4)   # penalty style -> NEGATIVE weight
    assert w["yaw_hold_zero_cmd"] == pytest.approx(0.8)  # needs MICRODUCK_YAW_HOLD_ZERO_CMD=1
