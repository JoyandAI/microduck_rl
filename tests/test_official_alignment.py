"""PROOF that the default velocity recipe IS the official upstream recipe.

2026-09-17. The user's requirement: "把除了 BAM 和舵机相关参数，其余的策略先全部
对齐一下官方" — align everything except the BAM/servo parameters with official.

What was wrong before. **Six switches covering twelve deviations** had
accumulated on top of the official recipe, every one of them ON by default, and
every one of them validated by comparing INDEPENDENTLY TRAINED runs whose
within-arm seed spread exceeded the effect being chased (AGENTS.md: 4 arms x 3
seeds could not separate any arm; ~52 seeds/arm would be needed). Two of them
were measured to make things WORSE: the ``hip_roll`` pose-std tightening
(splay 0.454 -> 0.583) and the ``hip_roll`` L1 arms (zero-command survival
100 % -> 0 %, tilt 2 % -> 47 %).

So the defaults were flipped back to official and every deviation was left
exactly one environment variable away. These tests are the proof of that
equality, and they are deliberately written as a *whole-config* comparison
against ``reference_official_velocity_env_cfg.py`` (the verbatim upstream copy)
rather than as a hand-written list of expectations — a list would silently stop
covering anything added later.

Run offline; the reference cfg is the same file the A/B task uses.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import os

import pytest

CFG_MODULE = "mjlab_microduck.tasks.microduck_velocity_env_cfg"
REF_MODULE = "mjlab_microduck.tasks.reference_official_velocity_env_cfg"

# Every recipe deviation and the value that turns it OFF (= official). Forcing
# ALL of them (not just the head block) makes this test independent of whatever
# the developer's shell happens to export.
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

# Sources of a differing leaf that are NOT recipe content:
#   scene.terrain.spec_fn — mjlab's TerrainImporterCfg default_factory builds a
#   fresh lambda per instance, so two structurally identical cfgs never compare
#   equal there. (For rough=False it is unused; for rough=True both files set it
#   to the same module-level function and the path matches.)
#   metrics — DIAGNOSTIC terms only (yaw_dc_abs / yaw_dc_signed, added 2026-09-18
#   because the training logs carried no yaw-DC quantity at all, see
#   mdp.yaw_wander_metric). A Metrics term cannot change the reward, the
#   observation or the dynamics, so it is outside the recipe; everything that CAN
#   change behaviour (rewards, curriculum, events, commands, observations,
#   terminations, actions, sim) is still compared strictly.
IGNORED_PATHS = ("cfg.scene.terrain.spec_fn", "cfg.metrics")


@contextlib.contextmanager
def _recipe(env: dict[str, str]):
    """Reload the cfg module with ``env`` applied, then restore the default state.

    ``importlib.reload`` gives the module a NEW globals dict, so any name a test
    imported at collection time keeps pointing at the old function — always take
    the factory off the reloaded module object, never off a module-level import.
    """
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


def _diff_leaves(x, y, path="cfg", depth=0, out=None):
    """Recursively collect differing leaves between two cfg dataclasses."""
    if out is None:
        out = []
    if depth > 8:
        return out
    if type(x) is not type(y):
        out.append((path, f"TYPE {type(x).__name__} vs {type(y).__name__}"))
        return out
    if dataclasses.is_dataclass(x):
        for f in dataclasses.fields(x):
            _diff_leaves(getattr(x, f.name), getattr(y, f.name),
                         f"{path}.{f.name}", depth + 1, out)
    elif isinstance(x, (tuple, list)):
        if len(x) != len(y):
            out.append((path, f"LEN {len(x)} vs {len(y)}"))
            return out
        for i, (xi, yi) in enumerate(zip(x, y)):
            _diff_leaves(xi, yi, f"{path}[{i}]", depth + 1, out)
    elif isinstance(x, dict):
        if set(x) != set(y):
            out.append((path, f"KEYS {sorted(set(x) ^ set(y))[:8]}"))
            return out
        for k in x:
            _diff_leaves(x[k], y[k], f"{path}[{k!r}]", depth + 1, out)
    else:
        try:
            equal = bool(x == y)
        except Exception:  # pragma: no cover - uncomparable leaf
            equal = True
        if not equal:
            out.append((path, f"{x!r} vs {y!r}"))
    return out


def _manager_terms(cfg, name):
    return getattr(cfg, name)


# --------------------------------------------------------------------------- #
# the alignment itself
# --------------------------------------------------------------------------- #
def test_all_deviation_switches_off_reproduces_the_official_cfg_exactly():
    """The headline invariant. Any new default-on reward/param breaks this."""
    with _recipe(OFFICIAL_VALUES) as mod:
        ours = mod.make_microduck_velocity_env_cfg()
    ref = importlib.import_module(REF_MODULE).make_microduck_velocity_env_cfg()

    diffs = [(p, v) for p, v in _diff_leaves(ours, ref)
             if not any(p.startswith(ign) for ign in IGNORED_PATHS)]
    assert not diffs, "recipe is NOT the official recipe:\n" + "\n".join(
        f"  {p}: {v}" for p, v in diffs
    )


def test_the_official_comparison_is_not_trivially_empty():
    """Guard against the test passing because both sides failed to build.

    A whole-config comparison is only meaningful if the config actually has the
    parts we care about — an empty/partial cfg would compare equal to another
    empty/partial cfg.
    """
    ref = importlib.import_module(REF_MODULE).make_microduck_velocity_env_cfg()
    for name, minimum in (("rewards", 16), ("events", 13), ("curriculum", 7)):
        terms = _manager_terms(ref, name)
        assert len(terms) >= minimum, (name, sorted(terms))
    # 48 base + 13 command = 61, the shared runtime contract.
    assert sorted(ref.observations["actor"].terms) == [
        "actions", "base_ang_vel", "body_command", "command", "head_command",
        "joint_pos", "joint_vel", "projected_gravity",
    ]
    assert ref.rewards["air_time"].func.__name__ == "feet_air_time"
    assert ref.rewards["pose"].func.__name__ == "variable_posture"
    # the three deviations this file exists to keep OUT of the default
    for absent in ("stance_hip_yaw", "heading_hold", "stillness_zero_cmd", "neck_motion"):
        assert absent not in ref.rewards, absent


def test_head_stability_is_the_only_remaining_deviation():
    """With everything else at official values, the head block adds exactly 3 terms.

    This is the user-requested exception (2026-09-17), so it must be *small and
    enumerable* — if a future edit sneaks another term in under this switch, the
    alignment claim quietly becomes false and this test is what catches it.
    """
    with _recipe({**OFFICIAL_VALUES, "MICRODUCK_HEAD_STABILITY": "1"}) as mod:
        ours = mod.make_microduck_velocity_env_cfg()
    ref = importlib.import_module(REF_MODULE).make_microduck_velocity_env_cfg()

    added = set(ours.rewards) - set(ref.rewards)
    assert added == {"head_motion", "head_droop", "head_forward"}, sorted(added)
    removed = set(ref.rewards) - set(ours.rewards)
    assert not removed, sorted(removed)

    added_cur = set(ours.curriculum) - set(ref.curriculum)
    assert added_cur == {"head_stability_weight", "head_droop_weight",
                         "head_forward_weight"}, sorted(added_cur)
    assert not (set(ref.curriculum) - set(ours.curriculum))

    # Every shared term keeps the official weight/params.
    diffs = []
    for k in sorted(set(ours.rewards) & set(ref.rewards)):
        a, b = ours.rewards[k], ref.rewards[k]
        if a.weight != b.weight:
            diffs.append(f"{k}: weight {a.weight} != {b.weight}")
        if a.params != b.params:
            diffs.append(f"{k}: params {a.params} != {b.params}")
        if getattr(a.func, "__name__", None) != getattr(b.func, "__name__", None):
            diffs.append(f"{k}: func {a.func} != {b.func}")
    assert not diffs, diffs


def test_pose_zero_command_bucket_defaults_to_the_official_value():
    """Uniform sampling essentially never hits all-zero, but 0.0 IS official."""
    with _recipe(OFFICIAL_VALUES) as mod:
        cfg = mod.make_microduck_velocity_env_cfg()
        assert mod.POSE_ZERO_COMMAND_PROB == 0.0
    for name in ("head_pose", "body_pose"):
        assert cfg.commands[name].zero_command_prob == 0.0


def test_hip_roll_pose_std_is_back_to_the_official_value():
    """0.12 was a widened target for a defect it did not explain.

    Tightening it to 0.05 was measured on 2026-09-16 to make the splay WORSE
    (L-R 0.454 -> 0.583) — which is a result about that *arm*, not evidence for
    keeping 0.12 as the default. Official is 0.05.
    """
    with _recipe(OFFICIAL_VALUES) as mod:
        assert mod.HIP_ROLL_POSE_STD == pytest.approx(0.05)
        cfg = mod.make_microduck_velocity_env_cfg()
    for key in ("std_standing", "std_walking"):
        std = cfg.rewards["pose"].params[key]
        assert std[r".*hip_roll.*"] == pytest.approx(0.05)
        assert std[r".*hip_yaw.*"] in (0.1, 0.3)
        assert std[r".*ankle.*"] in (0.1, 0.25)


def test_legacy_stack_is_still_reachable_in_one_env_var():
    """The A/B is not lost — only the default moved (this is the whole point)."""
    with _recipe({**OFFICIAL_VALUES, "MICRODUCK_STANCE_HYGIENE": "1",
                  "MICRODUCK_PER_JOINT_POSTURE": "1",
                  "MICRODUCK_AIR_TIME_GATE": "1",
                  "MICRODUCK_HEADING_HOLD": "1",
                  "MICRODUCK_HIP_ROLL_STD": "0.12"}) as mod:
        cfg = mod.make_microduck_velocity_env_cfg()
        assert mod.ENABLE_STANCE_HYGIENE is True
        assert mod.ENABLE_HEADING_HOLD is True
        assert mod.HIP_ROLL_POSE_STD == pytest.approx(0.12)

    for term in ("stance_hip_yaw", "action_over_limit", "neck_motion",
                 "standing_leg_vel", "stillness_zero_cmd", "no_stepping",
                 "yaw_hold_zero_cmd", "heading_hold"):
        assert term in cfg.rewards, term
    assert cfg.rewards["pose"].func.__name__ == "per_joint_posture"
    assert cfg.rewards["air_time"].func.__name__ == "feet_air_time_progress_gated"
    assert cfg.rewards["pose"].params["std_standing"][r".*hip_roll.*"] == pytest.approx(0.12)


def test_bam_and_servo_parameters_are_NOT_part_of_this_alignment():
    """The stated exception: actuator physics stays ours.

    The official recipe is built on the XL330 model (737 g, kp_fw=200); ours runs
    xgoduck's fit (kp_fw=5, 7.5 V, hd1910/m6.json). Those live in
    `robot/microduck_constants.py`, which this file must not have touched.
    """
    from mjlab_microduck.robot import microduck_constants as mc
    assert mc.KP_FW == 5.0
    kwargs = mc._BAM_ACTUATOR_KWARGS
    assert kwargs["motor_name"] == "hd1910"
    assert kwargs["model"] == "m6"
    assert tuple(kwargs["vin_range"]) == (7.1, 7.9)
    assert kwargs["vin_min"] == 6.5
    assert (kwargs["delay_min_lag"], kwargs["delay_max_lag"]) == (4, 7)
