"""Regression checks for HD-1910 actuator and robot configurations."""
from __future__ import annotations

import math
from pathlib import Path

import mujoco
import pytest

from bam.actuators import actuators
from bam.model import load_model
from mjlab_microduck.robot.microduck_constants import _BAM_ACTUATOR_KWARGS

ROBOT_DIR = Path("src/mjlab_microduck/robot/microduck")

HD1910_KT_DATASHEET = 0.7358
HD1910_R_DATASHEET = 3.75       # Ohm    (6 V / 1.6 A stall) — datasheet, not xgoduck's fit
HD1910_KT = 0.6237611235393989
HD1910_R = 4.910191564179625
HD1910_ERROR_GAIN_RATIO = 1.321738701681577
HD1910_FRICTION_BASE = 0.015803365327573494
HD1910_LOAD_FRICTION_MOTOR = 0.06851344455161522
HD1910_MAX_VELOCITY = 100.0
HD1910_ERROR_GAIN = 0.166      # actuator class default; not stored in the JSON
HD1910_STALL_AT_6V = 1.176      # N*m    (12 kg.cm * 0.098) — datasheet, at datasheet kt
HD1910_VIN_SPEC = (4.0, 8.4)    # V      — HARD hardware limit, never 12 V
HD1910_VIN_RAIL = 7.5            # V      — nominal robot rail
HD1910_LOW_VOLT_ALARM = 4.0     # V      (reg15=40: the servo's own under-voltage alarm)
HD1910_SERVO_MASS_G = 21.0      # +/- 2 g

P_SLOPE_XGODUCK = 5.0 * HD1910_ERROR_GAIN * HD1910_ERROR_GAIN_RATIO

CMD_DELAY_RANGE_S = (0.023, 0.031)
SUBSTEP_S = 0.005
ENV_STEP_S = 0.02           # decimation 4 x 0.005 (locked by a test below)

MODELS = (
    "robot_walk.xml",
    "robot_allcollisions.xml",
    "robot_allcollisions_rollers.xml",
    "robot_walk_backlash.xml",
    "robot_allcollisions_backlash.xml",
    "robot_allcollisions_rollers_backlash.xml",
)
HEAD_BODIES = {"neck", "neck_pitch", "yaw_roll_motion", "jaw_soft"}
LOAD_FRICTION_TIERS = {"m3", "m4", "m5", "m6"}


def test_hd1910_actuator_law_is_registered() -> None:
    assert "hd1910" in actuators, sorted(actuators)


def test_hd1909_alias_resolves_to_the_same_servo() -> None:
    assert "hd1909" in actuators, sorted(actuators)
    canonical = load_model(motor_name="hd1910", model="m5")
    alias = load_model(motor_name="hd1909", model="m5")
    assert alias.actuator_name == canonical.actuator_name
    assert alias.kt.value == canonical.kt.value
    assert alias.R.value == canonical.R.value
    assert alias.friction_base.value == canonical.friction_base.value
    assert alias.error_gain.value == canonical.error_gain.value


def test_bam_kwargs_point_at_hd1910() -> None:
    assert _BAM_ACTUATOR_KWARGS["motor_name"] == "hd1910"
    assert _BAM_ACTUATOR_KWARGS["kp_fw"] == 5.0  # P gain of xgoduck's fit
    assert _BAM_ACTUATOR_KWARGS["target_names_expr"] == (r"^(?!passive_).*",)
    assert _BAM_ACTUATOR_KWARGS["model"] in LOAD_FRICTION_TIERS, _BAM_ACTUATOR_KWARGS["model"]


def test_training_actuator_is_the_xgoduck_m6_not_xl330() -> None:
    assert _BAM_ACTUATOR_KWARGS["motor_name"] == "hd1910"
    assert _BAM_ACTUATOR_KWARGS["model"] == "m6"
    assert _BAM_ACTUATOR_KWARGS["kp_fw"] == 5.0
    model = load_model(motor_name="hd1910", model="m6")
    assert model.actuator_name == "sts3215", model.actuator_name
    assert not hasattr(model, "kd")


def test_the_params_bam_loads_are_the_xgoduck_m6_identification() -> None:
    import json
    raw = json.loads(Path("vendor/bam/bam/params/hd1910/m6.json").read_text(encoding="utf-8"))
    assert "xgoduck" in raw["source"]
    model = load_model(motor_name="hd1910", model=_BAM_ACTUATOR_KWARGS["model"])
    assert model.actuator_name == "sts3215", model.actuator_name
    assert model.friction_base.value == pytest.approx(HD1910_FRICTION_BASE, rel=1e-9)
    assert model.load_friction_motor.value == pytest.approx(HD1910_LOAD_FRICTION_MOTOR, rel=1e-9)
    assert model.error_gain_ratio.value == pytest.approx(HD1910_ERROR_GAIN_RATIO, rel=1e-9)
    assert model.max_velocity.value == pytest.approx(HD1910_MAX_VELOCITY, rel=1e-9)


def test_registered_tasks_carry_the_hd1910_actuator() -> None:
    import mjlab_microduck.tasks  # noqa: F401  (the import registers the tasks)
    from mjlab.tasks.registry import list_tasks, load_env_cfg

    wanted = [
        "Mjlab-Velocity-Flat-MicroDuck",
        "Mjlab-Velocity-Flat-Backlash-MicroDuck",
        "Mjlab-Velocity-Flat-Rollers-MicroDuck",
        "Mjlab-SitStand-Flat-MicroDuck",
    ]
    available = set(list_tasks())
    checked, offenders = 0, []
    for task_id in wanted:
        if task_id not in available:
            continue
        cfg = load_env_cfg(task_id, play=False)
        for ent_name, ent in (cfg.scene.entities or {}).items():
            art = getattr(ent, "articulation", None)
            for act in (getattr(art, "actuators", ()) or ()):
                motor = getattr(act, "motor_name", None)
                mdl = getattr(act, "model", None)
                if motor is None and mdl is None:
                    continue
                checked += 1
                if (motor, mdl) != ("hd1910", "m6"):
                    offenders.append(f"{task_id}:{ent_name}:{motor}/{mdl}")
    assert checked > 0, "no actuator cfgs found — the registry walk stopped working"
    assert not offenders, offenders


def test_actuator_cfgs_are_only_built_from_the_hd1910_kwargs() -> None:
    import ast

    names = {"FrictionDRBamActuatorCfg", "BacklashEncoderBamActuatorCfg"}
    src_root = Path("src/mjlab_microduck")
    offenders, sites = [], 0
    for py in src_root.rglob("*.py"):
        tree = ast.parse(py.read_text(errors="replace"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or getattr(node.func, "id", None) not in names:
                continue
            sites += 1
            kws = [k.arg for k in node.keywords]
            if "motor_name" in kws or "model" in kws:
                offenders.append(f"{py}:{node.lineno}: literal servo kwargs {kws}")
            elif not any(k.arg is None and getattr(k.value, "id", "") == "_BAM_ACTUATOR_KWARGS"
                         for k in node.keywords):
                offenders.append(f"{py}:{node.lineno}: built without **_BAM_ACTUATOR_KWARGS")
    assert sites > 0, "no actuator construction sites found — the walk stopped working"
    assert not offenders, offenders


def test_vin_is_the_7v5_rail_and_never_exceeds_spec() -> None:
    lo, hi = _BAM_ACTUATOR_KWARGS["vin_range"]
    assert HD1910_VIN_SPEC[0] <= lo < hi <= HD1910_VIN_SPEC[1], (lo, hi)
    assert lo < HD1910_VIN_RAIL < hi, (lo, hi)
    assert hi - lo <= 1.0, "DR band far wider than a 'regulated 7.5 V +/-5%' rail"
    assert lo > 6.0, (lo, hi)
    vmin = _BAM_ACTUATOR_KWARGS["vin_min"]
    assert vmin >= HD1910_LOW_VOLT_ALARM
    assert vmin <= lo


def test_vin_drop_resistance_is_sane() -> None:
    lo, hi = _BAM_ACTUATOR_KWARGS["vin_drop_resistance_range"]
    assert 0.0 <= lo < hi
    assert math.isclose(hi, 0.2 * 0.3459739511711113, rel_tol=0.02)


def test_delay_lags_are_physics_substeps_and_bracket_the_fit() -> None:
    lo_s = _BAM_ACTUATOR_KWARGS["delay_min_lag"] * SUBSTEP_S
    hi_s = _BAM_ACTUATOR_KWARGS["delay_max_lag"] * SUBSTEP_S
    assert lo_s < hi_s
    assert lo_s <= CMD_DELAY_RANGE_S[1], (lo_s, hi_s)
    assert hi_s >= CMD_DELAY_RANGE_S[0], (lo_s, hi_s)
    assert hi_s <= 0.08, hi_s
    assert lo_s >= 0.015, lo_s


def test_env_step_is_50hz() -> None:
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck", play=True)
    step = cfg.sim.mujoco.timestep * cfg.decimation
    assert math.isclose(step, ENV_STEP_S, rel_tol=1e-6), step
    assert math.isclose(cfg.sim.mujoco.timestep, SUBSTEP_S, rel_tol=1e-6)


def test_robot_actuator_cfg_uses_hd1910() -> None:
    from mjlab.tasks.registry import load_env_cfg

    for task in ("Mjlab-Velocity-Flat-MicroDuck", "Mjlab-SitStand-Flat-MicroDuck"):
        act = load_env_cfg(task, play=True).scene.entities["robot"].articulation.actuators[0]
        assert act.motor_name == "hd1910", task
        assert act.kp_fw == 5.0, task
        assert act.vin_range == _BAM_ACTUATOR_KWARGS["vin_range"], task


def test_hd1910_params_load_and_match_identification() -> None:
    m = load_model(motor_name="hd1910", model=_BAM_ACTUATOR_KWARGS["model"])
    assert m.actuator_name == "sts3215"
    assert math.isclose(m.kt.value, HD1910_KT, rel_tol=1e-6)
    assert math.isclose(m.R.value, HD1910_R, rel_tol=1e-6)
    assert math.isclose(m.error_gain_ratio.value, HD1910_ERROR_GAIN_RATIO, rel_tol=1e-6)
    assert m.max_velocity.value == pytest.approx(HD1910_MAX_VELOCITY)
    assert hasattr(m, "load_friction_motor")


def test_p_channel_matches_the_xgoduck_identification() -> None:
    m = load_model(motor_name="hd1910", model=_BAM_ACTUATOR_KWARGS["model"])
    assert not hasattr(m, "kd")
    m.actuator.kp = _BAM_ACTUATOR_KWARGS["kp_fw"]
    p_slope = (
        float(m.actuator.kp)
        * float(m.actuator.error_gain)
        * float(m.error_gain_ratio.value)
    )
    assert math.isclose(p_slope, P_SLOPE_XGODUCK, rel_tol=1e-6), p_slope


def test_forcerange_matches_the_loaded_identification() -> None:
    m = load_model(motor_name="hd1910", model=_BAM_ACTUATOR_KWARGS["model"])
    at_6v = 6.0 * m.kt.value / m.R.value
    assert at_6v == pytest.approx(6.0 * HD1910_KT / HD1910_R, rel=1e-9)
    datasheet = 6.0 * HD1910_KT_DATASHEET / HD1910_R_DATASHEET
    assert math.isclose(datasheet, HD1910_STALL_AT_6V, rel_tol=0.02), datasheet
    on_rail = _BAM_ACTUATOR_KWARGS["vin_range"][1] * m.kt.value / m.R.value
    assert on_rail == pytest.approx(
        _BAM_ACTUATOR_KWARGS["vin_range"][1] * HD1910_KT / HD1910_R, rel=1e-9
    )
    assert math.isclose(on_rail / at_6v, _BAM_ACTUATOR_KWARGS["vin_range"][1] / 6.0,
                        rel_tol=0.02), on_rail


@pytest.mark.parametrize("fname", MODELS)
def test_model_masses_match_hd1910(fname: str) -> None:
    path = ROBOT_DIR / fname
    if not path.exists():
        pytest.skip(f"{fname} missing")
    model = mujoco.MjModel.from_xml_path(str(path))
    total_g = sum(model.body_mass.tolist()) * 1000.0

    bak = path.with_suffix(path.suffix + ".bak_servoswap")
    if bak.exists():
        base = mujoco.MjModel.from_xml_path(str(bak))
        delta_g = total_g - sum(base.body_mass.tolist()) * 1000.0
        assert math.isclose(delta_g, 14 * (HD1910_SERVO_MASS_G - 18.0), abs_tol=2.0), (
            f"{fname}: servo mass delta {delta_g:.1f} g, expected +42 g"
        )
    else:
        assert 770.0 < total_g < 790.0, total_g


@pytest.mark.parametrize("fname", MODELS)
def test_head_assembly_mass(fname: str) -> None:
    path = ROBOT_DIR / fname
    if not path.exists():
        pytest.skip(f"{fname} missing")
    model = mujoco.MjModel.from_xml_path(str(path))
    head_g = sum(
        model.body_mass[b]
        for b in range(model.nbody)
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) in HEAD_BODIES
    ) * 1000.0
    assert 285.0 < head_g < 293.0, head_g


def test_inertia_scaled_with_mass() -> None:
    path = ROBOT_DIR / "robot_allcollisions.xml"
    txt = path.read_text(encoding="utf-8")
    assert 'mass="0.208224"' in txt, "trunk_base mass not the expected +3g/servo value"
    m = mujoco.MjModel.from_xml_path(str(path))
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    assert m.body_mass[bid] > 0.199224  # heavier after the swap
    assert all(v > 0 for v in m.body_inertia[bid])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
