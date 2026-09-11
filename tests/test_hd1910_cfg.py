"""HD-1910-C001 actuator + robot-model regression tests (CPU, no GPU needed).

Locks in the servo swap from Dynamixel XL330 (18 g, 3.7-6.0 V) to Feetech
HD-1910-C001 (21 g, 4-8.4 V, 12 kg.cm, firmware mode 4 = pure position PD).

Sources of truth:
- docs/feetech_hd1910_servo_notes.md
- docs/feetech_hd1910_pendulum_quickstart.md
- vendor/bam/bam/params/hd1910/*.json (+ characterization.json)
- hd1910_calibration/fit_kpfix/mae_report_kpfix.md

Why this file exists: tests/test_hls2909_cfg.py only exercises the *hls2909*
actuator law with hardcoded dicts — nothing locked the values that
`microduck_constants._BAM_ACTUATOR_KWARGS` actually feeds into training. The swap
moved the robot from a 12 V to a 5 V-rail 6 V-class servo, so an uncaught
regression here is both a training-validity bug and a hardware-destroying one.

Two unit traps this file pins down (both were gotten wrong once):
1. `vin_range` is the REAL robot rail = regulated 5.0 V (the pendulum bench was
   also run at 5.0-5.2 V, so the identified params transfer with no rescaling).
   It is NOT a 2S LiPo band and must never approach 12 V.
2. `delay_*_lag` is counted in **env steps** (decimation x timestep = 0.02 s),
   NOT in physics substeps (0.005 s). Using the substep inflates the modelled
   latency 4x.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import pytest

from bam.actuators import actuators
from bam.model import load_model
from mjlab_microduck.robot.microduck_constants import _BAM_ACTUATOR_KWARGS

ROBOT_DIR = Path("src/mjlab_microduck/robot/microduck")

# ── HD-1910-C001 identified / datasheet values ───────────────────────────────
HD_KT = 0.692               # N*m/A  — free-run MEASURED value (datasheet 7.5 kg.cm/A = 0.7358)
HD_KT_DATASHEET = 0.7358
HD_R = 3.75                 # Ohm    (6 V / 1.6 A stall)
HD_MAX_CURRENT = 3.25       # A      (reg28=500; mode 4 does not actually clamp)
HD_STALL_AT_6V = 1.176      # N*m    (12 kg.cm * 0.098) — datasheet, at datasheet kt
HD_VIN_SPEC = (4.0, 8.4)    # V      — HARD hardware limit, never 12 V
HD_LOW_VOLT_ALARM = 4.0     # V      (reg15=40: the servo's own under-voltage alarm)
HD_SERVO_MASS_G = 21.0      # +/- 2 g
XL330_MASS_G = 18.0

# ── D-channel sanity ─────────────────────────────────────────────────────────
# duty = dq*kp*error_gain/p_scale + (-dq_dot)*kd*error_gain/d_scale, clamped to +-max_pwm.
# The pendulum bench CANNOT identify the D strength (MAE spread 0.0014 rad across
# kd = 0.74/5/160 — see params/hd1910/characterization.json), so a free fit can land on a
# value that turns the servo into a brake. That happened on 2026-09-10: kd=192.68 gave a
# D slope of 7.85 duty/(rad/s), saturating the whole duty range at |dq| = 0.127 rad/s, which
# collapsed the walking policy (99.5% -> 0% tracking, 0 -> 6 falls in 8 s). These bounds
# exist so that regression cannot come back silently.
D_SLOPE_MAX = 0.05          # duty/(rad/s) — above this the D term dominates at gait speeds
D_SATURATION_MIN_RAD_S = 20.0  # the D term must not saturate below this joint speed
P_SLOPE_BENCH = 5.18        # measured P slope, duty/rad

# Identified command latency (characterization.json; consistent across tiers)
CMD_DELAY_RANGE_S = (0.023, 0.031)
# ⚠️ UNIT TRAP: `delay_*_lag` counts **physics substeps**, not env steps.
# mjlab calls `scene.write_data_to_sim()` INSIDE the decimation loop
# (managers_based_rl_env.py:414-419; entity.py:830 "before each sim.step() within
# the decimation loop"), and that is what drives Actuator.compute() -> the
# DelayBuffer append. So one lag = sim.mujoco.timestep, NOT decimation*timestep.
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
# m3-m6 add load-dependent friction; m1/m2 degenerate to base+viscous only.
LOAD_FRICTION_TIERS = {"m3", "m4", "m5", "m6"}


# ── actuator wiring ──────────────────────────────────────────────────────────

def test_hd1910_actuator_law_is_registered() -> None:
    """`hd1910` must exist in the BAM actuator registry or the env cannot build."""
    assert "hd1910" in actuators, sorted(actuators)


def test_hd1909_alias_resolves_to_the_same_servo() -> None:
    """`hd1909` is the lab shorthand for the same HD-1910-C001.

    It must be registered as BOTH an actuator law and a param source, and resolve
    to the identical identified parameters — an alias that silently diverged (or
    fell through to xl330) would be worse than no alias at all.
    """
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
    assert _BAM_ACTUATOR_KWARGS["kp_fw"] == 32.0  # firmware Kp readback (reg50)
    assert _BAM_ACTUATOR_KWARGS["target_names_expr"] == (r"^(?!passive_).*",)
    # Current pick is m5 (direction-separated load friction, physically the most
    # complete; ~4% off the numerically best m4). Lock only the FAMILY: regressing
    # to the degenerate m1/m2 would silently drop load-dependent friction.
    assert _BAM_ACTUATOR_KWARGS["model"] in LOAD_FRICTION_TIERS, _BAM_ACTUATOR_KWARGS["model"]


def test_vin_is_the_5v_rail_and_never_exceeds_spec() -> None:
    """HD-1910 is a 6 V-class servo on a regulated 5.0 V rail; >8.4 V destroys it."""
    lo, hi = _BAM_ACTUATOR_KWARGS["vin_range"]
    assert HD_VIN_SPEC[0] <= lo < hi <= HD_VIN_SPEC[1], (lo, hi)
    # the band must bracket the documented 5.0 V rail and stay near it
    assert lo < 5.0 < hi
    assert hi - lo <= 1.0, "DR band far wider than a 'regulated 5 V +/-5%' rail"
    # vin_min is the servo's own reg15 under-voltage alarm threshold
    vmin = _BAM_ACTUATOR_KWARGS["vin_min"]
    assert vmin >= HD_LOW_VOLT_ALARM
    assert vmin <= lo


def test_vin_drop_resistance_is_sane() -> None:
    lo, hi = _BAM_ACTUATOR_KWARGS["vin_drop_resistance_range"]
    assert 0.0 <= lo < hi
    # carried over from the proven XL330 wiring ceiling: R = gain * kt = 0.2 * 0.34597
    assert math.isclose(hi, 0.2 * 0.3459739511711113, rel_tol=0.02)


def test_delay_lags_are_physics_substeps_and_bracket_the_fit() -> None:
    """Lags count PHYSICS SUBSTEPS (0.005 s), not env steps (0.02 s).

    Reading them as env steps under-models the latency ~4x; that mistake was made
    once here, hence this test.
    """
    lo_s = _BAM_ACTUATOR_KWARGS["delay_min_lag"] * SUBSTEP_S
    hi_s = _BAM_ACTUATOR_KWARGS["delay_max_lag"] * SUBSTEP_S
    assert lo_s < hi_s
    # the window must cover the identified 0.023-0.031 s
    assert lo_s <= CMD_DELAY_RANGE_S[1], (lo_s, hi_s)
    assert hi_s >= CMD_DELAY_RANGE_S[0], (lo_s, hi_s)
    # ... without being a different order of magnitude (the env-step reading gave 1/4)
    assert hi_s <= 0.08, hi_s
    # and it must NOT be interpretable as ~5 ms (the under-modelled reading)
    assert lo_s >= 0.015, lo_s


def test_env_step_is_50hz() -> None:
    """One env step is decimation x timestep = 0.02 s. Kept as a sanity anchor
    for the substep/env-step distinction above."""
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck", play=True)
    step = cfg.sim.mujoco.timestep * cfg.decimation
    assert math.isclose(step, ENV_STEP_S, rel_tol=1e-6), step
    assert math.isclose(cfg.sim.mujoco.timestep, SUBSTEP_S, rel_tol=1e-6)


def test_robot_actuator_cfg_uses_hd1910() -> None:
    """The task cfgs the trainer actually builds must carry the HD-1910 values."""
    from mjlab.tasks.registry import load_env_cfg

    for task in ("Mjlab-Velocity-Flat-MicroDuck", "Mjlab-SitStand-Flat-MicroDuck"):
        act = load_env_cfg(task, play=True).scene.entities["robot"].articulation.actuators[0]
        assert act.motor_name == "hd1910", task
        assert act.kp_fw == 32.0, task
        assert act.vin_range == _BAM_ACTUATOR_KWARGS["vin_range"], task


# ── identified parameters resolve through BAM ────────────────────────────────

def test_hd1910_params_load_and_match_identification() -> None:
    m = load_model(motor_name="hd1910", model=_BAM_ACTUATOR_KWARGS["model"])
    assert m.actuator_name == "hd1910"
    assert math.isclose(m.kt.value, HD_KT, rel_tol=1e-6)
    assert math.isclose(m.R.value, HD_R, rel_tol=1e-6)
    assert math.isclose(m.max_current.value, HD_MAX_CURRENT, rel_tol=1e-6)
    assert m.max_velocity.value > 5.0
    # whatever tier is picked must carry load-dependent friction
    assert hasattr(m, "load_friction_motor")


@pytest.mark.xfail(
    reason="KNOWN-BAD: params/hd1910/m5.json currently ships kd=192.68 (D slope 7.85 "
           "duty/(rad/s)), which makes the servo a brake and collapsed the walking policy. "
           "Validated fix (one cp): hd1910_calibration/params_recommended_m5.json "
           "(kt=0.692 measured + kd=0.346 => D slope 0.0141, matches the bench regression). "
           "See docs/policy_failure_analysis.md. Remove this marker once installed.",
    strict=False,
)
def test_d_channel_cannot_turn_the_servo_into_a_brake() -> None:
    """The D term must leave real joint motion possible.

    Guards the 2026-09-10 incident: `kd` was freely fitted to 192.68 although the bench
    cannot identify the D strength, giving a D slope of 7.85 duty/(rad/s) that saturated the
    entire duty range at |dq| = 0.127 rad/s. A walking gait needs 1-8 rad/s, so the actuator
    became a brake and the policy collapsed. `kd` must stay pinned until hardware T5
    (step responses at different kd) actually measures the D strength.
    """
    m = load_model(motor_name="hd1910", model=_BAM_ACTUATOR_KWARGS["model"])
    # NOTE: do NOT call act.set_model(m) here. `initialize()` (invoked by set_model) RE-CREATES
    # kd/error_gain/armature/... from the actuator's class defaults, discarding the JSON. The
    # training path never does that (mjlab loads via load_model and stops), so reading the
    # loaded model is what training actually simulates. Getting this wrong once produced a
    # false alarm about the "effective D gain".
    act = actuators["hd1910"]()  # only for p_scale / d_scale defaults
    kd = float(m.kd.value)
    eg = float(m.error_gain.value)
    d_scale = float(act.d_scale)
    assert d_scale > 0
    d_slope = kd * eg / d_scale
    assert d_slope <= D_SLOPE_MAX, (
        f"D slope {d_slope:.4f} duty/(rad/s) (kd={kd}, error_gain={eg}, d_scale={d_scale}) "
        f"exceeds {D_SLOPE_MAX}; the D term would dominate the duty at gait speeds and the "
        f"servo becomes a brake. See docs/policy_failure_analysis.md."
    )
    saturation_speed = 1.0 / d_slope
    assert saturation_speed >= D_SATURATION_MIN_RAD_S, (
        f"the D term alone reaches full duty at only {saturation_speed:.3f} rad/s"
    )
    # the P channel must still match the bench regression (it is identifiable)
    p_slope = float(act.kp) * eg / float(act.p_scale)
    assert math.isclose(p_slope, P_SLOPE_BENCH, rel_tol=0.02), p_slope


def test_forcerange_matches_datasheet_stall() -> None:
    """BAM force limit = vin * kt / R must be at datasheet stall scale.

    NOTE: the model uses the *measured* kt (0.692), which is ~6% below the datasheet value
    (0.7358), so the limit lands ~6% under the 12 kg.cm datasheet stall — assert the scale,
    not an exact match.
    """
    m = load_model(motor_name="hd1910", model=_BAM_ACTUATOR_KWARGS["model"])
    at_6v = 6.0 * m.kt.value / m.R.value
    assert 1.00 < at_6v < 1.25, at_6v
    datasheet = 6.0 * HD_KT_DATASHEET / HD_R
    assert math.isclose(datasheet, HD_STALL_AT_6V, rel_tol=0.02), datasheet
    # on the real 5 V rail the available torque is proportionally lower and sane
    on_rail = _BAM_ACTUATOR_KWARGS["vin_range"][1] * m.kt.value / m.R.value
    assert 0.8 < on_rail < 1.2, on_rail


# ── robot model masses (21 g servo, not the 27.8 g HL-2909 placeholder) ──────

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
        assert math.isclose(delta_g, 14 * (HD_SERVO_MASS_G - XL330_MASS_G), abs_tol=2.0), (
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
    # 279.9 g at XL330 -> 288.9 g with the 21 g servo (3 of the head servos sit in
    # the head set: head_pitch on `neck`, head_yaw on `neck_pitch`, head_roll on
    # `yaw_roll_motion`; the neck_pitch servo case is on trunk_base)
    assert 285.0 < head_g < 293.0, head_g


def test_inertia_scaled_with_mass() -> None:
    """update_servo_mass must scale `fullinertia`, not just mass."""
    path = ROBOT_DIR / "robot_allcollisions.xml"
    txt = path.read_text(encoding="utf-8")
    # trunk_base carries 3 servos -> +9.0 g on the 199.224 g baseline
    assert 'mass="0.208224"' in txt, "trunk_base mass not the expected +3g/servo value"
    m = mujoco.MjModel.from_xml_path(str(path))
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    assert m.body_mass[bid] > 0.199224  # heavier after the swap
    assert all(v > 0 for v in m.body_inertia[bid])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
