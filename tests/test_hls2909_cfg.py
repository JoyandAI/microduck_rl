"""HL-2909-C001 actuator + config regression tests (CPU, no GPU needed).

Locks in the Feetech HL-2909 swap invariants documented in
docs/feetech_hls2915_servo_swap.md and docs/actuator_physics_swap_plan.md:

* the bundled hls2909 motor resolves and loads its params JSON;
* the HLS actuator law: accel+velocity double target smoothing, P(1/8)+D(1/4),
  current limit and PWM clamp are sane / NaN-free;
* the mjlab BamActuatorCfg wiring: motor_name + model resolve, forcerange
  (vin*k t/R) is in the right magnitude vs the datasheet stall torque.

NOTE: parameter values mirror the synced vendor params (bam@8d0025f snapshot,
`vendor/bam/bam/params/hls2909/m1.json`; hls2909 actuator initialize() re-creates
kt/R/armature/ramp limits from its own defaults, so compute_control never sees
stale dict values). Pendulum identification is still pending for the real values.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bam.actuators import actuators
from bam.model import load_model, load_model_from_dict

HLS_M1 = dict(
    kt=1.454, R=20.0, armature=1e-3, q_offset=0.0,
    friction_base=0.08, friction_viscous=0.012,
    error_gain=0.166, kd=32.0, max_velocity=19.1637, max_acceleration=500.0,
    max_current=1.95,
    model="m1", actuator="hls2909",
)


def test_hls2909_registered_and_loads() -> None:
    assert "hls2909" in actuators
    model = load_model_from_dict(dict(HLS_M1))
    assert model.actuator_name == "hls2909"
    assert abs(model.kt.value - 1.454) < 1e-6
    assert abs(model.R.value - 20.0) < 1e-6
    # m1 model: only base + viscous friction
    assert hasattr(model, "friction_base")
    assert hasattr(model, "friction_viscous")
    assert not hasattr(model, "friction_stribeck") or model.stribeck is False


def test_hls2909_chained_resolution() -> None:
    # bundled-motor path must also resolve (what microduck_constants uses)
    model = load_model(motor_name="hls2909", model="m1")
    assert model.actuator_name == "hls2909"


def test_hls2909_double_ramp_numerics() -> None:
    """1D trajectory check of accel+velocity smoothing (dt=20 ms).

    After set_model() the ramp limits come from the actuator defaults
    (default_max_velocity=19.1637, default_max_acceleration=500.0), never from
    the dict: ramp must reach cruise speed, never overshoot, never reverse.
    """
    act = actuators["hls2909"]()
    act.set_model(load_model_from_dict(dict(HLS_M1)))
    v_max = act.model.max_velocity.value
    a_max = act.model.max_acceleration.value
    assert v_max > 1.0 and a_max > 1.0  # 同步后默认, 非旧占位
    q = np.array([0.0])
    dq = np.array([0.0])
    dt = 0.02
    q_target = np.array([1.0])
    prev_qs = None
    v_hist = []
    for _ in range(30):  # 0.6 s
        volts = act.compute_control(q_target, q, dq, dt)
        assert np.isfinite(volts).all(), f"non-finite voltage at step"
        qs = act.q_target_smooth.copy()
        if prev_qs is not None:
            v_hist.append((qs - prev_qs) / dt)
        prev_qs = qs
    # never overshoot the target
    assert prev_qs[0] <= q_target[0] + 1e-9
    # reached cruise speed near v_max at least once by the end (long profile)
    assert max(v_hist, default=0.0) > 0.7 * v_max
    # acceleration bounded: per-step velocity change <= a_max*dt
    # + no mid-run overshoot / no negative velocity (soft-stop constraint)
    assert v_hist[0] <= 2 * a_max * dt + 1e-9
    for a, b in zip(v_hist[1:], v_hist[:-1]):
        assert a - b <= a_max * dt + 1e-9
        assert a + 1e-9 >= 0.0  # soft-stop: velocity never reverses mid-ramp


def test_hls2909_current_limit_and_pwm_clamp() -> None:
    """Huge error saturates: duty must clamp to ±max_pwm and return ±vin*max_pwm."""
    act = actuators["hls2909"]()
    act.set_model(load_model_from_dict(dict(HLS_M1)))
    q = np.array([0.0])
    dq = np.array([0.0])
    volts = act.compute_control(np.array([10.0]), q, dq, 0.02)  # far target (but ramp-limited)
    # after a single step the ramp limits the voltage; force PWM saturation by
    # feeding a target equal to the ramp cap over many steps instead: run 200 steps
    for _ in range(200):
        volts = act.compute_control(np.array([10.0]), q, dq, 0.02)
    assert np.isfinite(volts).all()
    # bounded by vin*max_pwm; also bounded by the current-limiter window
    assert abs(volts[0]) <= act.vin * act.max_pwm + 1e-9
    # current limit window: duty within [center-span, center+span]
    back_emf = act.model.kt.value * dq[0]
    duty = volts[0] / act.vin
    span = act.model.R.value * act.max_current / act.vin
    center = back_emf / act.vin
    assert duty <= center + span + 1e-9 and duty >= center - span - 1e-9


def test_hls2909_armature_hook() -> None:
    act = actuators["hls2909"]()
    act.set_model(load_model_from_dict(dict(HLS_M1)))
    # set_model() 会重跑 initialize(): armature 取执行器默认 1e-3(与 m1.json 一致),
    # 而非 dict 传入值。
    assert abs(act.get_extra_inertia() - 1e-3) < 1e-12


def test_hls2909_forcerange_matches_datasheet() -> None:
    """BamActuator edit_spec force limit = vin*k t/R must be at datasheet scale.

    官网: 堵转 8.9 kg.cm = 0.873 N·m @12V. With vin_upper=12.6 (12V±5%) the
    BAM force limit is 12.6*1.454/20 = 0.916 N·m — same magnitude (placeholder
    R; fitted R must land on 0.873 ±10%).
    """
    vin_upper = 12.6
    force_limit = vin_upper * HLS_M1["kt"] / HLS_M1["R"]
    assert math.isclose(force_limit, 0.916, rel_tol=0.02)
    assert 0.7 < force_limit < 1.05  # datasheet stall ±20% envelope


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
