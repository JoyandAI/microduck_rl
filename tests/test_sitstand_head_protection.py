"""Sit-stand head protection: the head-tripod terminal rest (2026-09-16).

Context this locks in. The delivered sit-stand policy does not let its head sag —
it PLANTS it, and the pose pays:

  * head-ground contact force **3.16 N = 37 % of body weight** (feet carry 4.7 N),
  * the neck is *commanded* to -1.72 rad with 0.003 rad tracking error,
  * pinning the head makes the robot topple forward (tilt 38 -> 88 deg), so the
    35 deg forward lean DEPENDS on the head propping it,
  * that rest beats standing upright by **+3.69/step**, because the two
    trunk-HEIGHT terms saturate (0.995 / 0.967) and the only term that objects is
    the multiplicative ``posture_composite``, whose weight cannot outbid them,
  * the existing ``head_pose_bias`` penalty is worth -0.00084/step there, i.e.
    0.1 % of the gap — a weight increase of ~1000x would be needed, and would
    wreck every other term.

Separately, and measured on the current 7.5 V / 779 g model
(``scripts/sitstand_com_support_check.py``), the CENTRE-OF-MASS hypothesis is
FALSIFIED for the commanded pose: at HOME legs with the feet flat and trunk
z = 0.115 m the whole-body CoM sits **0.41 mm from the ankle axis** and
**17.1 mm inside** the foot support polygon, and folding the neck to its hard stop
moves the CoM 29.8 mm FORWARD and shrinks that margin to 0.96 mm. So the head-down
pose is not a balance fix; it is a posture the reward failed to charge for.

These tests pin the fix: the sensor and both reward terms are registered and ramp
from exactly zero, the sign conventions match the functions' outputs, the
STAND-and-ramp-settled gate opens where it must and stays shut during a
transition, and the commanded stand target really is statically standable.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from mjlab.managers import CurriculumTermCfg, RewardTermCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_sitstand_env_cfg import (
    HEAD_L1_WEIGHT,
    HEAD_FIX_RAMP_ITERS,
    HEAD_GROUND_FORCE_THRESHOLD_N,
    HEAD_GROUND_WEIGHT,
    NECK_LIMIT_PROXIMITY_MARGIN,
    NECK_LIMIT_PROXIMITY_WEIGHT,
    RAMP_DONE_EPS,
    make_microduck_sitstand_env_cfg,
)

REPO = Path(__file__).resolve().parents[1]
# Measured on the delivered sitstand_hd1910_8000.onnx, 60 s protocol:
# neck_pitch absolute angle at the STAND steady state, and neck_pitch's hard
# lower limit from robot_allcollisions.xml.
NECK_PARKED_RAD = -1.583
NECK_HARD_LO_RAD = -1.5707963267948966
NECK_HOME_RAD = 0.3491


# --------------------------------------------------------------------------- #
# cfg-level invariants
# --------------------------------------------------------------------------- #
def test_sitstand_cfg_registers_head_ground_sensor_and_terms():
    cfg = make_microduck_sitstand_env_cfg()
    names = [getattr(s, "name", None) for s in cfg.scene.sensors]
    assert "head_ground_contact" in names, names

    assert "head_ground" in cfg.rewards
    assert "neck_limit_proximity" in cfg.rewards
    for n in ("head_ground", "neck_limit_proximity"):
        assert isinstance(cfg.rewards[n], RewardTermCfg), n
        # ramped in by curriculum, so it must start at exactly zero
        assert cfg.rewards[n].weight == 0.0, n

    cur = cfg.curriculum["head_ground_weight"]
    assert isinstance(cur, CurriculumTermCfg)
    stages = cur.params["stages"]
    assert {s["reward_name"] for s in stages} == {"head_ground", "neck_limit_proximity"}
    assert cur.params["ramp_steps"] == HEAD_FIX_RAMP_ITERS * 24


def test_reward_weight_signs_match_what_the_functions_return():
    """The repo's two penalty conventions, bit four envs (AGENTS.md).

    ``posture_head_ground_penalty`` self-negates (returns <= 0) -> POSITIVE weight.
    ``joint_limit_proximity_standing`` is a stock cost (>= 0)    -> NEGATIVE weight.
    """
    assert HEAD_GROUND_WEIGHT > 0, "self-negating penalty must carry a POSITIVE weight"
    assert NECK_LIMIT_PROXIMITY_WEIGHT < 0, "stock cost must carry a NEGATIVE weight"

    cfg = make_microduck_sitstand_env_cfg()
    stages = {s["reward_name"]: s["weight"]
              for s in cfg.curriculum["head_ground_weight"].params["stages"]}
    assert stages["head_ground"] > 0
    assert stages["neck_limit_proximity"] < 0


def test_the_measured_parked_neck_is_inside_the_keepout_band():
    """The whole point: the term must bite at the angle the policy actually rests at.

    neck_pitch rests ON its hard lower stop (-1.583 vs a -1.5708 limit). The stock
    ``dof_pos_limits`` term only fires past the soft limit, i.e. the last ~7.5 % of
    range, so it reads ~0 there. The keep-out band must cover the resting angle.
    """
    soft_lo = NECK_HARD_LO_RAD + NECK_LIMIT_PROXIMITY_MARGIN
    assert NECK_PARKED_RAD < soft_lo, (NECK_PARKED_RAD, soft_lo)
    # ...and it must NOT tax the neutral pose the head is supposed to hold.
    assert NECK_HOME_RAD > soft_lo


# --------------------------------------------------------------------------- #
# stub env for the gated terms
# --------------------------------------------------------------------------- #
class _SensorData:
    def __init__(self, n, force_z):
        self.force = torch.zeros(n, 1, 3)
        self.force[:, 0, 2] = torch.as_tensor(force_z, dtype=torch.float32)


class _Sensor:
    def __init__(self, data):
        self.data = data


class _CmdTerm:
    def __init__(self, alpha):
        self.alpha = alpha


class _CmdMgr:
    def __init__(self, cmd, alpha):
        self._cmd = cmd
        self._alpha = alpha
        # The head_pose command is a separate 4-vector; all-zero = "hold HOME",
        # which is what the parked-neck tests need to reason about.
        self._head_cmd = torch.zeros(cmd.shape[0], 4)

    def get_command(self, name):
        return self._head_cmd if name == "head_pose" else self._cmd

    def get_term(self, _name):
        return _CmdTerm(self._alpha)


def _four_head_joints(n: int, neck_q: float) -> torch.Tensor:
    """(n, 4): neck_pitch = neck_q, the other three head joints at HOME (= 0)."""
    q = torch.zeros(n, 4)
    q[:, 0] = float(neck_q)
    return q


class _RobotData:
    """The head really has FOUR joints (neck_pitch, head_pitch, head_yaw,
    head_roll). The stub must model that, because the head reward functions take
    the command as a 4-vector: a 1-column `joint_pos` broadcasts against it into
    four copies of the same error and a "sum over joints" then silently reads 4x
    its true value. That is exactly what a first version of this stub did."""

    def __init__(self, n, q, limits):
        self.joint_pos = q
        self.joint_pos_limits = limits
        # HOME-relative reference: this stub reports angles already relative to
        # HOME, so default_joint_pos is 0 and `actual - default == actual`.
        self.default_joint_pos = torch.zeros_like(q)


class _Asset:
    def __init__(self, data):
        self.data = data


class _Scene:
    """mjlab's Scene is both a mapping (`scene["robot"]`) and has `.sensors`."""

    def __init__(self, d):
        self._d = d
        self.sensors = d

    def __getitem__(self, k):
        return self._d[k]

    def __contains__(self, k):
        return k in self._d


class _Env:
    """Fake env carrying only what the two gated terms touch."""

    def __init__(self, n=4, flag=0.0, alpha=0.0, force_z=0.0,
                 neck_q=NECK_PARKED_RAD, neck_lim=(NECK_HARD_LO_RAD, 1.0472)):
        self.num_envs = n
        self.device = torch.device("cpu")
        cmd = torch.zeros(n, 3)
        cmd[:, 0] = flag
        self.command_manager = _CmdMgr(cmd, torch.full((n,), float(alpha)))
        self._d = {
            "head_ground_contact": _Sensor(_SensorData(n, force_z)),
            "robot": _Asset(_RobotData(
                n,
                _four_head_joints(n, neck_q),
                torch.tensor([[list(neck_lim)]] * n, dtype=torch.float32),
            )),
        }
        self.scene = _Scene(self._d)
        # Pre-populate the head measurement-view cache that head_pose_tracking
        # normally fills from the real articulation, so the L1 term can be tested
        # without a MuJoCo model. Identity view (no backlash) = the plain servo.
        # FOUR ids, matching the real 4-joint head.
        self._head_pose_neck_ids = torch.arange(4, dtype=torch.long)
        self._head_pose_bl_ids = torch.arange(4, dtype=torch.long)
        self._head_pose_bl_mask = torch.zeros(4)

    def set(self, *, flag=None, alpha=None, force_z=None, neck_q=None):
        if flag is not None:
            self.command_manager._cmd[:, 0] = flag
        if alpha is not None:
            self.command_manager._alpha[:] = alpha
        if force_z is not None:
            self._d["head_ground_contact"].data.force[:, 0, 2] = force_z
        if neck_q is not None:
            self._d["robot"].data.joint_pos[:, 0] = neck_q
        return self


def _hg(env, **kw):
    return microduck_mdp.posture_head_ground_penalty(
        env, sensor_name="head_ground_contact", command_name="twist",
        ramp_eps=RAMP_DONE_EPS, force_threshold_n=HEAD_GROUND_FORCE_THRESHOLD_N, **kw)


def _nl(env, **kw):
    return microduck_mdp.joint_limit_proximity_standing(
        env, command_name="twist", ramp_eps=RAMP_DONE_EPS,
        margin=NECK_LIMIT_PROXIMITY_MARGIN,
        asset_cfg=type("C", (), {"name": "robot", "joint_ids": [0]})(), **kw)


# --------------------------------------------------------------------------- #
# posture_head_ground_penalty
# --------------------------------------------------------------------------- #
def test_head_ground_penalty_is_a_cost_and_scales_with_load():
    env = _Env(force_z=3.16)  # the measured tripod load
    out = _hg(env)
    assert torch.all(out <= 0), out
    # 3.16 N load, 0.5 N threshold -> -(3.16 - 0.5) = -2.66
    assert out[0].item() == pytest.approx(-(3.16 - HEAD_GROUND_FORCE_THRESHOLD_N), rel=1e-4)
    heavier = _hg(_Env(force_z=6.0))
    assert heavier[0].item() < out[0].item()


def test_head_ground_penalty_ignores_a_light_brush():
    assert torch.all(_hg(_Env(force_z=0.2)) == 0.0)


def test_head_ground_penalty_is_silent_while_seated():
    """A SIT command must not be charged: the head may legitimately rest on the floor."""
    env = _Env(flag=1.0, alpha=1.0, force_z=3.16)
    assert torch.all(_hg(env) == 0.0)


def test_head_ground_penalty_is_silent_mid_transition():
    """STAND commanded but the 2 s posture ramp has not settled -> head assist is FREE.

    This is the freedom the env deliberately protects (it removed the sit env's
    head-ground penalty precisely so a transition could use the head as a third
    contact point). The fix must buy back the REST pose without taxing the rise.
    """
    env = _Env(flag=0.0, alpha=0.9, force_z=3.16)  # still 90 % of the way from SIT
    assert torch.all(_hg(env) == 0.0), _hg(env)
    # ...and it opens once the ramp settles
    env.set(alpha=0.0)
    assert torch.all(_hg(env) < 0)


def test_head_ground_penalty_gate_is_exactly_at_the_documented_epsilon():
    env = _Env(flag=0.0, alpha=RAMP_DONE_EPS * 0.5, force_z=3.16)
    assert torch.all(_hg(env) < 0), "inside the epsilon band the gate must be open"
    env.set(alpha=RAMP_DONE_EPS * 2.0)
    assert torch.all(_hg(env) == 0.0), "outside the epsilon band the gate must be shut"


def test_head_ground_penalty_needs_no_sensor():
    """Defensive: an env without the sensor must not crash (and must pay nothing)."""
    env = _Env(force_z=3.16)
    env._d.pop("head_ground_contact")
    env.scene.sensors = env._d
    out = microduck_mdp.posture_head_ground_penalty(
        env, sensor_name="head_ground_contact", command_name="twist")
    assert out.shape == (env.num_envs,)
    assert torch.all(out == 0.0)


# --------------------------------------------------------------------------- #
# joint_limit_proximity_standing
# --------------------------------------------------------------------------- #
def test_neck_limit_term_fires_on_the_stop_and_is_gated():
    on_stop = _Env(neck_q=NECK_PARKED_RAD)
    off_stop = _Env(neck_q=NECK_HOME_RAD)
    assert torch.all(_nl(on_stop) > 0), _nl(on_stop)
    assert torch.all(_nl(off_stop) == 0.0), _nl(off_stop)
    # seated / mid-transition -> no charge
    assert torch.all(_nl(_Env(flag=1.0, alpha=1.0, neck_q=NECK_PARKED_RAD)) == 0.0)
    assert torch.all(_nl(_Env(flag=0.0, alpha=0.8, neck_q=NECK_PARKED_RAD)) == 0.0)


def test_neck_limit_term_is_linear_in_the_intrusion():
    """relu(soft_lo - q): proportional, and zero in the interior."""
    soft_lo = NECK_HARD_LO_RAD + NECK_LIMIT_PROXIMITY_MARGIN
    deep = _nl(_Env(neck_q=soft_lo - 0.3))[0].item()
    shallow = _nl(_Env(neck_q=soft_lo - 0.1))[0].item()
    assert deep == pytest.approx(0.3, abs=1e-4)
    assert shallow == pytest.approx(0.1, abs=1e-4)
    assert torch.all(_nl(_Env(neck_q=soft_lo + 0.01)) == 0.0)


# --------------------------------------------------------------------------- #
# the centre-of-mass verdict (the "是奖励还是重心" question)
# --------------------------------------------------------------------------- #
def _load_com_module():
    spec = importlib.util.spec_from_file_location(
        "sitstand_com_support_check", REPO / "scripts" / "sitstand_com_support_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_commanded_stand_pose_is_statically_standable():
    """The commanded target must be a real equilibrium — the 重心 question.

    Measured (and this test locks it): at HOME legs, head at HOME, feet flat, the
    CoM is 0.4 mm from the ankle axis and >10 mm inside the support polygon. If a
    future model/pose edit makes the STAND target unreachable, the policy would
    have to contort to balance and no reward change could fix it — so this is the
    regression test that distinguishes "reward is wrong" from "target is wrong".

    This is not hypothetical: HOME_FRAME's own docstring records that the OLD home
    pose put the CoM ~5 mm BEHIND the ankle axis and that this is exactly what made
    the standup policy droop its head as a counterweight.
    """
    mod = _load_com_module()
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(REPO / mod.ROBOT_XML))
    r = mod.analyse(m, mod.HOME, 0.0, 0.115, "HOME, feet flat")
    assert r["feet_flat"], r
    assert abs(r["com_over_ankle_offset_mm"]) < 5.0, r
    assert r["com_inside_margin_mm"] > 10.0, r
    # The neck servo has an order of magnitude of torque margin holding the head
    # up: this is NOT a "can't lift it" problem.
    assert r["neck_gravity_torque_Nm"] < 0.1, r


def test_folding_the_neck_makes_balance_worse_not_better():
    """Kills the "head as counterweight for a bad CoM" explanation at the target.

    Folding the neck to its hard stop moves the CoM forward and shrinks the
    support margin — so the head-down pose cannot be the policy repairing a CoM
    problem. It is only *standable* because the head props the lean.
    """
    mod = _load_com_module()
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(REPO / mod.ROBOT_XML))
    up = mod.analyse(m, mod.HOME, 0.0, 0.115, "head up")
    down = mod.analyse(m, mod.OBSERVED, 0.0, 0.115, "head at stop")
    assert up["feet_flat"] and down["feet_flat"]
    assert down["com_x_m"] > up["com_x_m"], (up["com_x_m"], down["com_x_m"])
    assert down["com_inside_margin_mm"] < up["com_inside_margin_mm"], (up, down)
    assert math.isclose(down["head_mass_g"], 288.9, abs_tol=2.0), down["head_mass_g"]

# --------------------------------------------------------------------------- #
# head_pose_error_l1_standing — the NON-DILUTED head price (2026-09-16)
# --------------------------------------------------------------------------- #
def test_head_l1_arm_is_off_by_default_and_env_switchable():
    import importlib
    cfg = make_microduck_sitstand_env_cfg()
    assert "head_pose_l1" not in cfg.rewards
    assert "head_l1_weight" not in cfg.curriculum


def test_head_l1_is_summed_not_averaged_over_joints():
    """THE reason this term exists.

    `head_pose_tracking` averages per-joint Gaussians, so one parked joint is
    diluted by three good ones (measured: a 110 deg neck error moves it only
    1.0 -> 0.667). The fix must price the parked joint at FULL strength, so the
    reduction has to be a SUM. This test fails if anyone "tidies" it to a mean.
    """
    env = _Env(n=1, neck_q=NECK_PARKED_RAD)
    out = microduck_mdp.head_pose_error_l1_standing(
        env, command_name="head_pose", ramp_eps=RAMP_DONE_EPS, deadband=0.10)
    # Only the neck is off, so |err| - deadband = |NECK_PARKED_RAD| - 0.10 and the
    # SUM over 4 joints (three of them at 0) equals that single joint's value.
    single = abs(NECK_PARKED_RAD) - 0.10
    assert out[0].item() == pytest.approx(-single, rel=2e-3), out[0].item()
    # The whole point: a MEAN would have divided it by 4. Assert the ratio, so the
    # test states the property ("summed, not averaged") rather than a magic number.
    mean_version = single / 4.0
    assert abs(out[0].item()) == pytest.approx(4.0 * mean_version, rel=1e-3)
    assert abs(out[0].item()) > 3.0 * mean_version, (
        f"value {out[0].item():.3f} looks averaged, not summed")


def test_head_l1_gradient_does_not_vanish_at_large_error():
    """The composite's gradient is multiplied away by its own collapsed factor
    (~-0.09/rad at the parked pose); an L1 has a CONSTANT slope, which is the
    whole point. Slope must stay ~1 per rad across the whole range."""
    # UNIFORMLY spaced samples: the slope is d(value)/dq, so unevenly spaced q
    # makes the finite differences differ even for a perfectly linear term.
    # (A first version used (-0.6, -1.0, -1.5, -1.9) -- steps of 0.4/0.5/0.4 --
    # and then asserted every difference was 0.4, which failed on a CORRECT term.)
    STEP = 0.4
    qs = [-0.4 - STEP * k for k in range(5)]          # -0.4 .. -2.0
    vals = []
    for q in qs:
        env = _Env(n=1, neck_q=q)
        vals.append(microduck_mdp.head_pose_error_l1_standing(
            env, command_name="head_pose", ramp_eps=RAMP_DONE_EPS, deadband=0.10)[0].item())
    diffs = [vals[i] - vals[i + 1] for i in range(len(vals) - 1)]
    for d in diffs:
        assert d == pytest.approx(STEP, rel=1e-3), (qs, vals, diffs)
    # and it must be linear, NOT saturating: the last step equals the first
    assert diffs[0] == pytest.approx(diffs[-1], rel=1e-3), diffs


def test_head_l1_deadband_makes_small_errors_free():
    for q in (0.0, 0.05, -0.08):
        env = _Env(n=1, neck_q=q)
        out = microduck_mdp.head_pose_error_l1_standing(
            env, command_name="head_pose", ramp_eps=RAMP_DONE_EPS, deadband=0.10)
        assert torch.all(out == 0.0), (q, out)


def test_head_l1_is_gated_to_the_standing_rest_only():
    """Seated and mid-transition must be free: the env deliberately allows the
    head as a third contact point during the manoeuvre."""
    seated = _Env(n=1, flag=1.0, alpha=1.0, neck_q=NECK_PARKED_RAD)
    mid = _Env(n=1, flag=0.0, alpha=0.8, neck_q=NECK_PARKED_RAD)
    f = lambda e: microduck_mdp.head_pose_error_l1_standing(   # noqa: E731
        e, command_name="head_pose", ramp_eps=RAMP_DONE_EPS, deadband=0.10)
    assert torch.all(f(seated) == 0.0)
    assert torch.all(f(mid) == 0.0)
    assert torch.all(f(_Env(n=1, flag=0.0, alpha=0.0, neck_q=NECK_PARKED_RAD)) < 0)


def test_head_l1_weight_sign_is_positive_and_ramps_from_zero():
    assert HEAD_L1_WEIGHT > 0, "self-negating penalty -> POSITIVE weight"
    import importlib, os
    os.environ["MICRODUCK_HEAD_L1"] = "1"
    try:
        mod = importlib.import_module("mjlab_microduck.tasks.microduck_sitstand_env_cfg")
        importlib.reload(mod)
        cfg = mod.make_microduck_sitstand_env_cfg()
        assert cfg.rewards["head_pose_l1"].weight == 0.0, "must start at 0"
        st = cfg.curriculum["head_l1_weight"].params["stages"][0]
        assert st["reward_name"] == "head_pose_l1" and st["weight"] > 0
    finally:
        os.environ.pop("MICRODUCK_HEAD_L1", None)
        importlib.reload(mod)
