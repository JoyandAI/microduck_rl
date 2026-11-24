"""Stance / actuator hygiene for the walking task (2026-09-11).

Context this locks in. A forensic re-sim of the exported walking policy
(`walk_hd1910_15000.onnx`, trained by run `2026-09-10_18-43-53`) measured:

  * ``hip_yaw`` **commanded** +70 deg (left) / +35 deg (right) against hard
    stops at +30 / +25 deg — 52 % of commands outside the joint range, the
    joint sitting on a stop 41 % of the time, raw network output reaching
    |a| = 10 rad where 1 rad is nominal. The resulting ~16 deg common-mode
    foot yaw is what makes the stance look twisted.
  * ``neck_pitch`` driven through a +-15.7 deg sine at the 2.15 Hz gait
    frequency — the commanded TARGET swings, so it is a policy output, not
    gravity sag. Clamping the neck actions to HOME drops the head's
    trunk-relative nod from 22.3 deg std to 2.5 deg while the robot still
    walks (0.172 of 0.200 m/s, 0 falls), so the motion is freely avoidable.

Both defects were essentially UNPRICED: mjlab's ``pose`` term is
``exp(-mean_j(err_j^2/std_j^2))`` over ten leg joints, saturated to 0.029 of a
~7.5/step total, and nothing else charged head motion (``body_ang_vel`` is
trunk-only, ``angular_momentum`` contributed 2e-6/step).

These tests pin the fix: the terms exist, their ramp starts at exactly zero,
their sign convention is the penalty one, and the two new mdp functions behave
as documented.
"""

import importlib
import os

import pytest
import torch

from mjlab.managers import CurriculumTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    ACTION_OVER_LIMIT_OVERSHOOT,
    AIR_TIME_PROGRESS_GATE_TURN_REF,
    AIR_TIME_PROGRESS_GATE_V_REF,
    ACTION_OVER_LIMIT_WEIGHT,
    ENABLE_STANCE_HYGIENE,
    NECK_MOTION_DEADBAND_RAD_S,
    NECK_MOTION_WEIGHT,
    NUM_STEPS_PER_ENV,
    STANCE_HIP_YAW_L1_DEADBAND,
    STANCE_HIP_YAW_L1_WEIGHT,
    make_microduck_velocity_env_cfg,
)

HYGIENE_TERMS = ("stance_hip_yaw", "action_over_limit", "neck_motion")

CFG_MODULE = "mjlab_microduck.tasks.microduck_velocity_env_cfg"
# 2026-09-17: the DEFAULT recipe was flipped back to the official upstream recipe
# (see the OFFICIAL ALIGNMENT block in the cfg module). This file pins the
# 2026-09-11 stance/actuator-hygiene stack, which is still fully supported and
# exactly one env var away — so it re-enables the stack for its own tests and
# restores the default module state afterwards. Without this fixture the tests
# below would be asserting the *old* default, not the feature.
_LEGACY_ENV = {
    "MICRODUCK_STANCE_HYGIENE": "1",
    "MICRODUCK_PER_JOINT_POSTURE": "1",
    "MICRODUCK_AIR_TIME_GATE": "1",
}


@pytest.fixture(autouse=True)
def _hygiene_on():
    mod = importlib.import_module(CFG_MODULE)
    saved = {k: os.environ.get(k) for k in _LEGACY_ENV}
    for k, v in _LEGACY_ENV.items():
        os.environ[k] = v
    importlib.reload(mod)
    try:
        yield mod
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(mod)


# --------------------------------------------------------------------------- #
# cfg-level invariants
# --------------------------------------------------------------------------- #
def test_velocity_cfg_registers_the_hygiene_terms(_hygiene_on):
    assert _hygiene_on.ENABLE_STANCE_HYGIENE is True  # re-enabled by the fixture
    cfg = make_microduck_velocity_env_cfg()
    for name in HYGIENE_TERMS:
        assert name in cfg.rewards, name
        assert isinstance(cfg.rewards[name], RewardTermCfg)
    # `stance_hip_roll` was deliberately NOT added: the identified static sag of
    # that joint under load is ~12 deg, so an L1 on it would tax the part the
    # servo cannot escape. Its pose std was relaxed instead.
    assert "stance_hip_roll" not in cfg.rewards


def test_hygiene_terms_start_at_zero_and_are_ramped():
    """AGENTS.md: a tax active while a skill is being explored makes 'do nothing' win."""
    cfg = make_microduck_velocity_env_cfg()
    for name in HYGIENE_TERMS:
        assert cfg.rewards[name].weight == 0.0, f"{name} must start at zero weight"

    cur = cfg.curriculum["stance_tax_weight"]
    assert isinstance(cur, CurriculumTermCfg)
    assert cur.func is microduck_mdp.staged_reward_weight_ramp
    assert cur.params["ramp_steps"] > 0
    ramped = {s["reward_name"]: s["weight"] for s in cur.params["stages"]}
    # every ramped term exists and starts at zero; the zero-command stillness
    # terms ride the same ramp (see their own test)
    assert set(HYGIENE_TERMS) <= set(ramped)
    assert set(ramped) <= set(cfg.rewards)
    # every ramped target is the penalty-signed constant the module exports
    assert ramped["stance_hip_yaw"] == STANCE_HIP_YAW_L1_WEIGHT
    assert ramped["action_over_limit"] == ACTION_OVER_LIMIT_WEIGHT
    assert ramped["neck_motion"] == NECK_MOTION_WEIGHT


def test_every_hygiene_weight_is_a_penalty():
    """AGENTS.md rule: every Episode_Reward/<penalty> must be <= 0."""
    for c in (STANCE_HIP_YAW_L1_WEIGHT, ACTION_OVER_LIMIT_WEIGHT, NECK_MOTION_WEIGHT):
        assert c < 0.0, c


def test_hygiene_targets_the_walking_defect_joints_only():
    cfg = make_microduck_velocity_env_cfg()

    yaw_cfg = cfg.rewards["stance_hip_yaw"].params["asset_cfg"]
    assert isinstance(yaw_cfg, SceneEntityCfg)
    pat = yaw_cfg.joint_names[0]
    assert "hip_yaw" in pat
    # must not accidentally include the passive/backlash joints
    assert pat.startswith(r"^(?!passive_)")

    # hip_pitch/knee/ankle are NOT taxed: walking physically requires them to
    # leave HOME, and that is exactly what saturated the stock pose term.
    for banned in ("hip_pitch", "knee", "ankle"):
        assert banned not in pat

    assert cfg.rewards["stance_hip_yaw"].params["deadband"] == STANCE_HIP_YAW_L1_DEADBAND
    assert cfg.rewards["neck_motion"].params["deadband"] == NECK_MOTION_DEADBAND_RAD_S
    assert cfg.rewards["action_over_limit"].params["overshoot"] == ACTION_OVER_LIMIT_OVERSHOOT


def test_pose_hip_roll_std_is_the_official_value_not_the_widened_one():
    """0.12 was a widened target for a defect it did not explain (2026-09-17).

    History, and why the arrow points this way: 0.12 was introduced 2026-09-11
    on the theory that hip_roll sags ~12 deg under load. That theory was later
    falsified twice over — the 12 deg was identified on the RETIRED 5 V rail, and
    the observed bias points OUTWARD while gravity sag points INWARD
    (docs/forensics/agent_A_reward_audit.md:88-90). Tightening the std back to
    0.05 was then measured to make the splay WORSE (L-R 0.454 -> 0.583,
    logs/queue_r3_status.txt) — but that is a result about one training arm, not
    evidence for keeping a non-official default. Official is 0.05.
    """
    cfg = make_microduck_velocity_env_cfg()
    for key in ("std_standing", "std_walking"):
        std = cfg.rewards["pose"].params[key]
        assert std[r".*hip_roll.*"] == pytest.approx(0.05), (key, std[r".*hip_roll.*"])


# --------------------------------------------------------------------------- #
# mdp function behaviour
# --------------------------------------------------------------------------- #
class _Data:
    def __init__(self, n, nj):
        self.joint_pos = torch.zeros(n, nj)
        self.default_joint_pos = torch.zeros(n, nj)
        self.joint_vel = torch.zeros(n, nj)
        self.joint_pos_limits = torch.zeros(n, nj, 2)
        self.joint_pos_limits[..., 0] = -1.0
        self.joint_pos_limits[..., 1] = 1.0


class _Asset:
    def __init__(self, data):
        self.data = data

    def find_joints(self, pattern):
        """Minimal stand-in: `_servo_joint_vel` resolves the servo-only view."""
        del pattern
        n = self.data.joint_pos.shape[1]
        return list(range(n)), [f"j{i}" for i in range(n)]


class _Scene:
    def __init__(self, asset):
        self._asset = asset

    def __getitem__(self, _):
        return self._asset


class _ActionTerm:
    def __init__(self, n, nj):
        self.raw_action = torch.zeros(n, nj)
        self.scale = 1.0
        self.offset = torch.zeros(nj)
        self.target_ids = torch.arange(nj)


class _ActionManager:
    def __init__(self, n, nj):
        self._t = _ActionTerm(n, nj)

    def get_term(self, _):
        return self._t


class _Env:
    def __init__(self, n=2, nj=14):
        self.num_envs = n
        self.device = "cpu"
        self.scene = _Scene(_Asset(_Data(n, nj)))
        self.action_manager = _ActionManager(n, nj)


def test_joint_deviation_l1_default_matches_the_original_l1():
    env = _Env()
    env.scene["robot"].data.joint_pos[0, :3] = torch.tensor([0.2, -0.3, 0.1])
    cfg = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*hip_yaw.*",))
    cfg.joint_ids = [0, 1]
    out = microduck_mdp.joint_deviation_l1(env, asset_cfg=cfg)
    assert torch.allclose(out, torch.tensor([0.5, 0.0]), atol=1e-6), out


def test_joint_deviation_l1_deadband_is_free_inside_and_linear_outside():
    env = _Env()
    cfg = SceneEntityCfg("robot", joint_names=(".*",))
    cfg.joint_ids = [0]
    d = env.scene["robot"].data
    for err, expected in ((0.0, 0.0), (0.05, 0.0), (0.10, 0.0), (0.30, 0.20)):
        d.joint_pos[0, 0] = err
        out = microduck_mdp.joint_deviation_l1(env, asset_cfg=cfg, deadband=0.10)
        assert abs(float(out[0]) - expected) < 1e-6, (err, float(out[0]))
    # never negative, so a negative weight stays a penalty
    assert float(microduck_mdp.joint_deviation_l1(env, asset_cfg=cfg, deadband=0.1)[0]) >= 0.0


def test_action_over_limit_fires_only_past_limit_plus_overshoot():
    env = _Env()
    d = env.scene["robot"].data
    d.joint_pos_limits[:, 0, 0] = -0.5
    d.joint_pos_limits[:, 0, 1] = +0.5
    am = env.action_manager.get_term("joint_pos")
    for target, expected in ((0.5, 0.0), (0.8, 0.0), (0.9, 0.1), (-0.8, 0.0), (-0.9, 0.1)):
        am.raw_action.zero_()
        am.raw_action[0, 0] = target
        out = microduck_mdp.action_over_limit_penalty(env, overshoot=0.3)
        assert abs(float(out[0]) - expected) < 1e-6, (target, float(out[0]))
    # at weight <= 0 this must never be positive
    assert float(microduck_mdp.action_over_limit_penalty(env)[0]) >= 0.0


def test_neck_motion_deadband_ignores_slow_aiming_and_charges_gait_pumping():
    """The whole point: a commanded head pose move is free, a gait pump is not."""
    env = _Env()
    d = env.scene["robot"].data
    neck = slice(5, 9)
    # A commanded re-aim at ~1.5 rad/s peak (a 1 rad move in ~1 s) must be free.
    d.joint_vel[:, neck] = 1.5
    slow = microduck_mdp.neck_joint_vel_deadband_l2(env, deadband=2.0)
    assert torch.allclose(slow, torch.zeros(2), atol=1e-9), slow
    # The measured defect: neck_pitch at ~3.8 rad/s peak => charged.
    d.joint_vel.zero_()
    d.joint_vel[:, 5] = 3.8
    pumped = microduck_mdp.neck_joint_vel_deadband_l2(env, deadband=2.0)
    assert abs(float(pumped[0]) - (3.8 - 2.0) ** 2) < 1e-6, pumped
    # zero deadband degenerates to the plain L2 version
    d.joint_vel[:, 5] = 3.8
    assert torch.allclose(
        microduck_mdp.neck_joint_vel_deadband_l2(env, deadband=0.0),
        microduck_mdp.neck_joint_vel_l2(env),
        atol=1e-6,
    )
    # sign: the term itself is >= 0 so a negative weight is a real penalty
    assert float(pumped[0]) >= 0.0


def test_staged_reward_weight_ramp_interpolates_and_returns_a_scalar():
    """The curriculum must be a real ramp, not a staircase, and a scalar return."""
    class _RM:
        def __init__(self):
            self.w = {"a": 7.0, "b": 7.0}

        def get_term_cfg(self, name):
            outer = self.w

            class _C:
                weight = 0.0

                def __set_name__(self, *_):
                    pass

            class _T:
                def __init__(self):
                    self.name = name

                @property
                def weight(self):
                    return outer[self.name]

                @weight.setter
                def weight(self, v):
                    outer[self.name] = v

            return _T()

    class _E:
        device = "cpu"

        def __init__(self, step):
            self.common_step_counter = step
            self.reward_manager = _RM()

    stages = [{"reward_name": "a", "weight": -1.0}, {"reward_name": "b", "weight": -0.5}]
    env = _E(0)
    out = microduck_mdp.staged_reward_weight_ramp(env, None, stages, ramp_steps=100)
    assert out.ndim == 0, "CurriculumManager does term_state.item()"
    assert env.reward_manager.w["a"] == 0.0 and env.reward_manager.w["b"] == 0.0

    env = _E(50)
    microduck_mdp.staged_reward_weight_ramp(env, None, stages, ramp_steps=100)
    assert abs(env.reward_manager.w["a"] - (-0.5)) < 1e-6
    assert abs(env.reward_manager.w["b"] - (-0.25)) < 1e-6

    env = _E(10_000)
    microduck_mdp.staged_reward_weight_ramp(env, None, stages, ramp_steps=100)
    assert abs(env.reward_manager.w["a"] - (-1.0)) < 1e-6  # clamped at target
    assert abs(env.reward_manager.w["b"] - (-0.5)) < 1e-6


def test_ramp_completes_well_before_the_gait_must_be_learned():
    cfg = make_microduck_velocity_env_cfg()
    ramp_iters = cfg.curriculum["stance_tax_weight"].params["ramp_steps"] / NUM_STEPS_PER_ENV
    assert 0 < ramp_iters <= 500, ramp_iters


# --------------------------------------------------------------------------- #
# per_joint_posture: the structural fix for the dead posture gradient
# --------------------------------------------------------------------------- #
class _PostureAsset:
    def __init__(self, n, nj):
        self.data = _Data(n, nj)

    def find_joints(self, _):
        nj = self.data.joint_pos.shape[1]
        return list(range(nj)), [f"j{i}" for i in range(nj)]


class _PostureEnv:
    """Enough of an env for per_joint_posture: asset + a twist command."""

    # Realistic per-joint stds taken from the velocity cfg: ankle 0.25,
    # hip_yaw 0.3, hip_pitch/knee 0.4.
    STDS = {r"j0": 0.25, r"j1": 0.30, r"j2": 0.40}

    def __init__(self, n=2, nj=3, speed=0.2):
        self.num_envs = n
        self.device = "cpu"
        self._asset = _PostureAsset(n, nj)
        self.scene = _Scene(self._asset)
        self.command_manager = _Cmd(n, speed)
        cfg = SceneEntityCfg("robot", joint_names=(".*",))
        cfg.joint_ids = list(range(nj))
        from mjlab.managers import RewardTermCfg as _RTC

        params = {
            "asset_cfg": cfg,
            "command_name": "twist",
            "std_standing": self.STDS,
            "std_walking": self.STDS,
            "std_running": self.STDS,
            "walking_threshold": 0.01,
            "running_threshold": 1.5,
        }
        self.posture = microduck_mdp.per_joint_posture(_RTC(func=None, weight=1.0, params=params), self)
        self._params = params


class _Cmd:
    def __init__(self, n, speed=0.2):
        self._c = torch.zeros(n, 3)
        self._c[:, 0] = speed

    def get_command(self, _):
        return self._c


def _posture(env):
    p = env._params
    return env.posture(
        env,
        std_standing=p["std_standing"],
        std_walking=p["std_walking"],
        std_running=p["std_running"],
        asset_cfg=p["asset_cfg"],
        command_name=p["command_name"],
        walking_threshold=p["walking_threshold"],
        running_threshold=p["running_threshold"],
    )


def test_per_joint_posture_is_one_at_home():
    env = _PostureEnv()
    assert torch.allclose(_posture(env), torch.ones(2), atol=1e-6)


def test_per_joint_posture_survives_one_runaway_joint_where_the_collective_dies():
    """The whole point: the joint walking MUST swing far cannot mute the rest.

    Reproduces the measured local stance with the cfg's own stds — an ankle-like
    joint 45 deg off (as the real ankle is, +44/+48 deg), a hip_yaw-like joint
    16 deg off (the measured duck stance), and a joint at HOME.
    """
    env = _PostureEnv()
    d = env.scene["robot"].data
    d.joint_pos[0, 0] = 0.79   # ankle-like: 45 deg, std 0.25
    d.joint_pos[0, 1] = 0.28   # hip_yaw-like: 16 deg, std 0.30
    d.joint_pos[0, 2] = 0.0    # a well-behaved joint

    mine = float(_posture(env)[0])
    # mjlab's own form: exp(-mean(err^2 / std^2)) — note the collective divide.
    std = torch.tensor([0.25, 0.30, 0.40])
    err = d.joint_pos[0] - d.default_joint_pos[0]
    collective = float(torch.exp(-torch.mean(torch.square(err / std))))
    # per-joint mean: (4.6e-5 + 0.419 + 1.0)/3
    assert abs(mine - 0.473) < 0.01, mine
    # collective form collapses the WHOLE term to 0.027 (measured 0.015 on the
    # real policy vs 0.946 for the official one) — a 17x difference in the
    # gradient the optimizer actually sees.
    assert mine > 15 * collective, (mine, collective)
    # and the well-behaved joints keep a real gradient instead of being /10'd away
    assert mine > 0.4


def test_per_joint_posture_is_still_zero_when_every_joint_is_off():
    env = _PostureEnv()
    env.scene["robot"].data.joint_pos[:] = 5.0  # everything 5 rad off
    assert float(_posture(env)[0]) < 1e-6


def test_velocity_cfg_uses_the_per_joint_posture():
    cfg = make_microduck_velocity_env_cfg()
    assert cfg.rewards["pose"].func is microduck_mdp.per_joint_posture
    # identical stds and weight to the official recipe: only the reducer changed
    assert cfg.rewards["pose"].weight == 1.0
    assert set(cfg.rewards["pose"].params) >= {
        "std_standing", "std_walking", "std_running", "asset_cfg",
        "command_name", "walking_threshold",
    }


# --------------------------------------------------------------------------- #
# air_time progress gate: closes the "march on the spot" farm
# --------------------------------------------------------------------------- #
class _SensorData:
    def __init__(self, n, nfeet=2, air=0.2):
        # both feet airborne inside [threshold_min, threshold_max]
        self.current_air_time = torch.full((n, nfeet), air)


class _Sensor:
    def __init__(self, data):
        self.data = data


class _RobotData:
    def __init__(self, n):
        self.root_link_lin_vel_b = torch.zeros(n, 3)
        self.root_link_ang_vel_b = torch.zeros(n, 3)


class _GateEnv:
    """Fake env carrying only what feet_air_time_progress_gated touches."""

    def __init__(self, n=1):
        self.num_envs = n
        self.extras = {"log": {}}
        self._sensor = _Sensor(_SensorData(n))
        self._robot = _Asset(_RobotData(n))
        self._cmd = torch.zeros(n, 3)
        self.scene = {"feet_ground_contact": self._sensor, "robot": self._robot}
        self.command_manager = _Cmd3(self._cmd)

    def set_cmd(self, vx=0.0, vy=0.0, wz=0.0):
        self._cmd[0] = torch.tensor([vx, vy, wz])

    def set_vel_b(self, vx=0.0, vy=0.0, wz=0.0):
        self._robot.data.root_link_lin_vel_b[0] = torch.tensor([vx, vy, 0.0])
        self._robot.data.root_link_ang_vel_b[0] = torch.tensor([0.0, 0.0, wz])


class _Cmd3:
    def __init__(self, t):
        self._t = t

    def get_command(self, _):
        return self._t


def _air(env, **kw):
    return microduck_mdp.feet_air_time_progress_gated(
        env,
        sensor_name="feet_ground_contact",
        threshold_min=0.125,
        threshold_max=0.300,
        command_name="twist",
        command_threshold=0.01,
        **kw,
    )


def test_air_time_gate_disabled_is_a_passthrough():
    env = _GateEnv()
    env.set_cmd(vx=0.2)
    env.set_vel_b(vx=0.0)  # not moving at all
    out = _air(env, vel_gate_ref=0.0, turn_gate_ref=0.0)
    assert float(out[0]) == 2.0, out  # both feet counted, no gate


def test_air_time_gate_pays_the_floor_when_marching_in_place():
    """The defect: command non-zero, body stationary -> must NOT pay full price."""
    env = _GateEnv()
    env.set_cmd(vx=0.2)
    env.set_vel_b(vx=0.0)
    floored = float(_air(env, vel_gate_ref=0.15, turn_gate_ref=0.3, gate_floor=0.25)[0])
    assert abs(floored - 2.0 * 0.25) < 1e-6, floored

    # ... and the floor is what keeps the skill discoverable: it is NOT zero.
    assert floored > 0.0

    # At/above the reference speed it pays in full again.
    env.set_vel_b(vx=0.15)
    full = float(_air(env, vel_gate_ref=0.15, turn_gate_ref=0.3, gate_floor=0.25)[0])
    assert abs(full - 2.0) < 1e-6, full

    # Half the reference speed -> halfway between floor and full.
    env.set_vel_b(vx=0.075)
    half = float(_air(env, vel_gate_ref=0.15, turn_gate_ref=0.3, gate_floor=0.25)[0])
    assert abs(half - 2.0 * (0.25 + 0.75 * 0.5)) < 1e-6, half


def test_air_time_gate_measures_progress_along_the_COMMANDED_direction():
    """Strafe and reverse must count; moving the WRONG way must not."""
    env = _GateEnv()
    kw = dict(vel_gate_ref=0.15, turn_gate_ref=0.3, gate_floor=0.25)

    # commanded +y (strafe left), moving +y -> full
    env.set_cmd(vy=0.2)
    env.set_vel_b(vy=0.15)
    assert abs(float(_air(env, **kw)[0]) - 2.0) < 1e-6

    # commanded +y, moving -y (wrong way) -> floor only
    env.set_vel_b(vy=-0.15)
    assert abs(float(_air(env, **kw)[0]) - 0.5) < 1e-6

    # commanded forward, moving sideways fast: no forward progress -> floor
    env.set_cmd(vx=0.2)
    env.set_vel_b(vy=0.5)
    assert abs(float(_air(env, **kw)[0]) - 0.5) < 1e-6


def test_air_time_gate_uses_yaw_rate_for_turn_in_place():
    """15% of envs get lin=0 + |wz|>0.4; a forward-only gate would silence them."""
    env = _GateEnv()
    kw = dict(vel_gate_ref=0.15, turn_gate_ref=0.3, gate_floor=0.25)

    env.set_cmd(wz=1.0)          # spin on the spot
    env.set_vel_b(wz=0.0)        # not spinning yet
    assert abs(float(_air(env, **kw)[0]) - 0.5) < 1e-6  # floor

    env.set_vel_b(wz=0.3)        # spinning at the reference rate
    assert abs(float(_air(env, **kw)[0]) - 2.0) < 1e-6

    env.set_vel_b(wz=-0.3)       # either direction counts
    assert abs(float(_air(env, **kw)[0]) - 2.0) < 1e-6


def test_air_time_gate_keeps_the_command_gate_and_telemetry():
    env = _GateEnv()
    env.set_cmd(vx=0.0, vy=0.0, wz=0.0)   # no command at all
    env.set_vel_b(vx=0.5)
    out = _air(env, vel_gate_ref=0.15, turn_gate_ref=0.3)
    assert float(out[0]) == 0.0, out          # command gate still applies
    assert "Metrics/air_time_mean" in env.extras["log"]


def test_velocity_cfg_air_time_matches_the_state_of_the_gate(_hygiene_on):
    """The gate is a wiring switch, not a new function: with it off the term IS
    mjlab's stock ``feet_air_time`` (that is the official recipe — see
    tests/test_official_alignment.py), with it on the progress-gated version is
    swapped in. This module's fixture turns the gate ON, so the else branch runs
    here; the "off" branch is pinned by the alignment test.
    """
    cfg = make_microduck_velocity_env_cfg()
    from mjlab.tasks.velocity import mdp as _vel_mdp

    v_ref = _hygiene_on.AIR_TIME_PROGRESS_GATE_V_REF
    t_ref = _hygiene_on.AIR_TIME_PROGRESS_GATE_TURN_REF
    if v_ref <= 0.0 and t_ref <= 0.0:
        assert cfg.rewards["air_time"].func is _vel_mdp.feet_air_time
        assert "vel_gate_ref" not in cfg.rewards["air_time"].params
    else:
        assert cfg.rewards["air_time"].func is microduck_mdp.feet_air_time_progress_gated
        assert cfg.rewards["air_time"].params["vel_gate_ref"] == v_ref
        assert 0.0 < cfg.rewards["air_time"].params["gate_floor"] < 1.0


# --------------------------------------------------------------------------- #
# zero-command stillness: "don't march, walk only when commanded"
# --------------------------------------------------------------------------- #
def test_velocity_cfg_registers_the_zero_command_stillness_terms():
    cfg = make_microduck_velocity_env_cfg()
    assert "stillness_zero_cmd" in cfg.rewards
    assert "no_stepping" in cfg.rewards
    # positive reward + penalty, per AGENTS.md's sign convention
    ramped = {s["reward_name"]: s["weight"]
              for s in cfg.curriculum["stance_tax_weight"].params["stages"]}
    assert ramped["stillness_zero_cmd"] > 0.0
    assert ramped["no_stepping"] < 0.0
    # both must be gated on the command, so the walking gait is untouched
    for name in ("stillness_zero_cmd", "no_stepping"):
        assert cfg.rewards[name].params["command_name"] == "twist"
        assert cfg.rewards[name].params["command_threshold"] == 0.01
        assert cfg.rewards[name].weight == 0.0, "must start at zero and ramp"


class _StillEnv:
    """Fake env for stillness_at_zero_command / no_stepping_penalty."""

    def __init__(self, n=2):
        self.num_envs = n
        self._cmd = torch.zeros(n, 3)
        self._vel = torch.zeros(n, 3)
        self._found = torch.ones(n, 2, dtype=torch.bool)

        class _D:
            pass

        d = _D()
        d.root_link_vel_w = self._vel
        a = _Asset(d)
        self._asset = a

        class _SensData:
            pass

        sd = _SensData()
        sd.found = self._found

        class _Sens:
            data = sd

        self._sens = _Sens()
        # `no_stepping_penalty` reads env.scene.sensors[name].data.found,
        # while `stillness_at_zero_command` reads env.scene["robot"].
        self.scene = _SceneWithSensors(
            {"feet_ground_contact": self._sens}, {"robot": a}
        )
        self.command_manager = _Cmd3(self._cmd)


class _SceneWithSensors(dict):
    """dict-like scene that also exposes .sensors (no_stepping uses that path)."""

    def __init__(self, sensors, *a, **kw):
        super().__init__(*a, **kw)
        self.sensors = sensors


def test_stillness_pays_only_when_commanded_to_stand():
    env = _StillEnv()
    kw = dict(command_name="twist", command_threshold=0.01, vel_std=0.1)
    # commanded to stand, body still -> full reward
    assert abs(float(microduck_mdp.stillness_at_zero_command(env, **kw)[0]) - 1.0) < 1e-6
    # commanded to stand, body moving -> reward collapses (marching is not paid)
    env._vel[0] = torch.tensor([0.2, 0.0, 0.0])
    assert float(microduck_mdp.stillness_at_zero_command(env, **kw)[0]) < 0.02
    # a movement command switches the term OFF entirely
    env._cmd[0] = torch.tensor([0.2, 0.0, 0.0])
    assert float(microduck_mdp.stillness_at_zero_command(env, **kw)[0]) == 0.0


def test_no_stepping_charges_airborne_feet_only_at_zero_command():
    env = _StillEnv()
    kw = dict(sensor_name="feet_ground_contact", command_name="twist",
              command_threshold=0.01)
    # standing commanded, both feet down -> no charge
    assert float(microduck_mdp.no_stepping_penalty(env, **kw)[0]) == 0.0
    # standing commanded, one foot up (marching) -> charged
    env._found[0, 0] = False
    assert float(microduck_mdp.no_stepping_penalty(env, **kw)[0]) == 1.0
    env._found[0, 1] = False
    assert float(microduck_mdp.no_stepping_penalty(env, **kw)[0]) == 2.0
    # a movement command switches it off: walking is never taxed
    env._cmd[0] = torch.tensor([0.2, 0.0, 0.0])
    assert float(microduck_mdp.no_stepping_penalty(env, **kw)[0]) == 0.0
    # return value is >= 0, so a negative weight is a real penalty
    assert float(microduck_mdp.no_stepping_penalty(env, sensor_name="feet_ground_contact")[0]) >= 0.0
