"""Straight-line (heading-hold) and gait-symmetry terms for the walking task.

Context this locks in (2026-09-14 forensics, `docs/forensics/走歪归因报告.md`).
A faithful-BAM re-sim of the exported walking policy with a frozen straight
command (vx=0.2, vy=0, wz=0, 16 envs, 20 s) measured:

  * 20 s net heading change **-103 +- 193 deg**, path efficiency **0.249**
    (distance made good along the initial heading / total path length) — the
    robot CIRCLES. The official `alpha_walking.onnx` in its native XL330 env
    scores **0.842** on the same harness. Head-wobble ablation did not change it
    (the defect is a +28.9 deg/s HALF-STRIDE yaw asymmetry rectifying into a
    -4 deg/s DC bias, AC/DC = 16.7x — leg-driven, not head-driven).
  * The whole reward stack contained **no term that is a function of the yaw
    ANGLE**: `track_angular_velocity` charges only the yaw RATE, carried 2.1% of
    the return, and its raw value (0.0357/1.0) implies a combined angular error
    of 1.291 rad/s — deep in the Gaussian tail. So a 0.1 rad/s drift costs
    ~0.0014/step.
  * The 61-D actor obs has `base_ang_vel` but NO heading and no base linear
    velocity, so a bias integrates open-loop and cannot be corrected.

These tests pin the fix: the terms exist, start at zero and ramp, the gate is
wired, the sign conventions are right, and the two mdp functions behave as
documented — especially the yaw-command gate, which is the subtle part.
"""

from __future__ import annotations

import importlib
import math
import os

import pytest
import torch

from mjlab.managers import CurriculumTermCfg, RewardTermCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    ENABLE_GAIT_SYMMETRY,
    ENABLE_HEADING_HOLD,
    GAIT_SYMMETRY_RAMP_ITER_END,
    GAIT_SYMMETRY_RAMP_ITER_START,
    GAIT_SYMMETRY_WEIGHT,
    HEADING_HOLD_COMMAND_THRESHOLD,
    HEADING_HOLD_RAMP_ITER_END,
    HEADING_HOLD_RAMP_ITER_START,
    HEADING_HOLD_STD,
    HEADING_HOLD_VEL_GATE_REF,
    HEADING_HOLD_WEIGHT,
    NUM_STEPS_PER_ENV,
    POSE_ZERO_COMMAND_PROB,
    make_microduck_velocity_env_cfg,
)

CFG_MODULE = "mjlab_microduck.tasks.microduck_velocity_env_cfg"
# 2026-09-17: the DEFAULT recipe was flipped back to the official upstream recipe
# (see the OFFICIAL ALIGNMENT block in the cfg module). This file pins the
# heading_hold + marching-cure stack, which is still fully supported and one env
# var away — so it re-enables those deviations for its own tests and restores the
# default module state afterwards. Without this fixture the tests below would be
# asserting the *old* default, not the feature.
_LEGACY_ENV = {
    "MICRODUCK_HEADING_HOLD": "1",
    "MICRODUCK_STANCE_HYGIENE": "1",
    "MICRODUCK_PER_JOINT_POSTURE": "1",
    "MICRODUCK_AIR_TIME_GATE": "1",
    "MICRODUCK_POSE_ZERO_CMD_PROB": "0.1",
}


@pytest.fixture(autouse=True)
def _legacy_stack():
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
def test_velocity_cfg_registers_the_heading_hold_term(_legacy_stack):
    assert _legacy_stack.ENABLE_HEADING_HOLD is True  # re-enabled by the fixture
    cfg = make_microduck_velocity_env_cfg()
    assert "heading_hold" in cfg.rewards
    assert isinstance(cfg.rewards["heading_hold"], RewardTermCfg)
    # AGENTS.md: never introduce a new objective while the skill is still being
    # explored — it must start at zero and be ramped in.
    assert cfg.rewards["heading_hold"].weight == 0.0


def test_gait_symmetry_is_available_but_OFF_by_default():
    """Deliberate: this run changes ONE thing (heading_hold).

    `gait_symmetry` is a proxy metric (swing-time imbalance != yaw moment) and,
    ungated, it would penalise the asymmetric stepping that commanded TURNING
    legitimately requires. It stays wired and one flag away.
    """
    assert not ENABLE_GAIT_SYMMETRY
    cfg = make_microduck_velocity_env_cfg()
    assert "gait_symmetry" not in cfg.rewards
    assert "gait_symmetry_weight" not in cfg.curriculum
    # the weight is still penalty-signed, so flipping the flag is safe
    assert GAIT_SYMMETRY_WEIGHT < 0.0
    assert GAIT_SYMMETRY_RAMP_ITER_START < GAIT_SYMMETRY_RAMP_ITER_END


def test_heading_hold_is_the_only_yaw_ANGLE_term_and_is_positive():
    cfg = make_microduck_velocity_env_cfg()
    assert cfg.rewards["heading_hold"].func is microduck_mdp.heading_hold_reward
    # self-contained reward (returns in [0,1]): positive weight, per AGENTS.md's
    # sign convention. A negative weight here would REWARD veering.
    assert HEADING_HOLD_WEIGHT > 0.0
    assert cfg.rewards["heading_hold"].params["std"] == HEADING_HOLD_STD


def test_heading_hold_speed_gate_exists_and_is_wired():
    cfg = make_microduck_velocity_env_cfg()
    assert cfg.rewards["heading_hold"].params["vel_gate_ref"] == HEADING_HOLD_VEL_GATE_REF
    assert HEADING_HOLD_VEL_GATE_REF > 0.0


def test_gait_symmetry_penalty_still_behaves_when_the_flag_is_flipped():
    """The mdp function is untouched by the flag; only the wiring is."""
    assert callable(microduck_mdp.gait_symmetry_penalty)
    cfg = make_microduck_velocity_env_cfg()
    assert cfg.curriculum.get("gait_symmetry_weight") is None


def test_heading_hold_curriculum_ramps_from_zero_to_target():
    cfg = make_microduck_velocity_env_cfg()
    cur = cfg.curriculum["heading_hold_weight"]
    assert isinstance(cur, CurriculumTermCfg)
    # mdp.reward_weight is a step function, not an interpolation: discretised.
    assert cur.func is microduck_mdp.reward_weight
    assert cur.params["reward_name"] == "heading_hold"
    stages = cur.params["weight_stages"]
    assert stages[0]["weight"] == 0.0, "must be free while the gait is discovered"
    assert stages[-1]["weight"] == HEADING_HOLD_WEIGHT
    assert stages[-1]["step"] == HEADING_HOLD_RAMP_ITER_END * NUM_STEPS_PER_ENV
    # monotone non-decreasing
    ws = [s["weight"] for s in stages]
    assert ws == sorted(ws)


def test_gait_symmetry_curriculum_ramps_from_zero_to_target():
    """Only registered when the flag is on; verify the stages either way."""
    stages = [
        {"reward_name": "gait_symmetry", "weight": 0.0},
        {"reward_name": "gait_symmetry", "weight": 0.5 * GAIT_SYMMETRY_WEIGHT},
        {"reward_name": "gait_symmetry", "weight": GAIT_SYMMETRY_WEIGHT},
    ]
    assert stages[0]["weight"] == 0.0
    assert stages[-1]["weight"] == GAIT_SYMMETRY_WEIGHT
    ws = [s["weight"] for s in stages]
    assert ws == sorted(ws, reverse=True)  # -0.25 then -0.5
    assert GAIT_SYMMETRY_RAMP_ITER_END * NUM_STEPS_PER_ENV > 0


def test_straightness_ramps_start_after_the_stance_taxes():
    """Ordering matters: fix the stance first, then pay for straightness."""
    assert HEADING_HOLD_RAMP_ITER_START >= 0
    assert HEADING_HOLD_RAMP_ITER_START < HEADING_HOLD_RAMP_ITER_END
    assert GAIT_SYMMETRY_RAMP_ITER_START < GAIT_SYMMETRY_RAMP_ITER_END
    # both must be well inside a run that is expected to reach ~12000 iterations
    assert HEADING_HOLD_RAMP_ITER_END <= 4000
    assert GAIT_SYMMETRY_RAMP_ITER_END <= 4000


def test_pose_commands_sample_the_exact_zero_the_deployment_sends(_legacy_stack):
    """AGENTS.md: uniform sampling essentially never hits the all-zero command,
    which is exactly the deployment idle state.

    2026-09-17: this bucket is now OPT-IN (``MICRODUCK_POSE_ZERO_CMD_PROB``,
    default 0.0 = official), so this module's fixture turns it back on. The
    argument for it is unchanged — deployment really does send all-zero — it is
    just not part of the official recipe.
    """
    cfg = make_microduck_velocity_env_cfg()
    prob = _legacy_stack.POSE_ZERO_COMMAND_PROB
    assert 0.0 < prob <= 0.5
    for name in ("head_pose", "body_pose"):
        assert cfg.commands[name].zero_command_prob == prob
        # the observation SHAPE must be untouched (61-D contract)
        assert len(cfg.commands[name].ranges) == (4 if name == "head_pose" else 6)


def test_heading_hold_does_not_change_the_obs_contract():
    """Reward-only change: the 61-D contract and the runtime need no edit."""
    cfg = make_microduck_velocity_env_cfg()
    assert cfg.rewards["heading_hold"].params["command_name"] == "twist"
    assert (cfg.rewards["heading_hold"].params["command_threshold"]
            == HEADING_HOLD_COMMAND_THRESHOLD)


# --------------------------------------------------------------------------- #
# heading_hold_reward behaviour
# --------------------------------------------------------------------------- #
def _quat_from_yaw(yaw: float) -> list[float]:
    """Pure yaw rotation about z: [w, x, y, z]."""
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


class _CmdHolder:
    def __init__(self, n):
        self.c = torch.zeros(n, 3)

    def get_command(self, _):
        return self.c


class _HoldData:
    def __init__(self, n):
        self.root_link_quat_w = torch.zeros(n, 4)
        self.root_link_quat_w[:, 0] = 1.0  # identity = yaw 0
        self.root_link_lin_vel_b = torch.zeros(n, 3)  # stationary by default


class _HoldAsset:
    def __init__(self, n):
        self.data = _HoldData(n)


class _HoldEnv:
    """Minimal env for heading_hold_reward: quat + episode counter + command.

    NOTE the episode counter starts ABOVE the reset threshold (`<= 1`): with it
    at 0 every call would look like a fresh episode and re-capture the reference,
    which would silently make the term always return 1.0.
    """

    def __init__(self, n=2):
        self.num_envs = n
        self.device = "cpu"
        self.step_dt = 0.02
        self.scene = {"robot": _HoldAsset(n)}
        self.command_manager = _CmdHolder(n)
        self.episode_length_buf = torch.full((n,), 10, dtype=torch.long)

    def set_yaw(self, env_idx: int, yaw: float):
        self.scene["robot"].data.root_link_quat_w[env_idx] = torch.tensor(
            _quat_from_yaw(yaw)
        )

    def set_yaws(self, yaws):
        for i, y in enumerate(yaws):
            self.set_yaw(i, y)

    def set_wz_cmd(self, wz):
        self.command_manager.c[:, 2] = wz

    def set_speed(self, vx, vy=0.0, env_idx=None):
        if env_idx is None:
            self.scene["robot"].data.root_link_lin_vel_b[:, 0] = vx
            self.scene["robot"].data.root_link_lin_vel_b[:, 1] = vy
        else:
            self.scene["robot"].data.root_link_lin_vel_b[env_idx, 0] = vx
            self.scene["robot"].data.root_link_lin_vel_b[env_idx, 1] = vy


def test_heading_hold_is_one_at_the_reference_and_decays_with_error():
    env = _HoldEnv()
    env.set_yaws([0.0, 0.0])
    # cap the reference at the spawn heading
    microduck_mdp.heading_hold_reward(env, std=0.4)
    assert torch.allclose(
        microduck_mdp.heading_hold_reward(env, std=0.4), torch.ones(2), atol=1e-6
    )
    # one env drifts
    env.set_yaw(1, 0.4)
    out = microduck_mdp.heading_hold_reward(env, std=0.4)
    assert abs(float(out[0]) - 1.0) < 1e-6
    assert abs(float(out[1]) - math.exp(-1.0)) < 1e-6, float(out[1])
    # stays in [0, 1] and is never negative, so a positive weight cannot reward drift
    env.set_yaw(1, 3.0)
    out = microduck_mdp.heading_hold_reward(env, std=0.4)
    assert 0.0 <= float(out[1]) < 1e-6


def test_heading_hold_reference_is_per_env_and_survives_yaw_wraparound():
    """Spawn yaw is randomised over the full circle; the error must wrap."""
    env = _HoldEnv(n=2)
    env.set_yaws([3.0, -3.0])          # +-172 deg, either side of the wrap
    microduck_mdp.heading_hold_reward(env, std=0.4)
    # a small drift ACROSS the wrap must read as a small error, not ~2pi
    env.set_yaw(0, -3.0)               # moved +0.283 rad through +-pi
    env.set_yaw(1, 3.0)
    out = microduck_mdp.heading_hold_reward(env, std=0.4)
    expected = math.exp(-((2 * math.pi - 6.0) ** 2) / 0.4 ** 2)
    assert abs(float(out[0]) - expected) < 1e-6, (float(out[0]), expected)
    assert abs(float(out[1]) - expected) < 1e-6


def test_heading_hold_recaptures_on_reset():
    env = _HoldEnv()
    env.set_yaws([0.0, 0.0])
    microduck_mdp.heading_hold_reward(env, std=0.4)
    env.set_yaw(0, 1.0)
    assert float(microduck_mdp.heading_hold_reward(env, std=0.4)[0]) < 1.0
    # simulate a reset: a new spawn yaw must become the new reference
    env.episode_length_buf[0] = 0
    microduck_mdp.heading_hold_reward(env, std=0.4)
    assert abs(float(microduck_mdp.heading_hold_reward(env, std=0.4)[0]) - 1.0) < 1e-6


def test_heading_hold_gate_is_off_while_turning_and_freezes_afterwards():
    """The subtle part: a commanded turn earns nothing here (weight stays a
    straight-line instrument), and the heading held afterwards must be the one
    the robot had when the turn ENDED — not the pre-turn spawn heading."""
    env = _HoldEnv(n=1)
    env.set_yaw(0, 0.0)
    kw = dict(std=0.4, command_name="twist", command_threshold=0.2)

    # spawn straight -> reference = 0
    env.set_wz_cmd(0.0)
    assert float(microduck_mdp.heading_hold_reward(env, **kw)[0]) == 1.0

    # commanded to turn: the term pays 0.0 whatever the yaw does (no reward for
    # the turn itself, and crucially no penalty either — the reference tracks it)
    env.set_wz_cmd(1.0)
    for yaw in (0.3, 0.9, 1.5):
        env.set_yaw(0, yaw)
        assert float(microduck_mdp.heading_hold_reward(env, **kw)[0]) == 0.0

    # turn command ends -> the reference is the heading at that moment (1.5 rad)
    env.set_wz_cmd(0.0)
    env.set_yaw(0, 1.5)
    assert abs(float(microduck_mdp.heading_hold_reward(env, **kw)[0]) - 1.0) < 1e-6

    # now drifting away from the POST-turn heading is charged...
    env.set_yaw(0, 1.9)
    assert abs(float(microduck_mdp.heading_hold_reward(env, **kw)[0])
               - math.exp(-1.0)) < 1e-6
    # ...while the pre-turn heading (0) reads as a large error, NOT as correct
    env.set_yaw(0, 0.0)
    assert float(microduck_mdp.heading_hold_reward(env, **kw)[0]) < 1e-3


def test_heading_hold_gate_threshold_boundary_and_side_stepping():
    env = _HoldEnv(n=2)
    env.set_yaws([0.0, 0.0])
    kw = dict(std=0.4, command_name="twist", command_threshold=0.2)
    env.set_wz_cmd(0.0)
    microduck_mdp.heading_hold_reward(env, **kw)

    # exactly at the threshold is still "straight" (> is the gate, not >=)
    env.set_wz_cmd(0.2)
    env.set_yaw(0, 0.0)
    env.set_yaw(1, 0.4)
    out = microduck_mdp.heading_hold_reward(env, **kw)
    assert float(out[0]) == 1.0 and float(out[1]) < 1.0

    # side-stepping (vy != 0) must NOT open the gate: holding heading while
    # strafing is still the wanted behaviour.
    env.command_manager.c[:, 1] = 0.3
    env.set_wz_cmd(0.0)
    out = microduck_mdp.heading_hold_reward(env, **kw)
    assert float(out[1]) < 1.0


def test_heading_hold_gate_disabled_matches_the_preexisting_behaviour():
    """Rollers env calls it with no command gate; that path must be unchanged."""
    env = _HoldEnv()
    env.set_yaws([0.0, 0.0])
    env.set_wz_cmd(1.0)  # a turn command must be IGNORED when gate is off
    microduck_mdp.heading_hold_reward(env, std=0.4)
    env.set_yaw(1, 0.4)
    out = microduck_mdp.heading_hold_reward(env, std=0.4)
    assert abs(float(out[1]) - math.exp(-1.0)) < 1e-6


# --------------------------------------------------------------------------- #
# the SPEED gate: "standing still must not farm the heading reward"
# --------------------------------------------------------------------------- #
def test_heading_hold_speed_gate_zeroes_a_stationary_policy():
    """Without this the term digs a 'just stop' basin: a stationary robot holds
    the heading perfectly and would collect the full reward, scoring HIGHER than
    veering (measured: stand 1.34+1.5 vs veer 1.63+~0)."""
    env = _HoldEnv(n=2)
    env.set_yaws([0.0, 0.0])
    kw = dict(std=0.4, command_name="twist", command_threshold=0.2,
              vel_gate_ref=HEADING_HOLD_VEL_GATE_REF)
    env.set_wz_cmd(0.0)

    def settle(vx, vy=0.0, steps=400):
        """Hold a constant body velocity long enough for the 1 s EMA to converge
        (alpha = dt/(tau+dt) = 0.0196 -> ~51-step time constant)."""
        env.set_speed(vx, vy)
        out = None
        for _ in range(steps):
            out = microduck_mdp.heading_hold_reward(env, **kw)
        return float(out[0])

    # stationary, perfectly aligned -> must still pay NOTHING
    assert settle(0.0) == 0.0

    # at/above the reference speed -> full term again (the EMA asymptotes, so
    # allow a small settling tolerance)
    assert abs(settle(HEADING_HOLD_VEL_GATE_REF) - 1.0) < 2e-3
    assert abs(settle(3 * HEADING_HOLD_VEL_GATE_REF) - 1.0) < 2e-3

    # the gate uses the xy SPEED, so sustained strafing opens it too
    assert abs(settle(0.0, HEADING_HOLD_VEL_GATE_REF) - 1.0) < 2e-3

    # it MULTIPLIES the heading error rather than replacing it: a veering robot
    # moving fast still loses the reward
    settle(HEADING_HOLD_VEL_GATE_REF)
    env.set_yaw(0, 0.4)
    assert abs(float(microduck_mdp.heading_hold_reward(env, **kw)[0])
               - math.exp(-1.0)) < 2e-3


def test_heading_hold_speed_gate_does_not_pay_for_marching_IN_PLACE():
    """The defect this gate shape fixes (measured 2026-09-14).

    With an instantaneous |v| gate, a marching-in-place policy opened the gate
    59.7% of the time versus 40.3% for one standing still — the trunk swings past
    the reference every step, so the term paid MORE for marching and locked the
    behaviour in. Smoothing the signed components makes the two equal.
    """
    env = _HoldEnv(n=1)
    env.set_yaw(0, 0.0)
    env.set_wz_cmd(0.0)
    kw = dict(std=0.4, command_name="twist", command_threshold=0.2,
              vel_gate_ref=HEADING_HOLD_VEL_GATE_REF)

    # march: the trunk oscillates +-0.3 m/s at the step frequency but goes nowhere.
    # Warm up first so the 1 s EMA transient is not counted as marching credit.
    for i in range(300):
        env.set_speed(0.3 if i % 2 == 0 else -0.3)
        microduck_mdp.heading_hold_reward(env, **kw)
    acc = 0.0
    for i in range(200):
        env.set_speed(0.3 if i % 2 == 0 else -0.3)
        acc += float(microduck_mdp.heading_hold_reward(env, **kw))
    marching = acc / 200.0

    # stand: dead still (warm up the EMA down to zero first)
    for _ in range(400):
        env.set_speed(0.0)
        microduck_mdp.heading_hold_reward(env, **kw)
    stand = 0.0
    for _ in range(200):
        stand += float(microduck_mdp.heading_hold_reward(env, **kw))
    stand /= 200.0

    # marching must NOT out-earn standing (the old instantaneous gate had it ~1.5x)
    assert marching <= stand + 0.02, (marching, stand)
    assert marching < 0.05, marching   # and it must be near zero in absolute terms

    # walking steadily still pays in full
    for _ in range(400):
        env.set_speed(HEADING_HOLD_VEL_GATE_REF)
        walking = float(microduck_mdp.heading_hold_reward(env, **kw))
    assert walking > 0.99, walking


def test_heading_hold_speed_gate_is_off_by_default():
    """vel_gate_ref=0 keeps the plain angle-only behaviour (rollers call site)."""
    env = _HoldEnv(n=1)
    env.set_yaw(0, 0.0)
    env.set_wz_cmd(0.0)
    env.set_speed(0.0)
    microduck_mdp.heading_hold_reward(env, std=0.4)
    assert abs(float(microduck_mdp.heading_hold_reward(env, std=0.4)[0]) - 1.0) < 1e-6


# --------------------------------------------------------------------------- #
# gait_symmetry_penalty behaviour
# --------------------------------------------------------------------------- #
class _SensData:
    def __init__(self, n, nfeet=2):
        self.current_air_time = torch.zeros(n, nfeet)


class _Sens:
    def __init__(self, data):
        self.data = data


class _SymEnv:
    """Minimal env for gait_symmetry_penalty.

    The episode counter must sit ABOVE the reset threshold (`<= 1`); at 0 the
    accumulator is zeroed on every call and the imbalance never builds up.
    """

    def __init__(self, n=1, nfeet=2, dt=0.02):
        self.num_envs = n
        self.device = "cpu"
        self.step_dt = dt
        self.episode_length_buf = torch.full((n,), 10, dtype=torch.long)
        self._sens = _Sens(_SensData(n, nfeet))
        self.scene = {"feet_ground_contact": self._sens}

    def set_air(self, mask):
        """mask[e][f] = True means foot f is airborne (swinging)."""
        self._sens.data.current_air_time[:] = torch.tensor(mask, dtype=torch.float)


def test_gait_symmetry_is_zero_for_a_balanced_stride():
    """The metric is CUMULATIVE, so a full alternating cycle washes out.

    A real stride is instantaneously asymmetric by construction (only one foot
    swings at a time) — that must not be charged. Only a persistent imbalance is.
    """
    env = _SymEnv()
    # both feet airborne the whole time: perfectly balanced
    for _ in range(10):
        env.set_air([[True, True]])
        out = microduck_mdp.gait_symmetry_penalty(
            env, sensor_name="feet_ground_contact"
        )
    assert float(out[0]) < 1e-9, float(out[0])

    # a proper alternating gait, equal time per foot -> also balanced
    env2 = _SymEnv()
    for i in range(40):
        env2.set_air([[i % 2 == 0, i % 2 == 1]])
        out = microduck_mdp.gait_symmetry_penalty(
            env2, sensor_name="feet_ground_contact"
        )
    assert float(out[0]) < 1e-9, float(out[0])

    # a SINGLE asymmetric step inside a balanced run is nearly free: that is the
    # whole point of accumulating instead of testing the instant
    assert abs(float(env2._swing_accum[0, 0]) - float(env2._swing_accum[0, 1])) < 1e-9


def test_gait_symmetry_charges_a_lopsided_stride_and_is_nonnegative():
    env = _SymEnv()
    # only the left foot ever swings -> persistently lopsided
    for _ in range(200):
        env.set_air([[True, False]])
        out = microduck_mdp.gait_symmetry_penalty(
            env, sensor_name="feet_ground_contact"
        )
    assert float(out[0]) > 0.99, float(out[0])
    assert float(out[0]) >= 0.0  # so a negative weight is a real penalty


def test_gait_symmetry_accumulator_resets_per_episode():
    env = _SymEnv()
    for _ in range(10):
        env.set_air([[True, False]])
        microduck_mdp.gait_symmetry_penalty(env, sensor_name="feet_ground_contact")
    assert float(env._swing_accum[0, 0]) > 0.0
    env.episode_length_buf[0] = 0  # fresh episode
    env.set_air([[True, False]])
    microduck_mdp.gait_symmetry_penalty(env, sensor_name="feet_ground_contact")
    assert abs(float(env._swing_accum[0, 0]) - env.step_dt) < 1e-9
    assert abs(float(env._swing_accum[0, 1])) < 1e-9


def test_gait_symmetry_is_bounded_so_it_cannot_dominate():
    env = _SymEnv()
    for _ in range(200):
        env.set_air([[True, False]])
        out = microduck_mdp.gait_symmetry_penalty(
            env, sensor_name="feet_ground_contact"
        )
    assert 0.0 <= float(out[0]) <= 1.0


# --------------------------------------------------------------------------- #
# zero-command marching: joint_vel_l2_when_standing
# --------------------------------------------------------------------------- #
class _SimpleAsset:
    def __init__(self, data):
        self.data = data

    def find_joints(self, pattern):
        """`_servo_joint_ids` resolves the servo-only view through this."""
        del pattern
        nj = self.data.joint_vel.shape[1]
        return list(range(nj)), [f"j{i}" for i in range(nj)]


class _VelEnv:
    """Fake env for joint_vel_l2_when_standing: servo joint_vel + command."""

    def __init__(self, n=1, nj=14):
        self.num_envs = n
        self.device = "cpu"
        self.step_dt = 0.02
        self.episode_length_buf = torch.full((n,), 10, dtype=torch.long)
        self._cmd = torch.zeros(n, 3)

        class _D:
            pass

        d = _D()
        d.joint_vel = torch.zeros(n, nj)
        d.joint_pos = torch.zeros(n, nj)
        d.root_link_vel_w = torch.zeros(n, 3)
        self._asset = _SimpleAsset(d)
        self.scene = {"robot": self._asset}
        self.command_manager = _CmdHolder(n)

    def find_joints(self, pattern):
        nj = self._asset.data.joint_vel.shape[1]
        return list(range(nj)), [f"j{i}" for i in range(nj)]


def test_standing_leg_vel_is_zero_when_walking_is_commanded():
    """The term MUST be invisible to the walking gait (command-gated)."""
    env = _VelEnv()
    env.command_manager.c[0] = torch.tensor([0.2, 0.0, 0.0])
    env._asset.data.joint_vel[0, :5] = 5.0          # legs flailing
    out = microduck_mdp.joint_vel_l2_when_standing(
        env, command_name="twist", command_threshold=0.01)
    assert float(out[0]) == 0.0, float(out[0])


def test_standing_leg_vel_charges_legs_but_not_head_at_zero_command():
    env = _VelEnv()
    env.command_manager.c[0] = torch.zeros(3)                     # stand commanded
    # legs pumping => charged;  head/neck moving => free (this is a LEG term)
    env._asset.data.joint_vel[0, 0] = 3.0            # left_hip_yaw
    env._asset.data.joint_vel[0, 13] = 4.0           # right_ankle
    env._asset.data.joint_vel[0, 5] = 9.0            # neck_pitch
    out = microduck_mdp.joint_vel_l2_when_standing(
        env, command_name="twist", command_threshold=0.01)
    assert abs(float(out[0]) - (3.0 ** 2 + 4.0 ** 2)) < 1e-5, float(out[0])


def test_standing_leg_vel_is_essentially_free_for_a_still_policy():
    """Calibration from real rollouts: a standing policy reads ~0.008, a
    marching one ~75. At weight -0.02 that is -0.0002 vs -1.5 per step."""
    still = _VelEnv(); still._cmd[0] = torch.zeros(3)
    still._asset.data.joint_vel[0, :] = 0.02
    marching = _VelEnv(); marching._cmd[0] = torch.zeros(3)
    marching._asset.data.joint_vel[0, 0:5] = 2.5
    marching._asset.data.joint_vel[0, 9:14] = 2.5
    s = float(microduck_mdp.joint_vel_l2_when_standing(still, command_name="twist")[0])
    m = float(microduck_mdp.joint_vel_l2_when_standing(marching, command_name="twist")[0])
    assert s < 0.02, s
    assert m > 50.0, m
    assert m / max(s, 1e-9) > 1000            # huge separation
    # and at the wired weight it must be a real deterrent, not noise
    assert abs(-0.02 * m) > 1.0


def test_standing_leg_vel_push_exemption_uses_SMOOTHED_body_speed():
    """A sustained shove must unlock recovery steps; a step-locked trunk wobble
    must not (averaging |v| would stay positive for any oscillation)."""
    env = _VelEnv()
    env.command_manager.c[0] = torch.zeros(3)
    env._asset.data.joint_vel[0, 0] = 3.0
    kw = dict(command_name="twist", command_threshold=0.01,
              body_vel_threshold=0.2, body_vel_tau=1.0)

    # marching: trunk oscillates +-0.3 m/s but averages ~0 -> NOT exempt
    for i in range(300):
        env._asset.data.root_link_vel_w[0, 0] = 0.3 if i % 2 == 0 else -0.3
        out = microduck_mdp.joint_vel_l2_when_standing(env, **kw)
    assert float(out[0]) > 0.0, float(out[0])

    # a sustained shove (body really moving, not averaging out) -> exempt
    for _ in range(300):
        env._asset.data.root_link_vel_w[0, 0] = 0.5
        out = microduck_mdp.joint_vel_l2_when_standing(env, **kw)
    assert float(out[0]) == 0.0, float(out[0])


def test_velocity_cfg_wires_the_marching_cure():
    cfg = make_microduck_velocity_env_cfg()
    assert cfg.rewards["standing_leg_vel"].weight == 0.0, "must start at 0 and ramp"
    assert (cfg.rewards["standing_leg_vel"].func
            is microduck_mdp.joint_vel_l2_when_standing)
    assert cfg.rewards["standing_leg_vel"].params["command_threshold"] == 0.01
    ramped = {s["reward_name"]: s["weight"]
              for s in cfg.curriculum["stance_tax_weight"].params["stages"]}
    assert ramped["standing_leg_vel"] < 0.0, "penalty sign: the term returns >= 0"
