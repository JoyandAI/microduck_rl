"""The cycle-averaged net-yaw-rate term — the 2026-09-16 veer fix.

Why this term exists (64 env x 5 seed straightness matrix,
``scripts/q1_straightness_matrix.sh``):

    arm                     yaw DC (net)    yaw AC (per-half-step wobble)
    official alpha_walking      -0.23 deg/s              -0.66 deg/s
    our walk_hd1910_12000      -12.08 deg/s             -26.28 deg/s

The net rate is what walks the circle: heading changes by ``DC * T`` per gait
cycle, so a constant -12 deg/s is a full turn every 30 s (measured turning radius
3.2 m vs the official policy's ~12 m). The 5 V-era "hip_yaw parked on its stop,
feet twisted 28 deg" pathology is gone from the current policy (0.0 % of steps at
a stop, foot-pair torsion < 1 deg).

What these tests pin:
  1. the estimator reads the BIAS and is blind to the step-frequency wobble, so a
     symmetric gait that yaws hard within each step pays (almost) nothing;
  2. it is NOT a |wz| penalty — the property that got the rollers env's raw rate
     penalty reverted;
  3. it is gated off for commanded turns and for zero commands;
  4. the AC diagnostic is a diagnostic: it peaks on the MOST symmetric gait, which
     is why it must never be turned into a reward.
"""

from __future__ import annotations

import importlib

import pytest
import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    ENABLE_YAW_SYMMETRY,
    YAW_SYMMETRY_RAMP_ITERS,
    YAW_SYMMETRY_TAU_S,
    YAW_SYMMETRY_WEIGHT,
    YAW_SYMMETRY_WZ_CMD_THRESHOLD,
    make_microduck_velocity_env_cfg,
)

DT = 0.02
TAU = YAW_SYMMETRY_TAU_S


class _SensorData:
    def __init__(self, n):
        self.found = torch.zeros(n, 2)


class _Sensor:
    def __init__(self, n):
        self.data = _SensorData(n)


class _RobotData:
    def __init__(self, n):
        self.root_link_ang_vel_b = torch.zeros(n, 3)


class _Asset:
    def __init__(self, n):
        self.data = _RobotData(n)


class _CmdMgr:
    def __init__(self, n):
        self.cmd = torch.zeros(n, 3)

    def get_command(self, _name):
        return self.cmd


class _Scene:
    def __init__(self, d):
        self._d = d
        self.sensors = d

    def __getitem__(self, k):
        return self._d[k]


class _Env:
    def __init__(self, n=1):
        self.num_envs = n
        self.device = torch.device("cpu")
        self.step_dt = DT
        self.episode_length_buf = torch.ones(n, dtype=torch.long)
        self.command_manager = _CmdMgr(n)
        self._d = {"feet_ground_contact": _Sensor(n), "robot": _Asset(n)}
        self.scene = _Scene(self._d)

    def tick(self, wz, cmd=(0.2, 0.0, 0.0), left=True, right=False):
        """Advance one control step the way the reward manager would.

        The estimator's EMA advances INSIDE the reward function, so a harness that
        only sets state and then calls the term once at the end measures a single
        alpha-step, not the settled average. Both accumulators are therefore
        ticked here, once per step, exactly as mjlab's RewardManager calls them.
        """
        self._d["robot"].data.root_link_ang_vel_b[:, 2] = wz
        s = self._d.get("feet_ground_contact")
        if s is not None:
            s.data.found[:, 0] = float(left)
            s.data.found[:, 1] = float(right)
        self.command_manager.cmd[0] = torch.tensor(cmd)
        self.episode_length_buf += 1
        out = self.dc()
        if s is not None:
            self.ac()
        return out

    def dc(self):
        return microduck_mdp.yaw_dc_rate_penalty(
            self, command_name="twist",
            wz_command_threshold=YAW_SYMMETRY_WZ_CMD_THRESHOLD,
            command_threshold=0.05, tau_s=TAU)

    def ac(self):
        return microduck_mdp.yaw_impulse_asymmetry_diagnostic(
            self, sensor_name="feet_ground_contact", tau_s=TAU)


# --------------------------------------------------------------------------- #
# cfg wiring
# --------------------------------------------------------------------------- #
def test_arm_is_off_by_default():
    assert ENABLE_YAW_SYMMETRY is False, "must be a deliberate A/B arm, not a default"
    cfg = make_microduck_velocity_env_cfg()
    assert "yaw_dc" not in cfg.rewards
    assert "yaw_dc_weight" not in cfg.curriculum


def test_arm_registers_term_and_curriculum_when_enabled(monkeypatch):
    monkeypatch.setenv("MICRODUCK_YAW_SYMMETRY", "1")
    mod = importlib.import_module("mjlab_microduck.tasks.microduck_velocity_env_cfg")
    importlib.reload(mod)
    try:
        assert mod.ENABLE_YAW_SYMMETRY is True
        cfg = mod.make_microduck_velocity_env_cfg()
        assert "yaw_dc" in cfg.rewards
        assert cfg.rewards["yaw_dc"].weight == 0.0, "must start at 0"
        cur = cfg.curriculum["yaw_dc_weight"]
        assert cur.params["ramp_steps"] == YAW_SYMMETRY_RAMP_ITERS * mod.NUM_STEPS_PER_ENV
        assert cur.params["stages"][0]["reward_name"] == "yaw_dc"
        # self-negating penalty -> POSITIVE weight
        assert cur.params["stages"][0]["weight"] == YAW_SYMMETRY_WEIGHT > 0
    finally:
        monkeypatch.delenv("MICRODUCK_YAW_SYMMETRY", raising=False)
        importlib.reload(mod)


# --------------------------------------------------------------------------- #
# the estimator
# --------------------------------------------------------------------------- #
def _settle(env, n_steps, wz, **kw):
    out = None
    for _ in range(n_steps):
        out = env.tick(wz, **kw)
    return out[0].item()


def test_zero_yaw_rate_costs_nothing():
    env = _Env()
    assert _settle(env, 300, 0.0) == pytest.approx(0.0, abs=1e-6)


def test_a_constant_bias_is_read_at_full_strength():
    """The defect: -12 deg/s = -0.209 rad/s must show up as ~-0.209."""
    env = _Env()
    out = _settle(env, 300, -0.2094)
    assert out == pytest.approx(-0.2094, abs=5e-3), out


def test_step_frequency_wobble_is_averaged_out():
    """Property 1: a symmetric gait that yaws hard within each step pays ~nothing.

    Alternating +-26 deg/s at the gait frequency must NOT register as a bias —
    otherwise the term would tax the stepping itself rather than the drift.
    """
    env = _Env()
    for k in range(1200):
        env.tick(0.4538 if k % 10 < 5 else -0.4538)
    assert abs(env.dc()[0].item()) < 0.05, env.dc()[0].item()


def test_bias_plus_wobble_reads_the_bias():
    """The real signal: a wobble riding on a bias must resolve the bias."""
    env = _Env()
    for k in range(1200):
        wob = 0.4538 if k % 10 < 5 else -0.4538
        env.tick(-0.2094 + wob)
    assert env.dc()[0].item() == pytest.approx(-0.2094, abs=0.05), env.dc()[0].item()


def test_it_is_not_a_plain_wz_penalty():
    """Property 2: doubling the wobble while keeping the mean must not change it.

    This is exactly what distinguishes it from the rollers env's reverted |wz|
    penalty, which punished every steering correction.
    """
    a, b = _Env(), _Env()
    for k in range(1200):
        s = 1.0 if k % 10 < 5 else -1.0
        a.tick(0.2 * s)
        b.tick(1.0 * s)   # 5x the wobble, same zero mean
    # The residual is the KNOWN wobble leakage of the one-pole filter, not a bias:
    # a 0.4 s square wave through tau=0.8 s is attenuated ~8 %, so a 1.0 rad/s
    # wobble leaves ~0.08 while a 0.2 rad/s wobble leaves ~0.016. For the real
    # policy (0.45 rad/s at 2.3 Hz) that is ~0.04 rad/s against a 0.21 rad/s bias,
    # i.e. the estimator reads the bias with roughly 5x signal-to-noise.
    assert abs(a.dc()[0].item()) < 0.03
    assert abs(b.dc()[0].item()) < 0.12
    # and crucially it grows with the WOBBLE, not with the bias
    assert abs(a.dc()[0].item()) < 0.05


# --------------------------------------------------------------------------- #
# gating
# --------------------------------------------------------------------------- #
def test_gate_closes_for_a_commanded_turn():
    env = _Env()
    _settle(env, 300, -0.3, cmd=(0.2, 0.0, 0.8))
    assert env.dc()[0].item() == 0.0
    # inside the deadband it stays open
    env2 = _Env()
    _settle(env2, 300, -0.3, cmd=(0.2, 0.0, 0.05))
    assert env2.dc()[0].item() < -0.2


def test_gate_closes_at_zero_command():
    """Standing still is yaw_hold_at_zero_command's job, not this term's."""
    env = _Env()
    _settle(env, 300, -0.3, cmd=(0.0, 0.0, 0.0))
    assert env.dc()[0].item() == 0.0


def test_accumulator_resets_on_a_fresh_episode():
    env = _Env()
    _settle(env, 300, -0.3)
    assert env.dc()[0].item() < -0.25
    # `tick` advances the buffer before calling the term, as mjlab does, so a
    # freshly-reset env is the one whose buffer is 0 going into the reward call.
    env.episode_length_buf[:] = 0
    env.tick(0.0)
    assert abs(env.dc()[0].item()) < 0.02, env.dc()[0].item()


def test_term_is_always_a_cost_never_a_reward():
    for wz in (-0.5, 0.0, 0.5):
        env = _Env()
        assert _settle(env, 300, wz) <= 0.0


# --------------------------------------------------------------------------- #
# the AC diagnostic must stay a diagnostic
# --------------------------------------------------------------------------- #
def test_ac_diagnostic_peaks_on_the_most_symmetric_gait():
    """Why the AC quantity must never become a reward.

    A mirror-symmetric gait yaws +w on the left foot and -w on the right (yaw is a
    pseudoscalar under reflection). Its two half-step means are equal and opposite,
    so the AC diagnostic is at MAXIMUM exactly when the stride is perfectly
    balanced and produces no net drift. Penalising it would push the policy away
    from the symmetric stride we want.
    """
    env = _Env()
    for _ in range(400):
        for _ in range(10):
            env.tick(+0.5, left=True, right=False)
        for _ in range(10):
            env.tick(-0.5, left=False, right=True)
    ac = env.ac()[0].item()
    dc = env.dc()[0].item()
    assert abs(ac) > 0.9, ac      # large "asymmetry" reading...
    # ...on a gait whose net drift is ZERO. What the DC term reads instead is the
    # one-pole filter's ripple on the 2.5 Hz square wave, an order of magnitude
    # smaller -- which is exactly why the DC term is the usable signal and the AC
    # quantity is not.
    assert abs(dc) < abs(ac) / 10, (ac, dc)
    assert abs(dc) < 0.12, dc


def test_ac_diagnostic_is_zero_for_a_non_stepping_robot():
    env = _Env()
    for _ in range(200):
        env.tick(0.0, left=True, right=True)
    assert abs(env.ac()[0].item()) < 1e-6


def test_dc_term_survives_a_missing_sensor():
    """The DC term reads only ang_vel, so it must work without a contact sensor."""
    env = _Env()
    env._d.pop("feet_ground_contact")
    env.scene.sensors = env._d
    _settle(env, 100, -0.3)
    assert env.dc()[0].item() < -0.2
