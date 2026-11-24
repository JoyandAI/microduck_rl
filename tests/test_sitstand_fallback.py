"""Sitstand seated fall-back hardening (2026-10): cfg wiring + mdp-term semantics.

Real robot: an abrupt head move while seated tipped it BACKWARD and the neck
servo overloaded holding the head. The fix = fast head-command jerks and
backward tip kicks while seated (curriculum-ramped from 0) + a neck-torque
cost gated on leaning back in the sit. These tests lock the signs/gates.
"""

import math

import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_sitstand_env_cfg import (
    make_microduck_sitstand_env_cfg,
)

# ── cfg ───────────────────────────────────────────────────────────────────────


def test_neck_strain_is_cost_with_negative_weight():
    r = make_microduck_sitstand_env_cfg().rewards["neck_strain_leaned"]
    # neck_torque_when_tilted returns ≥ 0 (mjlab cost style) → weight must be < 0.
    assert r.func is microduck_mdp.neck_torque_when_tilted
    assert r.weight < 0


def test_perturbation_events_start_disabled_and_ramp():
    cfg = make_microduck_sitstand_env_cfg()
    jerk, tip = cfg.events["seated_head_jerk"], cfg.events["seated_tip"]
    assert jerk.mode == "interval" and tip.mode == "interval"
    assert jerk.params["prob"] == 0.0
    assert tip.params["ang_vel_range"] == (0.0, 0.0)
    for name, key in (("seated_head_jerk_prob", "prob"), ("seated_tip_range", "ang_vel_range")):
        stages = cfg.curriculum[name].params["param_stages"]
        assert stages[0]["step"] == 0
        first = stages[0]["params"][key]
        assert first == 0.0 or first == (0.0, 0.0)
        # Introduced only after the sit/rise exist (≥ iter 1500).
        assert stages[1]["step"] >= 1500 * 24
        last = stages[-1]["params"][key]
        assert (last > 0) if isinstance(last, float) else (last[1] > 0)


# ── fakes ─────────────────────────────────────────────────────────────────────


class _Data:
    def __init__(self, n, nu=14):
        self.actuator_force = torch.zeros(n, nu)
        self.projected_gravity_b = torch.tensor([[0.0, 0.0, -1.0]]).repeat(n, 1)
        self.root_link_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1)
        self.root_link_vel_w = torch.zeros(n, 6)


class _Asset:
    def __init__(self, n):
        self.data = _Data(n)
        self.written = None

    def find_actuators(self, _keys, preserve_order=False):
        return [5, 6, 7, 8], ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]

    def write_root_link_velocity_to_sim(self, vel, env_ids):
        self.written = (vel.clone(), env_ids.clone())


class _Scene:
    def __init__(self, asset):
        self._asset = asset

    def __getitem__(self, _):
        return self._asset


class _Term:
    def __init__(self, n, dim):
        self._command = torch.zeros(n, dim)
        self.alpha = None
        self.dim = dim


class _Cfg:
    ranges = ((-1.0, 1.0),) * 4


class _CmdMgr:
    def __init__(self, n):
        self.twist = _Term(n, 3)
        self.twist.alpha = torch.zeros(n)
        self.head = _Term(n, 4)
        self.head.cfg = _Cfg()

    def get_term(self, name):
        return self.twist if name == "twist" else self.head

    def get_command(self, name):
        return self.get_term(name)._command


class _Env:
    def __init__(self, n):
        self.num_envs = n
        self.device = "cpu"
        self.scene = _Scene(_Asset(n))
        self.command_manager = _CmdMgr(n)

    def seat(self, ids):
        self.command_manager.twist._command[ids, 0] = 1.0
        self.command_manager.twist.alpha[ids] = 1.0


def _lean(env, i, deg, backward=True):
    a = math.radians(deg)
    sx = -math.sin(a) if backward else math.sin(a)  # trunk +x forward: back-lean → g_x < 0
    env.scene._asset.data.projected_gravity_b[i] = torch.tensor([sx, 0.0, -math.cos(a)])


# ── neck_torque_when_tilted ───────────────────────────────────────────────────


def test_neck_cost_only_when_seated_and_leaned_back():
    env = _Env(5)
    env.scene._asset.data.actuator_force[:, 5:9] = 0.15  # overload-level torque
    env.seat([0, 1, 2, 3])
    _lean(env, 0, 3.0)                    # seated upright      → 0
    _lean(env, 1, 30.0)                   # seated, leaned back → full cost
    _lean(env, 2, 30.0, backward=False)   # seated, leaned fwd  → 0 (head-assist path)
    _lean(env, 3, 18.5)                   # mid tilt ramp       → partial
    _lean(env, 4, 30.0)                   # STANDING target     → 0
    c = microduck_mdp.neck_torque_when_tilted(env)
    assert c[0] == 0.0 and c[2] == 0.0 and c[4] == 0.0
    assert torch.isclose(c[1], torch.tensor((0.15 / 0.1) ** 2))
    assert 0.0 < c[3] < c[1]
    assert (c >= 0).all()


# ── seated_backward_tip ───────────────────────────────────────────────────────


def test_tip_kick_pitches_backward_and_only_seated():
    env = _Env(4)
    env.seat([0, 1])
    microduck_mdp.seated_backward_tip(
        env, torch.arange(4), ang_vel_range=(3.0, 3.0), lateral_frac=0.0
    )
    vel, ids = env.scene._asset.written
    assert ids.tolist() == [0, 1]
    # identity orientation: body y == world y; nose-up (backward) = −ω_y
    assert torch.allclose(vel[:, 4], torch.tensor([-3.0, -3.0]))
    assert torch.allclose(vel[:, :3], torch.zeros(2, 3))


def test_tip_kick_disabled_at_zero_range():
    env = _Env(2)
    env.seat([0, 1])
    microduck_mdp.seated_backward_tip(env, torch.arange(2), ang_vel_range=(0.0, 0.0))
    assert env.scene._asset.written is None


# ── seated_head_command_jerk ──────────────────────────────────────────────────


def test_head_jerk_touches_only_seated_envs():
    torch.manual_seed(0)
    env = _Env(6)
    head = env.command_manager.head._command
    head[:] = 0.5
    env.seat([0, 1, 2])
    microduck_mdp.seated_head_command_jerk(env, torch.arange(6), prob=1.0, flip_frac=1.0)
    assert torch.allclose(head[:3], torch.full((3, 4), -0.5))  # flipped
    assert torch.allclose(head[3:], torch.full((3, 4), 0.5))   # standing untouched


def test_head_jerk_resample_stays_in_ranges_and_zero_prob_noop():
    torch.manual_seed(0)
    env = _Env(64)
    env.seat(list(range(64)))
    head = env.command_manager.head._command
    microduck_mdp.seated_head_command_jerk(env, torch.arange(64), prob=0.0)
    assert (head == 0).all()
    microduck_mdp.seated_head_command_jerk(env, torch.arange(64), prob=1.0, flip_frac=0.0)
    assert (head.abs() <= 1.0).all() and (head != 0).any()
