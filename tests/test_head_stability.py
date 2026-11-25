"""Head-stability terms: "头要稳定 … 不能低头" (user requirement, 2026-09-17).

Two terms, both ON by default and both ramped in on the same schedule as the
official ``head_pose_bias`` (free until iteration 600, full weight at 1500):

  * ``head_motion`` — ``neck_joint_vel_deadband_l2``, deadband 2.0 rad/s.
    Calibration from measurement: the official policy's neck cadence amplitude is
    0.75 deg at the 2.15 Hz step frequency => ~0.18 rad/s peak == FREE; the
    broken policy runs 16.4 deg => ~3.87 rad/s peak == charged. So the deadband
    sits between "the official gait" and "the defect", and the tax is escapable.
  * ``head_droop`` — ``head_droop_penalty``, one-sided L1 on the head being held
    BELOW its commanded pitch, deadband 0.15 rad. The measured defect it prices:
    neck_pitch parked at -1.934 rad from HOME (logs/axis/THREE_AXIS_SUMMARY.txt)
    where the official sits at -0.127.

Why these are not redundant with the official head terms is argued in the
docstrings; what this file pins is that the WIRING cannot silently regress:
signs, deadbands, ramp endpoints, and the mdp function's semantics.
"""

import importlib

import pytest
import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    HEAD_DROOP_DEADBAND,
    HEAD_DROOP_WEIGHT,
    HEAD_MOTION_DEADBAND_RAD_S,
    HEAD_MOTION_WEIGHT,
    HEAD_STABILITY_RAMP_ITER_END,
    HEAD_STABILITY_RAMP_ITER_MID,
    HEAD_STABILITY_RAMP_ITER_START,
    NUM_STEPS_PER_ENV,
    make_microduck_velocity_env_cfg,
)

CFG_MODULE = "mjlab_microduck.tasks.microduck_velocity_env_cfg"


# --------------------------------------------------------------------------- #
# cfg-level wiring
# --------------------------------------------------------------------------- #
def test_head_stability_is_on_by_default_and_registers_both_terms():
    cfg = make_microduck_velocity_env_cfg()
    assert cfg.rewards["head_motion"].func is microduck_mdp.neck_joint_vel_deadband_l2
    assert cfg.rewards["head_droop"].func is microduck_mdp.head_droop_penalty
    assert cfg.rewards["head_motion"].params["deadband"] == HEAD_MOTION_DEADBAND_RAD_S
    assert cfg.rewards["head_droop"].params["deadband"] == HEAD_DROOP_DEADBAND
    assert cfg.rewards["head_droop"].params["command_name"] == "head_pose"


def test_head_motion_is_a_penalty_and_head_droop_is_a_one_sided_cost():
    """Sign convention (AGENTS.md): the logged value must be <= 0.

    ``neck_joint_vel_deadband_l2`` returns >= 0 -> NEGATIVE weight.
    ``head_droop_penalty`` returns <= 0 -> POSITIVE weight.
    Getting either backwards turns the term into a reward for the defect.
    """
    assert HEAD_MOTION_WEIGHT < 0.0
    assert HEAD_DROOP_WEIGHT > 0.0


def test_both_head_terms_start_at_zero_and_share_the_official_schedule():
    """AGENTS.md: no attempt-tax while the gait is still being discovered."""
    cfg = make_microduck_velocity_env_cfg()
    assert cfg.rewards["head_motion"].weight == 0.0
    assert cfg.rewards["head_droop"].weight == 0.0

    for cur_name, reward, target in (
        ("head_stability_weight", "head_motion", HEAD_MOTION_WEIGHT),
        ("head_droop_weight", "head_droop", HEAD_DROOP_WEIGHT),
    ):
        cur = cfg.curriculum[cur_name]
        assert cur.func is microduck_mdp.reward_weight
        assert cur.params["reward_name"] == reward
        stages = cur.params["weight_stages"]
        assert stages[0] == {"step": 0, "weight": 0.0}
        assert stages[-1]["step"] == HEAD_STABILITY_RAMP_ITER_END * NUM_STEPS_PER_ENV
        assert stages[-1]["weight"] == pytest.approx(target)
        # monotone towards the target (allowing for a negative target)
        mags = [abs(s["weight"]) for s in stages]
        assert mags == sorted(mags)

    # same timeline as the official head_pose_bias ramp — one clock, one story
    bias = cfg.curriculum["head_pose_bias_weight"].params["weight_stages"]
    assert [s["step"] for s in bias] == [0,
                                         HEAD_STABILITY_RAMP_ITER_START * NUM_STEPS_PER_ENV,
                                         HEAD_STABILITY_RAMP_ITER_MID * NUM_STEPS_PER_ENV,
                                         HEAD_STABILITY_RAMP_ITER_END * NUM_STEPS_PER_ENV]


def test_head_stability_keeps_the_61d_obs_contract():
    """Reward-only: no obs/command change, so the runtime needs no edit."""
    cfg = make_microduck_velocity_env_cfg()
    assert len(cfg.observations["actor"].terms) == 8
    assert len(cfg.commands["head_pose"].ranges) == 4
    assert len(cfg.commands["body_pose"].ranges) == 6


def test_head_stability_can_be_switched_off_for_a_clean_official_ab(monkeypatch):
    monkeypatch.setenv("MICRODUCK_HEAD_STABILITY", "0")
    mod = importlib.import_module(CFG_MODULE)
    importlib.reload(mod)
    try:
        assert mod.ENABLE_HEAD_STABILITY is False
        cfg = mod.make_microduck_velocity_env_cfg()
        assert "head_motion" not in cfg.rewards
        assert "head_droop" not in cfg.rewards
        assert "head_stability_weight" not in cfg.curriculum
        assert "head_droop_weight" not in cfg.curriculum
    finally:
        monkeypatch.delenv("MICRODUCK_HEAD_STABILITY", raising=False)
        importlib.reload(mod)


# --------------------------------------------------------------------------- #
# mdp.head_droop_penalty
# --------------------------------------------------------------------------- #
class _Data:
    def __init__(self, n, nj=6):
        self.joint_pos = torch.zeros(n, nj)
        self.default_joint_pos = torch.zeros(n, nj)


class _Asset:
    def __init__(self, data):
        self.data = data


class _Scene:
    def __init__(self, asset):
        self._asset = asset

    def __getitem__(self, _):
        return self._asset


class _Cmd:
    def __init__(self, n, dim=4):
        self.cmd = torch.zeros(n, dim)

    def get_command(self, _):
        return self.cmd


class _Env:
    """Minimal env with the head id cache pre-seeded (normally built from the asset)."""

    def __init__(self, n=2):
        self.num_envs = n
        self.device = torch.device("cpu")
        self.scene = _Scene(_Asset(_Data(n)))
        self.command_manager = _Cmd(n)
        self._head_pose_neck_ids = torch.tensor([0, 1, 2, 3])
        self._head_pose_bl_ids = torch.tensor([0, 0, 0, 0])
        self._head_pose_bl_mask = torch.zeros(4)


def _droop(env, **kw):
    return microduck_mdp.head_droop_penalty(env, **kw)


def test_droop_is_zero_when_the_head_is_at_the_command():
    env = _Env()
    assert torch.allclose(_droop(env), torch.zeros(2))


def test_droop_is_zero_when_the_head_is_ABOVE_the_command():
    """The whole point of one-sided: holding the head up must not be taxed,
    otherwise the term is just an unconditional 'pull to HOME' and fights the
    head_pose command (the documented reason `pose` excludes neck/head)."""
    env = _Env()
    env.scene._asset.data.joint_pos[:, 0] = 0.5   # neck_pitch raised
    env.scene._asset.data.joint_pos[:, 1] = 0.5   # head_pitch raised
    assert torch.allclose(_droop(env), torch.zeros(2))
    # ... and it stays free when the command asks for a look-DOWN that is obeyed
    env.command_manager.cmd[:, :2] = -1.0
    env.scene._asset.data.joint_pos[:, :2] = -1.0
    assert torch.allclose(_droop(env), torch.zeros(2))


def test_droop_charges_the_escapable_part_beyond_the_deadband():
    env = _Env()
    db = HEAD_DROOP_DEADBAND
    env.scene._asset.data.joint_pos[0, 0] = -(db - 0.01)   # env 0: inside deadband
    env.scene._asset.data.joint_pos[1, 1] = -(db + 0.30)   # env 1: 0.30 past it
    out = _droop(env)
    assert out[0].item() == pytest.approx(0.0, abs=1e-6)
    assert out[1].item() == pytest.approx(-0.30, abs=1e-6)
    assert (out <= 0).all(), "must be a cost (positive weight in the cfg)"


def test_droop_is_measured_against_the_COMMAND_not_against_HOME():
    """A commanded look-down is not droop; an uncommanded one is."""
    env = _Env()
    env.scene._asset.data.joint_pos[:, 0] = -1.0
    env.command_manager.cmd[:, 0] = 0.0
    charged = _droop(env)[0].item()
    env.command_manager.cmd[:, 0] = -1.0
    assert _droop(env)[0].item() == pytest.approx(0.0, abs=1e-6)
    assert charged < -0.5, charged


def test_droop_scales_with_the_magnitude_of_the_measured_defect():
    """Calibration against the real defect: neck_pitch -1.934 rad from HOME.

    At deadband 0.15 that is a cost of (1.934 - 0.15) * weight = -3.57/step at
    the final weight 2.0 — far more than the defect can pay for anywhere else in
    the stack. The official pose (-0.127) stays inside the deadband and pays 0.
    """
    env = _Env()
    env.scene._asset.data.joint_pos[:, 0] = -1.934
    raw = _droop(env)[0].item()
    assert raw == pytest.approx(-(1.934 - HEAD_DROOP_DEADBAND), abs=1e-6)
    assert raw * HEAD_DROOP_WEIGHT == pytest.approx(-3.568, abs=1e-2)

    env.scene._asset.data.joint_pos[:, 0] = -0.127
    assert _droop(env)[0].item() == pytest.approx(0.0, abs=1e-6)


def test_droop_reads_through_the_backlash_hinge_when_present():
    """The encoder sits on the OUTPUT side, so the term must measure the same
    quantity the policy sees (AGENTS.md: reward and obs must share a view)."""
    env = _Env()
    nj = 6
    env._head_pose_bl_ids = torch.tensor([4, 5, 0, 0])
    env._head_pose_bl_mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
    env.scene._asset.data.joint_pos = torch.zeros(2, nj)
    env.scene._asset.data.joint_pos[:, 4] = -0.6   # neck output link droops
    env.scene._asset.data.joint_pos[:, 5] = -0.6
    out = _droop(env)
    assert out[0].item() == pytest.approx(-2 * (0.6 - HEAD_DROOP_DEADBAND), abs=1e-6)


def test_motion_deadband_separates_the_official_gait_from_the_defect():
    """Numbers this pins: official 0.75 deg @2.15 Hz ~= 0.18 rad/s (free),
    broken 16.4 deg @2.15 Hz ~= 3.87 rad/s (charged)."""
    import math
    peak_official = math.radians(0.75) * 2 * math.pi * 2.15
    peak_broken = math.radians(16.4) * 2 * math.pi * 2.15
    assert peak_official < HEAD_MOTION_DEADBAND_RAD_S < peak_broken
    assert peak_official < 0.5
    assert peak_broken > 3.0


# --------------------------------------------------------------------------- #
# the sign convention, checked against the real model on CPU (no GPU needed)
# --------------------------------------------------------------------------- #
def test_negative_neck_pitch_really_lowers_the_head_in_the_model():
    """The term charges the NEGATIVE direction; prove that direction is 'down'.

    Everything about this term depends on which sign of neck_pitch droops the
    head. Rather than trust a sign in a docstring, this runs forward kinematics
    on the actual walk MJCF: with the legs at HOME, neck_pitch = -1.585 rad (the
    position the real policy parks at, -1.934 from HOME) must put the head body
    LOWER than HOME does — while neck_pitch = +1.047 rad (its upper limit,
    "head back") must not.

    Also pins the two reference heights the acceptance table quotes: HOME gives
    the nominal 232.6 mm head height.
    """
    import mujoco
    import pytest as _pytest

    from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_XML

    model = mujoco.MjModel.from_xml_path(str(MICRODUCK_WALK_XML))
    data = mujoco.MjData(model)

    def qadr(name):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return model.jnt_qposadr[jid]

    head_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
    home_legs = {
        "left_hip_roll": -0.0873, "right_hip_roll": 0.0873,
        "left_hip_pitch": -0.4579, "right_hip_pitch": 0.4579,
        "left_knee": -0.0049, "right_knee": 0.0049,
        "left_ankle": 0.4530, "right_ankle": -0.4530,
    }

    def head_z(neck_pitch):
        mujoco.mj_resetData(model, data)
        for name, value in home_legs.items():
            data.qpos[qadr(name)] = value
        data.qpos[qadr("neck_pitch")] = neck_pitch
        data.qpos[qadr("head_pitch")] = 0.3491  # HOME
        mujoco.mj_forward(model, data)
        return float(data.xpos[head_body][2])

    z_home = head_z(0.3491)
    z_drooped = head_z(-1.585)
    z_back = head_z(1.0472)

    assert z_home == _pytest.approx(0.2326, abs=1.5e-3), z_home
    # the defect direction is DOWN ...
    assert z_drooped < z_home - 0.05, (z_drooped, z_home)
    # ... and the term charges exactly that side (a raised/back head is not
    # "drooping", so it must not be charged even though it also sits lower
    # than the neutral pose: pitch is a rotation, not a height)
    assert z_back < z_home
    env = _Env()
    env.scene._asset.data.joint_pos[:, 0] = 1.0472 - 0.3491  # raised, rel HOME
    assert _droop(env)[0].item() == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# mdp.head_forward_penalty — "头要一直向前" (user requirement, 2026-09-17)
# --------------------------------------------------------------------------- #
def test_forward_is_free_when_the_head_points_straight_ahead():
    env = _Env()
    assert torch.allclose(microduck_mdp.head_forward_penalty(env), torch.zeros(2))


def test_forward_charges_both_signs_beyond_the_deadband():
    """Yaw is two-sided: left and right are equally off-forward (unlike droop)."""
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
        HEAD_FORWARD_DEADBAND_YAW,
    )
    env = _Env()
    env.scene._asset.data.joint_pos[0, 2] = +(HEAD_FORWARD_DEADBAND_YAW + 0.10)
    env.scene._asset.data.joint_pos[1, 2] = -(HEAD_FORWARD_DEADBAND_YAW + 0.10)
    out = microduck_mdp.head_forward_penalty(env)
    assert out[0].item() == pytest.approx(-0.10, abs=1e-6)
    assert out[1].item() == pytest.approx(-0.10, abs=1e-6)
    assert out[0].item() == pytest.approx(out[1].item(), abs=1e-9), "must be symmetric"
    assert (out <= 0).all(), "must be a cost (positive weight in the cfg)"


def test_forward_is_free_for_the_measured_official_head():
    """The deadbands must clear the OFFICIAL policy's own peaks, or this term is
    just a tax on a good gait. Measured peaks (mean + 2 sigma, both commands):
    walking yaw 0.104 / roll 0.088; standing yaw 0.064 / roll 0.043.
    """
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
        HEAD_FORWARD_DEADBAND_ROLL,
        HEAD_FORWARD_DEADBAND_YAW,
    )
    official_peak_yaw = max(0.014 + 2 * 0.045, 0.018 + 2 * 0.023)      # 0.104
    official_peak_roll = max(0.024 + 2 * 0.032, 0.035 + 2 * 0.004)     # 0.088
    assert HEAD_FORWARD_DEADBAND_YAW >= official_peak_yaw, HEAD_FORWARD_DEADBAND_YAW
    assert HEAD_FORWARD_DEADBAND_ROLL >= official_peak_roll, HEAD_FORWARD_DEADBAND_ROLL

    env = _Env()
    for i in range(2):
        env.scene._asset.data.joint_pos[i, 2] = official_peak_yaw
        env.scene._asset.data.joint_pos[i, 3] = official_peak_roll
    assert torch.allclose(microduck_mdp.head_forward_penalty(env), torch.zeros(2))


def test_forward_binds_on_the_measured_standing_roll_defect():
    """The one real defect this measured: our policy stands with head_roll at
    +0.119 rad (6.8 deg) while the official holds -0.035 (2 deg). The roll
    deadband must sit between the two so the defect is charged."""
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
        HEAD_FORWARD_DEADBAND_ROLL,
        HEAD_FORWARD_WEIGHT,
    )
    assert 0.035 + 2 * 0.004 < HEAD_FORWARD_DEADBAND_ROLL < 0.119
    env = _Env()
    env.scene._asset.data.joint_pos[:, 3] = 0.119
    charged = microduck_mdp.head_forward_penalty(env)[0].item() * HEAD_FORWARD_WEIGHT
    assert charged < 0.0, charged


def test_forward_does_not_fight_a_COMMANDED_look_sideways():
    """A commanded yaw must stay free, or this term and head_pose_tracking pull
    the same joint in opposite directions (the documented reason `pose` excludes
    neck/head). The gate is per-joint, so a commanded yaw does NOT excuse roll."""
    env = _Env()
    env.scene._asset.data.joint_pos[:, 2] = -1.0      # head turned hard right
    env.scene._asset.data.joint_pos[:, 3] = +0.5      # and tilted
    env.command_manager.cmd[:, 2] = -1.0              # ... which was commanded
    out = microduck_mdp.head_forward_penalty(env)
    # yaw free (commanded), roll still charged
    assert out[0].item() == pytest.approx(-(0.5 - 0.10), abs=1e-6)


def test_forward_gate_can_be_disabled_for_the_strict_reading():
    """"一直向前" read literally = charge even against a command (command_gate=0)."""
    env = _Env()
    env.scene._asset.data.joint_pos[:, 2] = -1.0
    env.command_manager.cmd[:, 2] = -1.0
    strict = microduck_mdp.head_forward_penalty(env, command_gate=0.0)
    assert strict[0].item() == pytest.approx(-(1.0 - 0.12), abs=1e-6)


def test_forward_reads_through_the_backlash_hinge():
    env = _Env()
    env._head_pose_bl_ids = torch.tensor([4, 5, 4, 5])
    env._head_pose_bl_mask = torch.tensor([1.0, 1.0, 1.0, 1.0])
    env.scene._asset.data.joint_pos = torch.zeros(2, 6)
    env.scene._asset.data.joint_pos[:, 4] = 0.40   # yaw output link, through play
    env.scene._asset.data.joint_pos[:, 5] = 0.40   # roll output link
    out = microduck_mdp.head_forward_penalty(env)
    # yaw channel: 0.40 - 0.12; roll channel: 0.40 - 0.10
    assert out[0].item() == pytest.approx(-((0.40 - 0.12) + (0.40 - 0.10)), abs=1e-6)


def test_forward_is_calibrated_against_the_measured_defect():
    """A W5/R3-class head thrash is 16-20 deg = 0.28-0.35 rad -> charged; the
    official baseline (1-3.5 deg) -> free. That separation is the whole design."""
    import math
    env = _Env()
    env.scene._asset.data.joint_pos[:, 2] = math.radians(20.0)   # thrashing
    env.scene._asset.data.joint_pos[:, 3] = 0.0                  # yaw only
    raw = microduck_mdp.head_forward_penalty(env)[0].item()
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_FORWARD_WEIGHT
    assert raw * HEAD_FORWARD_WEIGHT < -0.3, raw * HEAD_FORWARD_WEIGHT


# --------------------------------------------------------------------------- #
# yaw_wander_metric — the diagnostic agent E asked for (2026-09-18)
# --------------------------------------------------------------------------- #
class _MetricEnv:
    """Fake env: root ang-vel z + a twist command, both directly settable."""

    class _Data:
        def __init__(self, n):
            self.root_link_ang_vel_b = torch.zeros(n, 3)

    class _Asset:
        def __init__(self, data):
            self.data = data

    class _Scene:
        def __init__(self, asset):
            self._asset = asset

        def __getitem__(self, _):
            return self._asset

    class _Cmd:
        def __init__(self, n):
            self.cmd = torch.zeros(n, 3)

        def get_command(self, _):
            return self.cmd

    def __init__(self, n=2):
        self.num_envs = n
        self.device = torch.device("cpu")
        self.step_dt = 0.02
        self.episode_length_buf = torch.full((n,), 100, dtype=torch.long)
        asset = self._Asset(self._Data(n))
        self.scene = self._Scene(asset)
        self.command_manager = self._Cmd(n)


def _settle_metric(env, wz, steps=200, cmd_xy=0.2, cmd_wz=0.0):
    env.command_manager.cmd[:, 0] = cmd_xy
    env.command_manager.cmd[:, 2] = cmd_wz
    env.scene["robot"].data.root_link_ang_vel_b[:, 2] = wz
    out = None
    for _ in range(steps):
        out = microduck_mdp.yaw_wander_metric(env)
    return out


def test_wander_metric_reads_the_bias_not_the_oscillation():
    """A step-frequency wobble must average out; a DC bias must survive."""
    env = _MetricEnv()
    steady = _settle_metric(env, 0.20)                       # constant bias
    assert steady[0].item() == pytest.approx(0.20, abs=0.01)

    env2 = _MetricEnv()
    for i in range(400):                                     # zero-mean wobble
        env2.command_manager.cmd[:, 0] = 0.2
        env2.scene["robot"].data.root_link_ang_vel_b[:, 2] = 0.5 if i % 2 else -0.5
        out = microduck_mdp.yaw_wander_metric(env2)
    assert abs(out[0].item()) < 0.02, out[0].item()


def test_wander_metric_does_not_cancel_across_envs():
    """THE reason this metric exists: +bias in one env and -bias in another must
    NOT average to zero — that cancellation is exactly how a wandering policy
    looked 'aligned' on the signed mean."""
    env = _MetricEnv(n=2)
    env.command_manager.cmd[:, 0] = 0.2
    env.scene["robot"].data.root_link_ang_vel_b[0, 2] = +0.4
    env.scene["robot"].data.root_link_ang_vel_b[1, 2] = -0.4
    for _ in range(200):
        unsigned = microduck_mdp.yaw_wander_metric(env, signed=False)
        signed = microduck_mdp.yaw_wander_metric(env, signed=True)
    assert signed.mean().item() == pytest.approx(0.0, abs=0.02), "signed mean cancels"
    assert unsigned.mean().item() == pytest.approx(0.4, abs=0.02), "|.| must not"


def test_wander_metric_is_gated_to_straight_walking():
    env = _MetricEnv()
    _settle_metric(env, 0.3, cmd_wz=0.5)          # commanded turn -> not scored
    assert microduck_mdp.yaw_wander_metric(env).abs().max().item() == 0.0
    env = _MetricEnv()
    _settle_metric(env, 0.3, cmd_xy=0.0)          # standing -> not scored
    assert microduck_mdp.yaw_wander_metric(env).abs().max().item() == 0.0


def test_wander_metric_is_a_metric_and_not_a_reward():
    """It must not be able to change behaviour: no reward slot, metrics present."""
    cfg = make_microduck_velocity_env_cfg()
    assert "yaw_dc_abs" in cfg.metrics and "yaw_dc_signed" in cfg.metrics
    assert "yaw_wander" not in cfg.rewards


# --------------------------------------------------------------------------- #
# feet_distance_max_penalty — 站距上限（agent D 的处方②，2026-09-18）
# --------------------------------------------------------------------------- #
class _SiteEnv:
    """Fake env with two foot sites whose world xy positions we set directly."""

    class _Data:
        def __init__(self, n):
            self.site_pos_w = torch.zeros(n, 2, 3)

    class _Asset:
        def __init__(self, data):
            self.data = data

    class _Scene:
        def __init__(self, asset):
            self._asset = asset

        def __getitem__(self, _):
            return self._asset

    class _Cmd:
        def __init__(self, n):
            self.cmd = torch.zeros(n, 3)

        def get_command(self, _):
            return self.cmd

    def __init__(self, n=2, sep=0.130):
        self.num_envs = n
        self.device = torch.device("cpu")
        asset = self._Asset(self._Data(n))
        self.scene = self._Scene(asset)
        self.command_manager = self._Cmd(n)
        self.command_manager.cmd[:, 0] = 0.2          # walking
        self.scene["robot"].data.site_pos_w[:, 0, 0] = 0.0
        self.scene["robot"].data.site_pos_w[:, 0, 1] = +sep / 2
        self.scene["robot"].data.site_pos_w[:, 1, 0] = 0.0
        self.scene["robot"].data.site_pos_w[:, 1, 1] = -sep / 2

    def set_sep(self, sep):
        self.scene["robot"].data.site_pos_w[:, 0, 1] = +sep / 2
        self.scene["robot"].data.site_pos_w[:, 1, 1] = -sep / 2


_ASSET_CFG = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))


def test_feet_max_width_is_free_for_the_official_stance():
    """The official stands 65-84 mm apart; the deadband must clear that."""
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import FEET_MAX_WIDTH
    assert FEET_MAX_WIDTH > 0.084, FEET_MAX_WIDTH
    env = _SiteEnv(sep=0.084)
    out = microduck_mdp.feet_distance_max_penalty(env, FEET_MAX_WIDTH, _ASSET_CFG)
    assert torch.allclose(out, torch.zeros(2)), out


def test_feet_max_width_charges_our_measured_stance():
    """Our measured 130 mm must be charged, in proportion to the excess."""
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import FEET_MAX_WIDTH
    env = _SiteEnv(sep=0.130)
    out = microduck_mdp.feet_distance_max_penalty(env, FEET_MAX_WIDTH, _ASSET_CFG)
    assert out[0].item() == pytest.approx(-(0.130 - FEET_MAX_WIDTH), abs=1e-6)
    assert (out <= 0).all(), "self-negating: the cfg weight must be POSITIVE"


def test_feet_max_width_weight_sign_makes_the_term_a_cost():
    """AGENTS.md's double-negation trap: a NEGATIVE weight here would pay the
    policy for standing wider. The weighted value must be <= 0."""
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
        FEET_MAX_WIDTH, FEET_MAX_WIDTH_WEIGHT,
    )
    assert FEET_MAX_WIDTH_WEIGHT > 0.0, "self-negating penalty needs a POSITIVE weight"
    env = _SiteEnv(sep=0.150)
    raw = microduck_mdp.feet_distance_max_penalty(env, FEET_MAX_WIDTH, _ASSET_CFG)
    assert (raw * FEET_MAX_WIDTH_WEIGHT <= 0).all()


def test_feet_max_width_is_gated_to_walking():
    """Standing must not be taxed: the wide stance is a walking defect and
    standing is the behaviour that currently works."""
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import FEET_MAX_WIDTH
    env = _SiteEnv(sep=0.150)
    env.command_manager.cmd[:, 0] = 0.0
    out = microduck_mdp.feet_distance_max_penalty(env, FEET_MAX_WIDTH, _ASSET_CFG)
    assert torch.allclose(out, torch.zeros(2)), out


def test_feet_max_width_is_off_by_default_and_keeps_the_official_recipe():
    cfg = make_microduck_velocity_env_cfg()
    assert "feet_max_width" not in cfg.rewards


# --------------------------------------------------------------------------- #
# flight_penalty — 路线 A 的核心杠杆（2026-09-18）
# --------------------------------------------------------------------------- #
class _ContactEnv:
    class _Data:
        def __init__(self, n):
            self.found = torch.zeros(n, 2, dtype=torch.bool)

    class _Sensor:
        def __init__(self, data):
            self.data = data

    class _Cmd:
        def __init__(self, n):
            self.cmd = torch.zeros(n, 3)

        def get_command(self, _):
            return self.cmd

    def __init__(self, n=2):
        self.num_envs = n
        self.device = torch.device("cpu")
        self.scene = type("S", (), {})()
        self.scene.sensors = {"feet_ground_contact": self._Sensor(self._Data(n))}
        self.command_manager = self._Cmd(n)
        self.command_manager.cmd[:, 0] = 0.2
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)

    def set_contact(self, left, right):
        self.scene.sensors["feet_ground_contact"].data.found[:, 0] = left
        self.scene.sensors["feet_ground_contact"].data.found[:, 1] = right


def test_flight_penalty_charges_only_when_BOTH_feet_are_off():
    env = _ContactEnv()
    env.set_contact(True, False)          # single support = walking
    assert microduck_mdp.flight_penalty(env, "feet_ground_contact").abs().max().item() == 0.0
    env.set_contact(True, True)           # double support = standing/loading
    assert microduck_mdp.flight_penalty(env, "feet_ground_contact").abs().max().item() == 0.0
    env.set_contact(False, False)         # flight = running
    out = microduck_mdp.flight_penalty(env, "feet_ground_contact")
    assert (out < 0).all(), "self-negating: the cfg weight must be POSITIVE"


def test_flight_penalty_is_gated_to_walking():
    env = _ContactEnv()
    env.set_contact(False, False)
    env.command_manager.cmd[:, 0] = 0.0
    assert microduck_mdp.flight_penalty(env, "feet_ground_contact").abs().max().item() == 0.0


def test_flight_penalty_grace_steps_needs_consecutive_flight():
    env = _ContactEnv()
    env.set_contact(False, False)
    kw = dict(sensor_name="feet_ground_contact", grace_steps=2)
    assert microduck_mdp.flight_penalty(env, **kw).abs().max().item() == 0.0   # 1st
    assert microduck_mdp.flight_penalty(env, **kw).abs().max().item() == 0.0   # 2nd
    assert (microduck_mdp.flight_penalty(env, **kw) < 0).all()                 # 3rd -> charged


def test_flight_penalty_weight_sign_and_default_off():
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import FLIGHT_PENALTY_WEIGHT
    assert FLIGHT_PENALTY_WEIGHT > 0.0, "self-negating penalty needs a POSITIVE weight"
    cfg = make_microduck_velocity_env_cfg()
    assert "flight" not in cfg.rewards, "route-A lever must stay opt-in"


# --------------------------------------------------------------------------- #
# track_angular_velocity_z_only — 去稀释的偏航定价（agent A 的 4.2×，2026-09-18）
# --------------------------------------------------------------------------- #
class _YawEnv:
    class _Data:
        def __init__(self, n):
            self.root_link_ang_vel_b = torch.zeros(n, 3)

    class _Asset:
        def __init__(self, data):
            self.data = data

    class _Scene:
        def __init__(self, asset):
            self._asset = asset

        def __getitem__(self, _):
            return self._asset

    class _Cmd:
        def __init__(self, n, wz=0.0):
            self.cmd = torch.zeros(n, 3)
            self.cmd[:, 2] = wz

        def get_command(self, _):
            return self.cmd

    def __init__(self, n=2, cmd_wz=0.0):
        self.num_envs = n
        self.device = torch.device("cpu")
        asset = self._Asset(self._Data(n))
        self.scene = self._Scene(asset)
        self.command_manager = self._Cmd(n, cmd_wz)


def test_z_only_tracking_ignores_uncommanded_roll_and_pitch():
    """THE point of the term: the roll/pitch rates that dilute mjlab's version must
    not move it at all."""
    env = _YawEnv(cmd_wz=0.4)
    env.scene["robot"].data.root_link_ang_vel_b[:, 2] = 0.4     # perfect yaw track
    perfect = microduck_mdp.track_angular_velocity_z_only(env, 0.7071)
    assert perfect[0].item() == pytest.approx(1.0, abs=1e-6)
    # add a big roll/pitch rate: value must be UNCHANGED
    env.scene["robot"].data.root_link_ang_vel_b[:, 0] = 0.9
    env.scene["robot"].data.root_link_ang_vel_b[:, 1] = -0.9
    assert microduck_mdp.track_angular_velocity_z_only(env, 0.7071)[0].item() == pytest.approx(
        perfect[0].item(), abs=1e-9
    )


def test_z_only_tracking_is_strictly_stronger_than_the_diluted_official_term():
    """Quantify agent A's 4.2x: at the measured operating point the separated term
    must carry visibly more authority for the same yaw error."""
    from mjlab.tasks.velocity.mdp import track_angular_velocity as official
    env = _YawEnv(cmd_wz=0.4)
    env.scene["robot"].data.root_link_ang_vel_b[:, 0] = 0.62   # measured |w_xy|
    env.scene["robot"].data.root_link_ang_vel_b[:, 1] = 0.62
    env.scene["robot"].data.root_link_ang_vel_b[:, 2] = 0.4    # perfect yaw track
    official_v = official(env, std=0.7071, command_name="twist")
    z_only_v = microduck_mdp.track_angular_velocity_z_only(env, 0.7071)
    assert z_only_v[0].item() > 3.0 * official_v[0].item(), (z_only_v[0], official_v[0])


def test_z_only_tracking_is_off_by_default():
    cfg = make_microduck_velocity_env_cfg()
    assert "track_angular_velocity_z" not in cfg.rewards
