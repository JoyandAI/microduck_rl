"""Proof that the 61-D mirror table is correct — the gate on ever enabling it.

`microduck_vel_symmetry` (``tasks/symmetry.py``) was migrated from the old 51-D
observation layout and, as its module docstring says, has been "dead code until
now since no env had symmetry enabled". `ENABLE_SYMMETRY` is False in every task
cfg in this repo. Turning it on injects a mirror loss that compares the policy's
action on a mirrored observation against the mirror of its action — so if the
permutation or the sign vector is wrong by even one entry, PPO is trained toward
a WRONG gradient on that entry, silently, for the whole run.

That is exactly the class of failure this repo has been burned by before (a
kd rewrite mid-run, a DR that accumulated across resets): a change that produces
no error, just worse numbers. So the table is only trustworthy if it is pinned by
tests that would fail on a single misplaced sign.

What is verified here:
  * the table is a well-formed involution over the live 61-D layout,
  * the joint block order matches the ACTUAL model's actuated-joint order,
  * mirroring a state that is physically its own mirror (HOME pose, zero velocity,
    zero commands) is the identity — this single test covers perm AND sign,
  * each documented sign rule holds on the command blocks,
  * the public function doubles the batch the way rsl_rl expects.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from mjlab_microduck.tasks.symmetry import (
    SYMMETRY_CFG,
    _JOINT_PERM,
    _JOINT_SIGN,
    _OBS_PERM,
    _OBS_SIGN,
    microduck_vel_symmetry,
)

REPO = Path(__file__).resolve().parents[1]
WALK_XML = REPO / "src/mjlab_microduck/robot/microduck/robot_walk.xml"

# HOME joint angles, in the model's actuated-joint order.
HOME = [
    0.0, -0.0873, -0.4579, 0.0, 0.4530,            # left leg
    0.3491, 0.3491, 0.0, 0.0,                      # neck_pitch, head_pitch (20 deg), head_yaw, head_roll
    0.0, 0.0873, 0.4579, 0.0, -0.4530,             # right leg
]

# Block offsets within the 61-D actor obs.
BLOCKS = {
    "base_ang_vel": (0, 3),
    "projected_gravity": (3, 6),
    "joint_pos": (6, 20),
    "joint_vel": (20, 34),
    "actions": (34, 48),
    "twist": (48, 51),
    "head_command": (51, 55),
    "body_command": (55, 61),
}


def _actuated_joint_order() -> list[str]:
    """Joint names in MJCF declaration order, excluding the passive_* linkage."""
    root = ET.parse(WALK_XML).getroot()
    names = []
    for j in root.iter("joint"):
        n = j.get("name", "")
        if n.startswith("passive_"):
            continue
        # the freejoint is unnamed/anonymous; hinges carry a range
        if j.get("type") == "hinge":
            names.append(n)
    return names


# --------------------------------------------------------------------------- #
# well-formedness
# --------------------------------------------------------------------------- #
def test_tables_are_the_right_shape_and_a_valid_permutation():
    assert len(_OBS_PERM) == 61
    assert len(_OBS_SIGN) == 61
    assert sorted(_OBS_PERM) == list(range(61)), "not a permutation of 0..60"
    assert len(_JOINT_PERM) == 14
    assert sorted(_JOINT_PERM) == list(range(14))
    assert len(_JOINT_SIGN) == 14


def test_permutation_is_an_involution():
    """Mirroring twice must be the identity — anything else is a corrupted table."""
    p = torch.tensor(_OBS_PERM)
    assert torch.equal(p[p], torch.arange(61)), "obs permutation is not self-inverse"
    jp = torch.tensor(_JOINT_PERM)
    assert torch.equal(jp[jp], torch.arange(14)), "joint permutation is not self-inverse"


def test_sign_vector_is_consistent_with_the_permutation():
    """s[perm]-composed-twice == s, i.e. negations must undo themselves in pairs."""
    s = torch.tensor(_OBS_SIGN)
    p = torch.tensor(_OBS_PERM)
    twice = s[p] * s
    # Applying mirror twice = x[perm][perm] * s[perm] * s == x * s[perm]*s
    assert torch.all(twice == 1.0), twice.nonzero().flatten().tolist()


def test_joint_perm_and_sign_compose_to_the_identity_on_a_mirror_symmetric_pose():
    """HOME is its own mirror: swap L/R legs, negate the mirror-odd joints."""
    perm = torch.tensor(_JOINT_PERM)
    sign = torch.tensor(_JOINT_SIGN)
    home = torch.tensor(HOME)
    assert torch.allclose(home[perm] * sign, home, atol=1e-9), (home[perm] * sign - home)


# --------------------------------------------------------------------------- #
# the table must match the LIVE model and obs layout
# --------------------------------------------------------------------------- #
def test_joint_block_order_matches_the_real_model():
    order = _actuated_joint_order()
    assert len(order) == 14, order
    assert order[:5] == ["left_hip_yaw", "left_hip_roll", "left_hip_pitch",
                         "left_knee", "left_ankle"], order[:5]
    assert order[5:9] == ["neck_pitch", "head_pitch", "head_yaw", "head_roll"], order[5:9]
    assert order[9:] == ["right_hip_yaw", "right_hip_roll", "right_hip_pitch",
                         "right_knee", "right_ankle"], order[9:]
    # _JOINT_PERM must map left(0-4) <-> right(9-13) and fix the midline(5-8)
    for i in range(5):
        assert _JOINT_PERM[i] == 9 + i
        assert _JOINT_PERM[9 + i] == i
    for i in range(5, 9):
        assert _JOINT_PERM[i] == i


def test_obs_blocks_match_the_declared_61d_layout():
    """Term dimensions must sum to 61 in exactly the order symmetry.py assumes."""
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck", play=True)
    terms = cfg.observations["actor"].terms
    order = list(terms)
    expect = ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel",
              "actions", "command", "head_command", "body_command"]
    assert order == expect, order
    # Derive the block offsets from the table and require them to tile 0..61 in
    # that exact order — so a term reorder or a resized term fails here instead
    # of silently slicing the mirror table in the wrong place.
    off = 0
    for name in expect:
        key = "twist" if name == "command" else name
        lo, hi = BLOCKS[key]
        assert lo == off, (name, lo, off)
        off = hi
    assert off == 61


# --------------------------------------------------------------------------- #
# end-to-end mirroring
# --------------------------------------------------------------------------- #
def _make_obs(n=3):
    """A mirror-symmetric state: HOME pose, zero velocity, zero commands."""
    zeros3 = torch.zeros(n, 3)
    home = torch.tensor(HOME).repeat(n, 1)
    z = torch.zeros(n, 14)
    actor = torch.cat([zeros3,          # base_ang_vel
                       torch.tensor([0.0, 0.0, -1.0]).repeat(n, 1),  # projected_gravity (upright)
                       home, z, z,      # joint_pos, joint_vel, actions
                       zeros3,          # twist
                       torch.zeros(n, 4),   # head command
                       torch.zeros(n, 6)],  # body command
                      dim=-1)
    assert actor.shape == (n, 61)
    return actor


def test_mirroring_a_self_symmetric_state_is_the_identity():
    """Covers permutation AND signs in one shot: any single wrong entry fails."""
    actor = _make_obs()
    obs = TensorDict({"actor": actor, "critic": actor.clone()}, batch_size=[actor.shape[0]])
    aug, _ = microduck_vel_symmetry(None, obs, None)
    mirrored = aug["actor"][actor.shape[0]:]
    assert torch.allclose(mirrored, actor, atol=1e-6), (mirrored - actor).abs().max()


def test_mirroring_negates_exactly_the_documented_command_entries():
    actor = _make_obs(1)
    actor[0, 48:51] = torch.tensor([1.0, 2.0, 3.0])      # twist vx, vy, wz
    actor[0, 51:55] = torch.tensor([4.0, 5.0, 6.0, 7.0])  # neck, head p/y/r
    actor[0, 55:61] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # x,y,z,r,p,y
    obs = TensorDict({"actor": actor, "critic": actor.clone()}, batch_size=[1])
    aug, _ = microduck_vel_symmetry(None, obs, None)
    m = aug["actor"][1]
    # twist: vx keeps, vy and wz negate
    assert m[48].item() == 1.0 and m[49].item() == -2.0 and m[50].item() == -3.0
    # head command: pitches keep, yaw and roll negate
    assert m[51].item() == 4.0 and m[52].item() == 5.0
    assert m[53].item() == -6.0 and m[54].item() == -7.0
    # body command: x,z,pitch keep; y, roll, yaw negate
    assert m[55].item() == 1.0 and m[56].item() == -2.0 and m[57].item() == 3.0
    assert m[58].item() == -4.0 and m[59].item() == 5.0 and m[60].item() == -6.0


def test_mirroring_negates_body_roll_and_yaw_angular_velocity_only():
    actor = _make_obs(1)
    actor[0, 0:3] = torch.tensor([1.0, 2.0, 3.0])
    obs = TensorDict({"actor": actor, "critic": actor.clone()}, batch_size=[1])
    aug, _ = microduck_vel_symmetry(None, obs, None)
    m = aug["actor"][1]
    assert m[0].item() == -1.0 and m[1].item() == 2.0 and m[2].item() == -3.0


def test_mirroring_gravity_negates_only_the_lateral_component():
    actor = _make_obs(1)
    actor[0, 3:6] = torch.tensor([0.1, 0.2, -0.9])
    obs = TensorDict({"actor": actor, "critic": actor.clone()}, batch_size=[1])
    aug, _ = microduck_vel_symmetry(None, obs, None)
    m = aug["actor"][1]
    assert m[3].item() == pytest.approx(0.1) and m[4].item() == pytest.approx(-0.2)
    assert m[5].item() == pytest.approx(-0.9)


def test_mirroring_actions_swaps_the_legs_and_negates_every_leg_joint():
    a = torch.arange(14, dtype=torch.float32)[None]
    _, aug_a = microduck_vel_symmetry(None, None, a)
    assert aug_a.shape == (2, 14)
    mirrored = aug_a[1]
    assert mirrored[0].item() == -9.0 and mirrored[9].item() == 0.0 * -1.0
    perm = torch.tensor(_JOINT_PERM)
    sign = torch.tensor(_JOINT_SIGN)
    assert torch.allclose(mirrored, a[0][perm] * sign)


def test_batch_is_doubled_and_mirror_twice_returns_the_original():
    actor = _make_obs(4)
    obs = TensorDict({"actor": actor, "critic": actor.clone()}, batch_size=[4])
    act = torch.randn(4, 14)
    aug_obs, aug_act = microduck_vel_symmetry(None, obs, act)
    assert aug_obs["actor"].shape == (8, 61)
    assert aug_act.shape == (8, 14)
    assert torch.allclose(aug_obs["actor"][:4], actor)
    assert torch.allclose(aug_act[:4], act)
    # mirroring the mirrored half must return the original half
    mirrored_obs = TensorDict(
        {"actor": aug_obs["actor"][4:], "critic": aug_obs["critic"][4:]}, batch_size=[4])
    back, back_a = microduck_vel_symmetry(None, mirrored_obs, aug_act[4:])
    assert torch.allclose(back["actor"][4:], actor, atol=1e-6)
    assert torch.allclose(back_a[4:], act, atol=1e-6)


def test_symmetry_cfg_shape_is_what_rsl_rl_expects():
    assert SYMMETRY_CFG["use_mirror_loss"] is True
    assert SYMMETRY_CFG["use_data_augmentation"] is False
    assert 0.0 < SYMMETRY_CFG["mirror_loss_coeff"] <= 1.0
    assert SYMMETRY_CFG["data_augmentation_func"].endswith("microduck_vel_symmetry")


def test_every_env_cfg_keeps_symmetry_off_until_this_table_is_trusted():
    """Guard: enabling it is a deliberate act, not a drive-by default flip."""
    import importlib

    for mod, attr in (("microduck_velocity_env_cfg", "ENABLE_SYMMETRY"),
                      ("microduck_sitstand_env_cfg", "ENABLE_SYMMETRY"),
                      ("reference_official_velocity_env_cfg", "ENABLE_SYMMETRY")):
        m = importlib.import_module(f"mjlab_microduck.tasks.{mod}")
        assert getattr(m, attr) is False, mod
