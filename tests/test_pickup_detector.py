"""Pick-up detector: runtime-contract and logic tests (CPU, no sim)."""

import math

import numpy as np
import pytest
import torch

from mjlab_microduck.pickup.features import (
    CURRENT_SLICE, FEAT_DIM, FEATURE_NAMES, HOLD_RAMP_S, PAUSE_POSE_REL, PAUSED_IDX, CurrentSensorDR, paused_target_rel,
)
from mjlab_microduck.pickup.hand import CARRY, IDLE, LOWER, HandCfg, VirtualHand
from mjlab_microduck.pickup.model import WINDOW, PauseStateMachine, PickupNet, StateMachineCfg, export_onnx


def test_feature_layout():
    assert FEAT_DIM == 63 and len(FEATURE_NAMES) == FEAT_DIM
    assert FEATURE_NAMES[PAUSED_IDX] == "paused"
    assert FEATURE_NAMES[CURRENT_SLICE][0] == "cur_0" and FEATURE_NAMES[CURRENT_SLICE][-1] == "cur_13"


def test_pause_pose_is_mirror_symmetric():
    p = np.array(PAUSE_POSE_REL)
    left, right = p[0:5], p[9:14]
    # HOME is mirrored for every leg joint → a symmetric posture has opposite-sign offsets
    np.testing.assert_allclose(left, -right)
    assert p[7] == 0.0 and p[8] == 0.0  # no head yaw / roll


def test_paused_target_ramps_to_pause_pose():
    start = torch.full((2, 14), 0.3)
    t = torch.tensor([0.0, HOLD_RAMP_S + 0.1])
    out = paused_target_rel(t, start)
    torch.testing.assert_close(out[0], start[0])
    torch.testing.assert_close(out[1], torch.tensor(PAUSE_POSE_REL, dtype=torch.float32))


def test_current_dr_is_nonnegative_and_quantized():
    dr = CurrentSensorDR(4, "cpu")
    c = dr(torch.randn(4, 14))
    assert (c >= 0).all()
    torch.testing.assert_close(c, torch.round(c * 1000) / 1000)


def _hand_rollout(cfg, steps, feet_force_fn):
    h = VirtualHand(1, "cpu", cfg)
    pos = torch.tensor([[0.0, 0.0, 0.12]]); quat = torch.tensor([[1.0, 0, 0, 0]])
    h.start(torch.tensor([True]), pos, quat)
    phases = []
    for k in range(steps):
        tgt, q, active = h.step(0.02, pos, quat, torch.tensor([feet_force_fn(k, h)]), torch.tensor([0.75]), torch.tensor([True]))
        pos = tgt.clone()  # perfectly stiff follower
        phases.append(int(h.phase[0]))
        assert torch.isfinite(tgt).all() and torch.isfinite(q).all()
        torch.testing.assert_close(q.norm(dim=-1), torch.ones(1))
    return phases, h


def test_hand_lifts_then_sets_down_and_releases_on_touch():
    cfg = HandCfg(pickup_rate_hz=0.0, hold_s=(1.0, 1.0), lift_z=(0.2, 0.2), drop_prob=0.0, lower_speed=(0.5, 0.5), touch_release_s=(0.1, 0.1))
    # feet touch once the target is back near standing height
    phases, h = _hand_rollout(cfg, 300, lambda k, h: 10.0 if (h.phase[0] == LOWER and h.tgt_pos[0, 2] < 0.13) else 0.0)
    assert phases[0] == CARRY and LOWER in phases and phases[-1] == IDLE
    assert max(h.anchor[0, 2].item(), 0) == pytest.approx(0.32, abs=1e-5)


def test_hand_drop_releases_at_height():
    cfg = HandCfg(pickup_rate_hz=0.0, hold_s=(0.5, 0.5), drop_prob=1.0)
    phases, _ = _hand_rollout(cfg, 50, lambda k, h: 0.0)
    assert LOWER not in phases and phases[-1] == IDLE


def test_state_machine_hysteresis():
    c = StateMachineCfg()
    sm = PauseStateMachine(1, "cpu", c)
    # brief spike shorter than n_pause → no pause
    for _ in range(c.n_pause - 1):
        sm.update(torch.tensor([0.99]))
    sm.update(torch.tensor([0.1]))
    assert not sm.paused[0]
    for _ in range(c.n_pause):
        sm.update(torch.tensor([0.99]))
    assert sm.paused[0]
    # low p immediately after the pause is ignored until min_pause_s
    n_min = math.ceil(c.min_pause_s / sm.dt)
    for _ in range(n_min - 1):
        sm.update(torch.tensor([0.0]))
    assert sm.paused[0]
    for _ in range(c.n_resume + 1):
        sm.update(torch.tensor([0.0]))
    assert not sm.paused[0]


def test_net_shapes_and_onnx_contract(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    net = PickupNet()
    net.mask[CURRENT_SLICE] = 0.0
    x = torch.randn(3, WINDOW, FEAT_DIM)
    assert net(x).shape == (3,)
    # masked features have no influence
    x2 = x.clone(); x2[..., CURRENT_SLICE] += 5.0
    torch.testing.assert_close(net(x), net(x2))
    path = tmp_path / "det.onnx"
    export_onnx(net, str(path))
    s = ort.InferenceSession(str(path))
    assert s.get_inputs()[0].shape == [1, WINDOW, FEAT_DIM]
    p = s.run(None, {"features": x[:1].numpy()})[0]
    np.testing.assert_allclose(p, torch.sigmoid(net(x[:1])).detach().numpy(), atol=1e-5)


def test_head_grip_tracks_the_head_pose_and_starts_from_it():
    """v2: a head grip targets the HEAD's pose, relative to how it was grabbed — the hand must not
    snap a tilted head level at the grasp (that would be a yank no person does)."""
    cfg = HandCfg(pickup_rate_hz=0.0, head_grip_prob=1.0, orient_prob=0.0, yaw_turn_prob=0.0,
                  tilt_buckets=((0.0, 1.0),), drift_amp=(0.0, 0.0), tremor_amp=(0.0, 0.0),
                  tremor_rot_deg=(0.0, 0.0), yaw_rate=(0.0, 0.0), lift_z=(0.1, 0.1), shake_prob=0.0)
    h = VirtualHand(1, "cpu", cfg)
    trunk_pos, trunk_q = torch.tensor([[0.0, 0.0, 0.12]]), torch.tensor([[1.0, 0, 0, 0]])
    head_pos = torch.tensor([[0.02, 0.0, 0.22]])
    s, c = math.sin(0.35), math.cos(0.35)  # head pitched 40° (two 20° joints), yaw 0
    head_q = torch.tensor([[c, 0.0, s, 0.0]])
    h.start(torch.tensor([True]), trunk_pos, trunk_q, head_pos, head_q)
    assert int(h.grip[0]) == 1
    tgt, q, _ = h.step(0.02, trunk_pos, trunk_q, torch.tensor([0.0]), torch.tensor([0.75]),
                       torch.tensor([True]), head_pos, head_q)
    torch.testing.assert_close(tgt[0, :2], head_pos[0, :2], atol=1e-4, rtol=0)
    torch.testing.assert_close(q[0].abs(), head_q[0].abs(), atol=1e-4, rtol=0)


def test_sustained_orientation_reaches_upside_down():
    cfg = HandCfg(pickup_rate_hz=0.0, orient_prob=1.0, orient_pitch=(math.pi, math.pi), orient_roll=(0.0, 0.0),
                  orient_ramp_s=(0.5, 0.5), head_grip_prob=0.0, yaw_turn_prob=0.0, tilt_buckets=((0.0, 1.0),),
                  tremor_rot_deg=(0.0, 0.0), lift_s=(0.3, 0.3), hold_s=(10.0, 10.0))
    _, h = _hand_rollout(cfg, 100, lambda k, h: 0.0)
    w, x, y, z = h.tgt_quat[0].tolist()
    up_z = 1 - 2 * (x * x + y * y)  # world z of the body's z axis
    assert up_z < -0.99, up_z
