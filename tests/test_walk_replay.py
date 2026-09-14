"""Walk-angle record/replay regression tests (CPU, no pyserial, no GPU).

Locks the sim2servo contract of scripts/record_walk_angles.py and
scripts/replay_walk_angles.py:

* FT-SCS frame builders match the official doc examples byte-for-byte
  (docs/ftscs_protocol.md §4.1/4.2/4.6) — a checksum bug here would silently
  make every servo ignore the replay;
* rad → servo tick conversion matches duck-control/src/ftbus.rs
  `rad_to_position_count` (runtime authority);
* policy (14 joints) → wire (15 servos, mouth skipped) mapping;
* the seam-free loop window picks an integer number of gait periods.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import replay_walk_angles as rw  # noqa: E402  (no pyserial needed for these parts)
import record_walk_angles as rec  # noqa: E402

# doc §4.6 example: FF FF FE 20 83 2A 06 | 01 00 08 00 00 E8 03 | ... | 58
DOC_SYNC_WRITE = bytes.fromhex(
    "fffffe20832a060100080000e8030200080000e8030300080000e8030400080000e80358"
)


def test_sync_write_matches_protocol_doc_example() -> None:
    frame = rw.build_sync_write(
        [1, 2, 3, 4], 0x2A, [bytes.fromhex("00080000e803")] * 4
    )
    assert frame == DOC_SYNC_WRITE


def test_ping_and_read_match_protocol_doc_examples() -> None:
    # doc §4.1: FF FF 01 02 01 FB
    assert rw.build_ping(1) == bytes.fromhex("ffff010201fb")
    # doc §4.2: FF FF 01 04 02 38 02 BE
    assert rw.build_read(1, 0x38, 2) == bytes.fromhex("ffff0104023802be")


def test_write_matches_protocol_doc_examples() -> None:
    # doc §4.3 example 3 (broadcast, set ID=1 at addr 5): FF FF FE 04 03 05 01 F4
    assert rw.build_write(0xFE, 5, b"\x01") == bytes.fromhex("fffffe04030501f4")
    # doc §4.3 example 4 (ID 1, addr 0x2A, six data bytes): FF FF 01 09 03 2A 00 08 00 00 E8 03 D5
    assert rw.build_write(1, 0x2A, bytes.fromhex("00080000e803")) == \
        bytes.fromhex("ffff0109032a00080000e803d5")


def test_sync_write_15_servos_checksum_and_len() -> None:
    data = [t.to_bytes(2, "little") for t in [2048] * 15]
    frame = rw.build_sync_write(rw.WIRE_JOINT_IDS, rw.REG_GOAL_POSITION, data)
    # len byte = (2+1)*15 + 4 = 49
    assert frame[3] == 49
    # checksum is the last byte
    assert frame[-1] == rw.frame_checksum(0xFE, 49, 0x83, frame[5:-1])


def test_rad_to_tick_matches_runtime_convention() -> None:
    # ftbus.rs: count = (rad + pi) / (2*pi/4096), clamped to 0..4095
    assert rw.rad_to_tick(-np.pi) == 0
    assert rw.rad_to_tick(0.0) == 2048
    assert rw.rad_to_tick(np.pi) in (4095, 4096 - 1)  # round() may go 4096 → clamp
    assert rw.rad_to_tick(3.3) == 4095  # clamps, never wraps
    assert rw.rad_to_tick(-3.5) == 0
    # round-trips: measured home pose lands near 2048 for the zero joints
    assert abs(rw.rad_to_tick(0.3491) - 2048) < 300  # neck home tilt < 26 deg


def test_policy_to_wire_mapping_skips_mouth() -> None:
    assert rw.policy_to_wire_joint(0) == 0
    assert rw.policy_to_wire_joint(8) == 8
    assert rw.policy_to_wire_joint(9) == 10
    assert rw.policy_to_wire_joint(13) == 14
    for s in range(14):
        j = rw.policy_to_wire_joint(s)
        assert j != rw.MOUTH_INDEX
        assert int(rw.WIRE_JOINT_IDS[j]) in rw.WIRE_JOINT_IDS


def test_home_pose_wire_matches_runtime_default_position() -> None:
    # runtime model.rs DEFAULT_POSITION — copy semantics must not drift
    assert len(rw.HOME_POSE_WIRE) == 15
    assert abs(rw.HOME_POSE_WIRE[1] + 0.0873) < 1e-9      # left_hip_roll
    assert abs(rw.HOME_POSE_WIRE[14] + 0.4530) < 1e-9     # right_ankle
    # legs mirrored
    for l, r in ((1, 11), (2, 12), (3, 13), (4, 14)):
        assert abs(rw.HOME_POSE_WIRE[l] + rw.HOME_POSE_WIRE[r]) < 1e-9


def test_loop_window_uses_integer_number_of_periods() -> None:
    dt = 0.02
    t = np.arange(0.0, 8.0, dt)
    sig = np.sin(2 * np.pi * t / 0.38) + 0.1 * np.cos(2 * np.pi * t / 0.38 * 2)
    period, window = rec.loop_window(sig, dt)
    assert period >= 15 and period <= 23  # 0.38 s ± tolerance in steps
    assert window % period == 0           # integer cycle count → seam-free
    assert window <= len(sig)


def test_loop_window_flat_signal_returns_whole_file() -> None:
    period, window = rec.loop_window(np.zeros(400), 0.02)
    assert period == 0 and window == 400


def test_gait_freq_band_sane() -> None:
    dt = 0.02
    t = np.arange(0.0, 8.0, dt)
    sig = np.sin(2 * np.pi * 2.63 * t)
    f = rec.gait_freq_hz(sig, dt)
    assert 2.0 < f < 3.5
