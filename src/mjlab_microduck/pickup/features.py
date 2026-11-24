"""Per-tick feature vector + the paused-controller law — the runtime contract.

Everything here is what robotd can compute every 50 Hz tick, in the same units:

  [0:3]    gyro (rad/s, trunk IMU frame)                — ImuData.gyro
  [3:6]    projected gravity (unit vector, IMU frame)   — from the SFLP quaternion
  [6:20]   joint pos − HOME (rad)                       — same as obs[6:20]
  [20:34]  joint vel (rad/s)                            — same as obs[20:34]
  [34:48]  commanded target − HOME (rad)                — what robotd writes to the servos
  [48:62]  |present current| (A)                        — Sensors.currents_ma / 1000
  [62]     paused flag (1 while the detector holds the pose)

Joint order is the 14-servo ctrl order (0-4 left leg, 5-8 neck/head, 9-13 right leg).

While PAUSED the runtime ramps (over ``HOLD_RAMP_S``) from the last policy
target to ``PAUSE_POSE_REL`` and holds it — ``paused_target_rel`` is the exact
law. PAUSE_POSE_REL is velstand's own mean standing posture (zero command,
symmetrized): holding HOME on the floor at kp 200 TOPPLES in sim (50 % past
40° after 1 s, 93 % after 3 s); this pose buys time (11 % / 45 %) and lands
the robot in the stance the policy expects. Put-down detection must therefore
be FAST — the policy has to take over before a put-down robot tips.

(A hip-pitch "probe" sinusoid while paused was tried and dropped: in a
controlled sim bench the hand's compliance makes the trunk rock with the legs
too, so it did not separate held from on-floor, and it wiggles the robot.)
"""

from __future__ import annotations

import torch

NUM_JOINTS = 14
FEAT_DIM = 63
PAUSED_IDX = 62
CTRL_HZ = 50.0

LEG_IDS = (0, 1, 2, 3, 4, 9, 10, 11, 12, 13)  # servo ctrl indices
CURRENT_SLICE = slice(48, 62)

HOLD_RAMP_S = 0.3
# target − HOME (rad), ctrl order: velstand fhathosb@3750 standing at zero
# command, mean over 256 envs × 1 s, mirrored L/R, hip_yaw/head yaw/roll zeroed
PAUSE_POSE_REL = (
    0.0, -0.100, 0.038, -0.145, 0.044,      # left  hip_yaw, hip_roll, hip_pitch, knee, ankle
    -0.096, 0.008, 0.0, 0.0,                # neck_pitch, head_pitch, head_yaw, head_roll
    0.0, 0.100, -0.038, 0.145, -0.044,      # right hip_yaw, hip_roll, hip_pitch, knee, ankle
)

FEATURE_NAMES = (
    [f"gyro_{a}" for a in "xyz"] + [f"grav_{a}" for a in "xyz"]
    + [f"q_{i}" for i in range(14)] + [f"dq_{i}" for i in range(14)]
    + [f"tgt_{i}" for i in range(14)] + [f"cur_{i}" for i in range(14)] + ["paused"]
)
assert len(FEATURE_NAMES) == FEAT_DIM


def paused_target_rel(t_paused: torch.Tensor, target_at_pause: torch.Tensor) -> torch.Tensor:
    """Target − HOME while paused. ``t_paused`` (N,) seconds since the pause,
    ``target_at_pause`` (N, 14) the last policy target − HOME."""
    a = (t_paused / HOLD_RAMP_S).clamp(0.0, 1.0).unsqueeze(-1)
    pose = torch.tensor(PAUSE_POSE_REL, device=target_at_pause.device, dtype=target_at_pause.dtype)
    return target_at_pause * (1.0 - a) + pose * a


class CurrentSensorDR:
    """Sim current → what the XL330 reports: per-env/joint gain + offset, noise,
    1 mA quantization, absolute value (robotd stores |current|)."""

    def __init__(self, num_envs, device, gain=(0.7, 1.3), offset_ma=(0.0, 25.0), noise_ma=10.0):
        self.gain = torch.empty(num_envs, NUM_JOINTS, device=device).uniform_(*gain)
        self.offset = torch.empty(num_envs, NUM_JOINTS, device=device).uniform_(*offset_ma) / 1000.0
        self.noise = noise_ma / 1000.0

    def resample(self, env_ids, gain=(0.7, 1.3), offset_ma=(0.0, 25.0)):
        n = len(env_ids)
        dev = self.gain.device
        self.gain[env_ids] = torch.empty(n, NUM_JOINTS, device=dev).uniform_(*gain)
        self.offset[env_ids] = torch.empty(n, NUM_JOINTS, device=dev).uniform_(*offset_ma) / 1000.0

    def __call__(self, current_a: torch.Tensor) -> torch.Tensor:
        c = current_a * self.gain + torch.randn_like(current_a) * self.noise
        c = c.abs() + self.offset
        return torch.round(c * 1000.0) / 1000.0
