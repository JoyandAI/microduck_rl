"""Batched virtual hand: picks the robot up by the trunk, carries/tilts/shakes
it, then sets it down (lowers until the feet touch, lets go after a short
delay) or drops it.

Pure torch on (N,) state so the same code drives the MuJoCo-Warp data generator
and the single-env CPU demo. The hand is a mocap body welded to the trunk by a SOFT
equality constraint (``add_hand_to_spec``), solved implicitly by MuJoCo every
physics substep — a compliant grip, like a person. (An explicit PD wrench
updated at 50 Hz is unstable: through the soft servos the trunk's effective
inertia at high frequency is tiny, so any useful rotational damping overshoots.)
``VirtualHand.step`` only produces the hand's target pose + active mask.

Derived from the ``virtual_hand_pickup`` step event of the ``handled_robot``
branch, widened to how people actually handle the duck: any tilt (up to
upside-down), shaking, tremor, quick short lifts, and gentle set-downs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

IDLE, CARRY, LOWER = 0, 1, 2


@dataclass
class HandCfg:
    pickup_rate_hz: float = 0.08        # idle envs start a pick-up at this rate
    min_idle_s: float = 1.0             # after a release / episode start
    lift_z: tuple = (0.03, 0.35)        # above the trunk height at pick-up
    lift_s: tuple = (0.25, 3.0)          # v2: slow lifts too (a head grab is a gentle pull)
    hold_s: tuple = (0.5, 8.0)
    # tilt amplitude buckets (deg, probability): mostly casual, sometimes wild
    tilt_buckets: tuple = ((15.0, 0.35), (45.0, 0.30), (90.0, 0.25), (160.0, 0.10))
    tilt_freq_hz: tuple = (0.05, 0.6)
    yaw_rate: tuple = (-1.0, 1.0)
    drift_amp: tuple = (0.0, 0.25)      # m, smooth carry motion
    drift_freq_hz: tuple = (0.05, 0.5)
    shake_prob: float = 0.35
    shake_amp: tuple = (0.005, 0.03)    # m
    shake_freq_hz: tuple = (1.5, 5.0)
    tremor_amp: tuple = (0.0003, 0.002)  # m, physiological tremor 8-12 Hz
    tremor_rot_deg: tuple = (0.1, 0.8)
    drop_prob: float = 0.10             # let go at height instead of setting down
    lower_speed: tuple = (0.04, 0.4)    # m/s descent until touchdown
    putdown_tilt_deg: float = 8.0       # residual tilt at set-down
    touch_release_s: tuple = (0.05, 0.6)  # hand lets go this long after the feet touch
    touch_force_frac: float = 0.25      # feet carry this fraction of weight → "touched"
    max_lower_s: float = 6.0
    pickup_fallen: bool = True          # people pick fallen robots up too
    touch_press: tuple = (0.0, 0.01)    # m the hand keeps pushing down after touchdown
    # v2 (robot feedback 2026-10-03: lifted by the HEAD it never paused, turned 180° it resumed)
    head_grip_prob: float = 0.35        # grab the head (jaw_soft) instead of the trunk
    orient_prob: float = 0.35           # carry at a sustained orientation anywhere on the sphere
    orient_pitch: tuple = (-math.pi, math.pi)
    orient_roll: tuple = (-math.pi / 2, math.pi / 2)
    orient_ramp_s: tuple = (0.4, 2.0)   # how fast the hand turns the duck there
    yaw_turn_prob: float = 0.35         # a quick turn about the vertical, up to ±180°
    yaw_turn: tuple = (-math.pi, math.pi)
    yaw_turn_s: tuple = (0.4, 1.5)


def _u(n, lo_hi, device):
    lo, hi = lo_hi
    return torch.empty(n, device=device).uniform_(lo, hi)


def _quat_from_euler_xyz(roll, pitch, yaw):
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    return torch.stack([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dim=-1)


def _quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _quat_conj(q):
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _axis_angle(q):
    """Rotation vector of a unit quaternion (shortest path)."""
    q = torch.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    s = v.norm(dim=-1, keepdim=True)
    ang = 2.0 * torch.atan2(s, q[..., :1])
    return torch.where(s > 1e-8, v / s.clamp_min(1e-8) * ang, 2.0 * v)


def yaw_of(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class VirtualHand:
    """State machine IDLE → CARRY → (LOWER → touch → release | drop) → IDLE."""

    # smooth-noise channels: roll, pitch, x, y, z ; two sinusoids each
    _NCH = 5

    def __init__(self, num_envs: int, device, cfg: HandCfg | None = None):
        self.cfg = cfg or HandCfg()
        self.N, self.dev = num_envs, device
        z = lambda *s: torch.zeros(num_envs, *s, device=device)
        self.phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.t = z(); self.idle_t = z(); self.t_hold = z(); self.t_lift = z()
        self.anchor = z(3)            # carry anchor (lift target, no noise)
        self.z0 = z(); self.yaw = z(); self.yaw_rate = z()
        self.tilt_amp = z(); self.drift_amp = z()
        self.sin_a = z(self._NCH, 2); self.sin_f = z(self._NCH, 2); self.sin_p = z(self._NCH, 2)
        self.shake_amp = z(); self.shake_f = z(); self.shake_dir = z(3)
        self.trem_amp = z(); self.trem_rot = z(); self.trem_f = z(); self.trem_dir = z(6); self.trem_p = z()
        self.drop = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.lower_v = z(); self.touch_t = z(); self.touch_release = z(); self.touched = torch.zeros_like(self.drop)
        self.end_tilt = z(2); self.lower_from_tilt = z(2); self.lower_t0 = z()
        self.press = z(); self.z_touch = z()
        self.cur_tilt = z(2)
        self.forced = torch.zeros_like(self.drop)
        self.grip = torch.zeros(num_envs, dtype=torch.long, device=device)  # 0 trunk, 1 head
        self.base_q = torch.zeros(num_envs, 4, device=device); self.base_q[:, 0] = 1.0  # yaw-free frame at grasp
        self.orient = z(2); self.orient_s = z()
        self.yaw0 = z(); self.yaw_turn = z(); self.yaw_turn_t0 = z(); self.yaw_turn_s = z()

    # ------------------------------------------------------------------ api
    @property
    def active(self) -> torch.Tensor:
        return self.phase > IDLE

    def reset(self, env_ids):
        self.phase[env_ids] = IDLE
        self.t[env_ids] = 0.0
        self.idle_t[env_ids] = 0.0

    def select(self, pos, quat, pos_head=None, quat_head=None):
        """The pose of the body each env is gripped by (trunk, or head where ``grip`` is 1)."""
        if pos_head is None:
            return pos, quat
        head = (self.grip == 1).unsqueeze(-1)
        return torch.where(head, pos_head, pos), torch.where(head, quat_head, quat)

    def start(self, mask: torch.Tensor, pos: torch.Tensor, quat: torch.Tensor, pos_head=None, quat_head=None):
        """Begin a pick-up on ``mask`` envs (also used to force one in demos).
        ``pos/quat`` are the trunk's; ``pos_head/quat_head`` the head's, needed for head grips."""
        mask = mask & (self.phase == IDLE)
        n = int(mask.sum())
        if n == 0:
            return
        c, d = self.cfg, self.dev
        can_head = pos_head is not None
        self.grip[mask] = ((torch.rand(n, device=d) < c.head_grip_prob) & can_head).long()
        pos, quat = self.select(pos, quat, pos_head, quat_head)
        # yaw-free orientation of the gripped body at the grasp: the hand's tilts are relative to
        # it, so a head grip starts from the head's own pose instead of snapping it level. A trunk
        # grip stays absolute (identity) — picking a fallen robot up sets it back on its feet.
        yq = _quat_from_euler_xyz(torch.zeros(n, device=d), torch.zeros(n, device=d), -yaw_of(quat[mask]))
        bq = _quat_mul(yq, quat[mask])
        ident = torch.zeros_like(bq); ident[:, 0] = 1.0
        self.base_q[mask] = torch.where((self.grip[mask] == 1).unsqueeze(-1), bq, ident)
        orient = torch.rand(n, device=d) < c.orient_prob
        o = torch.stack([_u(n, c.orient_roll, d), _u(n, c.orient_pitch, d)], -1)
        self.orient[mask] = o * orient.unsqueeze(-1).float()
        self.orient_s[mask] = _u(n, c.orient_ramp_s, d)
        turn = torch.rand(n, device=d) < c.yaw_turn_prob
        self.yaw_turn[mask] = _u(n, c.yaw_turn, d) * turn.float()
        self.yaw_turn_s[mask] = _u(n, c.yaw_turn_s, d)
        self.yaw_turn_t0[mask] = _u(n, (0.5, 4.0), d)
        self.phase[mask] = CARRY
        self.t[mask] = 0.0
        self.t_hold[mask] = _u(n, c.hold_s, d)
        self.t_lift[mask] = _u(n, c.lift_s, d)
        self.z0[mask] = pos[mask, 2]
        anc = pos[mask].clone()
        anc[:, 2] = pos[mask, 2] + _u(n, c.lift_z, d)
        self.anchor[mask] = anc
        self.yaw[mask] = yaw_of(quat[mask]); self.yaw0[mask] = self.yaw[mask]
        self.yaw_rate[mask] = _u(n, c.yaw_rate, d) * (torch.rand(n, device=d) < 0.6).float()
        # tilt bucket
        amps = torch.tensor([b[0] for b in c.tilt_buckets], device=d)
        probs = torch.tensor([b[1] for b in c.tilt_buckets], device=d)
        k = torch.multinomial(probs, n, replacement=True)
        self.tilt_amp[mask] = torch.deg2rad(amps[k]) * torch.rand(n, device=d).sqrt()
        self.drift_amp[mask] = _u(n, c.drift_amp, d)
        a = torch.rand(n, self._NCH, 2, device=d)
        self.sin_a[mask] = a / a.sum(-1, keepdim=True)
        f = torch.empty(n, self._NCH, 2, device=d)
        f[:, :2].uniform_(*c.tilt_freq_hz); f[:, 2:].uniform_(*c.drift_freq_hz)
        self.sin_f[mask] = f
        self.sin_p[mask] = torch.rand(n, self._NCH, 2, device=d) * 2 * math.pi
        shake = torch.rand(n, device=d) < c.shake_prob
        self.shake_amp[mask] = _u(n, c.shake_amp, d) * shake.float()
        self.shake_f[mask] = _u(n, c.shake_freq_hz, d)
        sd = torch.randn(n, 3, device=d); self.shake_dir[mask] = sd / sd.norm(dim=1, keepdim=True)
        self.trem_amp[mask] = _u(n, c.tremor_amp, d)
        self.trem_rot[mask] = torch.deg2rad(_u(n, c.tremor_rot_deg, d))
        self.trem_f[mask] = _u(n, (8.0, 12.0), d)
        td = torch.randn(n, 6, device=d); self.trem_dir[mask] = td / td.norm(dim=1, keepdim=True)
        self.trem_p[mask] = torch.rand(n, device=d) * 2 * math.pi
        self.drop[mask] = torch.rand(n, device=d) < c.drop_prob
        self.lower_v[mask] = _u(n, c.lower_speed, d)
        self.touch_release[mask] = _u(n, c.touch_release_s, d)
        self.touched[mask] = False
        self.end_tilt[mask] = (torch.rand(n, 2, device=d) * 2 - 1) * math.radians(c.putdown_tilt_deg)
        self.press[mask] = _u(n, c.touch_press, d)

    def release(self, mask: torch.Tensor):
        """Force the end of the carry: set down (LOWER) — demo control."""
        m = mask & (self.phase == CARRY)
        self.t_hold[m] = self.t[m]

    def step(self, dt, pos, quat, feet_force, mass, upright, pos_head=None, quat_head=None):
        """Advance the hand one control step. Returns (target_pos (N,3),
        target_quat (N,4) wxyz, active (N,) bool) for the welded mocap hand —
        the target pose of the GRIPPED body (``grip``: 0 trunk, 1 head).
        ``feet_force`` = total normal force on the feet (N), ``mass`` = robot
        mass per env, ``upright`` = mask used to gate pick-ups of fallen robots."""
        c, d, N = self.cfg, self.dev, self.N

        # ── start pick-ups ──
        idle = self.phase == IDLE
        self.idle_t = torch.where(idle, self.idle_t + dt, torch.zeros_like(self.idle_t))
        elig = idle & (self.idle_t >= c.min_idle_s) & (upright | c.pickup_fallen)
        self.start(elig & (torch.rand(N, device=d) < c.pickup_rate_hz * dt), pos, quat, pos_head, quat_head)
        pos, quat = self.select(pos, quat, pos_head, quat_head)

        self.t = self.t + dt
        carry = self.phase == CARRY
        turn_a = ((self.t - self.yaw_turn_t0) / self.yaw_turn_s).clamp(0, 1)
        turn_a = 0.5 - 0.5 * torch.cos(math.pi * turn_a)
        self.yaw = torch.where(carry, self.yaw0 + self.yaw_rate * self.t + self.yaw_turn * turn_a, self.yaw)

        # ── carry → lower / drop ──
        done = carry & (self.t >= self.t_hold)
        if done.any():
            drop = done & self.drop
            self.phase[drop] = IDLE
            put = done & ~self.drop
            self.phase[put] = LOWER
            self.lower_t0[put] = self.t[put]
            self.lower_from_tilt[put] = self.cur_tilt[put]
            self.touched[put] = False
            self.touch_t[put] = 0.0
        lower = self.phase == LOWER
        # touchdown detection (the hand feels the floor) → release after a delay
        touch_now = lower & ~self.touched & (feet_force > c.touch_force_frac * mass * 9.81)
        self.z_touch = torch.where(touch_now, pos[:, 2], self.z_touch)
        self.touched = self.touched | touch_now
        self.touch_t = torch.where(lower & self.touched, self.touch_t + dt, self.touch_t)
        lt = self.t - self.lower_t0
        rel = lower & ((self.touched & (self.touch_t >= self.touch_release)) | (lt > c.max_lower_s))
        self.phase[rel] = IDLE
        lower = self.phase == LOWER
        carry = self.phase == CARRY
        active = carry | lower

        # ── target pose ──
        t = self.t.unsqueeze(-1).unsqueeze(-1)
        sn = (self.sin_a * torch.sin(2 * math.pi * self.sin_f * t + self.sin_p)).sum(-1)  # (N, 5)
        lift_a = (self.t / self.t_lift).clamp(0, 1)
        lift_a = 0.5 - 0.5 * torch.cos(math.pi * lift_a)  # smooth
        ramp = lift_a.unsqueeze(-1)
        orient_a = ((self.t - self.t_lift) / self.orient_s).clamp(0, 1)
        orient_a = (0.5 - 0.5 * torch.cos(math.pi * orient_a)).unsqueeze(-1)
        tilt = self.tilt_amp.unsqueeze(-1) * sn[:, :2] * ramp + self.orient * orient_a
        tgt = self.anchor.clone()
        tgt[:, :2] = tgt[:, :2] + self.drift_amp.unsqueeze(-1) * sn[:, 2:4] * ramp
        tgt[:, 2] = self.z0 + lift_a * (self.anchor[:, 2] - self.z0) + 0.3 * self.drift_amp * sn[:, 4] * lift_a
        # lowering: descend from the current carry pose at lower_v, tilt → end_tilt
        lo_a = (lt / 1.0).clamp(0, 1)
        tilt_lo = self.lower_from_tilt + (self.end_tilt - self.lower_from_tilt) * lo_a.unsqueeze(-1)
        z_lo = self.anchor[:, 2] - self.lower_v * lt
        z_lo = torch.where(self.touched, torch.maximum(z_lo, self.z_touch - self.press), z_lo)
        tilt = torch.where(lower.unsqueeze(-1), tilt_lo, tilt)
        tgt[:, 2] = torch.where(lower, z_lo, tgt[:, 2])
        self.cur_tilt = torch.where(carry.unsqueeze(-1), tilt, self.cur_tilt)
        # shake + tremor
        ph = 2 * math.pi * self.shake_f * self.t
        tgt = tgt + (self.shake_amp * torch.sin(ph)).unsqueeze(-1) * self.shake_dir * carry.float().unsqueeze(-1)
        tph = torch.sin(2 * math.pi * self.trem_f * self.t + self.trem_p)
        tgt = tgt + (self.trem_amp * tph).unsqueeze(-1) * self.trem_dir[:, :3]
        rot_trem = (self.trem_rot * tph).unsqueeze(-1) * self.trem_dir[:, 3:]
        q_tgt = _quat_from_euler_xyz(tilt[:, 0] + rot_trem[:, 0], tilt[:, 1] + rot_trem[:, 1], self.yaw + rot_trem[:, 2])
        q_tgt = _quat_mul(q_tgt, self.base_q)
        if lower.any():  # keep the robot's own yaw during set-down
            self.yaw = torch.where(lower, yaw_of(quat), self.yaw)

        self.tgt_pos, self.tgt_quat = tgt, q_tgt
        return tgt, q_tgt, active


def add_hand_to_spec(spec, trunk_body: str = "robot/trunk_base", timeconst: float = 0.03, dampratio: float = 1.0,
                     head_body: str | None = None):
    """Add the mocap hand + (inactive) soft welds hand↔trunk ("pickup_hand_weld") and, if
    ``head_body`` is given (default: the trunk's prefix + "jaw_soft"), hand↔head
    ("pickup_hand_weld_head"). Relative pose identity: the hand pose IS the gripped body's target."""
    import mujoco
    hand = spec.worldbody.add_body(name="pickup_hand", mocap=True)
    hand.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.012, 0, 0], contype=0, conaffinity=0,
                  rgba=[1.0, 0.6, 0.2, 0.0], group=5)
    if head_body is None:
        head_body = trunk_body.rsplit("trunk_base", 1)[0] + "jaw_soft"
    for name, body in (("pickup_hand_weld", trunk_body), ("pickup_hand_weld_head", head_body)):
        eq = spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                               name1="pickup_hand", name2=body, active=False)
        eq.name = name
        eq.solref = [timeconst, dampratio]
        eq.data[:] = 0.0
        eq.data[6] = 1.0   # relpose quat (w) = identity
        eq.data[10] = 1.0  # torquescale
    return spec
