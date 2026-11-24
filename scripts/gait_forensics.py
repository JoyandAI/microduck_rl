#!/usr/bin/env python3
"""Gait/posture forensics for a walking policy, in the FAITHFUL BAM training env.

Born from the 2026-09-11 question: "the walk looks weird — the legs splay apart
and the head tilts/wobbles way too much; how do we get the official look?"

`eval_onnx_bam.py` only answers "does it stay up / track velocity".  This script
answers "does it LOOK right", by measuring the two complaints directly and
against the geometry that causes them:

  SPLAY  — hip_roll/hip_yaw deviations from HOME (the stance is defined by HOME:
           left_hip_roll=-0.0873, right=+0.0873 = a 5 deg knees-in stance), plus
           the actual lateral distance between the two foot sites (world y and
           body-frame y).  A "splayed" gait shows up as hip_roll moving TOWARD
           zero/positive (legs swinging out) and a foot separation that grows.
  HEAD   — the head body's absolute roll/pitch in the world, split into
           DC bias (1 s EMA, the escapable gravity droop) and oscillation
           (residual, the part a 37%-of-body-mass head must produce while
           stepping).  Also head pitch RELATIVE to the trunk, which isolates
           neck control from whole-body pitching.

Everything is recorded to CSV so downstream analysis does not have to re-run
the sim.

Command handling: both the twist and the head_pose command are frozen (twist to
the requested values; head_pose to a fixed delta, default all-zero = "hold the
HOME head pose", which is what deployment feeds when no pose is requested).
Pass --head-cmd train to instead let head_pose resample like training does.

Usage:
    uv run python scripts/gait_forensics.py --onnx walk_hd1910_15000.onnx \
        --lin-vel-x 0.2 --seconds 20
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

CTRL_DT = 0.02  # 50 Hz

# HOME joint targets (microduck_constants.HOME_FRAME) — the "official" stance the
# posture reward pulls toward. hip_roll here is NOT zero: it is a deliberate
# 5 deg inward tilt so the sole sits flat.
HOME = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": -0.0873,
    "left_hip_pitch": -0.4579,
    "left_knee": 0.0,
    "left_ankle": 0.4530,
    "neck_pitch": 0.3491,
    "head_pitch": 0.3491,
    "head_yaw": 0.0,
    "head_roll": 0.0,
    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.0873,
    "right_hip_pitch": 0.4579,
    "right_knee": 0.0,
    "right_ankle": -0.4530,
}


def quat_to_rpy(q) -> tuple[float, float, float]:
    """(w,x,y,z) -> roll, pitch, yaw (rad)."""
    w, x, y, z = (float(v) for v in q)
    # roll (x-axis)
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    # pitch (y-axis), clamped
    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    # yaw (z-axis)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw


def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (axis*angle, radians). Small-angle safe."""
    c = (np.trace(R) - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    ang = math.acos(c)
    if ang < 1e-8:
        return np.array([0.0, 0.0, 0.0])
    if abs(math.pi - ang) < 1e-6:
        # near 180 deg: use the symmetric part
        A = (R + np.eye(3)) / 2.0
        ax = np.sqrt(np.maximum(np.diag(A), 0.0))
        n = np.linalg.norm(ax)
        return np.zeros(3) if n < 1e-9 else ax / n * ang
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v / (2.0 * math.sin(ang)) * ang


def dominant_freq(x: np.ndarray, dt: float) -> tuple[float, float]:
    """Dominant non-DC frequency and its amplitude (same units as x)."""
    x = np.asarray(x, dtype=float)
    v = x - x.mean()
    spec = np.abs(np.fft.rfft(v))
    f = np.fft.rfftfreq(len(v), dt)
    if len(spec) < 3:
        return 0.0, 0.0
    k = int(np.argmax(spec[1:]) + 1)
    return float(f[k]), float(spec[k] * 2.0 / len(v))


def zyx(R: np.ndarray) -> tuple[float, float, float]:
    """ZYX (yaw-pitch-roll) Euler angles of R. Returns (roll, pitch, yaw).

    Used instead of a raw rotation vector for the head: the trunk yaws a lot
    while walking (the policy drifts), and a rotation vector mixes that yaw
    into its other components. Factoring the yaw out first leaves pitch and
    roll that describe how far the head is tipped in the world, which is what
    "the head wobbles" means visually.
    """
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    if abs(math.cos(pitch)) < 1e-6:
        roll = math.atan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    else:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def ema(x: np.ndarray, tau_s: float, dt: float) -> np.ndarray:
    a = math.exp(-dt / tau_s)
    out = np.empty_like(x, dtype=float)
    acc = float(x[0])
    for i, v in enumerate(x):
        acc = a * acc + (1.0 - a) * float(v)
        out[i] = acc
    return out


def _unwrap_delta(x) -> float:
    """Total heading change (rad) accumulated step by step, wrap-safe.

    An Euler yaw in (-pi, pi] cannot be differenced end-to-end: every crossing of
    +-180 folds a full turn into the result. Summing wrapped per-step differences
    gives the true accumulated rotation.
    """
    import numpy as _np
    v = _np.asarray(x, dtype=float)
    if v.size < 2:
        return 0.0
    d = _np.diff(v)
    return float(_np.sum((d + _np.pi) % (2.0 * _np.pi) - _np.pi))


def amp_p2p(x: np.ndarray) -> float:
    return float(np.percentile(x, 97.5) - np.percentile(x, 2.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--task", default="Mjlab-Velocity-Flat-MicroDuck")
    ap.add_argument("--lin-vel-x", type=float, default=0.2)
    ap.add_argument("--lin-vel-y", type=float, default=0.0)
    ap.add_argument("--ang-vel-z", type=float, default=0.0)
    ap.add_argument("--warmup", type=float, default=2.0)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--params-json", default=None,
                    help="explicit BAM params JSON (the snapshot the policy trained with)")
    ap.add_argument("--head-cmd", default="zero", choices=("zero", "train"),
                    help="zero = freeze head_pose at all-zero delta (deployment idle); "
                         "train = resample like training")
    ap.add_argument("--csv", default=None, help="write raw per-step samples here")
    ap.add_argument("--json-out", default=None, help="write summary metrics here")
    ap.add_argument("--tag", default="", help="label printed in the report")
    ap.add_argument("--freeze-actions", default=None,
                    help="comma-separated joint=action pairs, e.g. 'neck_pitch=0,head_pitch=0'. "
                         "Overrides the policy's raw action for those joints with a constant "
                         "(action 0 == command the joint to its HOME target). Used to ablate "
                         "whether a motion the policy produces is load-bearing or gratuitous.")
    args = ap.parse_args()

    freeze: dict[str, float] = {}
    if args.freeze_actions:
        for part in args.freeze_actions.split(","):
            k, _, v = part.partition("=")
            freeze[k.strip()] = float(v) if v else 0.0

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg

    import onnxruntime as ort

    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = 1
    removed = cfg.events.pop("push_robot", None)
    print(f"[i] push_robot removed: {removed is not None}")

    # Freeze twist.
    tcfg = cfg.commands["twist"]
    tcfg.resampling_time_range = (1e9, 1e9)
    for attr, val in (("rel_standing_envs", 0.0), ("rel_heading_envs", 0.0)):
        if hasattr(tcfg, attr):
            setattr(tcfg, attr, val)
    if getattr(tcfg, "heading_command", False):
        tcfg.heading_command = False
        if hasattr(tcfg, "ranges") and hasattr(tcfg.ranges, "heading"):
            tcfg.ranges.heading = None

    # Freeze (or not) the head_pose command.
    hcfg = cfg.commands["head_pose"]
    if args.head_cmd == "zero":
        hcfg.resampling_time_range = (1e9, 1e9)
        # The initial value is zero (UniformPoseCommand._command starts at 0 and
        # only resamples on the first step); pushing the first resample past the
        # horizon keeps it at exactly zero = "hold HOME head pose".
    hcfg.zero_command_prob = 0.0

    if args.params_json:
        act = cfg.scene.entities["robot"].articulation.actuators[0]
        object.__setattr__(act, "json_path", args.params_json)
        object.__setattr__(act, "motor_name", None)
        object.__setattr__(act, "model", None)
        object.__setattr__(act, "_resolved_json_path", args.params_json)
        print(f"[i] BAM params overridden -> {args.params_json}")
    else:
        act = cfg.scene.entities["robot"].articulation.actuators[0]
        print(f"[i] BAM params (default) -> {getattr(act, '_resolved_json_path', None)}")

    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    env = RslRlVecEnvWrapper(env, clip_actions=None)
    unwrapped = env.unwrapped
    robot = unwrapped.scene.entities["robot"]

    sess = ort.InferenceSession(args.onnx)
    iname, oname = sess.get_inputs()[0].name, sess.get_outputs()[0].name
    assert sess.get_inputs()[0].shape[1] == 61, sess.get_inputs()[0].shape

    jn = list(robot.joint_names)
    jidx = {n: i for i, n in enumerate(jn)}
    missing = [k for k in HOME if k not in jidx]
    if missing:
        raise SystemExit(f"joints missing from model: {missing}")

    site_ids = {}
    for s in ("left_foot", "right_foot", "head_camera", "mouth_tip"):
        ids, names = robot.find_sites(s)
        if len(ids):
            site_ids[s] = int(ids[0])
    # head body: the head-roll body (bottom_head_shell / jaw_soft)
    body_names = list(robot.body_names)
    head_body, trunk_body = None, None
    for cand in ("bottom_head_shell", "jaw_soft"):
        if cand in body_names:
            head_body = body_names.index(cand)
    for cand in ("trunk_base", "trunk"):
        if cand in body_names:
            trunk_body = body_names.index(cand)
    print(f"[i] joints={len(jn)} sites={list(site_ids)} head_body={head_body} trunk_body={trunk_body}")

    term = unwrapped.command_manager.get_term("twist")
    hterm = unwrapped.command_manager.get_term("head_pose")

    obs_td = env.reset()[0]
    n_warm = int(round(args.warmup / CTRL_DT))
    n = int(round(args.seconds / CTRL_DT))

    # ---- Nominal reference frame ------------------------------------------
    # Right after reset the robot sits at HOME (the `init_state`), so the body
    # quaternions ARE the nominal reference. This is essential for the HEAD:
    # the head body's nominal frame is a pure -90 deg pitch, which is the Euler
    # gimbal-lock singularity — extracting roll/pitch/yaw from the head
    # quaternion directly returns garbage (roll and yaw wrap through +-180 deg).
    # Measure the head as the rotation vector of R_head @ R_head_home.T instead.
    R_head_home = quat_to_mat(robot.data.body_link_quat_w[0, head_body].cpu().numpy())
    R_trunk_home = quat_to_mat(robot.data.body_link_quat_w[0, trunk_body].cpu().numpy())
    R_rel_home = R_trunk_home.T @ R_head_home
    _bl_nom = R_trunk_home.T @ (
        robot.data.site_pos_w[0, site_ids["left_foot"]].cpu().numpy()
        - robot.data.body_link_pos_w[0, trunk_body].cpu().numpy()
    )
    _br_nom = R_trunk_home.T @ (
        robot.data.site_pos_w[0, site_ids["right_foot"]].cpu().numpy()
        - robot.data.body_link_pos_w[0, trunk_body].cpu().numpy()
    )
    nominal_sep = float(_bl_nom[1] - _br_nom[1])
    print(f"[i] nominal (HOME) foot separation in trunk frame: {nominal_sep*1000:.1f} mm")

    reset_kinds = Counter()
    rows = []
    for i in range(n_warm + n):
        term.command[:, 0] = args.lin_vel_x
        term.command[:, 1] = args.lin_vel_y
        term.command[:, 2] = args.ang_vel_z
        if args.head_cmd == "zero":
            hterm.command[:] = 0.0

        obs = obs_td["actor"].cpu().numpy()
        action = sess.run([oname], {iname: obs})[0].astype(np.float32)
        if freeze:
            for jn_f, val in freeze.items():
                action[0, jidx[jn_f]] = val
        obs_td, _, dones, _ = env.step(torch.from_numpy(action).to(args.device))
        if bool(dones[0]):
            reset_kinds["reset"] += 1

        if i < n_warm:
            continue

        q = robot.data.joint_pos[0].cpu().numpy().astype(float)
        qd = robot.data.joint_vel[0].cpu().numpy().astype(float)
        a = np.asarray(action[0], dtype=float)

        p_trunk = robot.data.body_link_pos_w[0, trunk_body].cpu().numpy().astype(float)
        qt_trunk = robot.data.body_link_quat_w[0, trunk_body].cpu().numpy().astype(float)
        p_head = robot.data.body_link_pos_w[0, head_body].cpu().numpy().astype(float)
        qt_head = robot.data.body_link_quat_w[0, head_body].cpu().numpy().astype(float)

        r_trunk, p_trunk_r, y_trunk = quat_to_rpy(qt_trunk)
        r_head, p_head_r, y_head = quat_to_rpy(qt_head)

        # Head deviation from its nominal attitude, as world-frame rotation
        # vector components: x = pitch (nod), y = roll (side tilt), z = yaw.
        R_head = quat_to_mat(qt_head)
        R_trunk = quat_to_mat(qt_trunk)
        dv_head = rotvec(R_head @ R_head_home.T)
        # ZYX decomposition of the same deviation: yaw is factored out first,
        # so pitch/roll stay clean even while the robot's heading drifts.
        dR_head = R_head @ R_head_home.T
        zr, zp, zy = zyx(dR_head)
        R_rel_now = R_trunk.T @ R_head
        dR_rel = R_rel_now @ R_rel_home.T
        # rotvec components are rotations ABOUT x, y, z. In the trunk frame
        # (x=forward, y=left, z=up at HOME): about-x = ROLL (side tilt),
        # about-y = PITCH (nod), about-z = YAW.
        # NOTE the SIGN: head_rel_trunk_pitch is anti-correlated (~-0.87) with
        # the raw neck_pitch/head_pitch joint deltas, because HOME already holds
        # the neck +20 deg back and the head +20 deg down; a positive joint delta
        # therefore reduces this world-relative nod. Read the magnitude, not the
        # sign, when comparing against joint-level tables.
        rr = rotvec(dR_rel)
        zrr, zrp, _ = zyx(dR_rel)

        row = {
            "t": (i - n_warm) * CTRL_DT,
            "trunk_z": float(p_trunk[2]),
            "trunk_roll": r_trunk,
            "trunk_pitch": p_trunk_r,
            "trunk_yaw": y_trunk,
            "trunk_tilt": math.degrees(math.acos(max(-1.0, min(1.0,
                            1.0 - 2.0 * (qt_trunk[1] ** 2 + qt_trunk[2] ** 2))))),
            "head_z": float(p_head[2]),
            # Euler on the head body is UNRELIABLE (nominal frame sits at the
            # singularity) — kept only for debugging, never for the verdict.
            "head_roll_euler": r_head,
            "head_pitch_euler": p_head_r,
            "head_dev_pitch": float(dv_head[0]),
            "head_dev_roll": float(dv_head[1]),
            "head_dev_yaw": float(dv_head[2]),
            "head_dev_angle": float(np.linalg.norm(dv_head)),
            # yaw-factored world-frame tilt of the head (the visually relevant one)
            "head_pitch_w": float(zp),
            "head_roll_w": float(zr),
            "head_yaw_w": float(zy),
            "head_rel_trunk_roll": float(rr[0]),   # about trunk x (side tilt)
            "head_rel_trunk_pitch": float(rr[1]),  # about trunk y (nod)
            "head_rel_trunk_yaw": float(rr[2]),
            "head_rel_trunk_angle": float(np.linalg.norm(rr)),
            "head_rel_trunk_pitch_z": float(zrp),
            "head_rel_trunk_roll_z": float(zrr),
            "quat_head_w": float(qt_head[0]), "quat_head_x": float(qt_head[1]),
            "quat_head_y": float(qt_head[2]), "quat_head_z": float(qt_head[3]),
            "quat_trunk_w": float(qt_trunk[0]), "quat_trunk_x": float(qt_trunk[1]),
            "quat_trunk_y": float(qt_trunk[2]), "quat_trunk_z": float(qt_trunk[3]),
        }
        for name, hi in HOME.items():
            row[f"q_{name}"] = float(q[jidx[name]])
            row[f"d_{name}"] = float(q[jidx[name]] - hi)
        for s, si in site_ids.items():
            sp = robot.data.site_pos_w[0, si].cpu().numpy().astype(float)
            row[f"site_{s}_x"] = float(sp[0])
            row[f"site_{s}_y"] = float(sp[1])
            row[f"site_{s}_z"] = float(sp[2])
        # foot separation in the TRUNK frame (removes yaw drift)
        if "left_foot" in site_ids and "right_foot" in site_ids:
            pl = robot.data.site_pos_w[0, site_ids["left_foot"]].cpu().numpy().astype(float)
            pr = robot.data.site_pos_w[0, site_ids["right_foot"]].cpu().numpy().astype(float)
            bl = R_trunk.T @ (pl - p_trunk)
            br = R_trunk.T @ (pr - p_trunk)
            row["foot_sep_body_y"] = float(bl[1] - br[1])
            row["foot_sep_x_body"] = float(bl[0] - br[0])
            row["foot_left_z_body"] = float(bl[2])
            row["foot_right_z_body"] = float(br[2])
            row["foot_lateral_gap"] = float(abs(bl[1]) + abs(br[1]))
        for k in range(len(jn)):
            row[f"a_{jn[k]}"] = float(a[k])
        rows.append(row)

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    R = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    nsteps = len(rows)
    dur = nsteps * CTRL_DT

    def s(k):
        return R[k]

    # ---------- GAIT ----------
    # step detection from left foot height (body frame)
    h = s("foot_left_z_body")
    thresh = (np.percentile(h, 5) + np.percentile(h, 95)) / 2
    above = h > thresh

    # ---------- HEAD ----------
    # Gaussian-equivalent amplitude: for x ~ N(0,s^2), E[exp(-(x/tol)^2)] =
    # 1/sqrt(1+2 s^2/tol^2); we report s (std) directly, plus the DC part.
    tau = 1.0
    head_pitch_ema = ema(s("head_pitch_w"), tau, CTRL_DT)
    head_roll_ema = ema(s("head_roll_w"), tau, CTRL_DT)
    head_pitch_osc = s("head_pitch_w") - head_pitch_ema
    head_roll_osc = s("head_roll_w") - head_roll_ema
    relp_ema = ema(s("head_rel_trunk_pitch"), tau, CTRL_DT)
    relr_ema = ema(s("head_rel_trunk_roll"), tau, CTRL_DT)
    neck_f, neck_a = dominant_freq(s("d_neck_pitch"), CTRL_DT)
    trunk_f, _ = dominant_freq(s("trunk_z"), CTRL_DT)
    foot_f, _ = dominant_freq(s("foot_left_z_body"), CTRL_DT)
    hiproll_f, _ = dominant_freq(s("d_left_hip_roll"), CTRL_DT)
    step_freq_gait = foot_f if foot_f > 0 else neck_f

    rep = {
        "tag": args.tag,
        "onnx": args.onnx,
        "task": args.task,
        "cmd": {"vx": args.lin_vel_x, "vy": args.lin_vel_y, "wz": args.ang_vel_z},
        "head_cmd": args.head_cmd,
        "params_json": args.params_json,
        "steps": nsteps,
        "duration_s": dur,
        "resets": int(reset_kinds["reset"]),
        "nominal_foot_sep_trunk_y_m": nominal_sep,
        "gait": {
            "gait_freq_hz": step_freq_gait,
            "neck_freq_hz": neck_f,
            "neck_amp_deg": math.degrees(neck_a),
            "trunk_bob_freq_hz": trunk_f,
            "hiproll_freq_hz": hiproll_f,
            "foot_clearance_body_z_m": float(np.percentile(h, 95) - np.percentile(h, 5)),
            "foot_sep_body_y_m": {
                "mean": float(np.mean(s("foot_sep_body_y"))),
                "min": float(np.min(s("foot_sep_body_y"))),
                "max": float(np.max(s("foot_sep_body_y"))),
                "p2p": amp_p2p(s("foot_sep_body_y")),
                "vs_nominal_mm": float(np.mean(s("foot_sep_body_y")) - nominal_sep) * 1000.0,
            },
            "duty_high_frac": float(np.mean(above)),
        },
        "trunk": {
            "z_mean": float(np.mean(s("trunk_z"))),
            "tilt_mean_deg": float(np.mean(s("trunk_tilt"))),
            "tilt_max_deg": float(np.max(s("trunk_tilt"))),
            "roll_mean_deg": math.degrees(float(np.mean(s("trunk_roll")))),
            "roll_std_deg": math.degrees(float(np.std(s("trunk_roll")))),
            "pitch_mean_deg": math.degrees(float(np.mean(s("trunk_pitch")))),
            "pitch_std_deg": math.degrees(float(np.std(s("trunk_pitch")))),
            "pitch_p2p_deg": math.degrees(amp_p2p(s("trunk_pitch"))),
            # ⚠️ UNWRAP before differencing. `trunk_yaw` is an Euler angle in
            # (-pi, pi], so `last - first` folds every +-180 crossing into a fake
            # +-360 jump: on 16 identical zero-command trajectories this field read
            # a median of -2.5 deg but a RANGE of -300..+3, while the unwrapped sum
            # of per-step differences read -44..+60 deg on the same data.
            # docs/walk_sitstand_training_queue.md quoted this field as a defect
            # ("zero-command yaw drift median -82 deg, range -284..+26") — that
            # number was manufactured by the wrap, not by the policy.
            "yaw_drift_deg": math.degrees(_unwrap_delta(s("trunk_yaw"))),
        },
        "head": {
            "z_mean": float(np.mean(s("head_z"))),
            "dev_angle_mean_deg": math.degrees(float(np.mean(s("head_dev_angle")))),
            "dev_angle_p2p_deg": math.degrees(amp_p2p(s("head_dev_angle"))),
            # --- world-frame tilt of the head, yaw factored out ---
            "pitch_dc_deg": math.degrees(float(np.mean(head_pitch_ema))),
            "pitch_std_deg": math.degrees(float(np.std(s("head_pitch_w")))),
            "pitch_osc_std_deg": math.degrees(float(np.std(head_pitch_osc))),
            "pitch_p2p_deg": math.degrees(amp_p2p(s("head_pitch_w"))),
            "pitch_freq_hz": dominant_freq(s("head_pitch_w"), CTRL_DT)[0],
            "pitch_amp_deg": math.degrees(dominant_freq(s("head_pitch_w"), CTRL_DT)[1]),
            "roll_dc_deg": math.degrees(float(np.mean(head_roll_ema))),
            "roll_std_deg": math.degrees(float(np.std(s("head_roll_w")))),
            "roll_osc_std_deg": math.degrees(float(np.std(head_roll_osc))),
            "roll_p2p_deg": math.degrees(amp_p2p(s("head_roll_w"))),
            "roll_freq_hz": dominant_freq(s("head_roll_w"), CTRL_DT)[0],
            "roll_amp_deg": math.degrees(dominant_freq(s("head_roll_w"), CTRL_DT)[1]),
            "yaw_std_deg": math.degrees(float(np.std(s("head_yaw_w")))),
            # --- head attitude RELATIVE TO THE TRUNK: isolates neck control ---
            "rel_trunk_pitch_dc_deg": math.degrees(float(np.mean(relp_ema))),
            "rel_trunk_pitch_std_deg": math.degrees(float(np.std(s("head_rel_trunk_pitch")))),
            "rel_trunk_pitch_p2p_deg": math.degrees(amp_p2p(s("head_rel_trunk_pitch"))),
            "rel_trunk_pitch_freq_hz": dominant_freq(s("head_rel_trunk_pitch"), CTRL_DT)[0],
            "rel_trunk_roll_dc_deg": math.degrees(float(np.mean(relr_ema))),
            "rel_trunk_roll_std_deg": math.degrees(float(np.std(s("head_rel_trunk_roll")))),
            "rel_trunk_roll_p2p_deg": math.degrees(amp_p2p(s("head_rel_trunk_roll"))),
            "rel_trunk_angle_std_deg": math.degrees(float(np.std(s("head_rel_trunk_angle")))),
            "rel_trunk_angle_p2p_deg": math.degrees(amp_p2p(s("head_rel_trunk_angle"))),
        },
        "splay": {},
        "joints_deg": {},
        "actions": {},
    }

    for name in HOME:
        d = s(f"d_{name}")
        qq = s(f"q_{name}")
        rep["joints_deg"][name] = {
            "mean_dev_from_home": math.degrees(float(np.mean(d))),
            "mean_abs_dev": math.degrees(float(np.mean(np.abs(d)))),
            "std": math.degrees(float(np.std(d))),
            "p2p": math.degrees(amp_p2p(d)),
            "min": math.degrees(float(np.min(qq))),
            "max": math.degrees(float(np.max(qq))),
            "home": math.degrees(HOME[name]),
        }

    # SPLAY summary: how far the legs are pushed OUT relative to HOME.
    # Positive "outward" for each side is defined as moving away from the HOME
    # inward tilt: left hip_roll increases toward 0/+, right hip_roll decreases.
    lr = s("d_left_hip_roll")
    rr = s("d_right_hip_roll")
    ly = s("d_left_hip_yaw")
    ry = s("d_right_hip_yaw")
    rep["splay"] = {
        "left_hip_roll_out_deg": {
            "mean": math.degrees(float(np.mean(lr))),
            "mean_abs": math.degrees(float(np.mean(np.abs(lr)))),
            "p2p": math.degrees(amp_p2p(lr)),
            "max": math.degrees(float(np.max(lr))),
        },
        "right_hip_roll_out_deg": {
            "mean": math.degrees(float(np.mean(rr))),
            "mean_abs": math.degrees(float(np.mean(np.abs(rr)))),
            "p2p": math.degrees(amp_p2p(rr)),
            "min": math.degrees(float(np.min(rr))),
        },
        # Symmetric abduction: both legs rotating outward together.
        "abduction_deg": {
            "mean": math.degrees(float(np.mean(0.5 * (lr - rr)))),
            "mean_abs": math.degrees(float(np.mean(np.abs(0.5 * (lr - rr))))),
            "p2p": math.degrees(amp_p2p(0.5 * (lr - rr))),
        },
        "toe_out_deg": {
            "left_mean": math.degrees(float(np.mean(ly))),
            "right_mean": math.degrees(float(np.mean(ry))),
            "left_abs": math.degrees(float(np.mean(np.abs(ly)))),
            "right_abs": math.degrees(float(np.mean(np.abs(ry)))),
        },
    }

    for k in jn:
        av = s(f"a_{k}")
        rep["actions"][k] = {
            "mean": float(np.mean(av)),
            "mean_abs": float(np.mean(np.abs(av))),
            "max_abs": float(np.max(np.abs(av))),
            "std": float(np.std(av)),
        }
    # Hard-limit occupancy: "parked on the stop" is the signature of a joint the
    # reward forgot to price (measured 48% of walking steps / 100% of standing
    # steps on hip_yaw in the 2026-09-10 run).
    lim = robot.data.joint_pos_limits[0].cpu().numpy()
    rep["limits"] = {}
    for k, name in enumerate(jn):
        q = s(f"q_{name}")
        lo, hi = float(lim[k, 0]), float(lim[k, 1])
        rep["limits"][name] = {
            "lo": lo, "hi": hi,
            "q_min": float(q.min()), "q_max": float(q.max()),
            "frac_within_1deg": float(np.mean((q < lo + 0.01745) | (q > hi - 0.01745))),
        }
    # action-rate (smoothness), in raw action units per step
    ar = []
    for k in jn:
        av = s(f"a_{k}")
        ar.append(np.mean(np.abs(np.diff(av))))
    rep["actions"]["mean_abs_rate"] = float(np.mean(ar))
    rep["actions"]["max_abs_rate"] = float(np.max(ar))

    # ---------- print ----------
    print(f"\n{'='*78}")
    print(f"GAIT FORENSICS  {args.tag or ''}")
    print(f"{'='*78}")
    print(f"  onnx      : {args.onnx}")
    print(f"  cmd       : vx={args.lin_vel_x:+.2f} vy={args.lin_vel_y:+.2f} wz={args.ang_vel_z:+.2f}   head_cmd={args.head_cmd}")
    print(f"  duration  : {dur:.2f}s ({nsteps} steps)   resets={rep['resets']}")
    print(f"  gait freq : {step_freq_gait:.2f} Hz (foot)  neck {neck_f:.2f} Hz  trunk-bob {trunk_f:.2f} Hz")
    print(f"  clearance : {rep['gait']['foot_clearance_body_z_m']*1000:.1f} mm")
    fs = rep["gait"]["foot_sep_body_y_m"]
    print(f"  foot sep y: mean={fs['mean']*1000:+.1f} min={fs['min']*1000:+.1f} max={fs['max']*1000:+.1f} mm  "
          f"(nominal {nominal_sep*1000:.1f} -> {fs['vs_nominal_mm']:+.1f} mm)")
    t = rep["trunk"]
    print(f"  trunk     : z={t['z_mean']*1000:.1f} mm  tilt={t['tilt_mean_deg']:.1f}deg (max {t['tilt_max_deg']:.1f})  "
          f"yaw drift={t['yaw_drift_deg']:+.1f}deg")
    print(f"              roll={t['roll_mean_deg']:+.1f}+-{t['roll_std_deg']:.1f}  pitch={t['pitch_mean_deg']:+.1f}+-{t['pitch_std_deg']:.1f} (p2p {t['pitch_p2p_deg']:.1f})")
    hd = rep["head"]
    print(f"  HEAD dev  : total dev mean={hd['dev_angle_mean_deg']:.1f}deg p2p={hd['dev_angle_p2p_deg']:.1f}deg")
    print(f"    world   : PITCH DC={hd['pitch_dc_deg']:+.1f} std={hd['pitch_std_deg']:.1f} amp={hd['pitch_amp_deg']:.1f} p2p={hd['pitch_p2p_deg']:.1f} @ {hd['pitch_freq_hz']:.2f}Hz")
    print(f"              ROLL  DC={hd['roll_dc_deg']:+.1f} std={hd['roll_std_deg']:.1f} amp={hd['roll_amp_deg']:.1f} p2p={hd['roll_p2p_deg']:.1f} @ {hd['roll_freq_hz']:.2f}Hz")
    print(f"    vs trunk: PITCH DC={hd['rel_trunk_pitch_dc_deg']:+.1f} std={hd['rel_trunk_pitch_std_deg']:.1f} p2p={hd['rel_trunk_pitch_p2p_deg']:.1f} @ {hd['rel_trunk_pitch_freq_hz']:.2f}Hz")
    print(f"              ROLL  DC={hd['rel_trunk_roll_dc_deg']:+.1f} std={hd['rel_trunk_roll_std_deg']:.1f} p2p={hd['rel_trunk_roll_p2p_deg']:.1f}")
    print(f"              TOTAL std={hd['rel_trunk_angle_std_deg']:.1f} p2p={hd['rel_trunk_angle_p2p_deg']:.1f}")
    sp = rep["splay"]
    print(f"  SPLAY     : L hip_roll dev={sp['left_hip_roll_out_deg']['mean']:+.1f}deg (|.|={sp['left_hip_roll_out_deg']['mean_abs']:.1f}, p2p {sp['left_hip_roll_out_deg']['p2p']:.1f})")
    print(f"              R hip_roll dev={sp['right_hip_roll_out_deg']['mean']:+.1f}deg (|.|={sp['right_hip_roll_out_deg']['mean_abs']:.1f}, p2p {sp['right_hip_roll_out_deg']['p2p']:.1f})")
    print(f"              abduction={sp['abduction_deg']['mean']:+.1f}deg |.|={sp['abduction_deg']['mean_abs']:.1f} p2p={sp['abduction_deg']['p2p']:.1f}")
    print(f"              toe-out L={sp['toe_out_deg']['left_mean']:+.1f} R={sp['toe_out_deg']['right_mean']:+.1f} deg (|.| L={sp['toe_out_deg']['left_abs']:.1f} R={sp['toe_out_deg']['right_abs']:.1f})")
    print(f"  action    : mean|a|={np.mean([rep['actions'][k]['mean_abs'] for k in jn]):.3f}  "
          f"mean rate={rep['actions']['mean_abs_rate']:.4f}  max rate={rep['actions']['max_abs_rate']:.4f}")
    yaws = [k for k in jn if "hip_yaw" in k]
    print(f"  limit park: " + "  ".join(
        f"{k.replace('_hip_yaw','')}={rep['limits'][k]['frac_within_1deg']*100:.1f}%" for k in yaws) +
        f"   |  max|a| over all joints = {max(rep['actions'][k]['max_abs'] for k in jn):.2f}")
    print("  per-joint deviation from HOME [deg] (mean_dev | mean|dev| std  p2p):")
    for name in HOME:
        j = rep["joints_deg"][name]
        print(f"    {name:<16} {j['mean_dev_from_home']:+7.2f} {j['mean_abs_dev']:7.2f} {j['std']:6.2f} {j['p2p']:6.2f}")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(rep, indent=2))
        print(f"\n[i] summary -> {args.json_out}")
    if args.csv:
        print(f"[i] raw     -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
