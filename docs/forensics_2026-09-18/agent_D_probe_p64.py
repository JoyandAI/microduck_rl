#!/usr/bin/env python3
"""P64 multi-env probe (vectorised): foot placement / clearance / duty / pitch.

Reproduces `scripts/eval_straightness.py`'s rollout loop EXACTLY (same cfg edits,
same seed, same command freeze, same override derivation) and adds channels that
script does not export. All device->host copies are batched; recording is
column-major (one (N,) array per step) so there is no per-env Python loop.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg

CTRL_DT = 0.02
LEG_JOINTS = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)
HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
ALL_JOINTS = LEG_JOINTS + HEAD_JOINTS
MODEL_XML = {
    "hd1910": ("file", "src/mjlab_microduck/robot/microduck/robot_walk.xml"),
    "preswap": ("file",
                "src/mjlab_microduck/robot/microduck/robot_walk.xml.bak_servoswap"),
}


def quats_to_mats(q):
    """(N,4) wxyz -> (N,3,3)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], axis=1)


def quats_to_rpy(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return roll, pitch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--task", default="Mjlab-Velocity-Flat-MicroDuck")
    ap.add_argument("--lin-vel-x", type=float, default=0.2)
    ap.add_argument("--warmup", type=float, default=2.0)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--override-model", choices=["hd1910", "preswap"], default=None)
    ap.add_argument("--override-actuator", choices=["hd1910", "xl330"], default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.events.pop("push_robot", None)
    tcfg = cfg.commands["twist"]
    tcfg.resampling_time_range = (1e9, 1e9)
    for a, v in (("rel_standing_envs", 0.0), ("rel_heading_envs", 0.0)):
        if hasattr(tcfg, a):
            setattr(tcfg, a, v)
    if getattr(tcfg, "heading_command", False):
        tcfg.heading_command = False
        if hasattr(tcfg.ranges, "heading"):
            tcfg.ranges.heading = None
    if hasattr(tcfg, "rel_turn_in_place_envs"):
        tcfg.rel_turn_in_place_envs = 0.0
    for cname in ("head_pose", "body_pose"):
        pc = cfg.commands.get(cname)
        if pc is None:
            continue
        pc.resampling_time_range = (1e9, 1e9)
        if hasattr(pc, "zero_command_prob"):
            pc.zero_command_prob = 1.0
        if hasattr(pc, "ranges") and isinstance(pc.ranges, (tuple, list)):
            pc.ranges = tuple((0.0, 0.0) for _ in pc.ranges)

    if args.override_model:
        import mujoco as _mj
        src = Path(MODEL_XML[args.override_model][1])
        if src.suffix == ".xml":
            xml = str(src.resolve())
        else:
            import tempfile
            d = Path(tempfile.mkdtemp(prefix="mjcf_override_"))
            (d / "robot_walk.xml").write_text(src.read_text())
            (d / "assets").symlink_to((src.parent / "assets").resolve())
            xml = str(d / "robot_walk.xml")
        rc = cfg.scene.entities["robot"]
        object.__setattr__(rc, "spec_fn", (lambda p=xml: _mj.MjSpec.from_file(p)))
    if args.override_actuator:
        from mjlab_microduck.robot.microduck_constants import _BAM_ACTUATOR_KWARGS as K
        ACT = {
            "hd1910": (f"vendor/bam/bam/params/hd1910/{K['model']}.json", float(K["kp_fw"]),
                       tuple(K["vin_range"]), float(K["vin_min"]),
                       tuple(K["vin_drop_resistance_range"]),
                       (int(K["delay_min_lag"]), int(K["delay_max_lag"]))),
            "xl330": ("vendor/bam/bam/params/xl330/m6.json", 200.0, (6.5, 8.2), 6.0,
                      (0.0, 0.2), (3, 6)),
        }[args.override_actuator]
        pj, kp, vin, vmin, vdrop, dly = ACT
        act = cfg.scene.entities["robot"].articulation.actuators[0]
        for k, v in (("json_path", pj), ("motor_name", None), ("model", None),
                     ("_resolved_json_path", pj), ("kp_fw", kp), ("vin_range", vin),
                     ("vin_min", vmin), ("vin_drop_resistance_range", vdrop),
                     ("delay_min_lag", dly[0]), ("delay_max_lag", dly[1])):
            object.__setattr__(act, k, v)
        print(f"[i] actuator {args.override_actuator}: {pj} kp_fw={kp} vin={vin} dly={dly}",
              flush=True)

    cfg.episode_length_s = 1.0e6
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=args.device),
                             clip_actions=None)
    unwrapped = env.unwrapped
    robot = unwrapped.scene.entities["robot"]
    N = args.num_envs
    sess = ort.InferenceSession(args.onnx)
    iname, oname = sess.get_inputs()[0].name, sess.get_outputs()[0].name
    in_shape = sess.get_inputs()[0].shape
    fixed_batch = isinstance(in_shape[0], int)

    term = unwrapped.command_manager.get_term("twist")
    bn = list(robot.body_names)
    trunk_id = bn.index("trunk_base")
    ank = {s: bn.index(n) for s, n in (("L", "ankle_left"), ("R", "ankle_right"))
           if n in bn}
    site_ids = {}
    for s in ("left_foot", "right_foot"):
        ids, _ = robot.find_sites(s)
        if len(ids):
            site_ids[s] = int(ids[0])
    try:
        foot_sensor = unwrapped.scene["feet_ground_contact"]
    except Exception:
        foot_sensor = None
    jids = []
    for jn in ALL_JOINTS:
        ids, _ = robot.find_joints(jn, preserve_order=True)
        jids.append(int(ids[0]))

    obs_td = env.reset()[0]
    n_warm = int(round(args.warmup / CTRL_DT))
    n_steps = int(round(args.seconds / CTRL_DT))
    keys = ("t", "x", "y", "z", "yaw", "wz", "tilt", "pitch", "roll", "vx_b", "vy_b",
            "phase", "cL", "cR", "lfy", "lfz", "rfy", "rfz", "yawL", "yawR", "jq")
    rec = {k: [] for k in keys}
    yaw_prev = np.zeros(N)
    yaw_unw = np.zeros(N)
    first = np.ones(N, dtype=bool)
    dead = np.zeros(N, dtype=bool)

    for i in range(n_warm + n_steps):
        term.command[:, 0] = args.lin_vel_x
        term.command[:, 1] = 0.0
        term.command[:, 2] = 0.0
        obs = obs_td["actor"].cpu().numpy()
        if fixed_batch and in_shape[0] != obs.shape[0]:
            action = np.concatenate(
                [sess.run([oname], {iname: obs[e:e + 1]})[0] for e in range(obs.shape[0])],
                axis=0).astype(np.float32)
        else:
            action = sess.run([oname], {iname: obs})[0].astype(np.float32)
        obs_td, _, dones, _ = env.step(torch.from_numpy(action).to(args.device))

        # ---- batched device->host ----
        pos = robot.data.body_link_pos_w[:, trunk_id].cpu().numpy().astype(np.float64)
        quat = robot.data.body_link_quat_w[:, trunk_id].cpu().numpy().astype(np.float64)
        vel = robot.data.root_link_vel_w[:, 0:3].cpu().numpy().astype(np.float64)
        wz_b = robot.data.root_link_ang_vel_b[:, 2].cpu().numpy().astype(np.float64)
        jq = robot.data.joint_pos[:, jids].cpu().numpy().astype(np.float64)
        pl = robot.data.site_pos_w[:, site_ids["left_foot"]].cpu().numpy().astype(np.float64)
        pr = robot.data.site_pos_w[:, site_ids["right_foot"]].cpu().numpy().astype(np.float64)
        al = robot.data.body_link_quat_w[:, ank["L"]].cpu().numpy().astype(np.float64)
        ar = robot.data.body_link_quat_w[:, ank["R"]].cpu().numpy().astype(np.float64)
        if foot_sensor is not None:
            f = foot_sensor.data.found.reshape(N, -1)
            if f.shape[1] == 1:
                f = f.repeat(2, axis=1)
            ctc = (f[:, :2] > 0).cpu().numpy()
        else:
            ctc = np.zeros((N, 2), dtype=bool)

        R = quats_to_mats(quat)                      # (N,3,3)
        roll, pitch = quats_to_rpy(quat)
        yaw0 = np.arctan2(R[:, 1, 0], R[:, 0, 0])
        d = np.where(first, 0.0, (yaw0 - yaw_prev + np.pi) % (2 * np.pi) - np.pi)
        yaw_unw += d
        yaw_prev = yaw0
        first[:] = False
        tilt = np.degrees(np.arccos(np.clip(R[:, 2, 2], -1, 1)))
        fl = np.einsum('nij,nj->ni', R.transpose(0, 2, 1), pl - pos)
        fr = np.einsum('nij,nj->ni', R.transpose(0, 2, 1), pr - pos)
        Rl = quats_to_mats(al); Rr = quats_to_mats(ar)
        Ml = np.einsum('nij,njk->nik', R.transpose(0, 2, 1), Rl)
        Mr = np.einsum('nij,njk->nik', R.transpose(0, 2, 1), Rr)
        yawL = np.degrees(np.arctan2(Ml[:, 1, 0], Ml[:, 0, 0]))
        yawR = np.degrees(np.arctan2(Mr[:, 1, 0], Mr[:, 0, 0]))
        phase = (ctc[:, 0] & ~ctc[:, 1]).astype(float) - (ctc[:, 1] & ~ctc[:, 0]).astype(float)
        yw = yaw_unw
        vxb = np.cos(yw) * vel[:, 0] + np.sin(yw) * vel[:, 1]
        vyb = -np.sin(yw) * vel[:, 0] + np.cos(yw) * vel[:, 1]

        # An env that terminates is teleported back to a fresh spawn by mjlab, so
        # from the step it resets on, its state is no longer the walk we set out
        # to measure. `dead` latches and every later sample for that env is NaN.
        # (`episode_length_s` is raised to 1e6 above, so a `done` here can only be
        # a real termination, never the timeout.)
        dead |= dones.cpu().numpy().astype(bool)
        if i < n_warm:
            continue
        rec["t"].append(np.where(dead, np.nan, (i - n_warm) * CTRL_DT))
        for k, v in (("x", pos[:, 0]), ("y", pos[:, 1]), ("z", pos[:, 2]), ("yaw", yaw_unw),
                     ("wz", wz_b), ("tilt", tilt), ("pitch", np.degrees(pitch)),
                     ("roll", np.degrees(roll)), ("vx_b", vxb), ("vy_b", vyb),
                     ("phase", phase), ("cL", ctc[:, 0].astype(float)),
                     ("cR", ctc[:, 1].astype(float)), ("lfy", fl[:, 1]), ("lfz", fl[:, 2]),
                     ("rfy", fr[:, 1]), ("rfz", fr[:, 2]), ("yawL", yawL), ("yawR", yawR),
                     ("jq", jq)):
            dv_ = dead[:, None] if np.ndim(v) == 2 else dead
            rec[k].append(np.where(dv_, np.nan, v))

    out = {k: np.array(v, dtype=np.float64) for k, v in rec.items()}   # (T,N)
    np.savez_compressed(args.out, **out, joint_names=np.array(ALL_JOINTS))

    def stat(a):
        a = np.asarray(a, dtype=float)
        return dict(mean=float(np.nanmean(a)), median=float(np.nanmedian(a)),
                    std=float(np.nanstd(a)))
    # per-env series: out[k][:, e]
    dur = np.array([np.count_nonzero(~np.isnan(out["t"][:, e])) for e in range(N)]) * CTRL_DT
    valid = dur > 1.0
    wz = out["wz"] * 180.0 / np.pi
    cL = np.nan_to_num(out["cL"]) > 0.5
    cR = np.nan_to_num(out["cR"]) > 0.5

    def agg(fn):
        return stat(np.array([fn(e) for e in range(N) if valid[e]], dtype=float))

    def td(c):
        return np.array([np.count_nonzero(c[e, 1:] & ~c[e, :-1]) for e in range(N)], float)

    tdL, tdR = td(cL), td(cR)
    dv = dur[valid]
    summ = {
        "tag": args.tag, "onnx": args.onnx, "num_envs": N, "seed": args.seed,
        "override_model": args.override_model,
        "override_actuator": args.override_actuator,
        "n_envs_valid": int(valid.sum()),
        "walking_envs": int(np.count_nonzero(valid & (tdL + tdR >= 10))),
        "gait": {
            "step_hz": stat((tdL[valid] + tdR[valid]) / dv / 2.0),
            "step_hz_L": stat(tdL[valid] / dv),
            "duty_L": agg(lambda e: cL[e][:int(dur[e] / CTRL_DT)].mean()),
            "duty_R": agg(lambda e: cR[e][:int(dur[e] / CTRL_DT)].mean()),
            "double_support": agg(lambda e: (cL[e] & cR[e])[:int(dur[e] / CTRL_DT)].mean()),
            "flight_frac": agg(lambda e: (~cL[e] & ~cR[e])[:int(dur[e] / CTRL_DT)].mean()),
            "clearance_L_mm": agg(lambda e: (np.nanpercentile(out["lfz"][:, e], 95) - np.nanpercentile(out["lfz"][:, e], 5)) * 1000),
            "clearance_R_mm": agg(lambda e: (np.nanpercentile(out["rfz"][:, e], 95) - np.nanpercentile(out["rfz"][:, e], 5)) * 1000),
            "swing_peak_L_mm": agg(lambda e: np.nanpercentile(out["lfz"][:, e], 98) * 1000),
            "swing_peak_R_mm": agg(lambda e: np.nanpercentile(out["rfz"][:, e], 98) * 1000),
            "stance_z_L_mm": agg(lambda e: np.nanmean(out["lfz"][cL[:, e], e]) * 1000 if cL[:, e].any() else np.nan),
        },
        "posture": {
            "trunk_z_mm": agg(lambda e: np.nanmean(out["z"][:, e]) * 1000),
            "tilt_deg": agg(lambda e: np.nanmean(out["tilt"][:, e])),
            "pitch_dc_deg": agg(lambda e: np.nanmean(out["pitch"][:, e])),
            "pitch_std_deg": agg(lambda e: np.nanstd(out["pitch"][:, e])),
            "roll_dc_deg": agg(lambda e: np.nanmean(out["roll"][:, e])),
        },
        "feet": {
            "foot_y_L_mm": agg(lambda e: np.nanmean(out["lfy"][:, e]) * 1000),
            "foot_y_R_mm": agg(lambda e: np.nanmean(out["rfy"][:, e]) * 1000),
            "stance_y_L_mm": agg(lambda e: np.nanmean(out["lfy"][cL[:, e], e]) * 1000 if cL[:, e].any() else np.nan),
            "stance_y_R_mm": agg(lambda e: np.nanmean(out["rfy"][cR[:, e], e]) * 1000 if cR[:, e].any() else np.nan),
            "foot_y_range_L_mm": agg(lambda e: (np.nanmax(out["lfy"][:, e]) - np.nanmin(out["lfy"][:, e])) * 1000),
            "foot_y_LR_sep_mm": agg(lambda e: (np.nanmean(out["lfy"][:, e]) - np.nanmean(out["rfy"][:, e])) * 1000),
            "ankle_yaw_L_deg": agg(lambda e: np.nanmean(out["yawL"][:, e])),
            "ankle_yaw_R_deg": agg(lambda e: np.nanmean(out["yawR"][:, e])),
            "ankle_yaw_LR_deg": agg(lambda e: np.nanmean(out["yawL"][:, e]) - np.nanmean(out["yawR"][:, e])),
        },
        "hip_roll": {
            "splay_LR_deg": agg(lambda e: abs(np.degrees(np.nanmean(out["jq"][:, e, ALL_JOINTS.index("left_hip_roll")]) + np.nanmean(out["jq"][:, e, ALL_JOINTS.index("right_hip_roll")]))) / 2.0),
            "dev_L_deg": agg(lambda e: abs(np.degrees(np.nanmean(out["jq"][:, e, ALL_JOINTS.index("left_hip_roll")])) + 0.0873)),
            "dev_R_deg": agg(lambda e: abs(np.degrees(np.nanmean(out["jq"][:, e, ALL_JOINTS.index("right_hip_roll")])) - 0.0873)),
        },
        "yaw": {
            "yaw_dc_body_deg_s": agg(lambda e: np.nanmean(wz[:, e])),
            "yaw_rate_std_deg_s": agg(lambda e: np.nanstd(wz[:, e])),
            "net_yaw_deg": agg(lambda e: (lambda y: y[-1] - y[0] if y.size else np.nan)(out["yaw"][:, e][~np.isnan(out["yaw"][:, e])])),
            "yaw_ac_lock_deg_s": agg(lambda e: np.nanmean(wz[:, e] * np.nan_to_num(out["phase"][:, e]))),
        },
        "travel": {
            "vx_body_all_mps": agg(lambda e: np.nanmean(out["vx_b"][:, e])),
            "vy_body_all_mps": agg(lambda e: np.nanmean(out["vy_b"][:, e])),
        },
    }
    Path(args.out).with_suffix(".json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
