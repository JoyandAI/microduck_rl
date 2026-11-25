#!/usr/bin/env python3
"""Post-process the straightness CSVs: robust aggregate table + hip_yaw limits.

Why a separate post-processor: the raw per-step CSVs written by
`scripts/eval_straightness_factorial.py` are the source of truth, so every table
in the report can be recomputed WITHOUT re-running the GPU sim. This script
re-derives the aggregate metrics and computes the hip_yaw mechanical-limit
occupancy from the CSV using the MECHANICAL limits read from the XML **by joint
name** (the mjlab `robot.data.joint_pos_limits` are the *soft* limits, factor
0.9, and `u.sim.model.jnt_range` is indexed by MUJOCO joint id, which is shifted
by one relative to the entity's joint list because of the free joint).

Usage:
  uv run python scripts/straightness_tables.py --glob 'docs/forensics/artifacts/straightness/*.csv'
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

REPO = Path("/home/joyandai/microduck_rl")
XMLS = {
    "hd1910": REPO / "src/mjlab_microduck/robot/microduck/robot_walk.xml",
    "xl330": REPO / "src/mjlab_microduck/robot/microduck/robot_walk_preswap.xml",
}
CTRL_DT = 0.02


def hard_limits(xml: Path) -> dict:
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(xml))
    out = {}
    for name in ("left_hip_yaw", "right_hip_yaw", "left_hip_roll", "right_hip_roll"):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        out[name] = (float(m.jnt_range[jid][0]), float(m.jnt_range[jid][1]))
    return out


def load_csv(path: Path):
    import csv as _csv
    with open(path) as f:
        rd = list(_csv.DictReader(f))
    cols = {k: np.array([float(r[k]) for r in rd]) for k in rd[0]}
    envs = np.array([int(r["env"]) for r in rd])
    return cols, envs


def per_env_metrics(cols, envs, xml_key: str):
    lim = hard_limits(XMLS[xml_key])
    out = {}
    for e in sorted(set(envs.tolist())):
        sel = envs == e
        d = {k: v[sel] for k, v in cols.items()}
        n = len(d["t"])
        if n < 25:
            continue
        t = d["t"]
        T = float(t[-1] - t[0])
        m: dict = {}
        # heading, unwrapped
        yaw = np.unwrap(d["yaw_raw"])
        yaw = yaw - yaw[0]
        m["heading_net_deg"] = math.degrees(yaw[-1])
        m["heading_rate_deg_s"] = math.degrees(yaw[-1]) / T
        m["heading_slope_deg_s"] = math.degrees(np.polyfit(t, yaw, 1)[0])
        m["heading_quad_deg_s2"] = math.degrees(2.0 * np.polyfit(t, yaw, 2)[0])
        m["heading_abs_max_deg"] = math.degrees(np.max(np.abs(yaw)))
        m["heading_t_at_absmax_s"] = float(t[int(np.argmax(np.abs(yaw)))])
        absy = np.abs(yaw)
        m["heading_final_over_max"] = float(absy[-1] / absy.max())
        m["heading_corrected_after_max_deg"] = math.degrees(absy.max() - absy[-1])
        lin = np.polyfit(t, yaw, 1)
        m["heading_osc_std_deg"] = math.degrees(np.std(yaw - np.polyval(lin, t)))
        m["wz_world_mean_deg_s"] = math.degrees(np.mean(d["wz_w"]))
        m["wz_world_std_deg_s"] = math.degrees(np.std(d["wz_w"]))
        m["wz_body_mean_deg_s"] = math.degrees(np.mean(d["wz_b"]))
        m["wz_body_std_deg_s"] = math.degrees(np.std(d["wz_b"]))
        # windowed yaw rates
        for k in range(4):
            a, b = int(k * n / 4), int((k + 1) * n / 4)
            m[f"yawrate_win{k}_deg_s"] = math.degrees(np.polyfit(t[a:b], yaw[a:b], 1)[0])
        # path in the initial-facing frame
        yaw0 = float(d["yaw_raw"][0])
        u = np.array([math.cos(yaw0), math.sin(yaw0)])
        nn = np.array([-math.sin(yaw0), math.cos(yaw0)])
        xy = np.stack([d["trunk_x"], d["trunk_y"]], axis=1)
        rel = xy - xy[0]
        fwd, lat = rel @ u, rel @ nn
        m["forward_m"] = float(fwd[-1])
        m["lateral_m"] = float(lat[-1])
        m["path_length_m"] = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
        m["lateral_per_m"] = float(lat[-1] / fwd[-1]) if abs(fwd[-1]) > 1e-6 else float("nan")
        m["path_heading_deg"] = math.degrees(math.atan2(lat[-1], fwd[-1]))
        c = rel - rel.mean(axis=0, keepdims=True)
        dv = np.linalg.svd(c, full_matrices=False)[2][0]
        perp = c @ np.array([-dv[1], dv[0]])
        m["path_line_rms_mm"] = float(np.sqrt(np.mean(perp ** 2))) * 1000.0
        m["path_line_max_mm"] = float(np.max(np.abs(perp))) * 1000.0
        # turning vs crabbing
        spd = np.linalg.norm(np.stack([d["vx_w"], d["vy_w"]], axis=1), axis=1)
        ok = spd > 0.02
        if ok.sum() > 5:
            vd = np.unwrap(np.arctan2(d["vy_w"][ok], d["vx_w"][ok]))
            m["vel_dir_net_deg"] = math.degrees(vd[-1] - vd[0])
        m["crab_angle_deg"] = math.degrees(math.atan2(np.mean(d["vy_b"]), np.mean(d["vx_b"])))
        m["vx_body_mean"] = float(np.mean(d["vx_b"]))
        m["vx_body_std"] = float(np.std(d["vx_b"]))
        m["vy_body_mean"] = float(np.mean(d["vy_b"]))
        m["trunk_z_mean"] = float(np.mean(d["trunk_z"]))
        m["trunk_tilt_mean_deg"] = math.degrees(float(np.mean(
            np.arccos(np.clip(1 - 2 * (d["qx"] ** 2 + d["qy"] ** 2), -1, 1)))))
        if "com_off_y_b" in d:
            m["com_off_x_b_mm"] = float(np.median(d["com_off_x_b"])) * 1000
            m["com_off_y_b_mm"] = float(np.median(d["com_off_y_b"])) * 1000
        # per-leg joints
        for j in ("left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee",
                  "left_ankle", "right_hip_yaw", "right_hip_roll", "right_hip_pitch",
                  "right_knee", "right_ankle"):
            q = d["q_" + j]
            m["q_" + j + "_mean_deg"] = math.degrees(np.mean(q))
            m["q_" + j + "_std_deg"] = math.degrees(np.std(q))
            m["q_" + j + "_dev_deg"] = math.degrees(np.mean(q - HOME[j]))
        # feet
        for side, km in (("L", "l"), ("R", "r")):
            mask = d["contact_" + km] > 0.5
            m["duty_" + side] = float(np.mean(mask))
            rise = np.where((~mask[:-1]) & mask[1:])[0] + 1
            fall = np.where(mask[:-1] & (~mask[1:]))[0] + 1
            m["n_steps_" + side] = int(len(rise))
            per = np.diff(rise) if len(rise) > 1 else np.array([])
            if len(per):
                m["step_period_" + side + "_s"] = float(np.mean(per) * CTRL_DT)
            sw = [(rise[rise > a][0] - a) * CTRL_DT for a in fall if len(rise[rise > a])]
            m["swing_time_" + side + "_s"] = float(np.mean(sw)) if sw else float("nan")
            if len(rise) > 1:
                sl = np.stack([np.diff(d["trunk_x"][rise]), np.diff(d["trunk_y"][rise])],
                              axis=1) @ u
                m["step_length_" + side + "_m"] = float(np.mean(sl))
                m["foot_y_body_at_td_" + side + "_mm"] = float(
                    np.mean(d["foot_" + km + "_y_b"][rise])) * 1000
                m["foot_y_body_at_td_" + side + "_std_mm"] = float(
                    np.std(d["foot_" + km + "_y_b"][rise])) * 1000
            m["force_" + side + "_fz_mean_N"] = float(np.mean(d["force_" + km + "_z"]))
            m["force_" + side + "_fz_peak_N"] = float(np.max(d["force_" + km + "_z"]))
            m["force_" + side + "_peak_N"] = float(np.max(d["force_" + km]))
        mL, mR = m["force_L_fz_mean_N"], m["force_R_fz_mean_N"]
        if abs(mL) + abs(mR) > 1e-9:
            m["contact_force_share_L"] = mL / (mL + mR)
        # hip_yaw MECHANICAL limit occupancy (from the XML, by name)
        for j in ("left_hip_yaw", "right_hip_yaw"):
            lo, hi = lim[j]
            q = d["q_" + j]
            m[j + "_hi_deg"] = math.degrees(hi)
            m[j + "_lo_deg"] = math.degrees(lo)
            m[j + "_frac_time_at_hi_1deg"] = float(np.mean(q > hi - math.radians(1)))
            m[j + "_frac_time_at_limit_1deg"] = float(np.mean(
                (q > hi - math.radians(1)) | (q < lo + math.radians(1))))
            km = "l" if j.startswith("left") else "r"
            rise = np.where((d["contact_" + km][:-1] <= 0.5) & (d["contact_" + km][1:] > 0.5))[0] + 1
            if len(rise):
                qs = q[rise]
                m[j + "_frac_steps_at_hi_1deg"] = float(np.mean(qs > hi - math.radians(1)))
                m[j + "_frac_steps_at_limit_1deg"] = float(np.mean(
                    (qs > hi - math.radians(1)) | (qs < lo + math.radians(1))))
            m[j + "_mean_deg"] = math.degrees(np.mean(q))
            m[j + "_absmean_deg"] = math.degrees(np.mean(np.abs(q)))
            m[j + "_p95_abs_deg"] = math.degrees(np.percentile(np.abs(q), 95))
            m[j + "_max_deg"] = math.degrees(np.max(q))
            m[j + "_min_deg"] = math.degrees(np.min(q))
            # "step" = interval between this foot's touchdowns; per-step peak |q|
            if len(rise) > 1:
                pk = [np.max(np.abs(q[rise[k]:rise[k + 1]])) for k in range(len(rise) - 1)]
                m[j + "_step_peak_absmean_deg"] = math.degrees(np.mean(pk))
        out[int(e)] = m
    return out


import csv as _csv_mod  # noqa: E402
from collections import Counter  # noqa: E402

HOME = {
    "left_hip_yaw": 0.0, "left_hip_roll": -0.0873, "left_hip_pitch": -0.4579,
    "left_knee": 0.0, "left_ankle": 0.4530,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0873, "right_hip_pitch": 0.4579,
    "right_knee": 0.0, "right_ankle": -0.4530,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="docs/forensics/artifacts/straightness/*.csv")
    ap.add_argument("--xml", default="hd1910", choices=tuple(XMLS))
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    table = {}
    for f in files:
        p = Path(f)
        cols, envs = load_csv(p)
        pe = per_env_metrics(cols, envs, args.xml)
        table[p.stem] = pe
        keys = sorted({k for m in pe.values() for k in m})
        summ = {}
        for k in keys:
            v = [m[k] for m in pe.values() if isinstance(m.get(k), (int, float)) and m[k] == m[k]]
            if v:
                summ[k] = (float(np.mean(v)), float(np.std(v)), len(v))
        print(f"\n### {p.stem}   ({len(pe)} envs, {len(cols['t'])} rows)")
        for k in ("heading_net_deg", "heading_slope_deg_s", "heading_quad_deg_s2",
                  "heading_abs_max_deg", "heading_final_over_max", "heading_t_at_absmax_s",
                  "wz_world_mean_deg_s", "wz_world_std_deg_s",
                  "yawrate_win0_deg_s", "yawrate_win1_deg_s", "yawrate_win2_deg_s",
                  "yawrate_win3_deg_s",
                  "forward_m", "lateral_m", "lateral_per_m", "path_length_m",
                  "path_heading_deg", "path_line_rms_mm", "crab_angle_deg",
                  "vel_dir_net_deg", "vx_body_mean", "vy_body_mean",
                  "trunk_z_mean", "trunk_tilt_mean_deg",
                  "left_hip_yaw_mean_deg", "right_hip_yaw_mean_deg",
                  "left_hip_yaw_frac_time_at_limit_1deg",
                  "right_hip_yaw_frac_time_at_limit_1deg",
                  "left_hip_yaw_frac_steps_at_limit_1deg",
                  "right_hip_yaw_frac_steps_at_limit_1deg",
                  "left_hip_yaw_frac_steps_at_hi_1deg",
                  "right_hip_yaw_frac_steps_at_hi_1deg",
                  "left_hip_yaw_max_deg", "right_hip_yaw_max_deg",
                  "left_hip_yaw_min_deg", "right_hip_yaw_min_deg",
                  "com_off_y_b_mm", "contact_force_share_L", "duty_L", "duty_R"):
            if k in summ:
                mu, sd, n = summ[k]
                print(f"    {k:<44} {mu:+10.4f} +- {sd:8.4f}   (n={n})")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(table, indent=2, default=str))
        print(f"\n[i] -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
