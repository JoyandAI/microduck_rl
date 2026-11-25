#!/usr/bin/env python3
"""Measure ALL THREE capabilities of a walking policy in one pass, per policy.

Why (user, 2026-09-17): "分析一下如何把走直、不外八、静态站稳几个模型能结合到一起，
也就是对齐官方". The three capabilities have been optimised one at a time and each
attempt traded the others away, so the question is whether they are in conflict at
all -- and that is only answerable if they are measured on the SAME rollout
protocol, side by side, with the official as the reference row.

Axis A -- WALKS STRAIGHT (vx = 0.2 m/s, 15 s, seed 0, 1 env):
    yaw rate DC (deg/s), net heading change (deg), forward speed (m/s).
    The DC rate is the robust number; turning radius is a ratio whose denominator
    (yaw rate) approaches zero, so it swings 5x between repeats of one policy and
    is reported for information only.

Axis B -- NO SPLAY (same rollout):
    L-R hip_roll deviation from HOME (the mirrored-abduction measure), and the
    lateral distance between the two ankle bodies. **These two disagree** on the
    current arms (R3 cut the angle but widened the feet to 56 mm; R4 matched the
    official's 27 mm but kept a 0.195 rad angle difference), so both are printed
    and neither is called "the" splay on its own.

Axis C -- STATIC STABILITY (vx = 0.0 m/s, 15 s, seed 0, 1 env):
    net heading drift (deg), tilt (deg), trunk height (mm) and its spread, total
    horizontal displacement (mm), and mean foot clearance (mm). Clearance is the
    metric this repo settled on for "is it marching in place" (19.4 mm marching
    vs 0.5-4.2 mm standing) -- yaw drift and displacement alone cannot tell a
    stable stance from a robot that is slowly walking away.

Every policy is run in the hardware era it was trained for: the official
alpha_walking is PRE-SWAP (XL330, kp_fw 200, 6.5-8.2 V, 737 g) and must be run
with `preswap`, or it is being measured on a 27 %-weaker, 33 %-slower servo than
it trained with.

Usage:
    uv run python scripts/policy_axis_matrix.py \
        --policy official:/home/joyandai/microduck/policies/alpha_walking.onnx:preswap \
        --policy ck7000:walk_hd1910_7000.onnx \
        --policy ck12000:walk_hd1910_12000.onnx \
        --policy R3_hrl1:logs/rsl_rl/velocity/<run>/<run>.onnx
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import onnxruntime as ort
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg

from mjlab_microduck.tasks import mdp as mdp_mod

PRESWAP_XML = "src/mjlab_microduck/robot/microduck/robot_walk_preswap.xml"
XL330_JSON = "vendor/bam/bam/params/xl330/m6.json"


def apply_preswap(cfg) -> None:
    import mujoco
    robot = cfg.scene.entities["robot"]
    object.__setattr__(robot, "spec_fn", (lambda p=PRESWAP_XML: mujoco.MjSpec.from_file(p)))
    act = robot.articulation.actuators[0]
    for k, v in (("json_path", XL330_JSON), ("motor_name", None), ("model", None),
                 ("_resolved_json_path", XL330_JSON), ("kp_fw", 200.0),
                 ("vin_range", (6.5, 8.2)), ("vin_min", 6.0),
                 ("vin_drop_resistance_range", (0.0, 0.2)),
                 ("delay_min_lag", 3), ("delay_max_lag", 6)):
        object.__setattr__(act, k, v)


def build(seed: int, device: str, preswap: bool):
    cfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck", play=True)
    cfg.scene.num_envs = 1
    cfg.seed = seed
    cfg.episode_length_s = 1000.0
    if preswap:
        apply_preswap(cfg)
    cfg.events.pop("push_robot", None)
    cfg.curriculum.pop("push_magnitude", None)
    # Freeze the twist command AND the pose commands: head_pose/body_pose resample
    # every 2-5 s and the head is 37 % of body mass, so a live pose command can
    # yaw the trunk and would contaminate Axis C (which is ABOUT yaw drift).
    for cname in ("twist", "head_pose", "body_pose"):
        pc = cfg.commands.get(cname)
        if pc is None:
            continue
        pc.resampling_time_range = (1e9, 1e9)
        if cname == "twist":
            for a in ("rel_standing_envs", "rel_heading_envs", "rel_turn_in_place_envs"):
                if hasattr(pc, a):
                    setattr(pc, a, 0.0)
            if getattr(pc, "heading_command", False):
                pc.heading_command = False
                # mjlab raises "ranges.heading is set but heading_command=False" --
                # disabling the command is not enough, the range must go too.
                # eval_straightness.py does this; the first version here did not.
                if hasattr(pc.ranges, "heading"):
                    pc.ranges.heading = None
        else:
            if hasattr(pc, "zero_command_prob"):
                pc.zero_command_prob = 1.0
            if hasattr(pc, "ranges") and isinstance(pc.ranges, (tuple, list)):
                pc.ranges = tuple((0.0, 0.0) for _ in pc.ranges)
    return cfg


def rollout(onnx: str, vx: float, seconds: float, warmup: float, seed: int,
            device: str, preswap: bool) -> dict:
    cfg = build(seed, device, preswap)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=device), clip_actions=None)
    un = env.unwrapped
    robot = un.scene.entities["robot"]
    sess = ort.InferenceSession(onnx)
    iname, oname = sess.get_inputs()[0].name, sess.get_outputs()[0].name
    sid = mdp_mod._servo_joint_ids(un, robot)
    names = [robot.joint_names[i] for i in sid]
    home = robot.data.default_joint_pos[0, sid].cpu().numpy()
    tb, _ = robot.find_bodies("trunk_base", preserve_order=True)
    ti = int(tb[0])
    lb, _ = robot.find_bodies("ankle_left", preserve_order=True)
    rb, _ = robot.find_bodies("ankle_right", preserve_order=True)
    li, ri = int(lb[0]), int(rb[0])

    obs = env.reset()[0]
    term = un.command_manager.get_term("twist")

    def step():
        nonlocal obs
        term.command[:, 0] = vx
        term.command[:, 1] = 0.0
        term.command[:, 2] = 0.0
        a = sess.run([oname], {iname: obs["actor"].cpu().numpy()})[0].astype(np.float32)
        obs, _, dones, _ = env.step(torch.from_numpy(a).to(device))
        return bool(dones[0])

    for _ in range(int(round(warmup / 0.02))):
        step()

    def yaw_of(q):
        w, x, y, z = (float(v) for v in q)
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    p0 = robot.data.body_link_pos_w[0, ti, :2].cpu().numpy().copy()
    yaw_prev = yaw_of(robot.data.body_link_quat_w[0, 0])
    yaw_unwrapped = 0.0
    tilt, zs, width, clear, resets = [], [], [], [], 0
    q_acc = []
    for _ in range(int(round(seconds / 0.02))):
        if step():
            resets += 1
        q = robot.data.body_link_quat_w[0, 0]
        y = yaw_of(q)
        d = y - yaw_prev
        d = (d + math.pi) % (2 * math.pi) - math.pi   # wrap-safe per-step delta
        yaw_unwrapped += d
        yaw_prev = y
        w, x, yy, z = (float(v) for v in q)
        tilt.append(math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + yy * yy))))))
        pos = robot.data.body_link_pos_w[0]
        zs.append(float(pos[ti, 2]))
        width.append(float(pos[ri, 1] - pos[li, 1]))
        q_acc.append(robot.data.joint_pos[0, sid].cpu().numpy())
        # TerrainHeightSensor exposes `data.heights` [B, F] = per-frame vertical
        # clearance already (frame_z - hit_z), so no terrain-origin subtraction
        # and no ray_hits_z -- the first version guessed both and read nan.
        sf = un.scene.sensors.get("foot_height_scan") if hasattr(un.scene, "sensors") else None
        if sf is not None and getattr(sf.data, "heights", None) is not None:
            clear.append(float(sf.data.heights.mean()))
    p1 = robot.data.body_link_pos_w[0, ti, :2].cpu().numpy()
    q_acc = np.array(q_acc)
    return {
        "names": names, "home": home,
        "yaw_dc": math.degrees(yaw_unwrapped) / seconds,
        "net_yaw": math.degrees(yaw_unwrapped),
        "speed": float(np.linalg.norm(p1 - p0) / seconds),
        "disp_mm": float(np.linalg.norm(p1 - p0) * 1000),
        "tilt": float(np.mean(tilt)),
        "z_mm": float(np.mean(zs) * 1000), "z_sd_mm": float(np.std(zs) * 1000),
        "width_mm": float(np.mean(width) * 1000),
        "clear_mm": float(np.mean(clear) * 1000) if clear else float("nan"),
        "roll_mean": q_acc.mean(0),
        "roll_std": q_acc.std(0),
        "resets": resets,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", action="append", required=True, help="name:path[:preswap]")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--warmup", type=float, default=3.0)
    ap.add_argument("--vx", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    pols = []
    for spec in args.policy:
        p = spec.split(":")
        pols.append((p[0], p[1], len(p) > 2 and p[2] == "preswap"))

    A, C = {}, {}
    for name, path, pre in pols:
        print(f"[run] {name} preswap={pre}", flush=True)
        A[name] = rollout(path, args.vx, args.seconds, args.warmup, args.seed,
                          args.device, pre)
        C[name] = rollout(path, 0.0, args.seconds, args.warmup, args.seed,
                          args.device, pre)

    def row(n):
        a = A[n]["names"]
        li, ri = a.index("left_hip_roll"), a.index("right_hip_roll")
        h = A[n]["home"]
        return (A[n]["roll_mean"][li] - h[li]) - (A[n]["roll_mean"][ri] - h[ri])

    print("\n" + "=" * 100)
    print("AXIS A -- WALKS STRAIGHT (vx=%.2f, %.0fs, seed %d)" % (args.vx, args.seconds, args.seed))
    print("=" * 100)
    print(f"{'policy':<18}{'yawDC deg/s':>13}{'net yaw deg':>13}{'speed m/s':>11}{'resets':>8}")
    for n in A:
        print(f"{n:<18}{A[n]['yaw_dc']:>13.2f}{A[n]['net_yaw']:>13.1f}"
              f"{A[n]['speed']:>11.3f}{A[n]['resets']:>8d}")

    print("\n" + "=" * 100)
    print("AXIS B -- NO SPLAY (same rollout; the two measures DISAGREE, read both)")
    print("=" * 100)
    print(f"{'policy':<18}{'L-R hip_roll':>14}{'ankle sep mm':>14}{'tilt deg':>10}")
    for n in A:
        print(f"{n:<18}{row(n):>14.3f}{A[n]['width_mm']:>14.1f}{A[n]['tilt']:>10.1f}")

    print("\n" + "=" * 100)
    print("AXIS C -- STATIC STABILITY (vx=0.0, %.0fs, seed %d)" % (args.seconds, args.seed))
    print("=" * 100)
    print(f"{'policy':<18}{'net yaw deg':>13}{'tilt deg':>10}{'z mm':>9}{'z sd mm':>10}"
          f"{'drift mm':>10}{'foot clr mm':>13}{'resets':>8}")
    for n in C:
        print(f"{n:<18}{C[n]['net_yaw']:>13.1f}{C[n]['tilt']:>10.1f}{C[n]['z_mm']:>9.1f}"
              f"{C[n]['z_sd_mm']:>10.2f}{C[n]['disp_mm']:>10.1f}"
              f"{C[n]['clear_mm']:>13.2f}{C[n]['resets']:>8d}")

    print("\n判读：Axis A 只看 yawDC（转弯半径是比值，重复间会差 5 倍，不引用）；")
    print("      Axis B 两个量都要看，它们现在互相矛盾；")
    print("      Axis C 的 foot clearance 是仓库里定下的“原地踏步”判据（踏步 ~19 mm / 站定 0.5–4 mm）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
