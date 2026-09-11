#!/usr/bin/env python3
"""Faithful headless eval of an exported ONNX policy in the REAL training env.

Why this exists: `scripts/headless_eval.py` and `scripts/infer_policy.py` run the
simplified `scene.xml` (raw position actuators), NOT the BAM voltage model the
policy was trained against. And `scripts/record_walk_angles.py --engine env` is
faithful but builds the env with `play=True`, which turns `push_robot` into a
0.5-1.0 s interval (±0.3 m/s shoves) — the robot looks like it "falls
constantly" when it is really being shoved.

This script builds the real training env (BAM + hd1910 params) with pushes
DISABLED and a constant command, then reports:
  trunk z (mean/min), trunk tilt (deg), achieved vs commanded xy velocity,
  and how many env resets (falls) happened.

Usage:
    uv run python scripts/eval_onnx_bam.py --onnx walk.onnx \
        --task Mjlab-Velocity-Flat-MicroDuck --lin-vel-x 0.2 --seconds 8
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

CTRL_DT = 0.02  # 50 Hz


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--task", default="Mjlab-Velocity-Flat-MicroDuck")
    ap.add_argument("--lin-vel-x", type=float, default=0.2)
    ap.add_argument("--lin-vel-y", type=float, default=0.0)
    ap.add_argument("--ang-vel-z", type=float, default=0.0)
    ap.add_argument("--warmup", type=float, default=2.0)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--params-json", default=None,
                    help="explicit BAM params JSON (e.g. the snapshot the policy was trained with)")
    ap.add_argument("--keep-pushes", action="store_true",
                    help="keep play-mode pushes (0.5-1.0 s interval) to test robustness")
    args = ap.parse_args()

    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = 1
    if not args.keep_pushes:
        removed = cfg.events.pop("push_robot", None)
        print(f"[i] push_robot removed: {removed is not None}")

    # Freeze the command: otherwise `env.step()` -> command_manager.compute()
    # resamples it every step (and zeroes `rel_standing_envs` of the envs), so a
    # per-step write of term.command is silently clobbered and the policy is
    # being tested against RANDOM commands, not the one we asked for.
    tcfg = cfg.commands["twist"]
    tcfg.resampling_time_range = (1e9, 1e9)
    for attr, val in (("rel_standing_envs", 0.0), ("rel_heading_envs", 0.0)):
        if hasattr(tcfg, attr):
            setattr(tcfg, attr, val)
    if getattr(tcfg, "heading_command", False):
        tcfg.heading_command = False
        if hasattr(tcfg, "ranges") and hasattr(tcfg.ranges, "heading"):
            tcfg.ranges.heading = None
    print("[i] twist command frozen (no resampling, no standing/heading mask)")

    # Optional: point the BAM actuator at an explicit params JSON (e.g. the exact
    # snapshot a policy was TRAINED with). `_resolved_json_path` is resolved in
    # __post_init__ on a frozen dataclass, so override both it and `json_path`.
    if getattr(args, "params_json", None):
        act = cfg.scene.entities["robot"].articulation.actuators[0]
        object.__setattr__(act, "json_path", args.params_json)
        object.__setattr__(act, "motor_name", None)
        object.__setattr__(act, "model", None)
        object.__setattr__(act, "_resolved_json_path", args.params_json)
        print(f"[i] BAM params overridden -> {args.params_json}")

    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    env = RslRlVecEnvWrapper(env, clip_actions=None)
    unwrapped = env.unwrapped
    robot = unwrapped.scene.entities["robot"]

    sess = ort.InferenceSession(args.onnx)
    iname, oname = sess.get_inputs()[0].name, sess.get_outputs()[0].name
    assert sess.get_inputs()[0].shape[1] == 61, sess.get_inputs()[0].shape

    term = unwrapped.command_manager.get_term("twist")

    obs_td = env.reset()[0]
    n_warm = int(round(args.warmup / CTRL_DT))
    n = int(round(args.seconds / CTRL_DT))

    from collections import Counter
    reset_kinds = Counter()
    zs, tilts, vxs, vys, resets = [], [], [], [], 0
    body_id = 0
    for i in range(n_warm + n):
        term.command[:, 0] = args.lin_vel_x
        term.command[:, 1] = args.lin_vel_y
        term.command[:, 2] = args.ang_vel_z

        obs = obs_td["actor"].cpu().numpy()
        action = sess.run([oname], {iname: obs})[0].astype(np.float32)
        obs_td, _, dones, _ = env.step(torch.from_numpy(action).to(args.device))
        if bool(dones[0]):
            resets += 1
            # attribute the reset: termination vs timeout
            try:
                term_b = bool(unwrapped.reset_terminated[0])
                tout_b = bool(unwrapped.reset_time_outs[0])
            except Exception:
                term_b = tout_b = False
            if term_b:
                reset_kinds["terminated"] += 1
                # which termination term(s) fired
                for name in getattr(unwrapped.termination_manager, "active_terms", []):
                    try:
                        if bool(unwrapped.termination_manager.get_term(name).terminated[0]):
                            reset_kinds[f"term:{name}"] += 1
                    except Exception:
                        pass
            elif tout_b:
                reset_kinds["time_out"] += 1
            else:
                reset_kinds["other"] += 1
            if len(zs) > 60:
                print(f"[i] reset at step {i}: z={float(robot.data.body_link_pos_w[0, body_id][2]):.3f} "
                      f"tilt={tilts[-1]:.1f}deg kinds={dict(reset_kinds)}")

        if i < n_warm:
            continue
        # trunk height + tilt from the trunk body orientation
        pos = robot.data.body_link_pos_w[0, body_id]
        quat = robot.data.body_link_quat_w[0, body_id]  # (w,x,y,z)
        w, x, y, zq = (float(v) for v in quat)
        # trunk z-axis in world = R @ [0,0,1]
        zax_z = 1.0 - 2.0 * (x * x + y * y)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, zax_z))))
        zs.append(float(pos[2]))
        tilts.append(tilt)
        w4, x4, y4, z4 = (float(v) for v in quat)
        yaw = math.atan2(2 * (w4 * z4 + x4 * y4), 1 - 2 * (y4 * y4 + z4 * z4))
        # world-frame root link linear velocity -> yaw frame
        vw = robot.data.root_link_vel_w[0, 0:3]
        vwx, vwy = float(vw[0]), float(vw[1])
        vxs.append(math.cos(yaw) * vwx + math.sin(yaw) * vwy)
        vys.append(-math.sin(yaw) * vwx + math.cos(yaw) * vwy)

    def stat(a):
        a = np.array(a)
        return f"mean={a.mean():+.3f} min={a.min():+.3f} max={a.max():+.3f}"

    print("\n=== faithful BAM eval ===")
    print(f"  task            : {args.task}")
    print(f"  cmd             : vx={args.lin_vel_x} vy={args.lin_vel_y} wz={args.ang_vel_z}")
    print(f"  steps           : {len(zs)} ({len(zs)*CTRL_DT:.2f}s)")
    print(f"  trunk_z [m]     : {stat(zs)}")
    print(f"  trunk tilt [deg]: {stat(tilts)}")
    print(f"  achieved vx [m/s]: {stat(vxs)}   (cmd {args.lin_vel_x:+.2f})")
    print(f"  achieved vy [m/s]: {stat(vys)}   (cmd {args.lin_vel_y:+.2f})")
    print(f"  env resets/falls: {resets}  in {args.seconds:.1f}s   kinds={dict(reset_kinds)}")
    print(f"  tracking err |vx|: {abs(np.mean(vxs) - args.lin_vel_x):.3f} m/s")
    # A healthy gait can be slightly crouched (the working walk sits at
    # z ~ 0.094-0.100 vs the 0.115 standing height), so 0.10 was too strict.
    ok = np.mean(zs) > 0.090 and np.mean(tilts) < 15 and resets <= 1
    print(f"  VERDICT         : {'STANDS/WALKS OK' if ok else 'NOT OK'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
