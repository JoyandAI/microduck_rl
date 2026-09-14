#!/usr/bin/env python3
"""Eval an ONNX policy inside the REAL training env (BAM actuator, 61D obs).

Unlike scripts/headless_eval.py (scene.xml + XML PD approximation), this runs
the policy inside the mjlab ManagerBasedRlEnv the same way training does —
BAM actuator (hls2909 or xl330 per microduck_constants), 12V DR, obs layout
identical to training.

Usage:
    uv run python scripts/env_onnx_eval.py --onnx walk.onnx --lin-vel-x 0.0
"""

from __future__ import annotations

import argparse

import numpy as np
import onnxruntime as ort
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--task", default="Mjlab-Velocity-Flat-MicroDuck")
    ap.add_argument("--lin-vel-x", type=float, default=0.0)
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--motor-name", default=None, help="override motor_name (e.g. xl330)")
    ap.add_argument("--model", default=None, help="override bam model (e.g. m6)")
    ap.add_argument("--kp-fw", type=float, default=None, help="override kp_fw (takes effect!)")
    args = ap.parse_args()

    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = args.num_envs
    # BAM actuator field overrides (kp_fw is a real cfg field; json 'kp' is not)
    act = cfg.scene.entities["robot"].articulation.actuators[0]
    if args.motor_name:
        act.motor_name = args.motor_name
    if args.model:
        act.model = args.model
    if args.kp_fw is not None:
        act.kp_fw = args.kp_fw
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    env = RslRlVecEnvWrapper(env, clip_actions=None)
    unwrapped = env.unwrapped

    sess = ort.InferenceSession(args.onnx)
    iname = sess.get_inputs()[0].name
    oname = sess.get_outputs()[0].name

    obs_td = env.reset()[0]
    robot = unwrapped.scene.entities["robot"]

    zs, tilts = [], []
    steps = int(args.seconds / 0.02)
    for i in range(steps):
        # force-fix the command (the manager resamples it from its range otherwise)
        term = unwrapped.command_manager.get_term("twist")
        term.command[:, 0] = args.lin_vel_x
        term.command[:, 1] = 0.0
        term.command[:, 2] = 0.0

        obs = obs_td["actor"].cpu().numpy()
        action = sess.run([oname], {iname: obs})[0]
        obs_td, *_ = env.step(torch.from_numpy(action).to("cuda:0"))

        if i % 5 == 0:
            pos = robot.data.body_link_pos_w  # (N, 15, 3); trunk_base = index 0
            quat = robot.data.body_link_quat_w  # (N, 15, 4) [w,x,y,z]
            zs.append(float(pos[:, 0, 2].mean().item()))
            tilt = 2 * torch.arctan2(
                torch.hypot(quat[:, 0, 1], quat[:, 0, 2]),
                torch.abs(quat[:, 0, 3]) + 1e-9,
            )
            tilts.append(float(torch.rad2deg(tilt).mean()))

    z_mean, tilt_mean = float(np.mean(zs)), float(np.mean(tilts))
    ok = z_mean > 0.10 and tilt_mean < 50
    print(f"=== env_onnx_eval: {args.onnx}  cmd vx={args.lin_vel_x} ===")
    print(f"  trunk_z_mean  = {z_mean*1000:.1f} mm  (stand reference ~117 mm)")
    print(f"  tilt_mean     = {tilt_mean:.1f} deg")
    print(f"  STANDED_OK    = {'YES' if ok else 'NO'}")


if __name__ == "__main__":
    main()
