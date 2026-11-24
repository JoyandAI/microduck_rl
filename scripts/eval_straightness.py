"""Evaluate an exported walking policy in its registered training environment."""

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

import mjlab_microduck.tasks  # Register the robot tasks.


def root_pose(robot):
    xy = robot.data.root_link_pos_w[:, :2].cpu().numpy().copy()
    w, x, y, z = robot.data.root_link_quat_w.cpu().numpy().T
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return xy, yaw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--task", default="Mjlab-Velocity-Flat-Backlash-OpenMD")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=10.0, help="Measurement duration after warmup.")
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--lin-vel-x", type=float, default=0.2)
    parser.add_argument("--lin-vel-y", type=float, default=0.0)
    parser.add_argument("--ang-vel-z", type=float, default=0.0)
    parser.add_argument("--freeze-pose-cmd", action="store_true", help="Hold the head and body commands at zero.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--json", type=Path, help="Write the summary and all individual trajectories' metrics.")
    args = parser.parse_args()
    command = (args.lin_vel_x, args.lin_vel_y, args.ang_vel_z)
    if not all(map(math.isfinite, (*command, args.warmup, args.seconds))):
        parser.error("Velocity commands and time intervals must be finite.")
    if args.num_envs < 1 or args.seconds <= 0 or args.warmup < 0:
        parser.error("Use at least one environment, a positive duration and a nonnegative warmup.")

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    actor_input, = session.get_inputs()
    actor_output, = session.get_outputs()
    if (len(actor_input.shape) != 2 or actor_input.shape[1] != 61
            or len(actor_output.shape) != 2 or actor_output.shape[1] != 14):
        parser.error("Expected a 61-observation, 14-action ONNX actor.")
    batch = actor_input.shape[0]
    if isinstance(batch, int) and batch not in (1, args.num_envs):
        parser.error(f"This ONNX actor requires --num-envs {batch}.")

    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs, cfg.seed = args.num_envs, args.seed
    cfg.events.pop("push_robot", None)
    cfg.curriculum.pop("standing_envs", None)
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x, twist.ranges.lin_vel_y, twist.ranges.ang_vel_z = [(v, v) for v in command]
    twist.heading_command = False
    twist.ranges.heading = None
    twist.rel_standing_envs = twist.rel_heading_envs = twist.rel_world_envs = twist.rel_forward_envs = 0.0
    twist.rel_turn_in_place_envs = 0.0
    twist.resampling_time_range = (1e9, 1e9)
    if args.freeze_pose_cmd:
        for name in ("head_pose", "body_pose"):
            cfg.curriculum.pop(f"{name}_range", None)
            pose = cfg.commands[name]
            pose.ranges = tuple((0.0, 0.0) for _ in pose.ranges)
            pose.zero_command_prob = 1.0

    dt = cfg.sim.mujoco.timestep * cfg.decimation
    warmup_steps, steps = round(args.warmup / dt), round(args.seconds / dt)
    if steps < 1:
        parser.error(f"--seconds must cover at least one control step ({dt:g} s).")
    cfg.episode_length_s = max(cfg.episode_length_s, (warmup_steps + steps + 2) * dt)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=args.device), clip_actions=None)
    try:
        obs, _ = env.reset()
        robot = env.unwrapped.scene.entities["robot"]
        expected_cmd = torch.as_tensor(command, device=env.unwrapped.device)
        alive = np.ones(args.num_envs, dtype=bool)
        recorded = np.zeros(args.num_envs, dtype=int)
        along, across, heading = np.zeros((3, args.num_envs), dtype=float)
        for step in range(warmup_steps + steps):
            if not torch.allclose(obs["actor"][:, 48:51], expected_cmd.expand(args.num_envs, -1), atol=1e-5):
                raise ValueError("The actor observations do not contain the requested fixed twist.")
            if step == warmup_steps:
                last_xy, last_yaw = root_pose(robot)
                forward = np.column_stack((np.cos(last_yaw), np.sin(last_yaw)))
                lateral = np.column_stack((-np.sin(last_yaw), np.cos(last_yaw)))
            inputs = obs["actor"].cpu().numpy()
            if batch == 1 and args.num_envs > 1:
                actions = np.concatenate([session.run([actor_output.name], {actor_input.name: row[None, :]})[0]
                                          for row in inputs])
            else:
                actions, = session.run([actor_output.name], {actor_input.name: inputs})
            if not np.isfinite(actions).all():
                raise ValueError("The actor produced non-finite actions.")
            obs, _, done, _ = env.step(torch.from_numpy(actions).to(env.unwrapped.device))
            alive &= ~done.cpu().numpy().astype(bool)
            if step >= warmup_steps:
                xy, yaw = root_pose(robot)
                delta = xy[alive] - last_xy[alive]
                along[alive] += np.sum(delta * forward[alive], axis=1)
                across[alive] += np.sum(delta * lateral[alive], axis=1)
                turn = yaw[alive] - last_yaw[alive]
                heading[alive] += np.arctan2(np.sin(turn), np.cos(turn))
                recorded[alive] += 1
                last_xy, last_yaw = xy, yaw
            if not alive.any():
                break
    finally:
        env.close()

    rows = []
    for i, count in enumerate(recorded):
        t = float(count * dt)
        rows.append(dict(env=i, measured_s=t, survived=bool(alive[i]),
                         forward_m=float(along[i]), lateral_m=float(across[i]),
                         heading_change_deg=float(np.degrees(heading[i])),
                         forward_m_s=float(along[i] / t) if count >= steps / 2 else None))
    speeds = [r["forward_m_s"] for r in rows if r["forward_m_s"] is not None]
    summary = dict(task=args.task, seed=args.seed, num_envs=args.num_envs,
                   command=dict(vx=command[0], vy=command[1], wz=command[2]), freeze_pose_cmd=args.freeze_pose_cmd,
                   warmup_s=warmup_steps * dt, window_s=steps * dt, num_survived=int(alive.sum()),
                   median_measured_s=float(np.median(recorded) * dt), num_speed_samples=len(speeds),
                   median_forward_m_s=float(np.median(speeds)) if speeds else None)
    print(json.dumps(summary, indent=2, allow_nan=False))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({**summary, "per_env": rows}, indent=2, allow_nan=False) + "\n",
                             encoding="utf-8")


if __name__ == "__main__":
    main()
