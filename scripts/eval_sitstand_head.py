#!/usr/bin/env python3
"""Faithful BAM eval of a sitstand policy: does the head droop, and WHEN?

Motivation (user observation 2026-09-11): "头是在坐下的时候不下垂，但是起立的时候
还是下垂的" — the head holds fine while seated but sags once the robot stands up.

The suspect is the upright gate on `head_pose_bias`: with
`gate_height_low=0.09 / gate_height_high=0.11` and SIT_Z=0.060 / STAND_Z=0.115, the
penalty is OFF while seated and only ramps in above 0.09 m — and the tilt gate
(full at 20deg, zero at 45deg) exempts a leaning rise. If the policy learned to
keep the head low through the rise, the post-rise head must be lifted by the
penalty alone, which may not be enough.

This script drives the posture flag on a schedule in the REAL training env (BAM,
pushes off, command frozen) and prints, per phase:
  trunk z, tilt, head joints (relative to HOME) and the head tracking error.

Usage:
    uv run python scripts/eval_sitstand_head.py \
        --onnx sitstand_hd1910_15000.onnx --sit-schedule "0:3,1:4,0:4"
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict

import numpy as np
import onnxruntime as ort
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg

from mjlab_microduck.tasks import mdp as mdp_mod

CTRL_DT = 0.02  # 50 Hz
HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")


def parse_schedule(s: str) -> list[tuple[int, float]]:
    out = []
    for seg in s.split(","):
        flag, dur = seg.split(":")
        out.append((int(flag), float(dur)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--task", default="Mjlab-SitStand-Flat-MicroDuck")
    ap.add_argument("--sit-schedule", default="0:3,1:4,0:4",
                    help="comma-separated flag:seconds (0=stand, 1=sit)")
    ap.add_argument("--warmup", type=float, default=2.0)
    ap.add_argument("--episode-length-s", type=float, default=60.0,
                    help="override episode_length_s so the schedule cannot be cut off "
                         "by a timeout reset (default task value is 12 s, which silently "
                         "turned the last phase's steady window into a teleported spawn)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--params-json", default=None,
                    help="explicit BAM params JSON (e.g. the snapshot the policy was trained with)")
    args = ap.parse_args()

    schedule = parse_schedule(args.sit_schedule)

    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = 1
    total_s = sum(d for _, d in schedule)
    if args.episode_length_s > cfg.episode_length_s:
        print(f"[i] episode_length_s {cfg.episode_length_s} -> {args.episode_length_s}")
        cfg.episode_length_s = args.episode_length_s
    if args.warmup + total_s >= cfg.episode_length_s:
        raise SystemExit(
            f"schedule too long for the episode: warmup {args.warmup}s + {total_s}s >= "
            f"episode {cfg.episode_length_s}s -> a timeout reset would corrupt the last "
            f"phase. Raise --episode-length-s.")
    # Drop pushes AND the curriculum that drives them — a curriculum term that
    # references a removed event raises at env.reset().
    cfg.events.pop("push_robot", None)
    cfg.curriculum.pop("push_magnitude", None)
    tcfg = cfg.commands["twist"]
    tcfg.resampling_time_range = (1e9, 1e9)
    for a in ("rel_standing_envs", "rel_heading_envs"):
        if hasattr(tcfg, a):
            setattr(tcfg, a, 0.0)


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

    servo_ids = mdp_mod._servo_joint_ids(unwrapped, robot)
    joint_names = [robot.joint_names[i] for i in servo_ids]
    head_idx = [joint_names.index(n) for n in HEAD_JOINTS]

    obs_td = env.reset()[0]
    term = unwrapped.command_manager.get_term("twist")
    head_cmd_term = unwrapped.command_manager.get_term("head_pose")

    def step_once():
        obs = obs_td["actor"].cpu().numpy()
        action = sess.run([oname], {iname: obs})[0].astype(np.float32)
        return env.step(torch.from_numpy(action).to(args.device))

    # warmup (keep the first schedule flag)
    first_flag = schedule[0][0]
    for _ in range(int(round(args.warmup / CTRL_DT))):
        term.command[:, 0] = float(first_flag)
        obs_td, *_ = step_once()

    home = robot.data.default_joint_pos[0, servo_ids].cpu().numpy()
    phases: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    timeline = []

    t = 0.0
    step_i = 0
    for si, (flag, dur) in enumerate(schedule):
        label = f"seg{si}_{'SIT' if flag else 'STAND'}"
        n_steps = int(round(dur / CTRL_DT))
        steady_from = int(n_steps * 0.7)  # last 30% of the segment = settled
        for k in range(n_steps):
            term.command[:, 0] = float(flag)
            obs_td, _, dones, _ = step_once()
            t += CTRL_DT
            step_i += 1

            pos = robot.data.body_link_pos_w[0, 0]
            quat = robot.data.body_link_quat_w[0, 0]
            w, x, y, z = (float(v) for v in quat)
            zax = 1.0 - 2.0 * (x * x + y * y)
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, zax))))

            jp = robot.data.joint_pos[0, servo_ids].cpu().numpy()
            rel = jp - home
            hcmd = head_cmd_term.command[0].cpu().numpy()
            err = rel[head_idx] - hcmd
            eff_cmd = float(obs_td["actor"][0, -13].item()) if obs_td["actor"].shape[1] == 61 else -1.0

            bucket = phases[label]
            bucket["z"].append(float(pos[2]))
            bucket["tilt"].append(tilt)
            bucket["neck_rel"].append(float(rel[head_idx[0]]))
            bucket["head_rel"].append(float(rel[head_idx[1]]))
            bucket["neck_abs"].append(float(jp[head_idx[0]]))
            bucket["err"].append(float(np.mean(np.abs(err))))
            bucket["hcmd_neck"].append(float(hcmd[0]))
            if k >= steady_from:
                for key in ("z", "tilt", "neck_rel", "head_rel", "err", "hcmd_neck"):
                    phases[label]["steady_" + key].append(bucket[key][-1])
            if step_i % 50 == 0:  # every 1.0 s — integer modulo, not float
                timeline.append((t, label, float(pos[2]), tilt,
                                 float(rel[head_idx[0]]), float(rel[head_idx[1]]),
                                 float(np.mean(np.abs(err))), eff_cmd, bool(dones[0])))

    print("\n=== sitstand head behaviour (faithful BAM env, no pushes) ===")
    print(f"  onnx    : {args.onnx}")
    print(f"  schedule: {schedule}   (0=STAND, 1=SIT)")
    print()
    hdr = f"{'phase':<14}{'n':>5}{'z_mean':>9}{'tilt_mean':>11}{'neck_rel':>10}{'head_rel':>10}{'track_err':>11}"
    print(hdr)
    print("-" * len(hdr))
    for label in (f"seg{i}_{'SIT' if f else 'STAND'}" for i, (f, _) in enumerate(schedule)):
        d = phases.get(label)
        if not d:
            continue
        print(f"{label:<14}{len(d['z']):>5}{np.mean(d['z']):>9.3f}{np.mean(d['tilt']):>11.2f}"
              f"{np.mean(d['neck_rel']):>10.3f}{np.mean(d['head_rel']):>10.3f}"
              f"{np.mean(d['err']):>11.4f}")
    print()
    print("  --- STEADY STATE ONLY (last 30% of each segment) ---")
    hdr2 = f"{'phase':<14}{'z':>9}{'tilt':>9}{'neck_rel':>10}{'head_rel':>10}{'neck_abs':>10}{'hcmd_neck':>11}{'err':>9}"
    print(hdr2)
    print("-" * len(hdr2))
    for label in (f"seg{i}_{'SIT' if f else 'STAND'}" for i, (f, _) in enumerate(schedule)):
        d = phases.get(label)
        if not d or not d.get("steady_z"):
            continue
        g = lambda k: np.mean(d["steady_" + k])  # noqa: E731
        print(f"{label:<14}{g('z'):>9.3f}{g('tilt'):>9.1f}{g('neck_rel'):>10.3f}"
              f"{g('head_rel'):>10.3f}{np.mean(d['neck_abs'][-len(d['steady_z']):]):>10.3f}"
              f"{g('hcmd_neck'):>11.3f}{g('err'):>9.4f}")
    print()
    print("  neck_abs = absolute neck_pitch [rad]; HOME = +0.3491 (= +20deg).")

    print()
    print("  HOME neck_pitch/head_pitch = +0.3491 rad each; rel<0 means the head is BELOW")
    print("  its HOME pose, i.e. drooping. track_err is the mean |actual-cmd| over the")
    print("  4 head joints (that is exactly what head_pose_bias penalises).")

    # timeline around the transitions, to see WHEN the head sags
    print("\n  --- timeline (every ~1.25 s) ---")
    print(f"  {'t[s]':>6}{'phase':>14}{'z':>8}{'tilt':>8}{'neck_rel':>10}{'head_rel':>10}{'err':>9}{'sitflg':>7}")
    last = -99.0
    for (t, label, z, tilt, nr, hr, err, eff, done) in timeline:
        if t - last < 0.99:
            continue
        last = t
        print(f"  {t:>6.2f}{label:>14}{z:>8.3f}{tilt:>8.1f}{nr:>10.3f}{hr:>10.3f}{err:>9.4f}"
              f"{eff:>7.1f}" + ("   <-- RESET" if done else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
