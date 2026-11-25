#!/usr/bin/env python3
"""Is the hesitant sit a CAPABILITY limit or a POLICY CHOICE?

The open question after `docs/q1_q2_root_cause_2026-09-16.md` §5.1f. Measured
facts: the head-ground-fix policy, in ~1/3 of seeded draws, holds z ≈ 0.094 at a
constant ~20° forward lean for 5-10 s after the sit command, and in some draws
never descends inside 12 s. The pre-fix policy descends 0.106 -> 0.055 in 1-2 s in
24/24 draws. The two new reward terms are gated OFF while the sit flag is set, so
the mechanism is behavioural, and the "collapsed multiplicative head factor
removes the descent gradient" explanation was RETRACTED (§5.1f) because three
non-multiplicative height terms (weights 1.0 / 1.0 / 6.0) already price descent.

This probe separates the two remaining possibilities, which have opposite fixes:

  * CAPABILITY — the policy IS commanding a descent and the robot cannot execute
    it (torque limit, kinematic bind, foot configuration). Signal: |action| large
    and/or SATURATED while z does not fall, and the leg joints not changing.
    Fixing that is a physics/pose problem, not a reward weight.
  * POLICY CHOICE — the policy is holding a pose on purpose (the descent is not
    worth its cost). Signal: |action| SMALL and steady during the hover, i.e. the
    controller is happily parked, then a discrete switch at the moment it drops.

It prints, per step of the SIT segment, the trunk z / tilt, |action| stats, the
action on the leg joints that must extend to lower the trunk, and the BLOCKING
flag: whether any servo action sits at the actuator's own limit.

Usage:
    uv run python scripts/probe_sit_hover.py --onnx /tmp/sit_6000.onnx --seed 2
    uv run python scripts/probe_sit_hover.py --onnx /tmp/sit_6000.onnx --seed 0  # a fast draw
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

CTRL_DT = 0.02
LEG_JOINTS = ("hip_pitch", "knee", "ankle")   # the repo's actual names; "ankle_pitch" does not exist


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--task", default="Mjlab-SitStand-Flat-MicroDuck")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--sit-schedule", default="0:3,1:12,0:4")
    ap.add_argument("--warmup", type=float, default=2.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--every", type=int, default=25, help="print every N steps (25 = 0.5 s)")
    args = ap.parse_args()

    schedule = [(int(f), float(d)) for f, d in
                (s.split(":") for s in args.sit_schedule.split(","))]

    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    # MUST MATCH eval_sitstand_head.py's value EXACTLY. This env has
    # reverse-curriculum spawns keyed on episode progress, so a different
    # `episode_length_s` samples a different SPAWN MIX -- setting it to 120 here
    # (vs the evaluator's 60) spawned a fallen robot at t=0 and the probe then
    # "measured" a fall while the evaluator's timeline showed a clean stand.
    # Same artifact class as "an evaluator that hardcodes physics silently
    # measures the wrong machine": the probe must reproduce the evaluator's env
    # before its numbers can be compared with the evaluator's.
    if 60.0 > cfg.episode_length_s:
        cfg.episode_length_s = 60.0
    cfg.events.pop("push_robot", None)
    cfg.curriculum.pop("push_magnitude", None)
    tcfg = cfg.commands["twist"]
    tcfg.resampling_time_range = (1e9, 1e9)
    for a in ("rel_standing_envs", "rel_heading_envs"):
        if hasattr(tcfg, a):
            setattr(tcfg, a, 0.0)

    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=args.device), clip_actions=None)
    unwrapped = env.unwrapped
    robot = unwrapped.scene.entities["robot"]
    sess = ort.InferenceSession(args.onnx)
    iname, oname = sess.get_inputs()[0].name, sess.get_outputs()[0].name

    servo_ids = mdp_mod._servo_joint_ids(unwrapped, robot)
    names = [robot.joint_names[i] for i in servo_ids]
    leg_idx = [names.index(n) for n in LEG_JOINTS if n in names]
    print(f"# act={args.onnx}  seed={args.seed}")
    print(f"# leg action channels: {[names[i] for i in leg_idx]}")

    obs_td = env.reset()[0]
    term = unwrapped.command_manager.get_term("twist")

    def step(flag):
        term.command[:, 0] = float(flag)
        obs = obs_td["actor"].cpu().numpy()
        a = sess.run([oname], {iname: obs})[0].astype(np.float32)
        out = env.step(torch.from_numpy(a).to(args.device))
        return a[0], out

    # WARMUP. The observation MUST be fed back every step: the first version of
    # this probe discarded `step()`'s return value here, so the policy saw ONE
    # frozen frame for all 100 warmup steps while the robot moved underneath it --
    # and it fell, which the probe then reported as the policy's behaviour. Any
    # rollout harness that does not close the obs loop measures a different
    # controller than the one deployed.
    for _ in range(int(round(args.warmup / CTRL_DT))):
        _, (obs_td, _, _, _) = step(schedule[0][0])

    t = 0.0
    print(f"{'t':>6}{'z':>8}{'tilt':>7}{'|a|mean':>9}{'|a|max':>8}{'n_sat':>7}"
          + "".join(f"{n[:8]+' q':>11}" for n in [names[i] for i in leg_idx])
          + "".join(f"{n[:8]+' a':>11}" for n in [names[i] for i in leg_idx])
          + f"{'vz':>9}")
    prev_z = None
    for si, (flag, dur) in enumerate(schedule):
        for k in range(int(round(dur / CTRL_DT))):
            a, (obs_td, _, dones, _) = step(flag)
            t += CTRL_DT
            z = float(robot.data.body_link_pos_w[0, 0, 2])
            jp = robot.data.joint_pos[0, servo_ids].cpu().numpy()
            vz = float(robot.data.root_link_lin_vel_w[0, 2])
            q = robot.data.body_link_quat_w[0, 0]
            w, x, y, zz = (float(v) for v in q)
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))
            # Saturation: the actuator ctrlrange is deliberately wide (low-kp
            # servos need overshoot), so |a| > 1 is NOT saturation by itself.
            # Count how many channels sit at the limit of the OBSERVED range
            # instead -- a policy pinned against its own output bound is the
            # capability signature we are looking for.
            n_sat = int((np.abs(a) > 2.5).sum())
            dz = (z - prev_z) / args.every if (prev_z is not None and k % args.every == 0) else float("nan")
            if k % args.every == 0:
                print(f"{t:>6.2f}{z:>8.3f}{tilt:>7.1f}{np.abs(a).mean():>9.3f}{np.abs(a).max():>8.3f}"
                      f"{n_sat:>7d}"
                      + "".join(f"{float(jp[i]):>11.3f}" for i in leg_idx)
                      + "".join(f"{a[i]:>11.3f}" for i in leg_idx)
                      + f"{float(vz):>9.4f}")
            if dz == dz:
                prev_z = z
            if bool(dones[0]):
                print(f"{t:>6.2f}   RESET")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
