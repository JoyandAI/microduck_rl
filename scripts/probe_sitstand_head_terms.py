#!/usr/bin/env python3
"""Why do `head_pose_tracking` and `posture_composite`'s head factor disagree?

They are two Gaussians over the SAME head error, differing only in std (0.5 vs
0.40), so `tracking = exp(-e/0.25)` and `head_factor = exp(-e/0.16)` where `e` is
the same mean squared error. Those two numbers cannot be 0.656 and 0.022 at once.

Measured on `sitstand_headfix` ck5000 (STAND), `scripts/eval_reward_terms.py`:
    head_pose_tracking raw = 0.656
    posture_composite raw  = 0.0133   -> with the other three factors at
                                        0.895/0.943/0.717 that implies
                                        head_factor ~= 0.022
Both cannot be right, and the answer decides whether the sit-stand head needs a
NEW penalty term or whether the existing ones simply need more iterations -- i.e.
whether to spend another training run.

This probe therefore does not infer anything: it builds the real env, drives the
real exported ONNX, and prints, per step, the command, the actual joint angles,
and the two functions' RETURN VALUES side by side together with the error each
one implies.

Usage:
    uv run python scripts/probe_sitstand_head_terms.py --onnx out.onnx
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

from mjlab_microduck.tasks import mdp as microduck_mdp

CTRL_DT = 0.02
HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--task", default="Mjlab-SitStand-Flat-MicroDuck")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--warmup", type=float, default=2.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--params-json", default=None)
    args = ap.parse_args()

    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = 1
    cfg.events.pop("push_robot", None)
    cfg.curriculum.pop("push_magnitude", None)
    tcfg = cfg.commands["twist"]
    tcfg.resampling_time_range = (1e9, 1e9)
    for a in ("rel_standing_envs", "rel_heading_envs"):
        if hasattr(tcfg, a):
            setattr(tcfg, a, 0.0)
    if getattr(args, "params_json", None):
        act = cfg.scene.entities["robot"].articulation.actuators[0]
        for k, v in (("json_path", args.params_json), ("motor_name", None),
                     ("model", None), ("_resolved_json_path", args.params_json)):
            object.__setattr__(act, k, v)
    cfg.episode_length_s = 1e6

    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=args.device),
                             clip_actions=None)
    u = env.unwrapped
    robot = u.scene.entities["robot"]
    ids, _ = robot.find_joints(list(HEAD_JOINTS), preserve_order=True)
    neck_ids = torch.tensor(ids, device=u.device, dtype=torch.long)
    # share the id cache with the reward functions so they measure the same view
    u._head_pose_neck_ids = neck_ids
    u._head_pose_bl_ids = neck_ids
    u._head_pose_bl_mask = torch.zeros(len(ids), device=u.device)

    sess = ort.InferenceSession(args.onnx)
    iname, oname = sess.get_inputs()[0].name, sess.get_outputs()[0].name
    obs_td = env.reset()[0]
    term = u.command_manager.get_term("twist")
    term.command[:, 0] = 0.0                       # STAND

    print(f"probe: {args.onnx}")
    print(f"{'t':>5}{'hcmd':>26}{'actual-HOME':>26}{'neck_err':>9}"
          f"{'track':>8}{'comp':>8}{'implied_e_track':>16}{'implied_e_head':>15}")
    rows = []
    n = int(round((args.warmup + args.seconds) / CTRL_DT))
    for i in range(n):
        term.command[:, 0] = 0.0
        obs = obs_td["actor"].cpu().numpy()
        act = sess.run([oname], {iname: obs})[0].astype(np.float32)
        obs_td, _, _, _ = env.step(torch.from_numpy(act).to(args.device))
        if i * CTRL_DT < args.warmup:
            continue

        cmd = u.command_manager.get_command("head_pose")[0].detach()
        rel = (robot.data.joint_pos[0, neck_ids]
               - robot.data.default_joint_pos[0, neck_ids]).detach()
        err = rel - cmd
        mean_sq = float((err ** 2).mean())
        track = float(microduck_mdp.head_pose_tracking(
            u, command_name="head_pose", std=0.5)[0])
        comp = float(microduck_mdp.posture_composite(
            u, command_name="twist",
            sit_overrides={}, joint_indices=[0], sit_z=0.060, stand_z=0.115,
            head_std=0.40)[0])
        e_track = -0.25 * math.log(max(track, 1e-12))
        e_head = -0.16 * math.log(max(comp, 1e-12))
        rows.append((mean_sq, e_track, e_head))
        if i % 50 == 0:
            print(f"{i*CTRL_DT:>5.2f}"
                  f"{np.array2string(cmd.cpu().numpy(), precision=3):>26}"
                  f"{np.array2string(rel.cpu().numpy(), precision=3):>26}"
                  f"{float(rel[0]-cmd[0]):>9.3f}{track:>8.4f}{comp:>8.5f}"
                  f"{e_track:>16.4f}{e_head:>15.4f}")

    a = np.array(rows)
    print()
    print(f"  over {len(rows)} steps (means):")
    print(f"    mean(err^2) from the joints themselves : {a[:,0].mean():.4f}")
    print(f"    implied by head_pose_tracking (std .5) : {a[:,1].mean():.4f}")
    print(f"    implied by posture_composite  (std .4) : {a[:,2].mean():.4f}")
    print()
    print("  If the joint-derived value matches NEITHER, one of the two functions is")
    print("  not measuring the error this probe assumes (cache/backlash view, or a")
    print("  different command source); if it matches ONE, the other has the bug.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
