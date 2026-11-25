#!/usr/bin/env python3
"""Per-term reward accounting for an exported ONNX policy, in the faithful env.

Why this exists: `eval_onnx_bam.py` says whether the robot walks; wandb says
what the SHAPED reward was, but only while training (and only as episode sums
divided by episode length). To decide whether a posture defect is "priced"
you need the *raw* value of each reward term under the current policy — i.e.
what the term would pay at weight 1.0 — versus the term's actual weighted
contribution to the total.

That comparison is what exposed the 2026-09-11 finding: mjlab's `pose` term
returns exp(-mean_j(err_j^2/std_j^2)) over ten leg joints, so during walking the
ankle term alone (~10.9) saturates it; it contributes 0.0154 of a ~106/step
total (0.014%), and a 16 deg hip_yaw error inside that mean is worth ~0.0001.
No gradient -> the policy parked hip_yaw on its mechanical stop.

Usage:
    uv run python scripts/eval_reward_terms.py --onnx walk_hd1910_15000.onnx \
        --lin-vel-x 0.2 --seconds 12

    # force the hygiene terms on even though the run had them ramped in:
    uv run python scripts/eval_reward_terms.py --onnx walk_hd1910_15000.onnx \
        --no-curricula --hygiene-on --seconds 12
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

CTRL_DT = 0.02


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--task", default="Mjlab-Velocity-Flat-MicroDuck")
    ap.add_argument("--lin-vel-x", type=float, default=0.2)
    ap.add_argument("--lin-vel-y", type=float, default=0.0)
    ap.add_argument("--ang-vel-z", type=float, default=0.0)
    ap.add_argument("--warmup", type=float, default=2.0)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--params-json", default=None)
    ap.add_argument("--no-curricula", action="store_true",
                    help="drop all curricula so reward weights stay as configured")
    ap.add_argument("--hygiene-on", action="store_true",
                    help="set the posture/actuator-hygiene weights to their target "
                         "values (they are 0 at step 0 and ramped by a curriculum)")
    ap.add_argument("--weight", action="append", default=[],
                    metavar="NAME=VALUE", help="override a reward weight (repeatable)")
    ap.add_argument("--head-cmd", default="zero", choices=("zero", "train"))
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg
    import onnxruntime as ort

    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = 1
    cfg.events.pop("push_robot", None)

    tcfg = cfg.commands["twist"]
    tcfg.resampling_time_range = (1e9, 1e9)
    for attr, val in (("rel_standing_envs", 0.0), ("rel_heading_envs", 0.0)):
        if hasattr(tcfg, attr):
            setattr(tcfg, attr, val)
    if getattr(tcfg, "heading_command", False):
        tcfg.heading_command = False
        if hasattr(tcfg, "ranges") and hasattr(tcfg.ranges, "heading"):
            tcfg.ranges.heading = None
    if args.head_cmd == "zero":
        cfg.commands["head_pose"].resampling_time_range = (1e9, 1e9)

    if args.no_curricula:
        cfg.curriculum.clear()
        print("[i] curricula removed")

    # Ensure the hygiene terms exist even on a task cfg that predates them.
    if args.hygiene_on:
        from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
            ACTION_OVER_LIMIT_OVERSHOOT,
            ACTION_OVER_LIMIT_WEIGHT,
            NECK_MOTION_DEADBAND_RAD_S,
            NECK_MOTION_WEIGHT,
            STANCE_HIP_YAW_L1_DEADBAND,
            STANCE_HIP_YAW_L1_WEIGHT,
        )
        overrides = {
            "stance_hip_yaw": STANCE_HIP_YAW_L1_WEIGHT,
            "action_over_limit": ACTION_OVER_LIMIT_WEIGHT,
            "neck_motion": NECK_MOTION_WEIGHT,
        }
        missing = [k for k in overrides if k not in cfg.rewards]
        if missing:
            raise SystemExit(
                f"--hygiene-on requested but these reward terms are not in the task "
                f"cfg: {missing}.\n"
                f"Since 2026-09-17 the DEFAULT recipe is the official one and the "
                f"stance-hygiene stack is opt-in — re-run with "
                f"MICRODUCK_STANCE_HYGIENE=1 in the environment (see the OFFICIAL "
                f"ALIGNMENT block in microduck_velocity_env_cfg.py)."
            )
        for k, v in overrides.items():
            cfg.rewards[k].weight = v
        cfg.rewards["action_over_limit"].params["overshoot"] = ACTION_OVER_LIMIT_OVERSHOOT
        cfg.rewards["neck_motion"].params["deadband"] = NECK_MOTION_DEADBAND_RAD_S
        cfg.rewards["stance_hip_yaw"].params["deadband"] = STANCE_HIP_YAW_L1_DEADBAND
        print(f"[i] hygiene weights forced on: {overrides}")

    for spec in args.weight:
        name, _, val = spec.partition("=")
        name = name.strip()
        if name not in cfg.rewards:
            raise SystemExit(f"unknown reward term: {name}")
        cfg.rewards[name].weight = float(val)
        print(f"[i] weight override {name} = {val}")

    if args.params_json:
        act = cfg.scene.entities["robot"].articulation.actuators[0]
        object.__setattr__(act, "json_path", args.params_json)
        object.__setattr__(act, "motor_name", None)
        object.__setattr__(act, "model", None)
        object.__setattr__(act, "_resolved_json_path", args.params_json)

    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    env = RslRlVecEnvWrapper(env, clip_actions=None)
    unwrapped = env.unwrapped
    rm = unwrapped.reward_manager

    sess = ort.InferenceSession(args.onnx)
    iname, oname = sess.get_inputs()[0].name, sess.get_outputs()[0].name
    assert sess.get_inputs()[0].shape[1] == 61, sess.get_inputs()[0].shape

    term = unwrapped.command_manager.get_term("twist")
    hterm = unwrapped.command_manager.get_term("head_pose")

    obs_td = env.reset()[0]
    n_warm = int(round(args.warmup / CTRL_DT))
    n = int(round(args.seconds / CTRL_DT))

    weighted = defaultdict(list)
    total = []
    names = list(rm.active_terms)
    for i in range(n_warm + n):
        term.command[:, 0] = args.lin_vel_x
        term.command[:, 1] = args.lin_vel_y
        term.command[:, 2] = args.ang_vel_z
        if args.head_cmd == "zero":
            hterm.command[:] = 0.0
        obs = obs_td["actor"].cpu().numpy()
        action = sess.run([oname], {iname: obs})[0].astype(np.float32)
        obs_td, rew, dones, _ = env.step(torch.from_numpy(action).to(args.device))
        if i < n_warm:
            continue
        sr = rm._step_reward[0].detach().cpu().numpy()
        for j, nm in enumerate(names):
            weighted[nm].append(float(sr[j]))
        total.append(float(rew[0]))

    # Recover the configured weights as they stand at the END of the rollout
    # (curricula mutate them), so raw = weighted / weight is accurate.
    cur_w = {nm: float(rm.get_term_cfg(nm).weight) for nm in names}

    rows = []
    for nm in names:
        w = cur_w[nm]
        vals = np.array(weighted[nm])
        mean_w = float(vals.mean())
        raw = mean_w / w if w != 0.0 else float("nan")
        rows.append((nm, w, mean_w, raw))

    tot = float(np.mean(total))
    # `tot` is the dt-SCALED reward the wrapper returns (mjlab multiplies by the
    # control dt), while _step_reward holds unscaled per-term values. Use the
    # unscaled sum for shares so the percentages mean something.
    tot_unscaled = sum(r[2] for r in rows)
    print("\n" + "=" * 92)
    print(f"REWARD TERMS  onnx={args.onnx}")
    print(f"  cmd vx={args.lin_vel_x:+.2f} vy={args.lin_vel_y:+.2f} wz={args.ang_vel_z:+.2f}  "
          f"head_cmd={args.head_cmd}  ({n} steps, {n*CTRL_DT:.1f}s)")
    print(f"  TOTAL reward/step (as returned, dt-scaled) = {tot:+.5f}")
    print(f"  TOTAL reward per second (sum of terms)     = {tot_unscaled:+.5f}")
    print("=" * 92)
    print(f"  {'term':<28} {'weight':>9} {'per-second':>14} {'share %':>9} {'raw value':>12}")
    print("  " + "-" * 76)
    for nm, w, mean_w, raw in sorted(rows, key=lambda r: -abs(r[2])):
        share = 100.0 * mean_w / tot_unscaled if tot_unscaled != 0 else 0.0
        raws = "       n/a" if w == 0.0 else f"{raw:12.5f}"
        print(f"  {nm:<28} {w:9.4f} {mean_w:14.5f} {share:8.3f}% {raws}")
    pos = sum(r[2] for r in rows if r[2] > 0)
    neg = sum(r[2] for r in rows if r[2] < 0)
    print("  " + "-" * 76)
    print(f"  {'positive mass':<28} {'':>9} {pos:14.5f}")
    print(f"  {'negative mass':<28} {'':>9} {neg:14.5f}")
    bad = [r[0] for r in rows if "penalt" in r[0] or r[0] in
           ("action_rate_l2", "body_ang_vel", "angular_momentum", "foot_clearance",
            "foot_swing_height", "foot_slip", "self_collisions", "dof_pos_limits",
            "stance_hip_yaw", "stance_hip_roll", "neck_motion", "action_over_limit")]
    print("\n  sign audit (AGENTS.md: every penalty term must be <= 0):")
    for nm in bad:
        r = next((x for x in rows if x[0] == nm), None)
        if r is None:
            continue
        flag = "OK " if r[2] <= 0 else "!!!"
        print(f"    {flag} {nm:<28} weighted/step = {r[2]:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
