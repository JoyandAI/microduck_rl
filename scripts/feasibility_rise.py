#!/usr/bin/env python3
"""Feasibility test: can the robot physically RISE at all under a given actuator param set?

Why: a policy that never rises may be (a) badly rewarded, or (b) trying to do something the
actuator cannot do. Distinguishing those decides whether retraining is worth anything.

Method: in the REAL training env (BAM, pushes off), settle the robot SEATED, then command
STAND and drive the joints with a trivial controller that is known-good by construction:
`action = 0` means "servo target = HOME (the standing pose)", because mjlab's position action
is `target = default_pose + action * scale` and `default_pose` IS the HOME/stand pose. If the
trunk cannot climb to standing height under that, the actuator/task is infeasible and no
reward shaping will help. A scripted stand-pose hold like this is the natural upper bound on
what any policy could achieve.

Usage:
    uv run python scripts/feasibility_rise.py --params-json /tmp/policy_forensics/params_recommended.json
    uv run python scripts/feasibility_rise.py                       # whatever params are installed
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg

CTRL_DT = 0.02  # 50 Hz


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Mjlab-SitStand-Flat-MicroDuck")
    ap.add_argument("--params-json", default=None)
    ap.add_argument("--sit-seconds", type=float, default=4.0, help="settle seated first")
    ap.add_argument("--rise-seconds", type=float, default=6.0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = 1
    cfg.events.pop("push_robot", None)
    cfg.curriculum.pop("push_magnitude", None)
    tcfg = cfg.commands["twist"]
    tcfg.resampling_time_range = (1e9, 1e9)

    # Start from a SEATED state on every reset, otherwise the "rise" test begins from
    # whatever the 50/50 spawn mix gave us (often a prone flop) and measures nothing.
    if "set_ground_state" in cfg.events:
        pr = cfg.events["set_ground_state"].params
        pr["sitting_prob"] = 1.0
        pr["standing_prob"] = 0.0
        pr["face_down_prob"] = 0.0
        pr["face_up_prob"] = 0.0
        print("[i] spawn mix forced to SEATED")

    if args.params_json:
        act = cfg.scene.entities["robot"].articulation.actuators[0]
        object.__setattr__(act, "json_path", args.params_json)
        object.__setattr__(act, "motor_name", None)
        object.__setattr__(act, "model", None)
        object.__setattr__(act, "_resolved_json_path", args.params_json)
        print(f"[i] BAM params -> {args.params_json}")

    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    env = RslRlVecEnvWrapper(env, clip_actions=None)
    unwrapped = env.unwrapped
    robot = unwrapped.scene.entities["robot"]
    term = unwrapped.command_manager.get_term("twist")
    n_act = env.num_actions

    obs = env.reset()[0]

    def run(flag: float, action: np.ndarray, seconds: float, label: str):
        zs, tilts = [], []
        for _ in range(int(round(seconds / CTRL_DT))):
            term.command[:, 0] = flag
            a = torch.as_tensor(action, dtype=torch.float32, device=args.device)
            obs, _, _, _ = env.step(a.unsqueeze(0))
            pos = robot.data.body_link_pos_w[0, 0]
            q = robot.data.body_link_quat_w[0, 0]
            w, x, y, z = (float(v) for v in q)
            zs.append(float(pos[2]))
            tilts.append(math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y))))))
        print(f"  {label:<34} z: mean={np.mean(zs):.3f} min={np.min(zs):.3f} max={np.max(zs):.3f}"
              f"   tilt_mean={np.mean(tilts):5.1f}deg")
        return np.array(zs), np.array(tilts)

    print(f"\n=== feasibility: sit -> STAND, scripted 'target = HOME' (action = 0) ===")
    print(f"  task={args.task}  params={'installed' if not args.params_json else args.params_json}")

    # Spawn is forced seated; let it settle under ZERO action (servo target = HOME) while the
    # flag says SIT. Zero action is the "hold the standing pose" command, so this phase is a
    # strong pull toward standing -- if the robot still ends up seated/settled here, the
    # seated start is real. Report the settled state so the rise phase is interpretable.
    z_seat, tilt_seat = run(1.0, np.zeros(n_act, dtype=np.float32), args.sit_seconds,
                            "settle (spawn SEATED, flag=SIT, action=0)")
    print(f"     -> seated start check: z={z_seat[-1]:.3f} m, tilt={tilt_seat[-1]:.1f} deg"
          f"   (want z~0.06, tilt<25 for a valid seated start)")
    z_rise, tilt_rise = run(0.0, np.zeros(n_act, dtype=np.float32), args.rise_seconds,
                            "RISE attempt (flag=STAND, action=0)")

    print()
    print(f"  z after seated phase : {z_seat[-1]:.3f} m")
    print(f"  z at end of rise     : {z_rise[-1]:.3f} m   (peak {z_rise.max():.3f})")
    print(f"  standing height ref  : 0.115 m ; seated ref 0.060 m")
    feasible = z_rise.max() >= 0.100
    print(f"  FEASIBLE (can reach >=0.100 m by targeting HOME): {'YES' if feasible else 'NO'}")
    print()
    print("  Reading: if NO, the actuator/task cannot stand the robot even with a perfect")
    print("  'target = standing pose' command -> retraining under these params is pointless")
    print("  until the actuator params are fixed. If YES, a failure to rise is a policy/reward")
    print("  problem and retraining is worth doing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
