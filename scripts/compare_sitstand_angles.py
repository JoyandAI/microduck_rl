#!/usr/bin/env python3
"""Compare the 14 SERVO ANGLES at the standing rest: official vs ours.

Why (user feedback 2026-09-16): "站起来的头部还是向下垂的". The head-ground-force
metric said 0.000 N and the head BODY height said 144-161 mm, and both were
reported as passing -- but the acceptance bar for height was 140 mm, which this
script's author chose, and 144 mm is only 62 % of the nominal 232.6 mm HEAD
height. **A threshold picked to clear the current policy is not a measurement.**
The user is looking at the neck, and the neck had been parked on its hard stop
(-1.934 rad rel HOME) in every phase, at every checkpoint, in all 32 draws.

So this dumps what the eye sees: every servo angle at the standing rest, side by
side with the official `alpha_sitstand`, in the hardware era each was trained for.

The official policies are PRE-SWAP (XL330, kp_fw 200, 6.5-8.2 V, 737 g). Running
one on the current HD-1910 config measures a 27 %-weaker, 33 %-slower servo than
it was trained with -- the same "evaluator silently measures the wrong machine"
trap this repo has hit before -- so `preswap` is a per-policy flag here, not a
global one.

Usage:
    uv run python scripts/compare_sitstand_angles.py \
        --policy official:/home/joyandai/microduck/policies/alpha_sitstand.onnx:preswap \
        --policy ours_now:sitstand_hd1910_8000.onnx \
        --policy ours_fix:sitstand_headfix_7000.onnx \
        --seed 0
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

CTRL_DT = 0.02
HEAD = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
# The SIT-STAND task uses the ALLCOLLISIONS model (`MICRODUCK_ALLCOLLISIONS_XML`),
# not the walk model: its head/jaw collision geoms are what the head-tripod rests
# on. Swapping in `robot_walk_preswap.xml` (the walk-era file) changed the
# collision geometry under the policy and the official sit-stand simply FELL
# (measured trunk_z = -0.047, right_knee 1.445 rad off HOME) -- which read as
# "the official policy is worse" until the model was checked. Use the matching
# pre-swap ALLCOLLISIONS copy (byte-identical to the in-repo .bak_servoswap;
# the copy exists because MuJoCo picks its decoder from the extension and
# refuses `.bak_servoswap`).
PRESWAP_XML = "src/mjlab_microduck/robot/microduck/robot_allcollisions_preswap.xml"
XL330_JSON = "vendor/bam/bam/params/xl330/m6.json"


def apply_preswap(cfg) -> dict:
    import mujoco
    robot = cfg.scene.entities["robot"]
    object.__setattr__(robot, "spec_fn",
                       (lambda p=PRESWAP_XML: mujoco.MjSpec.from_file(p)))
    act = robot.articulation.actuators[0]
    object.__setattr__(act, "json_path", XL330_JSON)
    object.__setattr__(act, "motor_name", None)
    object.__setattr__(act, "model", None)
    object.__setattr__(act, "_resolved_json_path", XL330_JSON)
    object.__setattr__(act, "kp_fw", 200.0)
    object.__setattr__(act, "vin_range", (6.5, 8.2))
    object.__setattr__(act, "vin_min", 6.0)
    object.__setattr__(act, "vin_drop_resistance_range", (0.0, 0.2))
    object.__setattr__(act, "delay_min_lag", 3)
    object.__setattr__(act, "delay_max_lag", 6)
    return {"preswap": True}


def run_one(onnx: str, preswap: bool, schedule, warmup: float, seed: int,
            device: str) -> dict:
    cfg = load_env_cfg("Mjlab-SitStand-Flat-MicroDuck", play=True)
    cfg.scene.num_envs = 1
    cfg.seed = seed
    cfg.episode_length_s = 60.0
    if preswap:
        apply_preswap(cfg)
    cfg.events.pop("push_robot", None)
    cfg.curriculum.pop("push_magnitude", None)
    t = cfg.commands["twist"]
    t.resampling_time_range = (1e9, 1e9)
    for a in ("rel_standing_envs", "rel_heading_envs"):
        if hasattr(t, a):
            setattr(t, a, 0.0)

    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=device), clip_actions=None)
    un = env.unwrapped
    robot = un.scene.entities["robot"]
    sess = ort.InferenceSession(onnx)
    iname, oname = sess.get_inputs()[0].name, sess.get_outputs()[0].name

    sid = mdp_mod._servo_joint_ids(un, robot)
    names = [robot.joint_names[i] for i in sid]
    hb, _ = robot.find_bodies("jaw_soft", preserve_order=True)
    head_id = int(hb[0])
    home = robot.data.default_joint_pos[0, sid].cpu().numpy()

    obs = env.reset()[0]
    term = un.command_manager.get_term("twist")

    def step(flag):
        nonlocal obs
        term.command[:, 0] = float(flag)
        a = sess.run([oname], {iname: obs["actor"].cpu().numpy()})[0].astype(np.float32)
        obs, _, dones, _ = env.step(torch.from_numpy(a).to(device))

    for _ in range(int(round(warmup / CTRL_DT))):
        step(schedule[0][0])

    acc = defaultdict(list)
    label_last = None
    for si, (flag, dur) in enumerate(schedule):
        lab = f"seg{si}_{'SIT' if flag else 'STAND'}"
        n = int(round(dur / CTRL_DT))
        for k in range(n):
            step(flag)
            if k >= int(n * 0.7):  # last 30 % = settled
                acc[lab].append(
                    np.concatenate([robot.data.joint_pos[0, sid].cpu().numpy(),
                                    [float(robot.data.body_link_pos_w[0, head_id, 2]),
                                     float(robot.data.body_link_pos_w[0, 0, 2])]]))
    return {"names": names, "home": home, "acc": {k: np.mean(v, axis=0) for k, v in acc.items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", action="append", required=True,
                    help="name:path[:preswap]  (repeatable)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=float, default=2.0)
    ap.add_argument("--schedule", default="0:3,1:4,0:4")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    schedule = [(int(f), float(d)) for f, d in
                (s.split(":") for s in args.schedule.split(","))]
    pols = []
    for spec in args.policy:
        parts = spec.split(":")
        pols.append((parts[0], parts[1], len(parts) > 2 and parts[2] == "preswap"))

    out = {}
    for name, path, pre in pols:
        print(f"[run] {name}: {path}  preswap={pre}", flush=True)
        try:
            out[name] = run_one(path, pre, schedule, args.warmup, args.seed, args.device)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {type(e).__name__}: {e}")
    if not out:
        return 1

    ref = pols[0][0]
    names = out[ref]["names"]
    home = out[ref]["home"]
    # Report EVERY settled segment: which one matters depends on the schedule
    # ("0:3,1:4,0:4" -> seg2_STAND is the phase the user watched; a single-segment
    # "0:12" probe has only seg0_STAND, and hardcoding seg2 printed an empty table).
    segs = sorted({k for v in out.values() for k in v["acc"]})
    for seg in segs:
        _table(seg, names, home, out)
    return 0


def _table(seg, names, home, out) -> None:

    print(f"\n=== servo angles [rad] at {seg} (settled) — HOME is the robot's default pose ===")
    hdr = f"{'joint':<12}{'HOME':>9}" + "".join(f"{n:>13}" for n in out)
    print(hdr); print("-" * len(hdr))
    for i, jn in enumerate(names):
        row = f"{jn:<12}{home[i]:>9.3f}"
        for n in out:
            a = out[n]["acc"].get(seg)
            row += f"{a[i]:>13.3f}" if a is not None else f"{'--':>13}"
        print(row)

    print(f"\n=== deviation from HOME [rad] — NEGATIVE = folded below the default pose ===")
    print(hdr); print("-" * len(hdr))
    for i, jn in enumerate(names):
        row = f"{jn:<12}{0.0:>9.3f}"
        for n in out:
            a = out[n]["acc"].get(seg)
            row += f"{a[i] - home[i]:>13.3f}" if a is not None else f"{'--':>13}"
        print(row)

    print(f"\n=== head joints the eye actually sees (rel HOME, {seg}) ===")
    print(f"{'policy':<14}" + "".join(f"{h:>13}" for h in HEAD) + f"{'head_z_mm':>11}{'trunk_z':>9}")
    for n in out:
        a = out[n]["acc"].get(seg)
        if a is None:
            continue
        idx = [names.index(h) for h in HEAD]
        print(f"{n:<14}" + "".join(f"{a[i] - home[i]:>13.3f}" for i in idx)
              + f"{a[-2] * 1000:>11.1f}{a[-1]:>9.3f}")
    print(f"\n  nominal HEAD body height (HOME pose) is ~232.6 mm; a folded neck reads far below.")


if __name__ == "__main__":
    raise SystemExit(main())
