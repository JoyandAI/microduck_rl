#!/usr/bin/env python3
"""Official vs ours, WALKING: the 14 servo angles and the lateral stance width.

User observation (2026-09-16): "我们版本模型走路腿有点像两侧迈开" -- our legs look
splayed to both sides. That is a testable claim about the ACTION OUTPUT, so this
compares the commanded/achieved joint angles against the official `alpha_walking`
under the same frozen forward command, and prints the numbers that make "splayed"
objective:

  * per-joint mean +- std over a steady walking window (std is the gait's actual
    use of the joint; a joint with a big static offset and tiny std is being
    HELD in a pose, not used for the gait);
  * hip roll/yaw deviation from HOME, left and right separately -- a MIRRORED
    pair (left +, right -) is a splay/cross, a COMMON-mode pair is a lean;
  * the lateral distance between the two ankle bodies, and each foot's lateral
    offset from the trunk: absolute stance width, not a proxy.

Hardware eras differ between the two files and that changes what the SAME action
means, so each policy is run in its own era (`preswap` for the official: XL330,
kp_fw 200, 6.5-8.2 V, and the pre-swap WALK MJCF, which for the velocity task IS
the right model -- unlike sit-stand, whose model is allcollisions).

Usage:
    uv run python scripts/compare_walk_angles.py \
        --policy official:/home/joyandai/microduck/policies/alpha_walking.onnx:preswap \
        --policy ours:walk_hd1910_7000.onnx \
        --vx 0.2 --seed 0
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

PRESWAP_XML = "src/mjlab_microduck/robot/microduck/robot_walk_preswap.xml"
XL330_JSON = "vendor/bam/bam/params/xl330/m6.json"
LEGS = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")


def apply_preswap(cfg) -> None:
    import mujoco
    robot = cfg.scene.entities["robot"]
    object.__setattr__(robot, "spec_fn", (lambda p=PRESWAP_XML: mujoco.MjSpec.from_file(p)))
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


def run_one(onnx, preswap, vx, seconds, warmup, seed, device, action_scale=1.0,
            legs_lowpass=None, head_lowpass=None) -> dict:
    cfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck", play=True)
    cfg.scene.num_envs = 1
    cfg.seed = seed
    cfg.episode_length_s = 1000.0
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
    home = robot.data.default_joint_pos[0, sid].cpu().numpy()
    lb, _ = robot.find_bodies("ankle_left", preserve_order=True)
    rb, _ = robot.find_bodies("ankle_right", preserve_order=True)
    li, ri = int(lb[0]), int(rb[0])

    obs = env.reset()[0]
    term = un.command_manager.get_term("twist")

    # Runtime low-pass parity. /home/joyandai/microduck/robotd/src/control.rs:80-81
    # hardcodes head_lowpass = Some(0.5) and legs_lowpass = Some(0.7), applied as
    #     target = alpha * target + (1 - alpha) * previous_target
    # on the FINAL joint targets (control.rs:437-450), i.e. AFTER action_scale.
    # Training has no such filter (grep ACTION_LOW_PASS src/ -> nothing), so this
    # reproduces the deployed robot to measure what the mismatch costs.
    legs_idx = [i for i, n in enumerate(names) if not n.startswith(("neck", "head"))]
    head_idx = [i for i, n in enumerate(names) if n.startswith(("neck", "head"))]
    prev = {"legs": None, "head": None}
    last_action = None

    def lowpass(a):
        """``a`` is (1, 14) — keep the batch dim for env.step()."""
        if legs_lowpass is None and head_lowpass is None:
            return a
        row = a[0]
        for key, idx, alpha in (("legs", legs_idx, legs_lowpass),
                                ("head", head_idx, head_lowpass)):
            if alpha is None or prev[key] is None:
                continue
            row[idx] = alpha * row[idx] + (1.0 - alpha) * prev[key]
            prev[key] = row[idx].copy()
        return a

    def step():
        nonlocal obs, last_action
        term.command[:, 0] = vx
        a = sess.run([oname], {iname: obs["actor"].cpu().numpy()})[0].astype(np.float32)
        # The DEPLOYED runtime scales the raw ONNX output before adding it to the
        # home pose (robotd.toml: "targets <- home pose + action_scale x action",
        # default 0.9 for walk mode), while TRAINING uses scale=1.0
        # (microduck_velocity_env_cfg.py:799-801).  Reproducing the runtime here is
        # the only way to see whether that mismatch changes the gait.
        if action_scale != 1.0:
            a = a * action_scale
        # Seed the filter's previous target on the first step (the runtime's
        # `previous` is None only before the very first tick too).
        if prev["legs"] is None:
            prev["legs"] = a[0][legs_idx].copy()
            prev["head"] = a[0][head_idx].copy()
        a = lowpass(a)
        last_action = a
        obs, _, dones, _ = env.step(torch.from_numpy(a).to(device))
        return bool(dones[0])

    n_warm = int(round(warmup / 0.02))
    for _ in range(n_warm):
        step()
    q, width, tilt, falls = [], [], [], 0
    acts, yaws = [], []
    tb, _ = robot.find_bodies("trunk_base", preserve_order=True)
    ti = int(tb[0])
    p0 = robot.data.body_link_pos_w[0, ti, :2].cpu().numpy().copy()
    for _ in range(int(round(seconds / 0.02))):
        if step():
            falls += 1
        q.append(robot.data.joint_pos[0, sid].cpu().numpy())
        acts.append(last_action[0].copy())
        pos = robot.data.body_link_pos_w[0]
        width.append(float(pos[ri, 1] - pos[li, 1]))
        quat = robot.data.body_link_quat_w[0, 0]
        w, x, y, z = (float(v) for v in quat)
        tilt.append(math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2 * (x * x + y * y))))))
        yaws.append(math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))
    q = np.array(q)
    acts = np.array(acts)
    # Net yaw change: wrap-safe sum of per-step differences. NEVER difference the
    # wrapped angles directly -- gait_forensics.py once read +107 deg where the
    # wrap-safe sum read -14.3 deg on the identical rollout.
    dyaw = np.diff(np.array(yaws))
    dyaw = (dyaw + 180.0) % 360.0 - 180.0
    yaw_drift_deg = float(dyaw.sum())
    # Forward speed from DISPLACEMENT, not from root_link_lin_vel_w: the root is a
    # free joint and reading its world velocity returned ~0 for policies that were
    # demonstrably walking (the same checkpoint measured +0.20 m/s elsewhere), so
    # the first version of this table silently compared three STANDING policies.
    # eval_straightness.py already uses displacement ("ground-truth speed"); match it.
    p1 = robot.data.body_link_pos_w[0, ti, :2].cpu().numpy()
    return {"names": names, "home": home, "mean": q.mean(0), "std": q.std(0),
            "width": float(np.mean(width)), "tilt": float(np.mean(tilt)),
            "vx": float(np.linalg.norm(p1 - p0) / seconds), "falls": falls, "n": len(q),
            "acts": acts, "yaw_drift_deg": yaw_drift_deg}


def parse_spec(spec, default_scale):
    """Parse ``name:path[:preswap][:scale=<f>][:lp[=<legs>,<head>]]``.

    Per-policy flags, because the DEPLOYED action handling differs per policy and a
    single global flag cannot express "this arm as trained vs that arm as deployed".
    """
    parts = spec.split(":")
    if len(parts) < 2:
        raise SystemExit(f"--policy needs name:path, got {spec!r}")
    name, path = parts[0], parts[1]
    pre = False
    scale = default_scale
    lp_legs = lp_head = None
    for opt in parts[2:]:
        if opt == "preswap":
            pre = True
        elif opt.startswith("scale="):
            scale = float(opt.split("=", 1)[1])
        elif opt == "lp":
            lp_legs, lp_head = 0.7, 0.5      # robotd control.rs:80-81 defaults
        elif opt.startswith("lp="):
            lp_legs, lp_head = (float(x) for x in opt.split("=", 1)[1].split(","))
        else:
            raise SystemExit(f"unknown policy option {opt!r} in {spec!r}")
    return name, path, pre, scale, lp_legs, lp_head


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", action="append", required=True,
                    help="name:path[:preswap][:scale=<f>][:lp[=<legs>,<head>]]")
    ap.add_argument("--vx", type=float, default=0.2)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--warmup", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--action-scale", type=float, default=1.0,
                    help="DEFAULT scale for policies that do not carry a `scale=` flag. "
                         "Reproduces the deployed runtime (robotd.toml walk default 0.9); "
                         "training used 1.0.")
    ap.add_argument("--lowpass-legs", type=float, default=None,
                    help="Runtime legs_lowpass (control.rs default 0.7). Training has none.")
    ap.add_argument("--lowpass-head", type=float, default=None,
                    help="Runtime head_lowpass (control.rs default 0.5). Training has none.")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    pols = [parse_spec(s, args.action_scale) for s in args.policy]
    out = {}
    for name, path, pre, sc, lp_legs, lp_head in pols:
        print(f"[run] {name}: {path} preswap={pre} scale={sc} "
              f"lowpass={lp_legs}/{lp_head}", flush=True)
        out[name] = run_one(path, pre, args.vx, args.seconds, args.warmup, args.seed,
                            args.device, action_scale=sc,
                            legs_lowpass=lp_legs, head_lowpass=lp_head)
    ref = pols[0][0]
    names, home = out[ref]["names"], out[ref]["home"]

    print(f"\n=== walking at vx={args.vx} m/s, {args.seconds}s after {args.warmup}s warmup "
          f"(seed {args.seed}) ===")
    print(f"{'policy':<12}{'vx_mps':>9}{'tilt':>8}{'stance_mm':>11}{'resets':>8}")
    for n in out:
        d = out[n]
        print(f"{n:<12}{d['vx']:>9.3f}{d['tilt']:>8.1f}{d['width'] * 1000:>11.1f}{d['falls']:>8d}")

    print(f"\n=== servo angle MEAN [rad] over the walking window ===")
    hdr = f"{'joint':<12}{'HOME':>9}" + "".join(f"{n:>12}" for n in out)
    print(hdr); print("-" * len(hdr))
    for i, jn in enumerate(names):
        print(f"{jn:<12}{home[i]:>9.3f}" + "".join(f"{out[n]['mean'][i]:>12.3f}" for n in out))

    print(f"\n=== deviation from HOME [rad] — mirrored L/R = SPLAYED, same sign = LEANING ===")
    print(hdr); print("-" * len(hdr))
    for i, jn in enumerate(names):
        print(f"{jn:<12}{0.0:>9.3f}" + "".join(f"{out[n]['mean'][i] - home[i]:>12.3f}"
                                               for n in out))

    print(f"\n=== gait USE of each joint (std over the window) — big offset + tiny std = HELD, not used ===")
    print(hdr); print("-" * len(hdr))
    for i, jn in enumerate(names):
        print(f"{jn:<12}{0.0:>9.3f}" + "".join(f"{out[n]['std'][i]:>12.3f}" for n in out))

    print(f"\n=== hip roll/yaw, L vs R (the splay question, directly) ===")
    print(f"{'policy':<12}{'L_roll':>10}{'R_roll':>10}{'L+R':>9}{'L-R':>9}"
          f"{'L_yaw':>10}{'R_yaw':>10}")
    for n in out:
        d = out[n]
        li, ri = names.index("left_hip_roll"), names.index("right_hip_roll")
        ly, ry = names.index("left_hip_yaw"), names.index("right_hip_yaw")
        lr, rr = d["mean"][li] - home[li], d["mean"][ri] - home[ri]
        lyn, ryn = d["mean"][ly] - home[ly], d["mean"][ry] - home[ry]
        print(f"{n:<12}{lr:>10.3f}{rr:>10.3f}{lr + rr:>9.3f}{lr - rr:>9.3f}"
              f"{lyn:>10.3f}{ryn:>10.3f}")
    print("\n  L+R ~ 0 with |L-R| large  => pure SPLAY (both legs rotate outward).")
    print("  L+R large with L-R ~ 0    => both legs rotate the SAME way (lean).")

    # ── The DISCRIMINATOR for "is the turn commanded or physical?" ──────────
    # The robot is trained and evaluated in a MIRROR-SYMMETRIC simulator: one
    # shared BAM parameter table for all 14 joints, mirrored HOME (+-0.0873 rad
    # hip_roll), identical hip_yaw ranges, no per-side DR. So any asymmetry in
    # the policy's ACTIONS is learned by the POLICY, not injected by the servo
    # model -- that is what separates "actuator/calibration asymmetry" from
    # "the reward never priced net yaw, so the policy settled in an asymmetric
    # gait". Amplitude here is the per-step action excursion, not the qpos.
    print("\n=== ACTION asymmetry (the servo model is mirrored, so this is LEARNED) ===")
    print(f"{'policy':<14}{'lyaw_mean':>10}{'ryaw_mean':>10}{'lyaw_amp':>10}{'ryaw_amp':>10}"
          f"{'amp_ratio':>10}{'yaw_drift_deg':>14}")
    for n in out:
        d = out[n]
        li_y = names.index("left_hip_yaw")
        ri_y = names.index("right_hip_yaw")
        a = d["acts"]
        # amplitude = half peak-to-peak about the mean; both legs must be measured
        # the same way or a DC offset masquerades as amplitude.
        la = 0.5 * (a[:, li_y].max() - a[:, li_y].min())
        ra = 0.5 * (a[:, ri_y].max() - a[:, ri_y].min())
        ratio = la / ra if ra > 1e-9 else float("inf")
        print(f"{n:<14}{a[:, li_y].mean():>10.3f}{a[:, ri_y].mean():>10.3f}"
              f"{la:>10.3f}{ra:>10.3f}{ratio:>10.2f}{d['yaw_drift_deg']:>14.2f}")
    print("\n  ratio ~ 1.0 => the policy COMMANDS both hip_yaws symmetrically;")
    print("  ratio >> 1   => the policy itself commands an asymmetric yaw impulse")
    print("                  (a learned gait, NOT a servo-model asymmetry).")
    print("  yaw_drift_deg is the wrap-safe net yaw over the window (frozen wz=0).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
