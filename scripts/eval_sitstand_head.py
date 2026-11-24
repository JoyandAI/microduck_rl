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
# Trunk z at/below which the robot counts as SEATED. SIT_Z = 0.060; 0.065 is the
# acceptance bar used everywhere else (`queue_round2_report.py`, `sit_seed_sweep.sh`).
SIT_REACHED_Z = 0.065


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
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed. STRONGLY recommended: sitstand DR is NOT gated on "
                         "play=True (only the terrain curriculum and the push interval "
                         "are), so an unseeded run draws its own CoM/encoder-bias/mass/"
                         "friction and TWO RUNS OF THE SAME ONNX DISAGREE. Measured: "
                         "the same sitstand_headfix ck3000 ONNX reached z=0.058 on one "
                         "draw and only 0.104 on another -- 'it sits' and 'it cannot sit' "
                         "from one artifact, one sample each.")
    ap.add_argument("--params-json", default=None,
                    help="explicit BAM params JSON (e.g. the snapshot the policy was trained with)")
    args = ap.parse_args()

    schedule = parse_schedule(args.sit_schedule)

    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = 1
    if args.seed is not None:
        cfg.seed = args.seed
        print(f"[i] seed = {args.seed}")
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
    # jaw_soft carries the head collision geoms in robot_allcollisions.xml.
    _hb, _ = robot.find_bodies("jaw_soft", preserve_order=True)
    head_body_id = int(_hb[0])

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
        t_seg0 = t  # segment start, for TIME-TO-SIT (see below)
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
            # Head-ground LOAD. This is the quantity the 2026-09-16 fix prices
            # (`mdp.posture_head_ground_penalty`), and the acceptance table asks for
            # it directly: the defect is not "the head is low", it is "the head is
            # CARRYING THE ROBOT" (measured 3.16 N = 37 % of body weight pre-fix).
            # Absent sensor (older cfg) -> 0.0 so this script still runs.
            hg = 0.0
            if "head_ground_contact" in unwrapped.scene.sensors:
                f = unwrapped.scene.sensors["head_ground_contact"].data.force
                if f is not None:
                    hg = float(abs(f[0, 0, 2]))
            bucket["head_force_n"].append(hg)
            # Head BODY height. The joint angle alone cannot answer "is the head
            # drooping": after the 2026-09-16 fix the neck still parks near its
            # stop while the head carries ZERO load and the trunk stands upright,
            # so "head low" and "head bearing weight" are different questions and
            # need different numbers. Pre-fix the head was both low AND loaded.
            bucket["head_z"].append(float(robot.data.body_link_pos_w[0, head_body_id, 2]))
            # Extremes over the WHOLE segment. The steady-state mean answers "where
            # did it settle"; it cannot answer "did it ever get there", and those
            # differ: the fixed sit-stand policy reaches z=0.058 seated at t=6-7 s
            # but is already standing again by the end of the SIT segment, so its
            # steady mean (0.105) looks like "never sat down" when it plainly did.
            bucket["seg_min_z"].append(float(pos[2]))
            bucket["seg_max_z"].append(float(pos[2]))
            # TIME TO SIT. "Can it sit" and "does it sit promptly" are different
            # questions, and the measured answer differs: the fixed policy reaches
            # z=0.055 in SOME draws within ~1 s and in others only after 5-11 s of
            # holding a 20 deg forward lean at UNCHANGED height, while the pre-fix
            # policy does it in 1-2 s in 24/24 draws. A min-z-over-a-4 s-window
            # criterion cannot tell those apart -- it just calls the slow draws
            # "cannot sit". Record the time from the segment's first commanded step
            # to the first trunk z <= SIT_REACHED_Z.
            if not bucket["sit_t"] and float(pos[2]) <= SIT_REACHED_Z:
                bucket["sit_t"].append(t - t_seg0)
            if k >= steady_from:
                for key in ("z", "tilt", "neck_rel", "head_rel", "err", "hcmd_neck",
                            "head_force_n", "head_z"):
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
    hdr2 = (f"{'phase':<14}{'z':>9}{'tilt':>9}{'neck_rel':>10}{'head_rel':>10}"
            f"{'neck_abs':>10}{'hcmd_neck':>11}{'err':>9}{'headF_N':>9}{'head_z_mm':>10}")
    print(hdr2)
    print("-" * len(hdr2))
    for label in (f"seg{i}_{'SIT' if f else 'STAND'}" for i, (f, _) in enumerate(schedule)):
        d = phases.get(label)
        if not d or not d.get("steady_z"):
            continue
        g = lambda k: np.mean(d["steady_" + k])  # noqa: E731
        print(f"{label:<14}{g('z'):>9.3f}{g('tilt'):>9.1f}{g('neck_rel'):>10.3f}"
              f"{g('head_rel'):>10.3f}{np.mean(d['neck_abs'][-len(d['steady_z']):]):>10.3f}"
              f"{g('hcmd_neck'):>11.3f}{g('err'):>9.4f}"
              f"{g('head_force_n'):>9.3f}{g('head_z')*1000:>9.1f}")
    print()
    print()
    print("  --- DID IT ACHIEVE THE POSTURE? (extreme over the whole segment) ---")
    for label in (f"seg{i}_{'SIT' if f else 'STAND'}" for i, (f, _) in enumerate(schedule)):
        d = phases.get(label)
        if not d:
            continue
        tgt = 0.060 if "SIT" in label else 0.115
        st = d.get("sit_t") or []
        stx = (f"   reached {tgt if tgt < 0.1 else tgt:.3f} at t={st[0]:.2f} s "
               f"after the command" if st else "   NEVER reached it in this window")
        print(f"    {label:<14} z min {min(d['seg_min_z']):.3f}  max "
              f"{max(d['seg_max_z']):.3f}   (target {tgt:.3f}){stx if 'SIT' in label else ''}")
    print("  neck_abs = absolute neck_pitch [rad]; HOME = +0.3491 (= +20deg).")
    print("  headF_N  = |vertical head-ground contact force| [N]. The 2026-09-16 fix")
    print("             targets THIS: pre-fix it read ~3.16 N = 37% of body weight,")
    print("             i.e. the head was carrying the robot. Acceptance: ~0.")

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
