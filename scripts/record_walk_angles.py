#!/usr/bin/env python3
"""Record the servo joint-angle trajectories of a walking policy from the sim.

What this produces
------------------
A 50 Hz CSV (one row per control step) of the **14 policy joints**:

  time, target_<joint>×14, measured_<joint>×14, trunk_z_m

  - `target_*`: the joint angle target the policy commands
    (= home pose + action, exactly what the runtime writes to the servos),
  - `measured_*`: the angle the sim's servos actually reached (the BAM /
    MuJoCo closed-loop result),
  - plus a sidecar JSON with metadata (policy, command, default pose, joint
    names, wire servo IDs, dt, warmup/duration).

Use the target columns for a real-robot replay: the duck's HLS2909 servos are
position servos, so sending them the same 50 Hz target trajectory reproduces
the walk (see scripts/replay_walk_angles.py). The measured columns are what a
stiff position servo would actually have done if the sim matched the robot.

Engine
------
Default engine is the **real training env** (mjlab ManagerBasedRlEnv + BAM
actuator, 61D obs, same DR stack as training) — the faithful one. walk.onnx is
the XL330-era policy, so `--motor-name xl330 --model m6` (the defaults)
reproduces the env it was trained in; pass `--motor-name hls2909 --model m1`
for the current servo model. Falls back to a CPU "deployment scene" engine
(scene.xml + PolicyInference) if the training env cannot be built
(--device cpu is slow the first time: warp compiles its kernels once).

Usage:
    uv run python scripts/record_walk_angles.py --onnx walk.onnx --lin-vel-x 0.15
    uv run python scripts/record_walk_angles.py --onnx walk.onnx --lin-vel-x 0.15 \
        --duration 6 --warmup 2 --out walk_angles.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg

from mjlab_microduck.tasks import mdp as mdp_mod

CTRL_DT = 0.02  # 50 Hz

# Runtime wire order (15 servos incl. mouth): left leg, neck/head/mouth, right leg.
# Verbatim from duck-control/src/model.rs (pollen-robotics/microduck).
WIRE_JOINT_IDS = [20, 21, 22, 23, 24, 30, 31, 32, 33, 34, 10, 11, 12, 13, 14]
MOUTH_INDEX = 9  # policy has no mouth; replay fills this slot with 0


def resolve_device(device: str) -> str:
    """Pick the device, honoring 'auto' (cuda → cpu fallback)."""
    if device != "auto":
        return device
    try:
        torch.zeros(1, device="cuda:0")
        return "cuda:0"
    except Exception as e:
        print(f"[i] CUDA unavailable ({e}); falling back to cpu")
        return "cpu"


def onnx_meta(path: str) -> dict:
    sess = ort.InferenceSession(path)
    md = sess.get_modelmeta().custom_metadata_map
    joint_names = md.get("joint_names", "").split(",")
    default_pose = np.array(
        [float(x) for x in md.get("default_joint_pos", "0," * 14).split(",")]
    )
    return {
        "run_path": md.get("run_path", ""),
        "joint_names": joint_names,
        "default_pose": default_pose,
        "action_scale": float(md.get("action_scale", "1.0")),
    }


def gait_freq_hz(signal: np.ndarray, dt: float = CTRL_DT) -> float:
    """Dominant frequency of a signal (FFT), or NaN for a flat signal."""
    s = signal - signal.mean()
    if np.allclose(s, 0.0):
        return float("nan")
    n = len(s)
    spec = np.abs(np.fft.rfft(s))
    freqs = np.fft.rfftfreq(n, dt)
    if len(freqs) < 3:
        return float("nan")
    # ignore DC already removed; take the strongest bin in (0.5, 4) Hz — gait band
    band = (freqs >= 0.5) & (freqs <= 4.0)
    if not band.any():
        return float("nan")
    return float(freqs[band][np.argmax(spec[band])])


def loop_window(signal: np.ndarray, dt: float = CTRL_DT,
                min_period_s: float = 0.15, max_period_s: float = 1.0) -> tuple:
    """Best integer-cycle loop window [steps, steps] for a periodic signal.

    Autocorrelation peak gives the period; the loop then covers an integer
    number of periods from the start, so a replay loop has no seam pop (the
    last row of the window and the first row are one period apart).

    Returns (period_steps, window_steps); (0, len) when the signal is not
    periodic enough to trust.
    """
    s = signal - signal.mean()
    n = len(s)
    lo, hi = int(min_period_s / dt), int(max_period_s / dt) + 1
    hi = min(hi, n // 2)
    if n < 4 * hi or np.allclose(s, 0.0):
        return 0, n
    ac = np.correlate(s, s, "full")[n - 1:]
    if ac[0] <= 0.0:
        return 0, n
    best = max(range(lo, hi), key=lambda lag: ac[lag])
    if ac[best] < 0.5 * ac[0]:
        return 0, n  # weak periodicity — loop the whole file
    period, window = best, (n // best) * best
    return max(period, 1), max(window, period)


def run_env_engine(args, meta: dict) -> list[dict]:
    """Record inside the real training env (BAM actuator, mjlab, 61D obs)."""
    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = args.num_envs
    act = cfg.scene.entities["robot"].articulation.actuators[0]
    act.motor_name = args.motor_name
    act.model = args.model
    if args.kp_fw is not None:
        act.kp_fw = args.kp_fw
    vin = tuple(float(x) for x in args.vin_range.split(","))
    drop = tuple(float(x) for x in args.vin_drop_gain.split(","))
    act.vin_range = vin
    act.vin_drop_gain_range = drop
    act.vin_min = args.vin_min
    if hasattr(act, "delay_min_lag") and hasattr(act, "delay_max_lag"):
        act.delay_min_lag = args.delay_min
        act.delay_max_lag = args.delay_max

    device = resolve_device(args.device)
    print(f"[i] env device: {device} (motor={args.motor_name}, model={args.model}, "
          f"kp_fw={act.kp_fw}, vin={vin}V, drop={drop}, "
          f"delay={args.delay_min}-{args.delay_max} steps)")
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=None)
    unwrapped = env.unwrapped
    robot = unwrapped.scene.entities["robot"]

    sess = ort.InferenceSession(args.onnx)
    iname, oname = sess.get_inputs()[0].name, sess.get_outputs()[0].name
    if sess.get_inputs()[0].shape[1] != 61:
        print(f"[!] unexpected ONNX input width {sess.get_inputs()[0].shape}")

    servo_ids = mdp_mod._servo_joint_ids(unwrapped, robot)

    obs_td = env.reset()[0]
    home = robot.data.default_joint_pos[0, servo_ids].cpu().numpy()
    term = unwrapped.command_manager.get_term("twist")

    rows: list[dict] = []
    n_warm = int(round(args.warmup / CTRL_DT))
    n_rec = int(round(args.duration / CTRL_DT))
    t0 = time.time()
    for i in range(n_warm + n_rec):
        term.command[:, 0] = args.lin_vel_x
        term.command[:, 1] = args.lin_vel_y
        term.command[:, 2] = args.ang_vel_z

        obs = obs_td["actor"].cpu().numpy()
        action_batch = sess.run([oname], {iname: obs})[0].astype(np.float64)
        action = action_batch.reshape(-1)
        obs_td, *_ = env.step(torch.from_numpy(action_batch.astype(np.float32)).to(device))

        if i < n_warm:
            if i % 25 == 0:
                print(f"  warmup {i * CTRL_DT:.2f}/{args.warmup:.2f}s")
            continue

        measured = robot.data.joint_pos[0, servo_ids].cpu().numpy()
        target = home + action
        row = {"time": round((i - n_warm) * CTRL_DT, 4)}
        for k, name in enumerate(meta["joint_names"]):
            row[f"target_{name}"] = round(float(target[k]), 6)
        for k, name in enumerate(meta["joint_names"]):
            row[f"measured_{name}"] = round(float(measured[k]), 6)
        row["trunk_z_m"] = round(float(robot.data.body_link_pos_w[0, 0, 2].item()), 6)
        rows.append(row)

        if len(rows) % 25 == 0:
            el = time.time() - t0
            print(f"  recorded {len(rows)}/{n_rec} steps ({el:.1f}s wall)")
    return rows


def run_scene_engine(args, meta: dict) -> list[dict]:
    """Fallback: deployment scene (scene.xml) + PolicyInference (CPU MuJoCo)."""
    sys.path.insert(0, str(Path(__file__).parent))
    import mujoco
    from infer_policy import PolicyInference

    xml = "src/mjlab_microduck/robot/microduck/scene.xml"
    model = mujoco.MjModel.from_xml_path(xml)
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    from bam.model import load_model

    kt = load_model(motor_name=args.motor_name, model=args.model).kt.value
    tl = kt * 1.75
    model.actuator_forcerange[:, 0] = -tl
    model.actuator_forcerange[:, 1] = tl
    model.actuator_forcelimited[:] = 1

    if args.sitstand:
        pol = PolicyInference(model, data, sitstand_onnx_path=args.sitstand,
                              new_cmd_obs=True, use_projected_gravity=True)
        schedule = []
        for seg in args.sit_schedule.split(","):
            flag_s, dur_s = seg.split(":")
            schedule.append((int(flag_s), float(dur_s)))
        # warmup phase holds the FIRST flag of the schedule (settle the pose)
        first_flag = schedule[0][0]
        pol.sit_mode = bool(first_flag)
        pol._update_command()
        print(f"[i] sitstand: schedule {schedule} (warmup keeps flag={first_flag})")
    else:
        pol = PolicyInference(model, data, walking_onnx_path=args.onnx,
                              new_cmd_obs=True, use_projected_gravity=True)
        pol.set_vel_cmd(args.lin_vel_x, args.lin_vel_y, args.ang_vel_z)

    def sit_flag_at(t: float) -> int:
        """Posture flag for a given time into the scheduled cycle."""
        if not args.sitstand:
            return 0
        total = sum(d for _, d in schedule)
        rem = t % total
        for flag_s, dur_s in schedule:
            if rem < dur_s:
                return flag_s
            rem -= dur_s
        return schedule[0][0]

    qadr = int(model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")])
    data.qpos[qadr + 2] = 0.12
    for i, ji in enumerate(pol.joint_qpos_indices):
        data.qpos[ji] = pol.default_pose[i]
    data.ctrl[:] = pol.default_pose
    mujoco.mj_forward(model, data)

    rows: list[dict] = []
    n_warm = int(round(args.warmup / CTRL_DT))
    n_rec = int(round(args.duration / CTRL_DT))
    for i in range(n_warm + n_rec):
        if args.sitstand:
            flag = sit_flag_at((i - n_warm) * CTRL_DT)
            if bool(flag) != pol.sit_mode:
                pol.sit_mode = bool(flag)
                pol._update_command()
        action = pol.infer()
        pol.apply_action(action)
        if i >= n_warm:
            measured = data.qpos[pol.joint_qpos_indices].copy()
            target = pol.default_pose + action
            row = {"time": round((i - n_warm) * CTRL_DT, 4)}
            for k, name in enumerate(meta["joint_names"]):
                row[f"target_{name}"] = round(float(target[k]), 6)
            for k, name in enumerate(meta["joint_names"]):
                row[f"measured_{name}"] = round(float(measured[k]), 6)
            row["trunk_z_m"] = round(float(data.qpos[qadr + 2]), 6)
            rows.append(row)
        for _ in range(4):
            mujoco.mj_step(model, data)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--task", default="Mjlab-Velocity-Flat-MicroDuck")
    ap.add_argument("--engine", choices=("env", "scene", "auto"), default="auto",
                    help="env = mjlab training env (BAM); scene = CPU "
                         "deployment-scene engine (DEFAULT — the only engine "
                         "walk.onnx walks cleanly in: the mjlab env now builds "
                         "hls2909 masses/params, which mismatch the Sep-2 "
                         "xl330-era policy)")
    ap.add_argument("--device", default="auto", help="auto | cuda:0 | cpu")
    ap.add_argument("--lin-vel-x", type=float, default=0.15)
    ap.add_argument("--lin-vel-y", type=float, default=0.0)
    ap.add_argument("--ang-vel-z", type=float, default=0.0)
    ap.add_argument("--warmup", type=float, default=2.0,
                    help="seconds of walking before recording (gait settles)")
    ap.add_argument("--duration", type=float, default=8.0,
                    help="seconds of trajectory recorded (replay loops over this)")
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--motor-name", default="hd1910",
                    help="BAM motor to build the env with (default hd1910 = the servo the robot "
                         "runs now; use xl330/m6 only to replay pre-swap policies)")
    ap.add_argument("--model", default="m5", help="BAM friction model; m5 = HD-1910 pick (m6 = xl330)")
    ap.add_argument("--kp-fw", type=float, default=None)
    ap.add_argument("--vin-range", default="4.75,5.25",
                    help="per-env battery voltage DR, V (HD-1910 on a regulated 5 V rail; "
                         "the old xl330-era value was 6.5,8.2 / HLS 12V was 10.5,12.6)")
    ap.add_argument("--vin-drop-gain", default="0.0,0.2",
                    help="load-dependent voltage sag gain range, V/Nm")
    ap.add_argument("--vin-min", type=float, default=4.0)
    ap.add_argument("--delay-min", type=int, default=4)
    ap.add_argument("--delay-max", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="walk_angles.csv")
    ap.add_argument("--sitstand", type=str, default=None, metavar="ONNX",
                    help="record a sitstand cycle instead of walking: use this "
                         "policy with a posture-flag schedule (0=stand, 1=sit). "
                         "Requires the scene engine.")
    ap.add_argument("--sit-schedule", default="0:2.0,1:3.0,0:3.0",
                    help="comma-separated flag:seconds steps for the sitstand "
                         "recording (default '0:2.0,1:3.0,0:3.0' = stand 2s, sit 3s, "
                         "stand 3s; then loops back to the start pose if the last "
                         "flag is 0)")
    args = ap.parse_args()

    if args.sitstand and not (args.engine in ("scene", "auto")):
        raise SystemExit("--sitstand recording currently uses the scene engine "
                         "(the env is hls-config'd); drop --engine env")
    if args.sitstand:
        args.engine = "scene"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    meta = onnx_meta(args.onnx)
    print(f"[i] policy: {args.onnx}  (from {meta['run_path']})")
    print(f"[i] joints: {meta['joint_names']}")
    print(f"[i] cmd: vx={args.lin_vel_x} vy={args.lin_vel_y} wz={args.ang_vel_z} "
          f"| warmup {args.warmup}s + record {args.duration}s @ 50 Hz")
    print(f"[i] replay will map policy → wire IDs {WIRE_JOINT_IDS} "
          f"(mouth slot {MOUTH_INDEX} = 0 rad)")

    engine = args.engine
    if engine == "auto":
        engine = "scene"  # see --engine help: the mjlab env mismatches walk.onnx
    try:
        rows = (run_env_engine(args, meta) if engine == "env"
                else run_scene_engine(args, meta))
    except Exception as e:
        print(f"[!] {engine} engine failed: {e}")
        if engine != "scene":
            print("[i] retrying with the CPU scene engine...")
            try:
                rows = run_scene_engine(args, meta)
            except Exception as e2:
                print(f"[!] scene engine failed too: {e2}")
                return 1
        else:
            return 1

    if not rows:
        print("[!] no rows recorded")
        return 1

    # stats
    names = meta["joint_names"]
    tgt = np.array([[r[f"target_{n}"] for n in names] for r in rows])
    mea = np.array([[r[f"measured_{n}"] for n in names] for r in rows])
    print("\n=== record summary ===")
    print(f"  steps: {len(rows)}  ({rows[-1]['time']:.2f}s @ 50 Hz)")
    for k, name in enumerate(names):
        print(f"  {name:<15} target[{tgt[:,k].min():+.3f},{tgt[:,k].max():+.3f}] "
              f"range={tgt[:,k].max()-tgt[:,k].min():.3f}  "
              f"measured_std={mea[:,k].std():.3f} rad")
    freq = gait_freq_hz(tgt[:, 4])  # left_ankle
    period_steps, loop_steps = loop_window(tgt[:, 3])  # left_knee period
    print(f"  gait freq ~{freq:.2f} Hz; loop period {period_steps} steps "
          f"({period_steps*CTRL_DT:.3f}s) → clean loop window = first "
          f"{loop_steps} rows (seam-free)" if period_steps else
          f"  gait freq ~{freq:.2f} Hz; no stable period — replay loops the whole file")

    # sanity: falls / resets show up as big per-joint target steps or trunk_z dips
    # (walking-specific; a sitstand cycle legitimately has fast transitions and
    # a low trunk — the sit IS the low z)
    if not args.sitstand:
        d = np.abs(tgt[1:, :] - tgt[:-1, :])
        jumps = np.where(d.max(axis=1) > 0.6)[0]
        zs = np.array([r["trunk_z_m"] for r in rows])
        if jumps.size:
            print(f"[!] per-step target jump >0.6 rad at rows {jumps[:10].tolist()} — "
                  f"the episode may have reset mid-recording; re-run with a shorter duration")
        if zs.min() < 0.08:
            print(f"[!] trunk_z dropped to {zs.min()*1000:.0f} mm — the sim fell / "
                  f"reset; the trajectory is not a clean walk")
    else:
        zs = np.array([r["trunk_z_m"] for r in rows])
        print(f"  sitstand z range: {zs.min()*1000:.0f}~{zs.max()*1000:.0f} mm "
              f"(stand ≈ {zs.max()*1000:.0f}, sit ≈ {zs.min()*1000:.0f})")

    from infer_policy import DEFAULT_POSE  # exact HOME frame used by the policy
    exact_home = [float(x) for x in DEFAULT_POSE[:14]]
    # model fingerprint: the recording is only valid for this exact robot model
    import hashlib
    import re as _re
    scene_path = "src/mjlab_microduck/robot/microduck/scene.xml"
    xml_sha = hashlib.sha256(open(scene_path, "rb").read()).hexdigest()[:12]
    try:
        mass = float(sum(float(m) for m in _re.findall(
            r'mass="([0-9.]+)"',
            open("src/mjlab_microduck/robot/microduck/robot_walk.xml").read())))
    except Exception:
        mass = None
    out = Path(args.out)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    side = out.with_suffix(".json")
    side.write_text(json.dumps({
        "source": "sim recording",
        "engine": engine,
        "onnx": args.onnx,
        "onnx_run_path": meta["run_path"],
        "task": args.task,
        "motor_name": args.motor_name,
        "model": args.model,
        "vin_range": args.vin_range,
        "vin_drop_gain_range": args.vin_drop_gain,
        "vin_min": args.vin_min,
        "delay_lag": [args.delay_min, args.delay_max],
        "device": resolve_device(args.device) if engine == "env" else "cpu-scene",
        "model_mass_kg": mass,
        "robot_scene_sha256_12": xml_sha,
        "dt_s": CTRL_DT,
        "warmup_s": args.warmup,
        "duration_s": args.duration,
        "lin_vel_x": args.lin_vel_x,
        "lin_vel_y": args.lin_vel_y,
        "ang_vel_z": args.ang_vel_z,
        "joint_names": names,
        "default_pose_rad": exact_home,
        "wire_joint_ids": WIRE_JOINT_IDS,
        "mouth_index": MOUTH_INDEX,
        "gait_freq_hz": freq,
        "loop_period_steps": period_steps,
        "loop_window_steps": loop_steps,
        "dt_s_csv": CTRL_DT,
    }, indent=2, ensure_ascii=False))
    print(f"\n[+] saved {out} (+ {side})")


if __name__ == "__main__":
    main()
