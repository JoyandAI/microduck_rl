#!/usr/bin/env python3
"""Keyboard-driven **ONNX** policy in the FAITHFUL training env (BAM + HD-1910/XL330).

Why this exists
---------------
Two existing tools each do half of the job:

* `scripts/play_keyboard.py` -- faithful BAM env + keyboard, but it only loads
  rsl_rl **checkpoints** (`.pt`) through mjlab's `play`; there is no ONNX path.
* `scripts/infer_policy.py` -- loads ONNX (all the official policies) + keyboard,
  but runs `scene.xml` with raw MuJoCo position actuators, so a policy trained
  against the BAM voltage model does not reproduce there. Measured 2026-09-11 in
  that scene: official `alpha_walking.onnx` sits at trunk z = 114.6 mm with
  vx_body = +0.001 m/s -- i.e. it *stands still* while commanded 0.2 m/s, and
  `play_keyboard.py`'s docstring records the mirror-image failure for the local
  HD-1910 policy (`STANDED_OK = NO`). Neither is a policy statement; both are the
  actuator model being wrong.

This script is the missing combination: real training env (BAM actuator, pushes
disabled) + an ONNX file + terminal keyboard commands.

Official policies need `--preswap`
----------------------------------
`/home/joyandai/microduck/policies/alpha_*.onnx` were trained on the *pre-swap*
robot: XL330 servos (firmware `kp_fw = 200`), 6.5-8.2 V rail, the 737 g model.
On the current HD-1910 hardware (`kp_fw = 32`, 5 V, 779 g) the official walking
policy falls (3 resets / 20 s, `docs/forensics/agent_B_official_reference.md`).
`--preswap` reproduces its own configuration: the origin/develop MJCF kept in this
repo as `robot_walk.xml.bak_servoswap` (verified byte-identical to
`origin/develop`), `vendor/bam/bam/params/xl330/m6.json`, kp_fw 200, 6.5-8.2 V,
delay 3-6 substeps.

Measured with this script (seed 0, ``PLAY_KB_VX=0.2``, trunk quantities from the
trunk body, velocity in the robot's own yaw frame):

    official alpha_walking, --preswap : vx +0.09..0.10 m/s, z 118-119 mm, tilt 2.5 deg
    local walk_hd1910_15000 (current) : vx +0.18      m/s, z  96- 97 mm, tilt 4.2 deg

Both match the forensics numbers (official 119.8 mm, local 91-97 mm), which is the
point: the official gait reproduces here, and it is ~22 mm taller than the local
one. One caveat this script also measured: the official policy is *spawn
sensitive* -- at vx 0.2 it walks under some DR draws (seed 0, 1) and simply stands
still (z 114-115 mm, vx 0.00, no fall) under others (seed 2), so fix `--seed` when
comparing two policies.

Usage
-----
    # official walking policy, in the configuration it was trained for
    uv run python scripts/play_keyboard_onnx.py \
        --onnx  /home/joyandai/microduck/policies/alpha_walking.onnx \
        --preswap --seed 0

    # local policy on the current hardware (no --preswap)
    uv run python scripts/play_keyboard_onnx.py --onnx walk_hd1910_15000.onnx --seed 0

    # start already walking, and verify the pipeline without a human:
    PLAY_KB_VX=0.2 uv run python scripts/play_keyboard_onnx.py --onnx <onnx> --preswap
    # headless (no viewer) -- same telemetry, for GPU boxes without a display:
    uv run python scripts/play_keyboard_onnx.py --onnx <onnx> --preswap --headless --seconds 20
    # viewer smoke test without a human at the keyboard:
    uv run python scripts/play_keyboard_onnx.py --onnx <onnx> --preswap --viewer-steps 400

Keys (they go to THIS TERMINAL, not the viewer window):
    UP / DOWN      vx += / -= 0.05      (walk forward / backward)
    LEFT / RIGHT   vy += / -= 0.05      (strafe)
    A / E          wz -= / += 0.2       (turn right / left)
    SPACE          zero the command (neutral stand)
    1 / 2 / 3      snap vx to 0.1 / 0.2 / 0.3
    R              reset the environment(s)
    Q              quit
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]

# Pre-swap (XL330 era) configuration, from origin/develop:microduck_constants.py.
# `robot_walk_preswap.xml` is a copy of the in-repo `robot_walk.xml.bak_servoswap`
# backup, byte-identical to origin/develop's robot_walk.xml (checked with
# `git show origin/develop:...`), so no /tmp worktree is needed. The copy exists
# because MuJoCo picks its decoder from the file extension and refuses to load a
# `.bak_servoswap` name (`ValueError: could not decode content`); the file must
# also live in this directory for its relative `assets/*.stl` references to resolve.
PRESWAP_XML = REPO / "src/mjlab_microduck/robot/microduck/robot_walk_preswap.xml"
XL330_JSON = REPO / "vendor/bam/bam/params/xl330/m6.json"
PRESWAP_ACTUATOR = dict(
    kp_fw=200.0,          # XL330 firmware Kp (reg50). Mistaking this for the
                          # HD-1910's 32 is the single cause of "the official
                          # policy cannot stand" -- see the docstring above.
    vin_range=(6.5, 8.2),
    vin_min=6.0,
    vin_drop=(0.0, 0.2),
    delay=(3, 6),         # substeps (5 ms) -> 15-30 ms of actuator lag
)

# ── command state shared with the key callback ────────────────────────────────
VX_STEP, VY_STEP, WZ_STEP = 0.05, 0.05, 0.2
VX_MAX = VY_MAX = 0.4
WZ_MAX = 1.0


@dataclass
class CmdState:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    quit: bool = False

    def show(self) -> None:
        print(f"[cmd] vx={self.vx:+.2f}  vy={self.vy:+.2f}  wz={self.wz:+.2f} m/s,rad/s", flush=True)


STATE = CmdState()
STATE.vx = float(os.environ.get("PLAY_KB_VX", "0.0"))
STATE.vy = float(os.environ.get("PLAY_KB_VY", "0.0"))
STATE.wz = float(os.environ.get("PLAY_KB_WZ", "0.0"))

# GLFW key codes as delivered by mujoco.viewer's key_callback
KEY_SPACE = 32
KEY_R, KEY_Q = 82, 81
KEY_A, KEY_E = 65, 69
KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN = 263, 262, 265, 264
KEY_1, KEY_2, KEY_3 = 49, 50, 51


def clamp(v: float, lim: float) -> float:
    return max(-lim, min(lim, v))


def make_key_handler(env_unwrapped) -> "callable":
    """key_callback for the native viewer: mutates the command state in place."""

    def on_key(key: int) -> None:
        if key == KEY_UP:
            STATE.vx = clamp(STATE.vx + VX_STEP, VX_MAX)
        elif key == KEY_DOWN:
            STATE.vx = clamp(STATE.vx - VX_STEP, VX_MAX)
        elif key == KEY_LEFT:
            STATE.vy = clamp(STATE.vy + VY_STEP, VY_MAX)
        elif key == KEY_RIGHT:
            STATE.vy = clamp(STATE.vy - VY_STEP, VY_MAX)
        elif key == KEY_A:
            STATE.wz = clamp(STATE.wz - WZ_STEP, WZ_MAX)
        elif key == KEY_E:
            STATE.wz = clamp(STATE.wz + WZ_STEP, WZ_MAX)
        elif key == KEY_SPACE:
            STATE.vx = STATE.vy = STATE.wz = 0.0
            print("[cmd] zeroed (neutral stand)", flush=True)
            return
        elif key == KEY_1:
            STATE.vx = 0.1
        elif key == KEY_2:
            STATE.vx = 0.2
        elif key == KEY_3:
            STATE.vx = 0.3
        elif key == KEY_R:
            env_unwrapped.reset()
            print("[cmd] env reset", flush=True)
            return
        elif key == KEY_Q:
            STATE.quit = True
            print("[cmd] quit", flush=True)
            return
        else:
            return
        STATE.show()

    return on_key


class OnnxPolicy:
    """obs -> ONNX Runtime -> action tensor, with the export contract checked.

    The ONNX files in this family are `obs[1,61] -> actions[1,14]` with the obs
    normalizer baked into the graph (`scripts/export.py`), so nothing else is
    applied here: feeding raw actor observations is exactly what the runtime does.
    """

    def __init__(self, path: str, device: torch.device | str):
        import onnxruntime as ort

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"ONNX not found: {p}")
        self.sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name
        self.oname = self.sess.get_outputs()[0].name
        in_shape = self.sess.get_inputs()[0].shape
        out_shape = self.sess.get_outputs()[0].shape
        if in_shape[-1] != 61:
            raise ValueError(f"{p.name}: observation width is {in_shape[-1]}, expected 61")
        self.device = device
        print(f"[onnx] {p.name}: obs{in_shape} -> actions{out_shape}", flush=True)

    def __call__(self, obs) -> torch.Tensor:
        x = obs["actor"] if hasattr(obs, "keys") else obs
        arr = x.detach().cpu().numpy().astype(np.float32)
        y = self.sess.run([self.oname], {self.iname: arr})[0]
        return torch.as_tensor(y, dtype=torch.float32, device=self.device)


def apply_preswap(cfg, xml: str | None, params_json: str | None, kp_fw: float | None,
                  vin_range, vin_min, vin_drop, delay) -> dict:
    """Point the robot spec + BAM actuator at a chosen hardware era.

    `microduck_constants.py` resolves `json_path`/`kp_fw` in `__post_init__` on a
    frozen dataclass, so every field is re-set explicitly (same trick as
    `docs/forensics` harness and `scripts/gait_forensics.py --params-json`).
    """
    import mujoco

    rec: dict = {}
    robot = cfg.scene.entities["robot"]
    if xml:
        object.__setattr__(robot, "spec_fn", (lambda p=xml: mujoco.MjSpec.from_file(p)))
        rec["xml"] = str(xml)
    act = robot.articulation.actuators[0]
    if params_json:
        object.__setattr__(act, "json_path", str(params_json))
        object.__setattr__(act, "motor_name", None)
        object.__setattr__(act, "model", None)
        object.__setattr__(act, "_resolved_json_path", str(params_json))
        rec["params_json"] = str(params_json)
    if kp_fw is not None:
        object.__setattr__(act, "kp_fw", float(kp_fw))
        rec["kp_fw"] = float(kp_fw)
    if vin_range is not None:
        object.__setattr__(act, "vin_range", tuple(vin_range))
        rec["vin_range"] = tuple(vin_range)
    if vin_min is not None:
        object.__setattr__(act, "vin_min", float(vin_min))
        rec["vin_min"] = float(vin_min)
    if vin_drop is not None:
        object.__setattr__(act, "vin_drop_resistance_range", tuple(vin_drop))
        rec["vin_drop"] = tuple(vin_drop)
    if delay is not None:
        object.__setattr__(act, "delay_min_lag", int(delay[0]))
        object.__setattr__(act, "delay_max_lag", int(delay[1]))
        rec["delay"] = tuple(delay)
    return rec


def build_env(args):
    """Real training env, single env, pushes off, commands frozen for the keyboard."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = 1
    if args.seed is not None:
        cfg.seed = args.seed
    if args.episode_s is not None:
        cfg.episode_length_s = args.episode_s

    # play=True sets push_robot to a 0.5-1.0 s interval (vs 3-6 s in training):
    # fine for a robustness demo, useless for manual control. Drop the event AND
    # any curriculum that drives it (a curriculum referencing a removed event
    # raises at reset()).
    if "push_robot" in getattr(cfg, "events", {}):
        cfg.events.pop("push_robot")
        for cname in list(getattr(cfg, "curriculum", {})):
            if "push" in cname:
                cfg.curriculum.pop(cname)
        print("[env] play-mode pushes disabled (0.5-1.0 s interval)", flush=True)

    # Freeze every command term: a resample (or an episode reset) would silently
    # overwrite the keypresses. head/body are held at their neutral all-zero
    # command -- 0 is inside every training range, and a random head command
    # would make the robot swing its neck for reasons the operator cannot see.
    for name in ("twist", "head_pose", "body_pose"):
        tcfg = getattr(cfg, "commands", {}).get(name)
        if tcfg is None:
            continue
        tcfg.resampling_time_range = (1e9, 1e9)
        for attr, val in (("rel_standing_envs", 0.0), ("rel_heading_envs", 0.0),
                          ("rel_turn_in_place_envs", 0.0)):
            if hasattr(tcfg, attr):
                setattr(tcfg, attr, val)
        if getattr(tcfg, "heading_command", False):
            tcfg.heading_command = False
            if hasattr(tcfg, "ranges") and hasattr(tcfg.ranges, "heading"):
                tcfg.ranges.heading = None
    print("[env] twist/head_pose/body_pose frozen (no resampling)", flush=True)

    overrides: dict = {}
    if args.preswap:
        overrides = apply_preswap(
            cfg, args.xml or str(PRESWAP_XML), args.params_json or str(XL330_JSON),
            PRESWAP_ACTUATOR["kp_fw"] if args.kp_fw is None else args.kp_fw,
            PRESWAP_ACTUATOR["vin_range"], PRESWAP_ACTUATOR["vin_min"],
            PRESWAP_ACTUATOR["vin_drop"], PRESWAP_ACTUATOR["delay"])
    else:
        if args.xml or args.params_json or args.kp_fw is not None:
            overrides = apply_preswap(
                cfg, args.xml, args.params_json, args.kp_fw,
                tuple(args.vin) if args.vin else None, args.vin_min,
                tuple(args.vin_drop) if args.vin_drop else None,
                tuple(args.delay) if args.delay else None)
    if overrides:
        print("[env] hardware-era overrides: " + ", ".join(f"{k}={v}" for k, v in overrides.items()),
              flush=True)

    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    env = RslRlVecEnvWrapper(env, clip_actions=None)

    # Ground truth: what the BUILT actuator actually resolves to. The cfg writes
    # above are only proposals -- `bam/mjlab.py` lets cfg.kp_fw override the JSON,
    # and __post_init__ resolves paths on a frozen dataclass, so a silent no-op is
    # easy to create and invisible without this line.
    try:
        act_cfg = env.unwrapped.scene.entities["robot"].actuators[0].cfg
        print(f"[env] effective actuator: json={getattr(act_cfg, '_resolved_json_path', None)} "
              f"kp_fw={getattr(act_cfg, 'kp_fw', None)} vin={getattr(act_cfg, 'vin_range', None)} "
              f"vin_min={getattr(act_cfg, 'vin_min', None)} "
              f"delay=({getattr(act_cfg, 'delay_min_lag', None)},{getattr(act_cfg, 'delay_max_lag', None)})",
              flush=True)
    except Exception as e:  # never block the run on a diagnostic
        print(f"[env] actuator introspection failed: {e}", flush=True)
    return env


def install_command_injection(env):
    """Re-apply STATE after every command compute(), and print 1 Hz telemetry.

    Frozen resampling alone is not enough: an episode reset resamples commands,
    so the value is rewritten right after each `compute()` -- the only hook that
    runs every env step regardless of who else touched the buffer.
    """
    unwrapped = env.unwrapped

    def optional_term(name):
        try:
            return unwrapped.command_manager.get_term(name)
        except Exception:
            return None

    twist = unwrapped.command_manager.get_term("twist")
    head = optional_term("head_pose")
    body = optional_term("body_pose")

    acc = {"n": 0, "N": 0, "vx": 0.0, "VX": 0.0, "z": 0.0, "tilt": 0.0}

    def robot_state():
        rob = unwrapped.scene.entities["robot"]
        v = rob.data.root_link_vel_w[0, 0:3]
        q = rob.data.body_link_quat_w[0, 0]
        w, qx, qy, qz = (float(t) for t in q)
        yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        fwd = math.cos(yaw) * float(v[0]) + math.sin(yaw) * float(v[1])
        z = float(rob.data.body_link_pos_w[0, 0][2])
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (qx * qx + qy * qy)))))
        return fwd, z, tilt

    orig_compute = twist.compute

    def compute(dt):
        orig_compute(dt)
        twist.command[:, 0] = STATE.vx
        twist.command[:, 1] = STATE.vy
        twist.command[:, 2] = STATE.wz
        if head is not None:
            head.command[:] = 0.0
        if body is not None:
            body.command[:] = 0.0
        try:
            fwd, z, tilt = robot_state()
            acc["n"] += 1
            acc["N"] += 1
            acc["vx"] += fwd
            acc["VX"] += fwd
            acc["z"] += z
            acc["tilt"] += tilt
            if acc["n"] >= 200:  # 200 substeps = 50 env steps = 1 s
                n, N = acc["n"], acc["N"]
                ec = twist.command[0].tolist()
                print(f"[tele] key({STATE.vx:+.2f},{STATE.vy:+.2f},{STATE.wz:+.2f})"
                      f" env_cmd=({ec[0]:+.2f},{ec[1]:+.2f},{ec[2]:+.2f})"
                      f" | 1s vx={acc['vx'] / n:+.2f}"
                      f" | all-time vx={acc['VX'] / N:+.2f} m/s"
                      f" | z={acc['z'] / n * 1000:.0f} mm"
                      f" | tilt={acc['tilt'] / n:.1f} deg", flush=True)
                acc.update(n=0, vx=0.0, z=0.0, tilt=0.0)
        except Exception:
            pass  # never let telemetry break the control loop

    twist.compute = compute
    return twist


def run_headless(env, policy, seconds: float) -> int:
    """No viewer (no GL, no GPU-window): same telemetry, for CI / remote checks."""
    obs = env.reset()[0]
    n = int(round(seconds / env.unwrapped.step_dt))
    for _ in range(n):
        action = policy(obs)
        obs, _, dones, _ = env.step(action)
        if bool(dones[0]):
            print("[headless] env reset (fall or timeout)", flush=True)
    return 0


def main() -> int:
    global VX_STEP, VX_MAX
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, help="policy ONNX (obs[1,61] -> actions[1,14])")
    ap.add_argument("--task", default="Mjlab-Velocity-Flat-MicroDuck")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--preswap", action="store_true",
                    help="reproduce the XL330-era config (origin/develop MJCF 737 g + "
                         "xl330/m6.json + kp_fw 200 + 6.5-8.2 V + delay 3-6). "
                         "REQUIRED for /home/joyandai/microduck/policies/alpha_*.onnx")
    ap.add_argument("--xml", default=None, help="explicit robot MJCF override")
    ap.add_argument("--params-json", default=None, help="explicit BAM params JSON override")
    ap.add_argument("--kp-fw", type=float, default=None, help="explicit firmware Kp")
    ap.add_argument("--vin", nargs=2, type=float, default=None, help="battery voltage range")
    ap.add_argument("--vin-min", type=float, default=None)
    ap.add_argument("--vin-drop", nargs=2, type=float, default=None)
    ap.add_argument("--delay", nargs=2, type=int, default=None, help="actuator lag range, substeps")
    ap.add_argument("--seed", type=int, default=None,
                    help="env seed; DR (spawn pose, voltage, friction, IMU) makes runs "
                         "differ, so fix it when comparing two policies")
    ap.add_argument("--episode-s", type=float, default=1000.0,
                    help="episode length (default 1000 s so a reset means a FALL, not a timeout)")
    ap.add_argument("--viewer-steps", type=int, default=None,
                    help="quit after N viewer steps (smoke test without a human at the keyboard)")
    ap.add_argument("--headless", action="store_true", help="no viewer; run --seconds then exit")
    ap.add_argument("--seconds", type=float, default=20.0, help="headless run length")
    args = ap.parse_args()

    # ── the vx slot means DIFFERENT THINGS in the two task families ───────────
    # On the velocity task twist[0] is a forward SPEED, so a +-0.4 clamp is right
    # (VX_MAX above). On sit-stand the SAME slot carries the posture FLAG, which
    # is binary: 0 = STAND, 1 = SIT (`SitStandCommand`: "cmd = [sit_flag, 0, 0]",
    # and the obs feeds the raw flag). Keeping the walk clamp there was a real
    # bug in this script: UP moved the flag 0.05 at a time and could never pass
    # 0.4, so the commanded height topped out at stand_z + 0.4*(sit_z - stand_z)
    # = 0.093 m -- a half-crouch that CANNOT demonstrate sitting, and one that
    # sits right on top of the hesitant-hover height we were chasing. So for
    # sit-stand the keys become a latch: UP = sit, DOWN/SPACE = stand.
    if "SitStand" in args.task:
        VX_STEP, VX_MAX = 1.0, 1.0
        print("[play_keyboard_onnx] sit-stand task: UP = SIT (flag 1.0), "
              "DOWN / SPACE = STAND (flag 0.0)", flush=True)

    env = build_env(args)
    policy = OnnxPolicy(args.onnx, args.device)
    install_command_injection(env)

    print("=" * 78, flush=True)
    print(f"[play_keyboard_onnx] faithful BAM env + ONNX + keyboard", flush=True)
    print(f"  onnx      : {args.onnx}", flush=True)
    print(f"  hardware  : {'XL330 pre-swap (737 g, kp_fw 200, 6.5-8.2 V)' if args.preswap else 'current HD-1910'}", flush=True)
    print(f"  cmd start : vx={STATE.vx:+.2f} vy={STATE.vy:+.2f} wz={STATE.wz:+.2f}", flush=True)
    if not args.headless:
        print("  UP/DOWN vx +-0.05 | LEFT/RIGHT vy +-0.05 | A/E wz -+0.2", flush=True)
        print("  SPACE zero | 1/2/3 snap vx=0.1/0.2/0.3 | R reset | Q quit", flush=True)
        print("  keys go to THIS TERMINAL, not the viewer window", flush=True)
    print("=" * 78, flush=True)

    if args.headless:
        rc = run_headless(env, policy, args.seconds)
        env.close()
        return rc

    from mjlab.viewer import NativeMujocoViewer

    unwrapped = env.unwrapped
    viewer = NativeMujocoViewer(env, policy, frame_rate=60.0,
                                key_callback=make_key_handler(unwrapped))

    # Q = quit: BaseViewer.run() loops on is_running() and _interrupted.
    orig_is_running = viewer.is_running

    def is_running() -> bool:
        if STATE.quit:
            return False
        return orig_is_running()

    viewer.is_running = is_running  # type: ignore[method-assign]
    viewer.run(num_steps=args.viewer_steps)
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
