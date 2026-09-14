#!/usr/bin/env python3
"""Keyboard-driven viewer in the REAL training environment (BAM + HD-1910 params).

Why this exists:
- `uv run play` runs the faithful env but offers no keyboard command control. Its viser
  (browser) viewer has a mouse joystick (`velocity_command.py:create_gui`), and the native
  viewer exposes a `key_callback` hook that `play` never wires up for commands.
- `scripts/infer_policy.py` IS keyboard-driven but runs `scene.xml` with raw position
  actuators, NOT the BAM voltage model, so policies trained on BAM behave wrongly there
  (verified: an HD-1910 policy shows `STANDED_OK = NO` in that scene).

So this script = faithful env + keyboard commands, by reusing `play`'s own env/checkpoint
loading and only injecting two things:
  1. the twist command's resampling is frozen (a standing/heading mask or a periodic
     resample would otherwise overwrite your keypresses), and
  2. a key_callback on the native viewer that drives vx / vy / wz, re-applied on every
     `compute()` so an episode reset cannot silently re-randomise the command.

Usage:
    uv run python scripts/play_keyboard.py Mjlab-Velocity-Flat-MicroDuck \
        --checkpoint-file logs/rsl_rl/velocity/<run>/model_14999.pt

    # 也可以启动即带指令（不必先按键），便于快速确认链路：
    PLAY_KB_VX=0.2 uv run python scripts/play_keyboard.py Mjlab-Velocity-Flat-MicroDuck \
        --checkpoint-file <pt>

Keys (printed at startup; keys go to THIS TERMINAL, not the viewer window):
    UP / DOWN      vx += / -= 0.05      (walk forward / backward)
    LEFT / RIGHT   vy += / -= 0.05      (strafe)
    A / E          wz -= / += 0.2       (turn right / left)
    SPACE          zero the command (stand)
    1 / 2 / 3      snap vx to 0.1 / 0.2 / 0.3
    R              reset the environments
    Q              quit

Reading the telemetry (printed once per second):
    key(...)        what your keypresses currently hold
    env_cmd(...)    the command the ENVIRONMENT actually holds -- if this matches
                    `key(...)`, injection is working; if it does not, the env's
                    command manager is overriding you
    all-time vx     running mean forward速度 in the robot's own yaw frame
    z               mean trunk height (a standing/walking duck sits near 90-100 mm)

Why the velocity is reported in the yaw frame: this policy has weak heading hold
(measured: yaw drifts -162..+198 deg in 8 s even with wz=0), so the raw WORLD-frame
forward velocity averages to ~0 while the robot is genuinely walking forward.
Two other verified gotchas this script handles for you:
  * play-mode `push_robot` fires every 0.5-1.0 s (vs 3-6 s in training) and makes
    manual control impossible -> the event (and its curriculum) is dropped here;
  * the twist command keeps resampling and episode resets re-randomise it ->
    resampling is frozen and the command is rewritten on every substep.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

# ── command state shared with the key callback ────────────────────────────────
VX_STEP, VY_STEP, WZ_STEP = 0.05, 0.05, 0.2
VX_MAX = VY_MAX = 0.4
WZ_MAX = 1.0


@dataclass
class CmdState:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0

    def show(self) -> None:
        print(f"[cmd] vx={self.vx:+.2f}  vy={self.vy:+.2f}  wz={self.wz:+.2f} m/s,rad/s",
              flush=True)


STATE = CmdState()

# Optional preset so you can start already walking (and so the injection path can be
# verified without a human at the keyboard):
#   PLAY_KB_VX=0.2 PLAY_KB_WZ=0.5 uv run python scripts/play_keyboard.py ...
import os as _os

STATE.vx = float(_os.environ.get("PLAY_KB_VX", "0.0"))
STATE.vy = float(_os.environ.get("PLAY_KB_VY", "0.0"))
STATE.wz = float(_os.environ.get("PLAY_KB_WZ", "0.0"))

# GLFW key codes as delivered by mujoco.viewer's key_callback
KEY_SPACE = 32
KEY_R, KEY_Q = 82, 81
KEY_A, KEY_E = 65, 69
KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN = 263, 262, 265, 264
KEY_1, KEY_2, KEY_3 = 49, 50, 51


def clamp(v: float, lim: float) -> float:
    return max(-lim, min(lim, v))


def make_key_handler(env_unwrapped, term):
    """Return a key_callback that mutates the twist command in place."""

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
            print("[cmd] zeroed (stand)", flush=True)
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
            print("[cmd] quit requested — close the viewer window or Ctrl+C", flush=True)
            return
        else:
            return
        STATE.show()

    return on_key


def main() -> int:
    import mjlab.scripts.play as play_mod
    from mjlab.viewer import NativeMujocoViewer as _Native

    # ---- 1) freeze the twist command so keypresses are not overwritten --------
    _orig_load_env_cfg = play_mod.load_env_cfg

    def load_env_cfg_frozen(task_id, play=False):
        cfg = _orig_load_env_cfg(task_id, play=play)
        # `play=True` sets push_robot's interval to 0.5-1.0 s (vs 3-6 s in training).
        # That is fine for a robustness demo but makes MANUAL control useless -- the robot
        # is shoved over before you can react. Drop the event and any curriculum that
        # drives it (a curriculum referencing a removed event raises at reset()).
        if "push_robot" in getattr(cfg, "events", {}):
            cfg.events.pop("push_robot")
            for cname in list(getattr(cfg, "curriculum", {})):
                if "push" in cname:
                    cfg.curriculum.pop(cname)
            print("[play_keyboard] play-mode pushes disabled (0.5-1.0 s interval)", flush=True)
        term_cfg = getattr(cfg, "commands", {}).get("twist")
        if term_cfg is not None:
            term_cfg.resampling_time_range = (1e9, 1e9)
            for attr, val in (("rel_standing_envs", 0.0), ("rel_heading_envs", 0.0)):
                if hasattr(term_cfg, attr):
                    setattr(term_cfg, attr, val)
            if getattr(term_cfg, "heading_command", False):
                term_cfg.heading_command = False
                if hasattr(term_cfg, "ranges") and hasattr(term_cfg.ranges, "heading"):
                    term_cfg.ranges.heading = None
        return cfg

    play_mod.load_env_cfg = load_env_cfg_frozen

    # ---- 2) inject the key callback + force the command every step -----------
    class KeyboardNativeViewer(_Native):  # type: ignore[misc, valid-type]
        def __init__(self, env, policy, *a, **kw):
            unwrapped = env.unwrapped
            term = unwrapped.command_manager.get_term("twist")

            # Re-apply our command after every compute(): frozen resampling alone is
            # not enough, because an episode reset still resamples the command.
            orig_compute = term.compute

            # ~1 s telemetry, reported as a MEAN over the window. Instantaneous trunk
            # velocity swings wildly during a gait (foot-strike deceleration), so a
            # single sample is unreadable -- the mean is what matches the command.
            _acc = {"n": 0, "vx": 0.0, "vy": 0.0, "z": 0.0, "N": 0, "VX": 0.0}

            def compute(dt):
                orig_compute(dt)
                term.command[:, 0] = STATE.vx
                term.command[:, 1] = STATE.vy
                term.command[:, 2] = STATE.wz
                try:
                    import math

                    rob = env.unwrapped.scene.entities["robot"]
                    v = rob.data.root_link_vel_w[0, 0:3]
                    # Rotate into the YAW frame: this policy has poor heading hold
                    # (yaw drifts even at wz=0), so the raw world-frame x averages to
                    # ~0 while the robot is genuinely walking forward in its own frame.
                    q = rob.data.body_link_quat_w[0, 0]
                    w, qx, qy, qz = (float(t) for t in q)
                    yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
                    fwd = math.cos(yaw) * float(v[0]) + math.sin(yaw) * float(v[1])
                    lat = -math.sin(yaw) * float(v[0]) + math.cos(yaw) * float(v[1])
                    _acc["n"] += 1
                    _acc["N"] += 1
                    _acc["vx"] += fwd
                    _acc["VX"] += fwd
                    _acc["vy"] += lat
                    _acc["z"] += float(rob.data.body_link_pos_w[0, 0][2])
                    if _acc["n"] >= 200:  # 200 substeps = 50 env steps = 1 s
                        n, N = _acc["n"], _acc["N"]
                        # print the ENV's ACTUAL command too, so command injection is
                        # verified rather than assumed, plus a cumulative mean (a 1 s
                        # window of a gait is dominated by foot-strike oscillation).
                        ec = term.command[0].tolist()
                        print(f"[tele] key({STATE.vx:+.2f},{STATE.vy:+.2f},{STATE.wz:+.2f})"
                              f" env_cmd=({ec[0]:+.2f},{ec[1]:+.2f},{ec[2]:+.2f})"
                              f" | 1s vx={_acc['vx'] / n:+.2f}"
                              f" | all-time vx={_acc['VX'] / N:+.2f} m/s"
                              f" | z={_acc['z'] / n * 1000:.0f} mm", flush=True)
                        _acc.update(n=0, vx=0.0, vy=0.0, z=0.0)
                except Exception:
                    pass  # never let telemetry break the control loop

            term.compute = compute  # <-- install the wrapper (the whole point)
            kw.setdefault("key_callback", make_key_handler(unwrapped, term))
            print("=" * 74, flush=True)
            print("[play_keyboard] faithful BAM env + keyboard control", flush=True)
            print("  UP/DOWN vx +-0.05 | LEFT/RIGHT vy +-0.05 | A/E wz -+0.2", flush=True)
            print("  SPACE zero | 1/2/3 snap vx=0.1/0.2/0.3 | R reset | Q quit", flush=True)
            print("  keys go to THIS TERMINAL, not the viewer window", flush=True)
            print("=" * 74, flush=True)
            super().__init__(env, policy, *a, **kw)

    play_mod.NativeMujocoViewer = KeyboardNativeViewer

    # ---- 3) hand off to play's own CLI/env/checkpoint handling --------------
    play_mod.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
