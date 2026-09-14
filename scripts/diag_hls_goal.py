#!/usr/bin/env python3
"""Diagnose and FIX the HLS servo session registers, then test goal tracking.

Background: the replay's original single-write frame was malformed (an extra
data-length byte — FT-SCS WRITE is [addr, data...] with the length implied by
the frame), so every register write landed one byte off. That put the servos
in constant-speed mode (reg 33 = 1), truncated the angle limits, and made
goal-position writes do nothing. `build_write` is fixed now; this script
writes the session registers CORRECTLY, reads each one back to prove it, and
then tests whether the servo actually tracks a goal position.

Modes:
  --state   read-only: current register values vs intended.
  --setup   write the session registers (torque stays OFF), read-back verify.
  --single  setup → torque ON → one WRITE DATA goal-position → track present.
  --sync    setup → torque ON → SYNC WRITE goal-position (replay-style) → track.
  --batch N setup → torque ON → SYNC WRITE for the first N servos of the
            wire order (find the max group size the firmware accepts).

All motion is ±--move ticks (default ≈13°) and returns to the start position.

Usage:
    uv run --with pyserial python scripts/diag_hls_goal.py --state --id 23
    uv run --with pyserial python scripts/diag_hls_goal.py --setup --id 23
    uv run --with pyserial python scripts/diag_hls_goal.py --single --id 23
    uv run --with pyserial python scripts/diag_hls_goal.py --sync --id 23
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from replay_walk_angles import (  # noqa: E402
    MAX_TICK,
    SERIAL_AVAILABLE,
    WIRE_JOINT_IDS,
    build_sync_write,
    ping,
    read_u16,
    write_register,
)

REG_GOAL_POSITION = 42
REG_PRESENT_POSITION = 56
REG_MODE = 33
REG_ACCEL = 41
REG_GOAL_SPEED = 46
REG_TORQUE_LIMIT = 48
REG_TORQUE_ENABLE = 40
REG_ANGLE_LIMIT_MIN = 9
REG_ANGLE_LIMIT_MAX = 11
REG_MAX_TEMP = 13
REG_RESPONSE_LEVEL = 8
REG_MAX_VOLTAGE = 14
REG_MIN_VOLTAGE = 15
REG_KP = 50
REG_KD = 51
REG_KI = 52

# (addr, value_bytes, size_in_bytes, name) — the session registers the replay
# depends on (mirrors replay apply_startup + gains).
SESSION_REGS = [
    (REG_RESPONSE_LEVEL, b"\x01", 1, "response_level=1"),
    (REG_ANGLE_LIMIT_MIN, (0).to_bytes(2, "little"), 2, "angle_min=0"),
    (REG_ANGLE_LIMIT_MAX, (MAX_TICK).to_bytes(2, "little"), 2, "angle_max=4095"),
    (REG_MAX_TEMP, b"\x46", 1, "max_temp=70"),
    (REG_MAX_VOLTAGE, b"\x8c", 1, "max_voltage=14.0V"),
    (REG_MIN_VOLTAGE, b"\x64", 1, "min_voltage=10.0V"),
    (REG_MODE, b"\x00", 1, "mode=0 (position servo)"),
    (REG_ACCEL, b"\x00", 1, "accel=max"),
    (REG_GOAL_SPEED, (32767).to_bytes(2, "little"), 2, "goal_speed=max"),
    (REG_TORQUE_LIMIT, (980).to_bytes(2, "little"), 2, "torque_limit=98%"),
    (REG_KP, b"\x20", 1, "kp=32"),
    (REG_KD, b"\x20", 1, "kd=32"),
    (REG_KI, b"\x00", 1, "ki=0"),
]


def read_val(ser, pid: int, addr: int, size: int) -> int | None:
    if size == 1:
        from replay_walk_angles import read_regs
        b = read_regs(ser, pid, addr, 1)
        return b[0] if b else None
    v = read_u16(ser, pid, addr)
    return v


def setup_regs(ser, pid: int, verify: bool = True) -> dict:
    """Write session registers (torque OFF afterwards), optionally verify."""
    write_register(ser, pid, REG_TORQUE_ENABLE, b"\x00")  # no motion during setup
    summary = {}
    for addr, data, size, name in SESSION_REGS:
        ok = write_register(ser, pid, addr, data)
        want = int.from_bytes(data, "little")
        got = None
        if verify:
            got = read_val(ser, pid, addr, size)
            ok = ok and got == want
        summary[name] = (want, got, ok)
        if not ok:
            print(f"    [FAIL] {name}: wrote {want}, read back {got}")
        else:
            print(f"    [ok]   {name}: {want} (read back {got})")
    return summary


def read_state(ser, pid: int) -> None:
    print(f"[i] register state of id {pid} — intended values in brackets:")
    for addr, want_b, size, name in SESSION_REGS:
        got = read_val(ser, pid, addr, size)
        want = int.from_bytes(want_b, "little")
        mark = "" if got is None else ("✓" if got == want else f"✗ (want {want})")
        print(f"    {name:<24} = {got} {mark}")
    got = read_val(ser, pid, REG_PRESENT_POSITION, 2)
    print(f"    present(56)              = {got}")
    # factory limits (read-only, "出厂参数") that bound real speed
    from replay_walk_angles import read_regs
    for addr in (84, 85, 86, 81):
        b = read_regs(ser, pid, addr, 1)
        print(f"    factory reg {addr}        = {b[0] if b else None}")


def track(ser, pid: int, label: str, write_fn) -> None:
    start = read_val(ser, pid, REG_PRESENT_POSITION, 2)
    if start is None:
        raise SystemExit("[!] cannot read present position")
    target = int(np.clip(start + 150, 0, MAX_TICK))
    print(f"[i] ({label}) id={pid}: present={start} → target {target} "
          f"(+150 ticks ≈ 13°)")
    write_register(ser, pid, REG_TORQUE_ENABLE, b"\x01")
    time.sleep(0.2)
    t0 = time.perf_counter()
    write_fn(target)
    samples = []
    while time.perf_counter() - t0 < 1.5:
        v = read_val(ser, pid, REG_PRESENT_POSITION, 2)
        if v is not None:
            samples.append(v)
        time.sleep(0.05)
    if samples:
        arr = np.array(samples)
        moved = abs(int(arr[-1]) - start) >= 10
        print(f"    present: start={start} final={int(arr[-1])} "
              f"min={int(arr.min())} max={int(arr.max())} → "
              f"{'[OK] 跟踪目标' if moved else '[!!] 不动 —— 目标仍没生效'}")
    # back to start via single write (known-good)
    write_register(ser, pid, REG_GOAL_POSITION, start.to_bytes(2, "little"))
    time.sleep(0.8)
    print(f"    回位: 现在 {read_val(ser, pid, REG_PRESENT_POSITION, 2)} (目标 {start})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=None)
    ap.add_argument("--baud", type=int, default=1000000)
    ap.add_argument("--id", type=int, default=23, help="test servo (23 = left_knee)")
    ap.add_argument("--state", action="store_true")
    ap.add_argument("--setup", action="store_true")
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--profile", action="store_true",
                    help="big step + 4 s trajectory sampling (velocity profile); "
                         "keep the tested leg FREE, hold the duck by the body")
    ap.add_argument("--move", type=int, default=600,
                    help="step size for --profile (default 600 ticks ≈ 53°)")
    args = ap.parse_args()

    if not SERIAL_AVAILABLE:
        raise SystemExit("[!] pyserial missing: uv run --with pyserial python "
                         "scripts/diag_hls_goal.py")
    import serial
    from serial.tools import list_ports

    if args.port is None:
        ports = sorted({p.device for p in list_ports.comports()
                        if p.device.startswith(("/dev/ttyACM", "/dev/ttyUSB"))})
        if not ports:
            raise SystemExit("[!] no serial port")
        args.port = ports[0]
    ser = serial.Serial(args.port, args.baud, timeout=0.3)
    ser.reset_input_buffer()
    if not ping(ser, args.id):
        raise SystemExit(f"[!] id {args.id} no answer")

    try:
        if args.state or (not any((args.single, args.sync, args.batch, args.profile))
                          and not args.setup):
            read_state(ser, args.id)
            return
        if args.setup:
            print(f"[i] writing session registers on id {args.id} (torque OFF):")
            setup_regs(ser, args.id)
            return
        if args.profile:
            print(f"[i] correcting session registers on id {args.id} first:")
            setup_regs(ser, args.id)
            start = read_val(ser, args.id, REG_PRESENT_POSITION, 2)
            target = int(np.clip(start + args.move, 0, MAX_TICK))
            print(f"[i] (profile) id={args.id}: {start} → {target} "
                  f"(Δ{args.move:+d} ticks ≈ {args.move*360/4096:.1f}°), "
                  f"sample 4 s — 腿要悬空自由, 手别碰!")
            write_register(ser, args.id, REG_TORQUE_ENABLE, b"\x01")
            time.sleep(0.2)
            t0 = time.perf_counter()
            prev, t_prev = start, t0
            speeds = []
            rows = []
            while time.perf_counter() - t0 < 4.0:
                v = read_val(ser, args.id, REG_PRESENT_POSITION, 2)
                now = time.perf_counter()
                if v is not None:
                    rows.append((now - t0, v))
                    if prev is not None and now > t_prev:
                        speeds.append((v - prev) / (now - t_prev))
                    prev, t_prev = v, now
                time.sleep(0.05)
            print(f"    {'t(ms)':>6} {'tick':>6} {'deg':>6}  {'deg/s':>7}")
            for t, v in rows[::2]:
                print(f"    {t*1000:6.0f} {v:>6} {v*360/4096:6.1f}  {'':>7}")
            if speeds:
                s = np.array(speeds)
                s = s[s > 0]
                if len(s):
                    print(f"    peak {np.percentile(s, 95)*360/4096:6.1f} deg/s  "
                          f"(max sample {s.max()*360/4096:6.1f} deg/s)")
            print(f"    final: {rows[-1][1]} (target {target})")
            write_register(ser, args.id, REG_GOAL_POSITION, start.to_bytes(2, "little"))
            time.sleep(1.0)
            print(f"    回位: 现在 {read_val(ser, args.id, REG_PRESENT_POSITION, 2)} "
                  f"(目标 {start})")
            return
        # motion tests: correct the session registers first
        print(f"[i] correcting session registers on id {args.id} first:")
        bad = [n for n, (_, _, ok) in
               setup_regs(ser, args.id).items() if not ok]
        if bad:
            print(f"[!] {len(bad)} registers failed: {bad}")
        if args.single:
            track(ser, args.id, "single WRITE DATA",
                  lambda t: write_register(ser, args.id, REG_GOAL_POSITION,
                                           t.to_bytes(2, "little")))
        elif args.sync or args.batch:
            ids = (WIRE_JOINT_IDS if not args.batch else WIRE_JOINT_IDS[:args.batch])
            start = read_val(ser, args.id, REG_PRESENT_POSITION, 2)
            target = int(np.clip(start + 150, 0, MAX_TICK))
            print(f"[i] (sync) id={args.id}: present={start} → target {target} "
                  f"(+150 ticks ≈ 13°), group={ids}")
            write_register(ser, args.id, REG_TORQUE_ENABLE, b"\x01")
            time.sleep(0.2)
            t0 = time.perf_counter()
            tags_ticks = [read_val(ser, p, REG_PRESENT_POSITION, 2) for p in ids]
            ticks_out = [target if p == args.id else (v or 2048)
                         for p, v in zip(ids, tags_ticks)]
            ser.write(build_sync_write(
                ids, REG_GOAL_POSITION,
                [t.to_bytes(2, "little") for t in ticks_out]))
            samples = []
            while time.perf_counter() - t0 < 1.5:
                v = read_val(ser, args.id, REG_PRESENT_POSITION, 2)
                if v is not None:
                    samples.append(v)
                time.sleep(0.05)
            arr = np.array(samples)
            moved = abs(int(arr[-1]) - start) >= 10
            print(f"    present: start={start} final={int(arr[-1])} "
                  f"min={int(arr.min())} max={int(arr.max())} → "
                  f"{'[OK] 跟踪目标' if moved else '[!!] 不动 —— sync 帧仍没生效'}")
            write_register(ser, args.id, REG_GOAL_POSITION, start.to_bytes(2, "little"))
            time.sleep(0.8)
            print(f"    回位: 现在 {read_val(ser, args.id, REG_PRESENT_POSITION, 2)} "
                  f"(目标 {start})")
    finally:
        write_register(ser, args.id, REG_TORQUE_ENABLE, b"\x00")
        ser.close()
        print("[i] torque off, port closed")


if __name__ == "__main__":
    main()
