#!/usr/bin/env python3
"""Replay a recorded walk trajectory on the real duck's HL-2909-C001 servos.

Reads the CSV produced by scripts/record_walk_angles.py and streams the 14
joint target angles to the 15 FT-SCS servos (14 policy joints + mouth) at
50 Hz via one SYNC WRITE per tick — the duck should be HELD UP / unsupported
the whole time; it will pedal like it is walking.

Angle convention (must match the runtime, duck-control/src/ftbus.rs):
    servo tick = round((joint_rad + pi) * 4096 / (2*pi)), clamped to [0, 4095]
i.e. servo mid (2048) is joint angle 0, sign +1 on every joint. The servos
must have been zero-calibrated ("校到中位" / 位置偏移 reg 31) at the joint-zero
pose. `--verify-home` checks this: after ramping to the home pose it reads the
present positions back and prints per-joint deviation; `--apply-home-offset`
then bakes the measured deviation in as a constant per-joint correction (for
ducks zeroed at the HOME pose instead of the joint-zero pose).

Safety
------
* Torque stays OFF until the interactive confirm (or --yes).
* Ctrl+C at any time ramps back to the home pose; torque stays on unless
  --finish off.
* Every target is clamped to the single-turn register range, so a bad row can
  never wrap the servo.

Usage:
    uv run --with pyserial python scripts/replay_walk_angles.py \\
        --csv walk_angles.csv [--port /dev/ttyACM0] [--speed 1.0]
    # offline frame verification, opens nothing:
    uv run --with pyserial python scripts/replay_walk_angles.py --csv walk_angles.csv --dry-run
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

# pyserial is only needed for the real (non-dry-run) path; the protocol
# builders below are import-able without it (so unit tests can verify them).
try:
    import serial
    from serial.tools import list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    serial = None
    list_ports = None
    SERIAL_AVAILABLE = False

# ── FT-SCS protocol primitives (same formulas as scripts/read_hls_registers.py;
#    kept local so this script is standalone and import-able without pyserial) ──
HEAD = b"\xFF\xFF"
INST_PING = 0x01
INST_READ = 0x02


def frame_checksum(pid: int, length: int, instr: int, params: bytes) -> int:
    return (~(pid + length + instr + sum(params))) & 0xFF


def build_ping(pid: int) -> bytes:
    return HEAD + bytes([pid, 2, INST_PING]) + bytes([frame_checksum(pid, 2, INST_PING, b"")])


def build_read(pid: int, addr: int, nbytes: int) -> bytes:
    params = bytes([addr, nbytes])
    ln = len(params) + 2
    return HEAD + bytes([pid, ln, INST_READ]) + params + bytes(
        [frame_checksum(pid, ln, INST_READ, params)])


def read_frame(ser, timeout: float = 0.3):
    """Read one reply frame: FF FF ID LEN ERR? DATA... CHECK. → (id, err, data) | None."""
    ser.timeout = timeout
    head = ser.read(2)
    if len(head) != 2 or head != HEAD:
        return None
    pid = ser.read(1)
    ln = ser.read(1)
    if len(pid) != 1 or len(ln) != 1:
        return None
    body = ser.read(ln[0])  # ERR(1) + DATA + CHECK(1) == LEN
    if len(body) != ln[0]:
        return None
    err, data, chk = body[0], body[1:-1], body[-1]
    expect = (~(pid[0] + ln[0] + err + sum(data))) & 0xFF
    if chk != expect:
        return None
    return pid[0], err, data


def ping(ser, pid: int) -> bool:
    ser.reset_input_buffer()
    ser.write(build_ping(pid))
    return read_frame(ser) is not None


def read_regs(ser, pid: int, addr: int, nbytes: int):
    """Read nbytes from addr → bytes or None."""
    ser.reset_input_buffer()
    ser.write(build_read(pid, addr, nbytes))
    f = read_frame(ser, timeout=0.3)
    if f is None or f[0] != pid:
        return None
    return bytes(f[2][:nbytes])

# ── constants lifted from duck-control (runtime), keep in sync ───────────────
# Wire order (15 servos): left leg, neck/head/mouth, right leg.
WIRE_JOINT_IDS = [20, 21, 22, 23, 24, 30, 31, 32, 33, 34, 10, 11, 12, 13, 14]
MOUTH_INDEX = 9          # policy has no mouth; this wire slot gets 0.0 rad
RAD_PER_TICK = 2.0 * math.pi / 4096.0
MAX_TICK = 4095

# FT-SCS registers (HLS memory table)
REG_GOAL_POSITION = 42   # 2B, 0.087° (0.088°-ish) per tick, 0..4095 single-turn
REG_PRESENT_POSITION = 56
REG_MODE = 33
REG_ACCEL = 41
REG_GOAL_SPEED = 46
REG_TORQUE_LIMIT = 48
REG_TORQUE_ENABLE = 40
REG_KP = 50
REG_KD = 51
REG_KI = 52
REG_ANGLE_LIMIT_MIN = 9   # 2B, EEPROM
REG_ANGLE_LIMIT_MAX = 11  # 2B, EEPROM
REG_MAX_TEMPERATURE = 13
REG_RESPONSE_LEVEL = 8
REG_MAX_VOLTAGE = 14
REG_MIN_VOLTAGE = 15

# Default home pose (rad), wire order = runtime model.rs DEFAULT_POSITION.
WIRE_JOINT_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll", "mouth",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
HOME_POSE_WIRE = np.array([
    0.0,           # left_hip_yaw
    -0.0873,       # left_hip_roll
    -0.4579,       # left_hip_pitch
    -0.0049,       # left_knee
    0.4530,        # left_ankle
    0.3491,        # neck_pitch
    0.3491,        # head_pitch
    0.0,           # head_yaw
    0.0,           # head_roll
    0.0,           # mouth
    0.0,           # right_hip_yaw
    0.0873,        # right_hip_roll
    0.4579,        # right_hip_pitch
    0.0049,        # right_knee
    -0.4530,       # right_ankle
], dtype=np.float64)


def rad_to_tick(rad: float) -> int:
    """Joint radians → single-turn servo tick, clamped (ftbus.rs convention)."""
    count = int(round((rad + math.pi) / RAD_PER_TICK))
    return max(0, min(MAX_TICK, count))


def tick_to_rad(tick: int) -> float:
    return tick * RAD_PER_TICK - math.pi


def build_sync_write(ids: list[int], addr: int, data_per_id: list[bytes]) -> bytes:
    """FT-SCS SYNC WRITE frame (0x83, broadcast FE, no reply).

    Format per docs/ftscs_protocol.md §4.6:
      FF FF FE LEN 83 ADDR LEN2 [ID DATALEN2...] CHECK
    with LEN = (LEN2+1)*n + 4.
    """
    assert len(ids) == len(data_per_id)
    n = len(ids)
    l2 = len(data_per_id[0])
    assert all(len(d) == l2 for d in data_per_id)
    params = bytearray([addr, l2])
    for pid, d in zip(ids, data_per_id):
        params += bytes([pid]) + d
    ln = (l2 + 1) * n + 4
    return HEAD + bytes([0xFE, ln, 0x83]) + bytes(params) + bytes(
        [frame_checksum(0xFE, ln, 0x83, bytes(params))])


def build_write(pid: int, addr: int, data: bytes) -> bytes:
    """FT-SCS WRITE DATA frame (0x03), returns error/status reply.

    Protocol (ftscs_protocol.md §4.3): params = [addr, data...] — the data
    length is implied by the frame length, there is NO data-length byte.
    (A length byte here shifts every write by one address: writing MODE=0
    would set MODE=1, exactly the bug that put the servos in constant-speed
    mode and froze the replay.)
    """
    params = bytes([addr]) + bytes(data)
    ln = len(params) + 2
    return HEAD + bytes([pid, ln, 0x03]) + params + bytes(
        [frame_checksum(pid, ln, 0x03, params)])


def write_register(ser, pid: int, addr: int, data: bytes) -> bool:
    """Single WRITE DATA; True only on a reply with err byte == 0."""
    ser.reset_input_buffer()
    ser.write(build_write(pid, addr, data))
    f = read_frame(ser)
    if f is None:
        return False
    pid_r, err, _ = f
    if pid_r != pid or err != 0:
        return False
    return True


def read_u16(ser, pid: int, addr: int):
    b = read_regs(ser, pid, addr, 2)
    if b is None or len(b) < 2:
        return None
    return int(b[0]) | (int(b[1]) << 8)


def policy_to_wire_joint(policy_idx: int) -> int:
    """Policy joint index (14) → wire joint index (15, mouth skipped)."""
    return policy_idx if policy_idx < MOUTH_INDEX else policy_idx + 1


def apply_startup(ser) -> int:
    """Write + VERIFY the session registers on every servo (mirrors runtime
    HlsBusIo::apply_startup). Returns the number of mismatches found.

    The frames were once malformed (a stray data-length byte shifted every
    write by one address — that is how the duck ended up in constant-speed
    mode), so every register is read back and reported. EEPROM regs write
    best-effort (persist only while unlocked); session behavior is what the
    replay needs.
    """
    session = [
        (REG_MODE, b"\x00", 1, "mode=0"),
        (REG_ACCEL, b"\x00", 1, "accel=max"),
        (REG_GOAL_SPEED, (32767).to_bytes(2, "little"), 2, "goal_speed=max"),
        (REG_TORQUE_LIMIT, (980).to_bytes(2, "little"), 2, "torque_limit=98%"),
        (REG_ANGLE_LIMIT_MIN, (0).to_bytes(2, "little"), 2, "angle_min=0"),
        (REG_ANGLE_LIMIT_MAX, (MAX_TICK).to_bytes(2, "little"), 2, "angle_max=4095"),
        (REG_MAX_TEMPERATURE, b"\x46", 1, "max_temp=70C"),
        (REG_RESPONSE_LEVEL, b"\x01", 1, "response_level=1"),
        (REG_MAX_VOLTAGE, b"\x8c", 1, "max_voltage=14.0V"),
        (REG_MIN_VOLTAGE, b"\x64", 1, "min_voltage=10.0V"),
    ]
    bad = 0
    for pid in WIRE_JOINT_IDS:
        for addr, data, nbytes, what in session:
            wrote = write_register(ser, pid, addr, data)
            got = read_regs(ser, pid, addr, nbytes)
            want = int.from_bytes(data, "little")
            ok = (wrote and got is not None and len(got) >= nbytes
                  and int.from_bytes(got[:nbytes], "little") == want)
            if not ok:
                bad += 1
                wr = "ok" if wrote else "FAIL"
                print(f"[!] {what} on id {pid}: write={wr} readback={got}")
    return bad


def read_present_ticks(ser, ids: list[int]) -> list[int]:
    out = []
    for pid in ids:
        v = read_u16(ser, pid, REG_PRESENT_POSITION)
        out.append(v if v is not None else -1)
    return out


def print_zero_check(ids: list[int], present_ticks: list[int]) -> None:
    """Print present position vs the home-pose ticks per servo.

    This is the zero-convention check for a duck standing in its home pose:
    the ticks must match the table below (e.g. left_hip_pitch ≈ 1749, NOT
    2048) because the zero is the joint MECHANICAL zero, not the standing
    pose. A column full of 2048 means the servos were 校到中位 while the duck
    was standing — every joint then carries the home angle as a constant
    offset (up to ~26 deg) and the replay would be wrong.
    """
    home_tick = [rad_to_tick(float(v)) for v in HOME_POSE_WIRE]
    print(f"{'id':>3} {'joint':<15} {'present':>7} {'home':>6} {'dev':>5}  deg")
    for j, pid in enumerate(ids):
        got = present_ticks[j]
        if got < 0:
            print(f"{pid:>3} {WIRE_JOINT_NAMES[j]:<15}   <no reply>")
            continue
        want = home_tick[j]
        dev = (got - want) % 4096
        if dev > 2048:
            dev -= 4096
        print(f"{pid:>3} {WIRE_JOINT_NAMES[j]:<15} {got:>7} {want:>6} {dev:>+5}  "
              f"{abs(dev) * RAD_PER_TICK * 180 / math.pi:5.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--port", default=None, help="default: first ttyACM/ttyUSB")
    ap.add_argument("--baud", type=int, default=1000000)
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback rate (0.6 => 40 ms/row, command rate drops to 25 Hz)")
    ap.add_argument("--stretch", type=float, default=1.0,
                    help="slow the gait by resampling to full 50 Hz "
                         "(1.5 => 2.63 Hz gait becomes 1.75 Hz, command rate stays 50 Hz)")
    ap.add_argument("--loop", action="store_true", default=True)
    ap.add_argument("--no-loop", action="store_true", help="play through once, then home")
    ap.add_argument("--loop-window", default="auto",
                    help="auto = JSON loop_window_steps (seam-free); "
                         "none = whole file; N = first N CSV rows")
    ap.add_argument("--ramp-s", type=float, default=1.5,
                    help="seconds to interpolate from the current pose to HOME")
    ap.add_argument("--finish", choices=("home", "off"), default="home",
                    help="home = ramp to HOME and keep torque; off = torque off")
    ap.add_argument("--kp", type=int, default=32, help="reg 50 Kp (calibrated = 32)")
    ap.add_argument("--kd", type=int, default=32, help="reg 51 Kd (calibrated = 32)")
    ap.add_argument("--ki", type=int, default=0)
    ap.add_argument("--verify-home", action="store_true", default=True)
    ap.add_argument("--no-verify-home", action="store_true")
    ap.add_argument("--apply-home-offset", action="store_true",
                    help="bake the measured home deviation into every target as a "
                         "per-joint constant (for ducks zeroed at the HOME pose, "
                         "not the joint-zero pose)")
    ap.add_argument("--monitor-s", type=float, default=1.0,
                    help="seconds between reading back left_knee as a sanity check; 0=off")
    ap.add_argument("--yes", action="store_true", help="skip the interactive confirm")
    ap.add_argument("--dry-run", action="store_true",
                    help="offline: verify frames/checksums against the protocol doc "
                         "examples and print the replay plan — opens no port")
    args = ap.parse_args()

    # ---- read the recording -------------------------------------------------
    idx2wire = {i: policy_to_wire_joint(i) for i in range(14)}
    jnames = None
    meta = {}
    csvp = Path(args.csv)
    side = csvp.with_suffix(".json")
    if side.exists():
        try:
            meta = json.loads(side.read_text())
            jnames = meta.get("joint_names")
        except Exception:
            meta = {}

    with csvp.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if not rows:
        raise SystemExit(f"[!] {csvp} is empty")

    if jnames is None:
        jnames = [k[7:] for k in fieldnames if k.startswith("target_")]
    if len(jnames) != 14:
        raise SystemExit(f"[!] expected 14 target_* columns, found {len(jnames)}")
    tgt = np.array([[float(r[f"target_{n}"]) for n in jnames] for r in rows])
    times = np.array([float(r.get("time", i * 0.02)) for i, r in enumerate(rows)])
    print(f"[i] {csvp}: {len(rows)} rows @ {np.diff(times).mean()*1000:.1f} ms "
          f"({times[-1]-times[0]:.2f} s)")

    # loop window
    if args.no_loop:
        nloop = len(tgt)
        loop = False
    elif args.loop_window == "auto":
        nloop = int(meta.get("loop_window_steps") or len(tgt))
        loop = True
        if meta.get("loop_window_steps"):
            print(f"[i] loop window: first {nloop} rows "
                  f"({meta.get('loop_period_steps')}-step period, seam-free)")
    elif args.loop_window == "none":
        nloop = len(tgt)
        loop = True
    else:
        nloop = min(int(args.loop_window), len(tgt))
        loop = True
    nloop = max(nloop, 2)

    # wire targets (15) for every loop row
    wire_tgt = np.zeros((nloop, 15))
    for s in range(14):
        wire_tgt[:, idx2wire[s]] = tgt[:nloop, s]
    wire_tgt[:, MOUTH_INDEX] = 0.0

    # ---- slow-motion via 50 Hz resampling (keeps the command rate at 50 Hz) ----
    if args.stretch != 1.0:
        period = int(meta.get("loop_period_steps") or 0)
        if period > 0 and nloop % period == 0:
            n_per = nloop // period
            nper_new = max(3, int(round(period * args.stretch)))
            t_old = np.linspace(0.0, 1.0, period, endpoint=False)
            t_old_ext = np.concatenate([t_old, [1.0]])
            seg_old = wire_tgt[:period]
            seg_ext = np.vstack([seg_old, seg_old[:1]])
            t_new = np.linspace(0.0, 1.0, nper_new, endpoint=False)
            one = np.stack([np.interp(t_new, t_old_ext, seg_ext[:, c])
                            for c in range(15)], axis=1)
            wire_tgt = np.tile(one, (n_per, 1))
            nloop = int(nper_new * n_per)
            note = f"resampled {period}->{nper_new} steps/period, {n_per} periods"
        else:
            dur = times[min(nloop, len(times)) - 1] - times[0]
            t_old = np.linspace(0.0, dur, nloop, endpoint=False)
            t_old_ext = np.concatenate([t_old, [dur]])
            seg_ext = np.vstack([wire_tgt, wire_tgt[:1]])
            n_new = max(3, int(round(nloop * args.stretch)))
            t_new = np.linspace(0.0, dur, n_new, endpoint=False)
            wire_tgt = np.stack([np.interp(t_new, t_old_ext, seg_ext[:, c])
                                 for c in range(15)], axis=1)
            nloop = n_new
            note = f"resampled whole window x{args.stretch}"
        print(f"[i] --stretch {args.stretch}: {note}; "
              f"gait ≈ {meta.get('gait_freq_hz', 0)/args.stretch:.2f} Hz")

    # ---- dry run ------------------------------------------------------------
    if args.dry_run:
        print("\n=== dry-run: protocol frame verification (no port opened) ===")
        # doc §4.6 example: IDs 1-4, addr 0x2A, 6-byte payload each
        ex = build_sync_write([1, 2, 3, 4], 0x2A,
                              [bytes.fromhex("00080000e803")] * 4)
        want = bytes.fromhex("fffffe20832a060100080000e8030200080000e8030300080000e8030400080000e80358")
        print("  sync_write  :", ex.hex(" "))
        print("  doc example :", want.hex(" "))
        print("  MATCH" if ex == want else "  MISMATCH!")
        assert ex == want, "sync-write frame does not match the protocol doc"
        # doc §4.1 PING example: FF FF 01 02 01 FB
        print("  ping(1)     :", build_ping(1).hex(" "),
              "MATCH" if build_ping(1) == bytes.fromhex("ffff010201fb") else "MISMATCH!")
        # doc §4.2 READ example: FF FF 01 04 02 38 02 BE
        print("  read(1,38,2):", build_read(1, 0x38, 2).hex(" "),
              "MATCH" if build_read(1, 0x38, 2) == bytes.fromhex("ffff0104023802be")
              else "MISMATCH!")
        print("\n=== replay plan ===")
        print(f"  ids (wire order): {WIRE_JOINT_IDS}")
        dur = times[min(nloop, len(times)) - 1] - times[0]
        print(f"  cmd: speed={args.speed}x loop={loop} window={nloop} rows "
              f"({dur:.2f} s of gait)")
        print(f"  target range (rad): [{wire_tgt.min():+.3f}, {wire_tgt.max():+.3f}]")
        print(f"  target range (ticks): [{rad_to_tick(float(wire_tgt.min()))}, "
              f"{rad_to_tick(float(wire_tgt.max()))}] (mid=2048)")
        d = np.abs(np.diff(wire_tgt, axis=0)).max()
        print(f"  max per-tick demand: {d:.3f} rad/20ms -> {d/0.02:.1f} rad/s "
              f"(servo max ~19.2 rad/s, firmware rate-limits)")
        return 0

    # ---- find the port ------------------------------------------------------
    if not SERIAL_AVAILABLE:
        raise SystemExit("[!] pyserial missing: uv run --with pyserial python "
                         "scripts/replay_walk_angles.py --csv walk_angles.csv")
    if args.port is None:
        ports = sorted({p.device for p in list_ports.comports()
                        if p.device.startswith(("/dev/ttyACM", "/dev/ttyUSB"))},
                       key=lambda d: d)
        if not ports:
            raise SystemExit("[!] no serial port found — plug in the USB-TTL "
                             "(use --port; sudo group dialout?)")
        args.port = ports[0]
        print(f"[i] auto-selected port: {args.port}")
    ser = serial.Serial(args.port, args.baud, timeout=0.3)
    ser.reset_input_buffer()

    # ---- presence check + startup -------------------------------------------
    missing = [pid for pid in WIRE_JOINT_IDS if not ping(ser, pid)]
    if missing:
        ser.close()
        raise SystemExit(f"[!] servos missing: {missing} (12V power? signal wire?)")
    print(f"[i] all {len(WIRE_JOINT_IDS)} servos answered on {args.port} @ {args.baud}")

    print("[i] applying startup registers (mode/speed/torque limit/gains, with read-back)…")
    bad = apply_startup(ser)
    for pid in WIRE_JOINT_IDS:
        for addr, want in ((REG_KP, args.kp), (REG_KD, args.kd), (REG_KI, args.ki)):
            if not write_register(ser, pid, addr, bytes([want])):
                bad += 1
                print(f"[!] gain reg {addr} on id {pid}: write FAIL")
    if bad:
        print(f"[!] {bad} register writes mismatched — see above; the replay "
              f"will likely misbehave, fix first")
    else:
        print("[i] all session registers verified ok (mode=position, limits [0,4095], …)")

    present = read_present_ticks(ser, WIRE_JOINT_IDS)
    if any(t < 0 for t in present):
        print(f"[!] couldn't read present position on some joint: {present}")
        ser.close()
        raise SystemExit("aborting (a servo is not answering reads)")
    print("[i] zero-convention check (duck standing in its home pose):"
          " present should match 'home' — NOT all 2048!")
    print_zero_check(WIRE_JOINT_IDS, present)
    home_tick = [rad_to_tick(float(v)) for v in HOME_POSE_WIRE]

    # ---- home calibration check (zero convention) ---------------------------
    off = np.zeros(15)
    if args.verify_home and not args.no_verify_home:
        # ramp to home with torque OFF would just deflect; instead we ramp with
        # torque ON — do that after the confirm below. So the verify happens
        # post-ramp; here only print the assumption.
        print("[i] will verify the home-pose zero convention after the ramp "
              "(--verify-home)")

    if not args.yes:
        try:
            input("\n>>> 鸭子已拿稳/悬空, 舵机将要上电开扭力 — 回车开始, "
                  "Ctrl+C 随时回到 HOME:\n")
        except EOFError:
            pass

    # torque on
    for pid in WIRE_JOINT_IDS:
        write_register(ser, pid, REG_TORQUE_ENABLE, b"\x01")
    print("[i] torque ON")

    # ---- ramp present → HOME --------------------------------------------------
    steps = max(1, int(args.ramp_s / 0.02))
    ramps = np.linspace(np.array(present, dtype=float), np.array(home_tick, dtype=float),
                        steps + 1)
    t0 = time.perf_counter()
    for i in range(1, steps + 1):
        ticks = [int(max(0, min(MAX_TICK, int(round(v))))) for v in ramps[i]]
        ser.write(build_sync_write(WIRE_JOINT_IDS, REG_GOAL_POSITION,
                                   [t.to_bytes(2, "little") for t in ticks]))
        _sleep_until(t0 + i * 0.02)

    # ---- verify home ----------------------------------------------------------
    time.sleep(0.5)
    present_home = read_present_ticks(ser, WIRE_JOINT_IDS)
    if all(t >= 0 for t in present_home):
        for j, (got, want) in enumerate(zip(present_home, home_tick)):
            d = (got - want) % 4096
            if d > 2048:
                d -= 4096
            off[j] = d
        worst = float(np.abs(off).max())
        print(f"[i] home verify: worst deviation {worst:.1f} ticks "
              f"({worst * RAD_PER_TICK * 180 / math.pi:.1f} deg)")
        big = [f"{WIRE_JOINT_IDS[j]}:{off[j]:+.0f}" for j in range(15) if abs(off[j]) > 34]
        if big:
            print(f"[!] joints deviating >3 deg: {', '.join(big)}")
            print("    the servo zeros are probably NOT at the joint-zero pose "
                  "(校到中位 was done at HOME, not at the CAD zero pose).")
            if args.apply_home_offset:
                print(f"[i] applying per-joint offset {off.round(0).astype(int).tolist()}")
            elif args.yes:
                print("    continuing WITHOUT offset correction (--apply-home-offset "
                      "to bake it in)")
            else:
                ans = input("    apply the measured offset to every target? [y/N] ").strip().lower()
                args.apply_home_offset = ans in ("y", "yes")

    def to_tick(policy_rad15: np.ndarray) -> list[int]:
        ticks = [rad_to_tick(float(v)) for v in policy_rad15]
        if args.apply_home_offset:
            ticks = [int(max(0, min(MAX_TICK, t + int(off[j]))))
                     for j, t in enumerate(ticks)]
        return ticks

    # ---- playback loop ----------------------------------------------------------
    print(f"[i] playing {nloop} rows @ {(times[min(nloop, len(times))-1]-times[0])/nloop*1000:.2f} ms "
          f"x{args.speed} — loop={'ON' if loop else 'OFF'} (Ctrl+C → HOME)")
    try:
        li = 0
        next_monitor = 0.0
        while True:
            start = time.perf_counter()
            ticks = to_tick(wire_tgt[li % nloop])
            ser.write(build_sync_write(WIRE_JOINT_IDS, REG_GOAL_POSITION,
                                       [t.to_bytes(2, "little") for t in ticks]))
            if args.monitor_s > 0 and time.perf_counter() >= next_monitor:
                v = read_u16(ser, WIRE_JOINT_IDS[idx2wire[3]], REG_PRESENT_POSITION)  # left_knee
                if v is not None:
                    d = (v - ticks[idx2wire[3]]) % 4096
                    if d > 2048:
                        d -= 4096
                    print(f"    [mon] left_knee tick={v} (target {ticks[idx2wire[3]]}, "
                          f"Δ{d:+d} = {abs(d)*RAD_PER_TICK*180/math.pi:.1f} deg)")
                next_monitor = time.perf_counter() + args.monitor_s
            li += 1
            if not loop and li >= nloop:
                break
            period = 0.02 / args.speed
            _sleep_until(start + period)
    except KeyboardInterrupt:
        print("\n[stop] returning to HOME…")

    # ---- finish ----------------------------------------------------------------
    present_end = read_present_ticks(ser, WIRE_JOINT_IDS)
    if not loop or args.finish == "home":
        steps = max(1, int(args.ramp_s / 0.02))
        start_pos = np.array([t if t >= 0 else h for t, h in
                              zip(present_end, home_tick)], dtype=float)
        ramps = np.linspace(start_pos, np.array(home_tick, dtype=float), steps + 1)
        t0 = time.perf_counter()
        for i in range(1, steps + 1):
            ser.write(build_sync_write(
                WIRE_JOINT_IDS, REG_GOAL_POSITION,
                [int(max(0, min(MAX_TICK, int(round(v))))).to_bytes(2, "little")
                 for v in ramps[i]]))
            _sleep_until(t0 + i * 0.02)
        print("[i] at HOME")
    if args.finish == "off":
        for pid in WIRE_JOINT_IDS:
            write_register(ser, pid, REG_TORQUE_ENABLE, b"\x00")
        print("[i] torque OFF")
    ser.close()


def _sleep_until(t: float) -> None:
    dt = t - time.perf_counter()
    if dt > 0:
        time.sleep(min(dt, 0.05))


if __name__ == "__main__":
    main()
