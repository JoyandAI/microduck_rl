#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Microduck HL-2909 agent test battery (real robot, FT-SCS 1Mbps /dev/ttyACM0).

Subcommands: t1 t2 t3 t4 reset
Each command is independent and powers the touched servo(s) back down
(40=0, 55=0, 46=0) at exit. Red lines:
  - only write to ID 22 / ID 23 (T5 skipped for ID21: red line #2)
  - abort if temp>52C, current>1.0A sustained 0.7s, stall>0.6s un-recovered.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/joyandai/microduck_rl/scripts")
from read_hls_registers import (  # noqa: E402
    RPM_TO_RAD_S,
    read_regs,
    sm11,
    sm16,
    u16_le,
    u8,
    write_regs,
)
from calibrate_hls_bus import (  # noqa: E402
    CURRENT_LIMIT_A,
    DUTY_LSB,
    I_LSB,
    KPFW,
    LSB_TO_RAD,
    RESOLUTION,
    TEMP_ABORT_C,
    Safety,
    VEL_LSB_RAD_S,
    V_LSB,
    probe_range,
    read_sm11,
    read_sm16,
    read_state,
    read_u8,
    set_mode,
    set_torque,
    sm16_enc,
)

OUT = Path("/home/joyandai/microduck_rl/hls2909_calibration/agent_battery_20260907_130207")

# ── stricted red-line safety (stricter than Safety's 55C/1.2A) ──────────────
LINE_TEMP_C = 52.0
LINE_CUR_A = 1.0
LINE_CUR_BURST_S = 0.7


class Redline(Exception):
    pass


def cur_pos(ser, pid):
    r = read_regs(ser, pid, 56, 2)
    return u16_le(r or b"\x00\x00", 0)


def check_redline(ser, pid, t0, over_cur_since):
    t = read_u8(ser, pid, 63)
    if t is not None and t > LINE_TEMP_C:
        return over_cur_since, f"temp {t}C > {LINE_TEMP_C}C"
    c = read_sm16(ser, pid, 69)
    now = time.time()
    if c is not None:
        ca = abs(c) * I_LSB
        if ca > LINE_CUR_A:
            if over_cur_since is None:
                over_cur_since = now
            elif now - over_cur_since > LINE_CUR_BURST_S:
                return over_cur_since, f"cur {ca:+.2f}A > {LINE_CUR_A}A {LINE_CUR_BURST_S}s"
        else:
            over_cur_since = None
    return over_cur_since, None


def park(ser, pid):
    """Power down a servo: torque off + lock off + velocity 0 + goal=current."""
    try:
        p = cur_pos(ser, pid)
        write_regs(ser, pid, 42, [p & 0xFF, (p >> 8) & 0xFF])
        write_regs(ser, pid, 46, [0, 0])
        write_regs(ser, pid, 40, [0])
        write_regs(ser, pid, 55, [0])
    except Exception:
        pass


def ensure_mode0(ser, pid):
    set_torque(ser, pid, False); time.sleep(0.3)
    m = read_u8(ser, pid, 33)
    if m != 0:
        set_mode(ser, pid, 0); time.sleep(0.3)
        m2 = read_u8(ser, pid, 33)
        if m2 != 0:
            raise Redline(f"ID {pid}: could not set mode=0 (read {m2})")
    return m


def open_ser():
    import serial
    ser = serial.Serial("/dev/ttyACM0", 1_000_000, timeout=0.3)
    ser.reset_input_buffer()
    return ser


# ── T1: free-range probe + URDF joint mapping ───────────────────────────────
def t1(ser):
    res = {}
    for pid in (23, 22):
        print(f"\n=== T1 ID={pid} ===")
        safety = Safety(ser, pid)             # fresh instance per servo (90s watchdog)
        prev_mode = ensure_mode0(ser, pid)
        write_regs(ser, pid, 41, [254])
        write_regs(ser, pid, 46, [0xFF, 0x7F])  # 46 = 32767 全速
        set_torque(ser, pid, True); time.sleep(0.2)
        lo, hi = probe_range(ser, pid, safety)
        safety.check(strict=False)
        park(ser, pid)
        deg = (hi - lo) * 360.0 / RESOLUTION
        res[pid] = {
            "prev_mode": prev_mode,
            "lo_lsb": lo, "hi_lsb": hi,
            "range_lsb": hi - lo,
            "range_deg": round(deg, 2),
            "pos0_before": None,
        }
        print(f"[T1] ID={pid}: free range {lo}..{hi} LSB  ({deg:.1f}°), mode was {prev_mode}")
    return res


# ── T2: bus latency approximation (ID23) ─────────────────────────────────────
def t2(ser):
    pid = 23
    print(f"\n=== T2 ID={pid} bus latency ===")
    t0 = time.time()
    # write->response RTT: write reg 34 (1 byte, no-op same value)
    b34 = read_regs(ser, pid, 34, 1)
    w_rtt = []
    for _ in range(20):
        tt = time.perf_counter()
        ok = write_regs(ser, pid, 34, [b34[0] if b34 else 0])
        w_rtt.append((time.perf_counter() - tt) * 1000)
        if not ok:
            w_rtt.pop()
    # read RTT: read reg 63 (1 byte)
    r_rtt = []
    for _ in range(20):
        tt = time.perf_counter()
        read_regs(ser, pid, 63, 1)
        r_rtt.append((time.perf_counter() - tt) * 1000)

    # command->effect latency: write 42 = pos0+30, poll 56
    ensure_mode0(ser, pid)
    write_regs(ser, pid, 41, [254])
    write_regs(ser, pid, 46, [0xFF, 0x7F])   # fast reaction
    set_torque(ser, pid, True); time.sleep(0.2)
    pos0 = cur_pos(ser, pid)
    target = (pos0 + 30) % RESOLUTION
    effects = []
    for _ in range(20):
        p0 = cur_pos(ser, pid)
        goal = (p0 + 30) % RESOLUTION
        wt = time.perf_counter()
        write_regs(ser, pid, 42, [goal & 0xFF, (goal >> 8) & 0xFF])
        dt = None
        while time.perf_counter() - wt < 0.25:
            now = cur_pos(ser, pid)
            d = (now - p0) % RESOLUTION
            if d >= 2:                       # detected movement
                dt = (time.perf_counter() - wt) * 1000
                break
        if dt is not None:
            effects.append(dt)
        # return to start
        write_regs(ser, pid, 42, [p0 & 0xFF, (p0 >> 8) & 0xFF])
        time.sleep(0.02)
    park(ser, pid)

    out = {
        "write_rtt_ms": {"val": round(statistics.median(w_rtt), 3), "n": len(w_rtt)},
        "read_rtt_ms": {"val": round(statistics.median(r_rtt), 3), "n": len(r_rtt)},
        "cmd_effect_ms": {"val": round(statistics.median(effects), 3), "n": len(effects)},
        "all": {"w_med": [round(x, 3) for x in w_rtt],
                "r_med": [round(x, 3) for x in r_rtt],
                "e_med": [round(x, 3) for x in effects]},
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(f"[T2] write RTT med={out['write_rtt_ms']['val']}ms  read RTT med={out['read_rtt_ms']['val']}ms")
    print(f"[T2] cmd->effect med={out['cmd_effect_ms']['val']}ms  (n={out['cmd_effect_ms']['n']})")
    return out


# ── T3: velocity unit verification (ID22) ────────────────────────────────────
def t3(ser, free22):
    pid = 22
    print(f"\n=== T3 ID={pid} velocity unit ===")
    lo, hi = free22["lo_lsb"], free22["hi_lsb"]
    ensure_mode0(ser, pid)
    write_regs(ser, pid, 41, [254])
    set_torque(ser, pid, True); time.sleep(0.2)
    # pick a start such that +1500 stays in range (buffer 150 from +block)
    end = hi - 150
    start = end - 1500
    if start < lo + 150:
        start = lo + 150
        end = start + 1500
    results = {}
    for spd in (250, 80, 20):
        # settle at start (moderate velocity)
        write_regs(ser, pid, 46, sm16_enc(120))
        write_regs(ser, pid, 42, [start & 0xFF, (start >> 8) & 0xFF])
        for _ in range(40):
            time.sleep(0.05)
            if abs(cur_pos(ser, pid) - start) <= 3:
                break
        write_regs(ser, pid, 46, [0, 0]); time.sleep(0.1)
        s0 = cur_pos(ser, pid)
        # commanded velocity run
        write_regs(ser, pid, 46, sm16_enc(spd))
        write_regs(ser, pid, 42, [end & 0xFF, (end >> 8) & 0xFF])
        samples = []
        t_run = time.time()
        while time.time() - t_run < 1.0:
            t_now = time.perf_counter()
            pos = cur_pos(ser, pid)
            vel = read_sm16(ser, pid, 58)
            samples.append({"t": t_now, "pos": pos, "vel_lsb": vel})
            if pos >= end - 2:
                break
            time.sleep(0.006)
        # stop
        write_regs(ser, pid, 46, [0, 0])
        write_regs(ser, pid, 42, [s0 & 0xFF, (s0 >> 8) & 0xFF])
        time.sleep(0.2)
        # post-process: sliding-window dpos/dt, pick sustained max window
        ts = np.array([s["t"] for s in samples])
        ps = np.array([s["pos"] for s in samples])
        vs = np.array([s["vel_lsb"] for s in samples], dtype=float)
        # constant-velocity window: linear fit over windows; pick best slope window
        best = None
        for i in range(len(ts) - 5):
            tseg, pseg = ts[i:i + 6], ps[i:i + 6]
            if tseg[-1] - tseg[0] <= 0:
                continue
            slope = (pseg[-1] - pseg[0]) / (tseg[-1] - tseg[0])
            if slope <= 0:
                continue
            # R^2 of linear fit tells constancy
            A = np.column_stack([np.ones(6), tseg - tseg[0]])
            coef, *_ = np.linalg.lstsq(A, pseg, rcond=None)
            resid = pseg - A @ coef
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((pseg - pseg.mean()) ** 2)) + 1e-9
            r2 = 1 - ss_res / ss_tot
            if best is None or (slope > best[0] and r2 > 0.95):
                best = (slope, i, r2)
        slope, i0, r2 = best if best else (float("nan"), 0, float("nan"))
        omega_lsb_s = slope                      # LSB/s
        omega_rad_s = slope * (2 * np.pi / RESOLUTION)
        vel_read = np.nanmean(vs[i0:i0 + 6]) if best else float("nan")
        ratio = omega_rad_s / (vel_read * 0.0767) if (vel_read and vel_read != 0) else float("nan")
        results[spd] = {
            "cmd_lsb": spd, "start": int(s0), "end": int(end),
            "omega_lsb_s": float(omega_lsb_s),
            "omega_rad_s": float(omega_rad_s),
            "read58_vel_lsb": float(vel_read),
            "ratio_vs_0.0767": float(ratio),
            "r2": float(r2), "n_samples": len(samples),
        }
        print(f"[T3] cmd={spd}: dpos/dt={omega_lsb_s:.1f} LSB/s ({omega_rad_s:.3f} rad/s)"
              f"  vel58={vel_read:.1f} LSB  ratio={ratio:.3f}  R2={r2:.3f}")
    park(ser, pid)
    return results


# ── T4: R re-measure, stall, anti-ripple (ID23) ─────────────────────────────
def t4(ser, free23):
    pid = 23
    print(f"\n=== T4 ID={pid} R stall ===")
    lo, hi = free23["lo_lsb"], free23["hi_lsb"]
    ensure_mode0(ser, pid)
    # read current protection current (reg 44) to restore later
    p44a = read_regs(ser, pid, 44, 2)
    set_torque(ser, pid, False); time.sleep(0.2)
    # temporary current limit 0.975A → 150 LSB (keeps under 1.0A red-line)
    LIM = 150
    write_regs(ser, pid, 44, [LIM & 0xFF, (LIM >> 8) & 0xFF])
    time.sleep(0.2)
    write_regs(ser, pid, 41, [254])
    write_regs(ser, pid, 46, [0xFF, 0x7F])
    set_torque(ser, pid, True); time.sleep(0.2)
    # stall against + block: goal beyond hi (physically blocked)
    goal = hi + 400
    write_regs(ser, pid, 42, [goal & 0xFF, (goal >> 8) & 0xFF])
    time.sleep(1.0)                              # settle / heat
    # sample 80 (duty, vin, cur)
    samples = []
    over_cur_since = None
    t0 = time.time()
    for _ in range(80):
        d = read_sm11(ser, pid, 60)
        v = read_u8(ser, pid, 62)
        c = read_sm16(ser, pid, 69)
        pos = cur_pos(ser, pid)
        over_cur_since, red = check_redline(ser, pid, t0, over_cur_since)
        if red:
            park(ser, pid)
            raise Redline(red)
        if d is not None and v is not None and c is not None and c != 0:
            samples.append({"duty_lsb": d, "vin_lsb": v, "cur_lsb": c, "pos": pos})
        time.sleep(0.008)
    # restore reg44 + park
    if p44a:
        write_regs(ser, pid, 44, [p44a[0], p44a[1]])
    park(ser, pid)

    Rs = []
    for s in samples:
        vin = s["vin_lsb"] * V_LSB
        duty = s["duty_lsb"] * DUTY_LSB
        cur = s["cur_lsb"] * I_LSB
        if abs(cur) < 0.02:
            continue
        R = abs(duty * vin / cur)
        Rs.append({"R": R, "duty": duty, "vin": vin, "cur": cur, "pos": s["pos"]})
    Rvals = np.array([x["R"] for x in Rs])
    # filter |cur|>0.975A anti-ripple (name per task)
    fR = Rvals[Rvals <= np.inf]
    mask_cur = np.array([abs(x["cur"]) <= 0.975 for x in Rs])
    R_filter = Rvals[mask_cur]
    median_all = float(np.median(Rvals)) if len(Rvals) else float("nan")
    median_f = float(np.median(R_filter)) if len(R_filter) else float("nan")
    iqr_f = (float(np.percentile(R_filter, 75)) - float(np.percentile(R_filter, 25))) if len(R_filter) else float("nan")
    out = {
        "R_median_all_ohm": median_all,
        "R_median_filter975_ohm": median_f,
        "R_IQR_filter975_ohm": iqr_f,
        "R_25_75": [float(np.percentile(R_filter, 25)), float(np.percentile(R_filter, 75))] if len(R_filter) else [],
        "n_samples": len(Rvals), "n_filter975": int(mask_cur.sum()),
        "free_range": [lo, hi], "stall_goal": int(goal),
        "restored44": [p44a[0], p44a[1]] if p44a else None,
        "samples": Rs,
    }
    print(f"[T4] R median(all)={median_all:.3f}Ω  median(<=0.975A)={median_f:.3f}Ω  IQR={iqr_f:.3f}  n={len(Rvals)}/{out['n_filter975']}")
    return out


# ── reset: power down every servo we touched ─────────────────────────────────
def reset(ser):
    for pid in (21, 22, 23):
        park(ser, pid)
    print("[reset] ID21/22/23 torque off (40=0,55=0,46=0)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["t1", "t2", "t3", "t4", "reset"])
    args = ap.parse_args()
    ser = open_ser()
    try:
        if args.cmd == "reset":
            reset(ser)
            return 0
        free22 = {"lo_lsb": 0, "hi_lsb": 0}
        free23 = {"lo_lsb": 0, "hi_lsb": 0}
        if args.cmd == "t1":
            r = t1(ser)
            OUT.joinpath("t1_free_range.json").write_text(
                json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
            print("[T1] saved", OUT / "t1_free_range.json")
        elif args.cmd == "t2":
            r = t2(ser)
            OUT.joinpath("t2_latency.json").write_text(
                json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
            print("[T2] saved", OUT / "t2_latency.json")
        elif args.cmd == "t3":
            # re-read free range if present
            if OUT.joinpath("t1_free_range.json").exists():
                fr = json.loads(OUT.joinpath("t1_free_range.json").read_text())
                free22 = fr["22"]
            r = t3(ser, free22)
            OUT.joinpath("t3_velocity.json").write_text(
                json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
            print("[T3] saved", OUT / "t3_velocity.json")
        elif args.cmd == "t4":
            if OUT.joinpath("t1_free_range.json").exists():
                fr = json.loads(OUT.joinpath("t1_free_range.json").read_text())
                free23 = fr["23"]
            r = t4(ser, free23)
            OUT.joinpath("t4_R.json").write_text(
                json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
            print("[T4] saved", OUT / "t4_R.json")
    finally:
        reset(ser)
        ser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
