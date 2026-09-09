#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立摆锤台架 —— 自动记录脚本(你执行, 安全防护内置)。

台架: 单独一只 HL-2909(建议 ID=1) + 铝舵机臂(夹头) + 碳纤杆(Ø6×~200mm)
+ 端部砝码托(M6 + 砝码片 50/100/150g)。角度用舵机内部编码器(输出轴, 0.087°)。

已知量(全部由你称/量):
    m_tip  = 端部总质量(砝码+托盘+杆端夹头+螺丝)      [kg, 电子秤]
    m_arm  = 摆杆质量(均匀细杆)                       [kg, 电子秤]
    m_hub  = 联轴器/舵机臂/夹块(绕轴同转件)总质量      [kg, 电子秤]
    L      = 轴心 → 端部质量质心距离                   [m, 卡尺 ±1mm]
    r_hub  = 轴心 → m_hub 质心距离                     [m, 卡尺]
    q_zero = 杆垂直向下时的编码器值(脚本自动测)

几何 → Pendulum 等价(与 bam 动力学精确一致, 零改动):
    M     = m_tip·L² + m_arm·L²/3 + m_hub·r_hub²
    B_max = (m_tip + m_arm/2)·g·L + m_hub·g·r_hub
    L_eq  = M·g / B_max          m_eq = B_max² / (M·g²)     (arm_mass=0)

用法:
    /usr/bin/python3 scripts/record_pendulum_bench.py --port /dev/ttyACM0 --id 1 \
      --tip-mass 0.12 --arm-mass 0.012 --arm-length 0.20 --hub-mass 0.015 --hub-radius 0.025 \
      --trajectory lift_and_drop --trajectory sin_time_square --trajectory up_and_down \
      --tip-mass 0.05 --tip-mass 0.10 --tip-mass 0.15 --reps 3 \
      --out hls2909_calibration/bench

安全(与 calibrate_hls.py 同款): 临时限流 0.975A(reg44=150, 结束恢复运行前值);
温度 >50°C / |I|>1.0A×0.7s / 单条 30s 看门狗 → 立即断扭矩。
⚠ 摆锤摆动平面内手勿入; 端部砝码请勿随意加大(完整静态重力矩上限 0.35N·m)。
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_hls_registers import read_regs, write_regs, sm11, sm16, u16_le  # noqa: E402

# 轨迹(无 bam 依赖)
try:
    from bam.trajectory import trajectories as _traj
except Exception:

    def _cubic(kf, t):
        if t < kf[0][0]:
            return kf[0][1]
        if t > kf[-1][0]:
            return kf[-1][1]
        for k in range(len(kf) - 1):
            t0, x0, d0 = kf[k]
            t1, x1, d1 = kf[k + 1]
            if t0 <= t <= t1:
                h = t1 - t0
                s = (t - t0) / h
                return (x0 * (2 * s**3 - 3 * s**2 + 1) + x1 * (-2 * s**3 + 3 * s**2)
                        + (d0 * h) * (s**3 - 2 * s**2 + s) + (d1 * h) * (s**3 - s**2))
        return kf[-1][1]

    class _Tr:
        def __init__(self, dur, fn):
            self.duration = dur
            self._fn = fn

        def __call__(self, t):
            return self._fn(t)

    _traj = {
        "lift_and_drop": _Tr(6.0, lambda t: (_cubic([[0, 0, 0], [2, -math.pi / 2, 0]], t), t < 2.0)),
        "sin_time_square": _Tr(6.0, lambda t: (math.sin(t**2), True)),
        "up_and_down": _Tr(6.0, lambda t: (_cubic([[0, 0, 0], [3, math.pi / 2, 0], [6, 0.8 * math.pi / 2, 0]], t), True)),
        "sin_sin": _Tr(6.0, lambda t: (math.sin(t) * math.pi / 2 + math.sin(5 * t) * 0.5 * math.sin(2 * t), True)),
        "nothing": _Tr(6.0, lambda t: (0.0, False)),
    }

LSB_RAD = 2 * math.pi / 4096.0
G = 9.81
TEMP_ABORT_C = 50.0
CURRENT_LIMIT_A = 1.0
CURRENT_BURST_S = 0.7
WATCHDOG_S = 30.0
RECORD_DURATION_S = 7.0
SETTLE_S = 0.5


def pendulum_equiv(m_tip, m_arm, L, m_hub, r_hub):
    M = m_tip * L**2 + m_arm * L**2 / 3.0 + m_hub * r_hub**2
    B_max = (m_tip + m_arm / 2.0) * G * L + m_hub * G * r_hub
    L_eq = M * G / B_max
    m_eq = B_max**2 / (G**2 * M)
    return {"M_kgm2": M, "B_max_Nm": B_max, "L_eq_m": L_eq, "m_eq_kg": m_eq}


class Safety:
    def __init__(self, ser, pid, restore_limit_lsb):
        self.ser, self.pid = ser, pid
        self.restore_limit_lsb = restore_limit_lsb
        self.t0 = time.time()
        self._over = None

    def check(self, current_A=None, temp_C=None):
        now = time.time()
        if now - self.t0 > WATCHDOG_S:
            self.emergency(f"看门狗 {WATCHDOG_S:.0f}s")
        if temp_C is None:
            t = read_regs(self.ser, self.pid, 63, 1)
            temp_C = t[0] if t else None
        if temp_C is not None and temp_C > TEMP_ABORT_C:
            self.emergency(f"温度 {temp_C}°C > {TEMP_ABORT_C:.0f}°C")
        if current_A is None:
            i = read_regs(self.ser, self.pid, 69, 2)
            current_A = abs(sm16(i, 0)) * 0.0065 if i else None
        if current_A is not None:
            cur = abs(current_A)
            if cur > CURRENT_LIMIT_A:
                if self._over is None:
                    self._over = now
                elif now - self._over > CURRENT_BURST_S:
                    self.emergency(f"持续电流 {cur:.2f}A")
            else:
                self._over = None

    def emergency(self, reason):
        lo, hi = self.restore_limit_lsb & 0xFF, (self.restore_limit_lsb >> 8) & 0xFF
        for a, d in ((46, [0, 0]), (40, [0]), (55, [0]), (44, [lo, hi])):
            try:
                write_regs(self.ser, self.pid, a, d)
            except Exception:
                pass
        raise SystemExit(f"[安全中止] {reason}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--id", type=int, default=1, help="台架舵机 ID")
    ap.add_argument("--tip-mass", action="append", type=float, default=[],
                    help="端部总质量 m_tip [kg], 可多个(负载档)")
    ap.add_argument("--arm-mass", type=float, required=True, help="摆杆质量 m_arm [kg]")
    ap.add_argument("--arm-length", type=float, required=True, help="L: 轴心→端部质心 [m]")
    ap.add_argument("--hub-mass", type=float, default=0.0, help="联轴器/舵机臂质量 [kg]")
    ap.add_argument("--hub-radius", type=float, default=0.0, help="r_hub [m]")
    ap.add_argument("--trajectory", action="append", default=[],
                    choices=["lift_and_drop", "sin_time_square", "up_and_down", "sin_sin", "nothing"])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="hls2909_calibration/bench")
    ap.add_argument("--limit-a", type=float, default=0.975)
    ap.add_argument("--torque-budget", type=float, default=0.35,
                    help="最大静态重力矩 [Nm], 超过则拒绝运行(默认 0.35)")
    args = ap.parse_args()

    tip_masses = args.tip_mass or [0.05]
    trajs = args.trajectory or [
        "sin_time_square", "sin_sin", "lift_and_drop", "up_and_down"
    ]
    if min(tip_masses) <= 0 or args.arm_mass <= 0 or args.arm_length <= 0:
        ap.error("质量和摆长必须为正数")
    for m_tip in tip_masses:
        eq = pendulum_equiv(m_tip, args.arm_mass, args.arm_length,
                            args.hub_mass, args.hub_radius)
        if eq["B_max_Nm"] > args.torque_budget:
            ap.error(
                f"m_tip={m_tip:g}kg 的静态重力矩 {eq['B_max_Nm']:.3f}Nm "
                f">安全上限 {args.torque_budget:.3f}Nm; 减小砝码或摆长"
            )

    pid = args.id
    import serial
    ser = serial.Serial(args.port, 1_000_000, timeout=0.1)
    original_limit = r = read_regs(ser, pid, 44, 2)
    original_limit_lsb = u16_le(r, 0) if r else 300
    limit_lsb = int(round(args.limit_a / 0.0065))
    write_regs(ser, pid, 44, [limit_lsb & 0xFF, (limit_lsb >> 8) & 0xFF])
    print(f"[i] 临时限流 {args.limit_a:.3f}A (reg44={limit_lsb})")

    def r16(a):
        r = read_regs(ser, pid, a, 2)
        return None if r is None else u16_le(r, 0)

    safety = Safety(ser, pid, original_limit_lsb)

    def cleanup():
        if not ser.is_open:
            return
        lo, hi = original_limit_lsb & 0xFF, (original_limit_lsb >> 8) & 0xFF
        for addr, data in ((46, [0, 0]), (40, [0]), (55, [0]), (44, [lo, hi])):
            try:
                write_regs(ser, pid, addr, data)
            except Exception:
                pass

    atexit.register(cleanup)

    # 配置: 位置伺服
    write_regs(ser, pid, 46, [0, 0])
    write_regs(ser, pid, 40, [0])
    # 运行模式: 自动保持设备当前模式(HD-1910 出厂默认 4=纯位置PD/Sim2Real; HLS 默认 0)。
    # 不强制写 0 —— 强制切到模式 0 会改变控制律, 识别结果与部署不一致。
    mode_now = r16(33)
    mode_now_u = mode_now & 0xFF if mode_now is not None else 0
    if mode_now_u in (0, 4):
        print(f"[i] 运行模式 reg33={mode_now_u} (0=位置伺服/4=纯位置PD-Sim2Real), 保持不变")
    else:
        print(f"[!] 运行模式 reg33={mode_now_u} 超出已知语义(0/4), 写 0 继续")
        write_regs(ser, pid, 33, [0])
    # reg41=0 在官方 HLS 内存表中表示最大加速度，与部署配置保持一致。
    write_regs(ser, pid, 41, [0])
    write_regs(ser, pid, 46, [0xFF, 0x7F])
    time.sleep(0.5)

    # 零位(垂下)
    zeros = [r16(56) for _ in range(30)]
    zeros = [z for z in zeros if z is not None]
    q_zero = float(np.median(zeros))
    print(f"[i] 台架舵机 ID={pid}, 零位 q_zero={q_zero:.1f} LSB(摆杆垂直向下)")
    # HD-1910(mode4) 位置误差不做 ±2048 回绕: q_zero 不能落在编码器 0/4095 附近,
    # 否则目标会撞角度限位(实测 4094→338 走 -3755 LSB 长路径; 目标被钳位 [0,4095])。
    if q_zero < 300 or q_zero > 3796:
        print(f"[!] q_zero={q_zero:.0f} 接近编码器 0/4095 回绕区, mode4 下目标会走长路径/撞限位!"
              "请重新安装摆臂(让垂线落在 300~3796 之间)后重试。")
        return 1

    # 符号检测
    write_regs(ser, pid, 40, [1])
    time.sleep(0.1)
    goal = int(round(q_zero + 0.2 / LSB_RAD)) % 4096
    write_regs(ser, pid, 42, [goal & 0xFF, (goal >> 8) & 0xFF])
    time.sleep(0.5)
    now = r16(56)
    d = ((now - q_zero + 2048) % 4096) - 2048 if now is not None else 0
    sign = 1.0 if d >= 0 else -1.0
    print(f"[i] 符号检测: +0.2rad → Δ{d:+.0f} LSB → sign={sign:+.0f}")
    write_regs(ser, pid, 42, [int(q_zero) & 0xFF, (int(q_zero) >> 8) & 0xFF])
    time.sleep(0.4)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"id": pid, "q_zero_lsb": q_zero, "sign": sign,
                "geometry": dict(arm_mass=args.arm_mass, arm_length=args.arm_length,
                                 hub_mass=args.hub_mass, hub_radius=args.hub_radius),
                "limit_A": args.limit_a, "tip_cases": {}, "records": []}

    n_rec = 0
    total = len(tip_masses) * len(trajs) * args.reps
    for m_tip in tip_masses:
        # Each mass case requires a physical weight change. Stop torque and wait
        # for an explicit operator acknowledgement before starting that block.
        write_regs(ser, pid, 46, [0, 0])
        write_regs(ser, pid, 40, [0])
        write_regs(ser, pid, 55, [0])
        print(f"\n[换砝码] 请安装端部总质量 {m_tip * 1000:.1f} g，锁紧 M6 螺母，"
              "手动检查 ±90° 无碰撞。")
        try:
            answer = input("确认人员已离开摆动平面后输入 START: ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            print("\n[停止] 未确认换砝码，实验取消。")
            return 2
        if answer != "START":
            print("[停止] 未输入 START，实验取消。")
            return 2

        eq = pendulum_equiv(m_tip, args.arm_mass, args.arm_length, args.hub_mass, args.hub_radius)
        manifest["tip_cases"][f"{m_tip:g}"] = eq
        print(f"[i] m_tip={m_tip:g}kg → M={eq['M_kgm2']:.4g} B_max={eq['B_max_Nm']:.5f} "
              f"L_eq={eq['L_eq_m']*1000:.1f}mm m_eq={eq['m_eq_kg']*1000:.1f}g")
        for tname in trajs:
            traj = _traj[tname]
            for rep in range(args.reps):
                safety.t0 = time.time()
                safety._over = None
                safety.check()
                dt_loop = 0.006
                entries = []
                t_start = time.time()
                print(f"  [record] {tname} m_tip={m_tip:g} rep{rep}: ", end="", flush=True)
                for k in range(int(RECORD_DURATION_S / dt_loop)):
                    t = k * dt_loop
                    if t > getattr(traj, "duration", 6.0):
                        break
                    angle, enable = traj(t)
                    glb = int(round(q_zero + sign * angle / LSB_RAD)) % 4096
                    if enable:
                        # B 式写序(HD-1910 mode4 已验证 16/16 稳定): 每拍只写 目标42 + 扭矩40;
                        # 不写 reg46(速度) —— HLS 原序在 HD mode4 下实测不动。
                        write_regs(ser, pid, 42, [glb & 0xFF, (glb >> 8) & 0xFF])
                        write_regs(ser, pid, 40, [1])
                        # reg55 写锁: 不再逐拍写(与已验证协议一致)。
                    else:
                        write_regs(ser, pid, 40, [0])
                    t_now = time.time() - t_start
                    pos = r16(56)
                    if pos is None:
                        continue
                    r = read_regs(ser, pid, 60, 2)
                    duty = sm11(r, 0) * 0.001 if r else None
                    r = read_regs(ser, pid, 62, 1)
                    vin = r[0] * 0.1 if r else None
                    r = read_regs(ser, pid, 69, 2)
                    cur = sm16(r, 0) * 0.0065 if r else None
                    r = read_regs(ser, pid, 63, 1)
                    tmp = r[0] if r else None
                    delta_lsb = ((pos - q_zero + 2048) % 4096) - 2048
                    q = delta_lsb * LSB_RAD * sign
                    qg = float(angle)
                    dq = (q - entries[-1]["position"]) / max(t_now - entries[-1]["t"], 1e-6) \
                        if entries else 0.0
                    entries.append({"t": t_now, "position": float(q), "speed": float(dq),
                                    "goal_position": float(qg), "torque_enable": bool(enable),
                                    "duty": duty, "vin": vin, "current_A": cur, "temp": tmp})
                    safety.check(current_A=cur, temp_C=tmp)
                    time.sleep(max(0.0, dt_loop - (time.time() - t_start - t_now)))
                write_regs(ser, pid, 42, [int(q_zero) & 0xFF, (int(q_zero) >> 8) & 0xFF])
                write_regs(ser, pid, 40, [0])
                time.sleep(SETTLE_S)
                if len(entries) < 50:
                    print("FAIL(样本过少), 跳过")
                    continue
                dt_med = float(np.median([entries[i]["t"] - entries[i - 1]["t"]
                                          for i in range(1, len(entries))]))
                vin_samples = [e["vin"] for e in entries if e["vin"] is not None]
                vin_measured = float(np.median(vin_samples)) if vin_samples else 12.5
                log = {
                    "motor": "hls2909", "kp": 32, "vin": vin_measured, "dt": dt_med,
                    "mass": eq["m_eq_kg"], "arm_mass": 0.0, "length": eq["L_eq_m"],
                    "trajectory": tname, "tip_mass_kg": m_tip, "rep": rep,
                    "entries": [{k: entries[i][k] for k in
                                 ("position", "speed", "goal_position", "torque_enable")}
                                for i in range(len(entries))],
                    "telemetry": entries,
                }
                fn = f"{tname}_tip{m_tip:g}kg_rep{rep}.json"
                (out / fn).write_text(json.dumps(log, indent=2), encoding="utf-8")
                n_rec += 1
                manifest["records"].append({"file": fn, "trajectory": tname,
                                            "tip_mass_kg": m_tip, "rep": rep})
                print(f"OK n={len(entries)} dt={dt_med*1000:.1f}ms ({100*n_rec/total:.0f}%)")

    cleanup()
    atexit.unregister(cleanup)
    ser.close()
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] {n_rec} 条 → {out.resolve()}  "
          f"(扭矩已关, 限流已恢复 {original_limit_lsb * 0.0065:.3f}A)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
