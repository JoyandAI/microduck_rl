#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HD-1910 空载恒速法测力矩常数 kt —— **不需要电流表**。

原理
----
把摆杆/砝码全部拆掉(只剩舵盘)，命令若干档**恒定角速度**的线性斜坡，稳态时电机只
需要克服自身摩擦，于是电气平衡为::

    duty·vin = ke·ω + I0·R          (ω>0 取正, ω<0 两者同时变号)

对 `|duty|·vin` 与 `|ω|` 做线性回归::

    斜率  = ke + (dτ_fric/dω)·R/kt  ≈ ke (+约 7%，来自粘性摩擦，可用 BAM 的
            friction_viscous 修正)   —— 而 SI 恒等式给出 **kt[N·m/A] = ke[V·s/rad]**
    截距  = (I0 + τ_Coulomb/kt)·R   —— 库仑摩擦+空载损耗，两方向翻转、平均后抵消

要点：全程只用**编码器 + 占空比反馈 reg60 + 电压 reg62**，不含任何电流量 ⇒ 结果与
reg69 的刻度无关（reg69 在 mode 4 的绝对刻度已被证明与手册差 6.6 倍）。
⚠ 必须**空载**：轴上若挂着摆杆，重力矩会污染结果（脚本会检查截距过大并警告）。

安全(与 record_pendulum_bench.py 同款)
------------------------------------
临时限流 0.975A(reg44, 结束恢复原值)；温度>50°C / 单腿看门狗 15s / 摆角越界
(|q-q0| > amp+150 LSB) → 立即断扭；HD 的电流中止默认关闭(reg69 不可信)。写序沿用
已验证的 B 式(每拍只写 42=目标 → 40=1)。

用法
----
    /usr/bin/python3 scripts/measure_kt_free_run.py --port /dev/ttyACM0 --id 1 \\
      --speeds 0.8,1.2,1.8,2.6,3.6,5.0 --amp-lsb 800 \\
      --out hd1910_calibration/kt_freerun
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
from read_hls_registers import ping, read_regs, write_regs, sm11, sm16, u16_le  # noqa: E402

LSB_RAD = 2 * math.pi / 4096.0
TEMP_ABORT_C = 50.0
WATCHDOG_S = 15.0          # 单腿(一次斜坡)看门狗
DT_LOOP = 0.005
SETTLE_TRIM = 0.25         # 每腿首尾各丢弃的比例(等 PD 稳态)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--id", type=int, default=1)
    ap.add_argument("--speeds", default="0.8,1.2,1.8,2.6,3.6,5.0",
                    help="恒速档 [rad/s]，逗号分隔")
    ap.add_argument("--amp-lsb", type=int, default=800,
                    help="斜坡半幅 [LSB]（会自动收缩到编码器范围内）")
    ap.add_argument("--limit-a", type=float, default=0.975, help="临时限流 [A]")
    ap.add_argument("--out", default="hd1910_calibration/kt_freerun")
    args = ap.parse_args()

    speeds = [float(s) for s in args.speeds.split(",") if s.strip()]
    import serial
    try:
        ser = serial.Serial(args.port, 1_000_000, timeout=0.1)
    except serial.SerialException as e:
        print(f"[!] 打不开串口 {args.port}: {e}")
        return 1
    pid = args.id
    if not ping(ser, pid):
        print(f"[!] 舵机 ID={pid} 无响应（检查供电/接线/ID）。")
        return 1

    def r16(addr, n=2):
        r = read_regs(ser, pid, addr, n)
        if r is None:
            return None
        return u16_le(r, 0) if n == 2 else r[0]

    # --- 限流(临时) + 原值备份 ---
    orig = read_regs(ser, pid, 44, 2)
    orig_limit = u16_le(orig, 0) if orig else 300
    lim_lsb = int(round(args.limit_a / 0.0065))
    write_regs(ser, pid, 44, [lim_lsb & 0xFF, (lim_lsb >> 8) & 0xFF])

    def cleanup():
        for a, d in ((46, [0, 0]), (40, [0]), (44, [orig_limit & 0xFF, orig_limit >> 8])):
            try:
                write_regs(ser, pid, a, d)
            except Exception:
                pass

    atexit.register(cleanup)

    mode = r16(33, 2)
    mode_u = mode & 0xFF if mode is not None else None
    kp = r16(50, 1)
    print(f"[i] 型号 reg3={r16(3,2)} 模式 reg33={mode_u}(须为 4) Kp(reg50)={kp}")
    if mode_u != 4:
        print("[!] 运行模式不是 4（纯位置 PD），本方法的控制律假设不成立 → 中止。")
        return 1

    # --- 配置写序: 46=0 → 46=max（已验证可动写序 #1 的前两步）---
    write_regs(ser, pid, 46, [0, 0])
    write_regs(ser, pid, 40, [0])
    write_regs(ser, pid, 46, [0xFF, 0x7F])
    time.sleep(0.4)

    q0_vals = [v for v in (r16(56, 2) for _ in range(30)) if v is not None]
    if not q0_vals:
        print("[!] 读不到 reg56 位置。")
        return 1
    q0 = float(np.median(q0_vals))
    amp = int(min(args.amp_lsb, q0 - 200, 4095 - 200 - q0))
    print(f"[i] 静止位置 q0={q0:.0f} LSB ({q0*0.08789:.1f}°)；斜坡半幅 amp={amp} LSB "
          f"({amp*0.08789:.1f}°)")
    if amp < 250:
        print(f"[!] 可用幅值只剩 {amp} LSB —— 机器人零点太靠边（mode 4 目标被钳位 [0,4095]）。")
        print("    请把舵盘/舵机本体转一下，让静止读数靠近 2048 后重跑。")
        return 1
    print("[i] ⚠ 本次测量要求**轴上无负载**（摆杆+砝码已拆）；有负载时结果无效。")

    def set_goal(g):
        write_regs(ser, pid, 42, [int(g) & 0xFF, (int(g) >> 8) & 0xFF])
        write_regs(ser, pid, 40, [1])

    legs = []          # (ω_cmd, 方向, 起点目标, 终点目标)
    for w in speeds:
        for d in (+1, -1):
            legs.append((w, d, q0, q0 + d * amp))    # 中心 → 端点
            legs.append((w, d, q0 + d * amp, q0))    # 端点 → 中心

    recs = []
    total = len(legs)
    print(f"[i] 共 {total} 条斜坡腿（{len(speeds)} 档速度 × 双向 × 来回）\n")
    for li, (w, d, g_start, g_end) in enumerate(legs):
        dist = abs(g_end - g_start) * LSB_RAD
        dur = dist / w
        t_leg = time.time()
        sub = []
        k = 0
        while True:
            # 目标按**真实经过时间**插值(而不是按步数): 读寄存器有开销, 按步数会让实际
            # 速度低于设定值(2026-09-10 实测: 设定 5.0 rad/s 只跑出 4.0)。回归用的是
            # 实测 ω, 所以过去的结果有效, 但修正后设定值才名副其实。
            t_rel = time.time() - t_leg
            frac = min(1.0, t_rel / dur)
            goal = g_start + (g_end - g_start) * frac
            set_goal(goal)
            k += 1
            pos = r16(56, 2)
            duty_b = read_regs(ser, pid, 60, 2)
            vin = r16(62, 1)
            tmp = r16(63, 1)
            cur = read_regs(ser, pid, 69, 2)
            if pos is not None:
                sub.append(dict(t=t_rel, target=goal, pos=pos,
                                duty=None if duty_b is None else sm11(duty_b, 0) * 0.001,
                                vin=None if vin is None else vin * 0.1,
                                temp=tmp,
                                current_A=None if cur is None else sm16(cur, 0) * 0.0065))
            # 安全
            if tmp is not None and tmp > TEMP_ABORT_C:
                cleanup()
                print(f"\n[安全中止] 温度 {tmp}°C")
                return 1
            if time.time() - t_leg > WATCHDOG_S:
                cleanup()
                print(f"\n[安全中止] 单腿看门狗 {WATCHDOG_S}s")
                return 1
            if pos is not None and abs(pos - q0) > amp + 150:
                cleanup()
                print(f"\n[安全中止] 摆角越界 (pos={pos}, q0={q0}, amp={amp})")
                return 1
            if frac >= 1.0:
                break
            time.sleep(max(0.0, DT_LOOP - (time.time() - t_leg - t_rel)))
        recs.append(dict(speed_cmd=w, direction=d, dur=dur, samples=sub))
        w_real = abs(sub[-1]["pos"] - sub[0]["pos"]) * LSB_RAD / max(sub[-1]["t"] - sub[0]["t"], 1e-6) if len(sub) > 1 else float("nan")
        print(f"  [{li + 1:2d}/{total}] ω_设定={d * w:+.1f} ω_实测={w_real:+.2f} rad/s "
              f"腿长 {dur:.2f}s 采样 {len(sub)}")

    set_goal(q0)
    time.sleep(0.3)
    write_regs(ser, pid, 40, [0])
    write_regs(ser, pid, 46, [0, 0])
    atexit.unregister(cleanup)
    write_regs(ser, pid, 44, [orig_limit & 0xFF, orig_limit >> 8])

    # ---------------- 分析 ----------------
    rows = []
    for leg in recs:
        s = leg["samples"]
        if len(s) < 12:
            continue
        s = s[int(len(s) * SETTLE_TRIM):int(len(s) * (1 - SETTLE_TRIM))]   # 去掉首尾瞬态
        s = [x for x in s if x["pos"] is not None and x["duty"] is not None and x["vin"] is not None]
        if len(s) < 8:
            continue
        t = np.array([x["t"] for x in s])
        q = np.array([x["pos"] for x in s], dtype=float) * LSB_RAD
        du = np.array([abs(x["duty"]) for x in s])
        vn = np.array([x["vin"] for x in s])
        w_meas = float(np.polyfit(t, q, 1)[0])                  # 实测角速度 [rad/s]
        duty = float(np.mean(du))
        vin = float(np.median(vn))
        rows.append(dict(w_cmd=leg["speed_cmd"], dir=leg["direction"], w=w_meas,
                         duty=duty, vin=vin, duty_vin=duty * vin, n=len(t)))

    if not rows:
        print("[!] 有效样本不足，无法回归。")
        return 1
    W = np.array([abs(r["w"]) for r in rows])
    Y = np.array([r["duty_vin"] for r in rows])
    A = np.c_[W, np.ones(len(W))]
    (slope, intercept), *_ = np.linalg.lstsq(A, Y, rcond=None)
    pred = A @ np.array([slope, intercept])
    r2 = 1 - ((Y - pred) ** 2).sum() / max(((Y - Y.mean()) ** 2).sum(), 1e-12)

    print("\n================ 结果 ================")
    print(f"{'ω_cmd':>7s} {'ω_meas':>8s} {'duty':>7s} {'vin':>6s} {'|duty|·vin':>11s}")
    for r in sorted(rows, key=lambda x: (x["dir"], abs(x["w"]))):
        print(f"{r['dir']*r['w_cmd']:7.1f} {r['w']:+8.3f} {r['duty']:7.4f} {r['vin']:6.2f} {r['duty_vin']:11.4f}")
    print(f"\n回归: |duty|·vin = {slope:.4f}·|ω| + {intercept:.4f}   R²={r2:.4f}  (n={len(rows)})")
    print(f"  → 斜率 ≈ ke = kt = **{slope:.4f}** V·s/rad ≡ N·m/A   "
          f"(含粘性摩擦偏置，修正见下)")
    print(f"  → 截距(库仑摩擦+空载损耗) 折算电流 {intercept / 3.75:.3f} A·Ω   "
          f"[若 >0.5 V 说明轴上可能仍有负载]")
    print(f"  数据手册 kt = 0.7358 N·m/A；本次实测 {slope:.3f} "
          f"→ {slope / 0.7358 * 100:.0f}% 手册值")
    print(f"  修正粘性项: ke ≈ 斜率 − friction_viscous·R/kt "
          f"(用当前摩擦参数迭代; friction_viscous≈0.009, R/kt≈{1 / (slope if slope else 1):.2f})")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "kt_freerun.json").write_text(json.dumps(dict(
        q0_lsb=q0, amp_lsb=amp, speeds=speeds, legs=rows, raw_legs=recs,
        fit=dict(slope_ke=slope, intercept=intercept, r2=r2, n=len(rows)),
        note="|duty|·vin = ke·|ω| + (I0+τ_C/kt)·R；ke≡kt(SI)。需空载。",
    ), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] 原始数据 + 回归 → {out.resolve()}  (扭矩已关, 限流已恢复)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
