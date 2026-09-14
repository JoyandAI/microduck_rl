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
温度 >50°C / |I|>阈值×0.7s / |位置误差|>0.8rad×1.0s(卡死) / 单条 30s 看门狗 → 立即断扭矩。
⚠ HD-1910(mode4) 的 reg69 电流读数不可信(起步瞬态恒 ≈2.1A, 峰值可达 4A > 物理堵转
1.36A) → 该型号默认把电流中止关掉(--current-abort 0), 卡死检测与温度/看门狗为主;
HLS 保持 1.0A 阈值。见 /home/joyandai/bam/docs/identification/hd1910_servo_notes.md §6.5/6.6。
⚠ 摆锤摆动平面内手勿入; 端部砝码请勿随意加大(完整静态重力矩上限 0.35N·m)。
q_zero 超出按所选轨迹振幅实算的安全带(HD mode4 误差不回绕, 目标须整体落于 [0,4095])
时, 先自动用 reg31(位置偏移)居中到 2048(动态算值, 换舵机通用; 探针+双符号编码回退+
运动验证; 失败恢复原值), 成功才继续; 居中失败才要求重新安装摆臂。
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
# 位置误差卡死检测(HD-1910 mode 4 的电流反馈不可信, 用它兜底; HLS 上同样有效):
# 扭矩开启期间 |q - q_goal| 连续超过 JAM_ERR_RAD 达 JAM_HOLD_S 秒 → 判卡死并断扭。
JAM_ERR_RAD = 0.8
JAM_HOLD_S = 1.0
WATCHDOG_S = 30.0
RECORD_DURATION_S = 7.0


def pendulum_equiv(m_tip, m_arm, L, m_hub, r_hub):
    M = m_tip * L**2 + m_arm * L**2 / 3.0 + m_hub * r_hub**2
    B_max = (m_tip + m_arm / 2.0) * G * L + m_hub * G * r_hub
    L_eq = M * G / B_max
    m_eq = B_max**2 / (G**2 * M)
    return {"M_kgm2": M, "B_max_Nm": B_max, "L_eq_m": L_eq, "m_eq_kg": m_eq}


class _ScaledTraj:
    """按系数缩放摆角幅值的轨迹包装(保波形形状, 不做削顶)。

    用途: 台架有机械障碍时把峰值压到安全角以内。缩放只改幅值, 频率成分与
    波形形状不变; 拟合按日志里的 goal_position 回放, 因此缩放不影响辨识口径。
    """

    def __init__(self, base, k):
        self.base = base
        self.k = float(k)
        self.duration = getattr(base, "duration", RECORD_DURATION_S)

    def __call__(self, t):
        a, e = self.base(t)
        return a * self.k, e


def traj_peak_rad(tr):
    """单条轨迹(或包装后)在 1ms 网格上的全局峰值摆角 [rad]。"""
    dur = getattr(tr, "duration", RECORD_DURATION_S)
    amp = 0.0
    for k in range(int(dur * 1000) + 1):
        a, _ = tr(k * 0.001)
        amp = max(amp, abs(a))
    return amp


def zero_belt(trajs, margin_lsb=64):
    """按所选轨迹的真实振幅实算 q_zero 安全带 [lo, hi]。

    HD-1910(mode4) 位置误差不做 ±2048 回绕: 任一目标 q_zero ± max|angle| 都必须
    整体落在 [0, 4095] 内, 否则误差 >2048 LSB 会走长路径/撞限位
    (实测 4094→338 走 -3755 LSB; 目标被钳位 [0,4095])。
    1 ms 网格扫描轨迹取全局峰值 → amp_lsb, 安全带 = 峰值 ± margin。
    """
    amp = 0.0
    for tr in trajs:
        amp = max(amp, traj_peak_rad(tr))
    amp_lsb = int(math.ceil(amp / LSB_RAD))
    lo, hi = amp_lsb + margin_lsb, 4095 - amp_lsb - margin_lsb
    return lo, hi, amp_lsb


class Safety:
    def __init__(self, ser, pid, restore_limit_lsb, current_abort_a=CURRENT_LIMIT_A,
                 jam_err_rad=JAM_ERR_RAD, jam_hold_s=JAM_HOLD_S, hard_limit_rad=None):
        self.ser, self.pid = ser, pid
        self.restore_limit_lsb = restore_limit_lsb
        self.current_abort_a = float(current_abort_a)   # 0/None = 关闭电流中止
        self.jam_err_rad = None if jam_err_rad is None else float(jam_err_rad)
        self.jam_hold_s = float(jam_hold_s)
        self.hard_limit_rad = None if hard_limit_rad is None else float(hard_limit_rad)
        self.t0 = time.time()
        self._over = None
        self._jam = None

    def reset(self):
        self.t0 = time.time()
        self._over = None
        self._jam = None

    def check(self, current_A=None, temp_C=None, pos_err_rad=None, pos_rad=None):
        now = time.time()
        if now - self.t0 > WATCHDOG_S:
            self.emergency(f"看门狗 {WATCHDOG_S:.0f}s")
        # 防撞硬限: 与扭矩开关无关, 自由落体段同样生效(撞测试板就是这个后果)。
        if self.hard_limit_rad is not None and pos_rad is not None:
            if abs(pos_rad) > self.hard_limit_rad:
                self.emergency(f"摆角 {math.degrees(abs(pos_rad)):.1f}° 超过硬限 "
                               f"{math.degrees(self.hard_limit_rad):.1f}°")
        if temp_C is None:
            t = read_regs(self.ser, self.pid, 63, 1)
            temp_C = t[0] if t else None
        if temp_C is not None and temp_C > TEMP_ABORT_C:
            self.emergency(f"温度 {temp_C}°C > {TEMP_ABORT_C:.0f}°C")
        if self.current_abort_a:
            if current_A is None:
                i = read_regs(self.ser, self.pid, 69, 2)
                current_A = abs(sm16(i, 0)) * 0.0065 if i else None
            if current_A is not None:
                cur = abs(current_A)
                if cur > self.current_abort_a:
                    if self._over is None:
                        self._over = now
                    elif now - self._over > CURRENT_BURST_S:
                        self.emergency(f"持续电流 {cur:.2f}A")
                else:
                    self._over = None
        # 位置误差卡死检测: 真卡死时 PD 环误差会持续发散(而 HD 的电流读数不会)。
        if self.jam_err_rad is not None and pos_err_rad is not None:
            if abs(pos_err_rad) > self.jam_err_rad:
                if self._jam is None:
                    self._jam = now
                elif now - self._jam > self.jam_hold_s:
                    self.emergency(f"位置误差持续 {abs(pos_err_rad):.2f} rad "
                                   f"(>{self.jam_err_rad:.2f} rad × {self.jam_hold_s:.1f}s), 疑似卡死/超载")
            else:
                self._jam = None

    def emergency(self, reason):
        lo, hi = self.restore_limit_lsb & 0xFF, (self.restore_limit_lsb >> 8) & 0xFF
        for a, d in ((46, [0, 0]), (40, [0]), (55, [0]), (44, [lo, hi])):
            try:
                write_regs(self.ser, self.pid, a, d)
            except Exception:
                pass
        raise SystemExit(f"[安全中止] {reason}")


def center_zero_reg31(ser, pid, target=2048.0):
    """摆杆静止读数不在安全带时, 用 reg31(位置偏移)自动把读数动态校准到 target(2048)。

    全程无固定数值, 换舵机通用:
      0. 先写 reg31=0 读**原始**编码器基线(舵机可能带历史偏移 —— 上次居中后
         reg31≠0, 若直接探针, 增量是 (128 − 历史值) 而非 ±128);
      1. 探针 reg31=+128 → 判定"读数=(raw±off)%4096"的符号约定
         (±128 的符号幅值编码在 BIT15/BIT11 两种约定下相同, 探针本身无歧义);
      2. off = (target - q0) % 4096(减法约定取负), 幅值取 ≤2047 侧;
      3. 写入后读回验证(先试 BIT15 符号位, 再试 BIT11);
      4. 运动验证: 命令 +0.2 rad, 须短路径到位 —— 证明 mode4 控制用的是偏移后的值。

    任何一步失败都**恢复进入时的 reg31 原值**(不是写 0 —— 舵机可能已有历史偏移)。
    成功则保留新值并返回 (True, 说明, reg31值): reg31 通常为 EPROM 持久值,
    断电保留, 意味着该舵机中位已被校准到 2048, 下次运行无需再居中。

    注: HD-1910(mode4)实测**控制环不使用 reg31**(只偏置读数 —— 偏置后命令
    +0.2rad 目标, 位置纹丝不动) → 运动验证必然失败, 本函数在该固件上只作兜底
    (若遇到支持 reg31 的固件/舵机则自动启用); 失败时 main() 会打印物理对准建议。
    """
    r31_orig_r = read_regs(ser, pid, 31, 2)
    r31_orig = u16_le(r31_orig_r, 0) if r31_orig_r else 0

    def restore():
        write_regs(ser, pid, 31, [r31_orig & 0xFF, (r31_orig >> 8) & 0xFF])

    def read_pos(n=30):
        vals = []
        for _ in range(n):
            r = read_regs(ser, pid, 56, 2)
            if r:
                vals.append(u16_le(r, 0))
            time.sleep(0.03)
        return float(np.median(vals)) if vals else None

    write_regs(ser, pid, 46, [0, 0])
    write_regs(ser, pid, 40, [0])
    write_regs(ser, pid, 31, [0, 0])
    time.sleep(0.3)
    q0 = read_pos()
    if q0 is None:
        restore()
        return False, "读不到 reg56", None
    print(f"[i] 居中前静止读数 q0={q0:.1f} LSB (reg31 原值 0x{r31_orig:04X})")

    # 1) 探针: 判定加/减约定
    write_regs(ser, pid, 31, [0x80, 0x00])
    time.sleep(0.3)
    q1 = read_pos(10)
    if q1 is None:
        restore()
        return False, "探针读回 reg56 失败", None
    d = int(((q1 - q0 + 2048) % 4096) - 2048)
    print(f"[i] 探针 reg31=+128 → reg56 {q0:.0f}→{q1:.0f} (Δ{d:+d})")
    if abs(d) < 8 or abs(abs(d) - 128) > 8:
        restore()
        return False, f"reg31 对读数无预期效果(Δ{d:+d}), 疑似写入被拒/被忽略", None
    add = d > 0  # True: 读数 = (raw + off) % 4096

    # 2) 动态计算偏移(无固定数值)
    off = (int(round(target)) - int(round(q0))) % 4096
    if not add:
        off = (-off) % 4096
    if off == 2048:
        restore()
        return False, "需要 ±2048 临界偏移(符号幅值表示有歧义), 只能物理重装", None
    mag = off if off <= 2047 else 4096 - off
    sign = 0x8000 if off > 2047 else 0

    # 3) 写入 + 读回验证(先 BIT15 符号位, 再 BIT11)
    val = None
    for cand in (sign | mag, 0x0800 | mag):
        write_regs(ser, pid, 31, [cand & 0xFF, (cand >> 8) & 0xFF])
        time.sleep(0.3)
        q2 = read_pos()
        print(f"[i] 试 reg31=0x{cand:04X} → reg56={q2:.0f} (目标 {target:.0f})")
        if q2 is not None and abs(q2 - target) <= 8:
            val = cand
            break
    if val is None:
        restore()
        return False, "两种符号编码都未能校准到 2048", None

    # 4) 运动验证: 证明 mode4 控制用的是偏移后的值, 而非只改了读数
    goal = int(round(target + 0.2 / LSB_RAD))
    write_regs(ser, pid, 41, [0])
    write_regs(ser, pid, 46, [0xFF, 0x7F])
    write_regs(ser, pid, 42, [goal & 0xFF, (goal >> 8) & 0xFF])
    write_regs(ser, pid, 40, [1])
    time.sleep(1.2)
    q3 = read_pos(10)
    write_regs(ser, pid, 40, [0])
    if q3 is None or abs(q3 - goal) > 40 or abs(q3 - target) < 80:
        restore()
        return False, f"运动验证失败(reg56={q3:.0f}, 期望≈{goal}), mode4 疑似不认 reg31", None
    write_regs(ser, pid, 42, [int(round(target)) & 0xFF, (int(round(target)) >> 8) & 0xFF])
    return True, (f"reg31 自动居中成功(q0={q0:.0f} → {target:.0f}, reg31=0x{val:04X}, "
                  f"原值 0x{r31_orig:04X}; 已持久写入, 下次无需再居中), 运动验证通过"), val


SETTLE_QUIET_RAD = 0.02      # 稳位判据: |q - q_zero|
SETTLE_QUIET_VEL = 0.25      # 稳位判据: |dq| [rad/s]
SETTLE_HOLD_S = 0.25         # 需连续满足的时长
SETTLE_MAX_S = 3.0           # 稳位超时(超过就带残余继续, 并打印警告)


def settle_at_zero(ser, pid, q_zero, r16, safety):
    """记录前把摆杆**主动**稳到零位(扭矩保持 ON), 保证每条轨迹都从"静止下垂"起跑。

    原实现是"目标回零 + 断扭矩 + 睡 0.5s"(HLS 沿用): 断扭后摆杆自由摆动, 下一条
    记录就从**余摆中途**起跑, 而日志首拍速度被写成 0 → 仿真以"静止"起跑与真机
    不符(2026-09-10 实测 up_and_down rep1/2 `pos@t0=-0.635rad`, 首秒误差 0.654rad,
    占 PASS 预算 0.157rad 的 ~12%)。

    ⚠ 目标必须**逐拍重写**(与记录循环同款 B 式 42+40): 只写一次的"单发目标"
    实测不生效(2026-09-10 探针: 单发后摆杆 6s 仍停在旧目标处), 逐拍重写则从
    +1.16rad 在 0.52s 内收敛到 +0.006rad。写序前置两步沿用已验证可动写序 #1。

    返回 (稳位用时s, 残余偏差rad); 超时也返回并打印警告。
    """
    g = [int(q_zero) & 0xFF, (int(q_zero) >> 8) & 0xFF]
    write_regs(ser, pid, 46, [0, 0])
    write_regs(ser, pid, 46, [0xFF, 0x7F])
    t0 = time.time()
    quiet_since = None
    prev = None
    while time.time() - t0 < SETTLE_MAX_S:
        write_regs(ser, pid, 42, g)
        write_regs(ser, pid, 40, [1])
        pos = r16(56)
        t = time.time()
        if pos is not None:
            q = (((pos - q_zero + 2048) % 4096) - 2048) * LSB_RAD
            v = abs(q - prev[0]) / max(t - prev[1], 1e-6) if prev else 0.0
            prev = (q, t)
            if abs(q) < SETTLE_QUIET_RAD and v < SETTLE_QUIET_VEL:
                if quiet_since is None:
                    quiet_since = t
                elif t - quiet_since >= SETTLE_HOLD_S:
                    return t - t0, q
            else:
                quiet_since = None
        safety.check()
        time.sleep(0.03)
    q_now = prev[0] if prev else float("nan")
    print(f"\n  [!] 稳位超时 {SETTLE_MAX_S:.1f}s(残余 {q_now:+.3f} rad), 带残余继续记录",
          flush=True)
    return SETTLE_MAX_S, q_now


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
    ap.add_argument("--motor", default="auto", choices=["auto", "hls2909", "hd1910"],
                    help="日志 motor 标签(默认 auto: reg3=7946→hd1910, 其余→hls2909)")
    ap.add_argument("--current-abort", type=float, default=None,
                    help="电流中止阈值 [A](|I|>阈值持续 0.7s → 断扭); 0=关闭。"
                         "缺省: HLS=1.0; HD-1910=0(关闭, reg69 读数不可信, 见 hd1910_servo_notes.md §6.5/6.6)")
    ap.add_argument("--jam-err-rad", type=float, default=JAM_ERR_RAD,
                    help=f"位置误差卡死检测阈值 [rad](默认 {JAM_ERR_RAD}); 0=关闭")
    ap.add_argument("--jam-hold-s", type=float, default=JAM_HOLD_S,
                    help=f"位置误差持续时长 [s](默认 {JAM_HOLD_S})")
    ap.add_argument("--amp-max-deg", type=float, default=80.0,
                    help="摆角峰值上限 [度]; 峰值超过它的轨迹按比例整体缩放(保波形, 不削顶), "
                         "0=不缩放。本台架实测障碍: 正侧 +96.7°、负侧 −88.4°(占空比打到 100% "
                         "硬顶) → 默认 80°(对负侧留 8.4°、正侧留 16.7° 余量)")
    ap.add_argument("--hard-limit-deg", type=float, default=86.0,
                    help="摆角硬限(防撞台架/测试板) [度]; 0=关闭。预检: 所选轨迹峰值超过它"
                         "就拒绝运行; 运行中: 任一采样 |q| 超过它就立即断扭退出。"
                         "默认 104(实测 sin_sin 指令峰值 100.8°)")
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

    # ---- 幅值限幅(按台架实测障碍角) + 防撞预检(在打开串口之前, 失败即退出, 不碰硬件) ----
    amp_max_rad = None if not args.amp_max_deg else math.radians(args.amp_max_deg)
    trajmap = {}
    for n in trajs:
        base = _traj[n]
        pk = traj_peak_rad(base)
        k = 1.0 if (amp_max_rad is None or pk <= amp_max_rad) else amp_max_rad / pk
        trajmap[n] = _ScaledTraj(base, k)
    if amp_max_rad is not None:
        parts = []
        for n in trajs:
            k = trajmap[n].k
            tag = "" if k == 1.0 else f"(×{k:.3f})"
            parts.append(f"{n}={math.degrees(traj_peak_rad(trajmap[n])):.1f}°{tag}")
        print(f"[i] 摆角峰值上限 {args.amp_max_deg:.1f}°; 实际下发峰值: " + ", ".join(parts))
    hard_limit_rad = None if not args.hard_limit_deg else math.radians(args.hard_limit_deg)
    if hard_limit_rad is not None:
        peaks = {n: traj_peak_rad(trajmap[n]) for n in trajs}
        print("[i] 摆角硬限 %.1f° ; 所选轨迹峰值: %s" % (
            args.hard_limit_deg,
            ", ".join(f"{n}={math.degrees(p):.1f}°" for n, p in peaks.items())))
        over = {n: p for n, p in peaks.items() if p > hard_limit_rad}
        if over:
            print("[!] 以下轨迹的峰值摆角超过硬限(会打到台架/测试板), 拒绝运行:")
            for n, p in over.items():
                print(f"      {n}: {math.degrees(p):.1f}° > {args.hard_limit_deg:.1f}°"
                      f"(超出 {math.degrees(p) - args.hard_limit_deg:+.1f}°)")
            print("[!] 处理: 把障碍物移出摆动平面(①最好), 或降低该轨迹幅值, "
                  "或按实际碰撞角调整 --hard-limit-deg。")
            return 1

    pid = args.id
    import serial
    try:
        ser = serial.Serial(args.port, 1_000_000, timeout=0.1)
    except serial.SerialException as e:
        print(f"[!] 打不开串口 {args.port}: {e}")
        print("    检查: 设备插好没(`ls /dev/ttyACM*`)、是否被别的进程占用"
              "(`pgrep -af record_pendulum_bench`)、当前用户在不在 dialout 组。")
        return 1
    if not ping(ser, pid):
        print(f"[!] 台架舵机 ID={pid} 无响应(检查接线/供电/ID; 可用 "
              "`scripts/setup_bench_servo.py --find-id <id>` 扫描)。")
        ser.close()
        return 1
    original_limit = r = read_regs(ser, pid, 44, 2)
    original_limit_lsb = u16_le(r, 0) if r else 300
    limit_lsb = int(round(args.limit_a / 0.0065))
    write_regs(ser, pid, 44, [limit_lsb & 0xFF, (limit_lsb >> 8) & 0xFF])
    print(f"[i] 临时限流 {args.limit_a:.3f}A (reg44={limit_lsb})")

    def r16(a):
        r = read_regs(ser, pid, a, 2)
        return None if r is None else u16_le(r, 0)

    # 型号/有效增益: 只读, 用于日志标签与溯源(bam 拟合以 --actuator 为准)。
    # 必须早于 Safety 构造 —— 电流中止阈值按型号自动选定。
    model_id = r16(3)
    # ⚠️ reg50=Kp、reg51=Kd 是两个独立的 8 位寄存器；用 r16(50) 会把 kd 读进高字节
    # （32 + 40*256 = 10272），而 bam 的 load_log 会直接采用 log["kp"] → P 增益大 321 倍。
    # 只读 1 字节。
    _r = read_regs(ser, pid, 50, 1)
    kp_eff = int(_r[0]) if _r else 32
    if args.motor == "auto":
        motor = "hd1910" if model_id == 7946 else "hls2909"
    else:
        motor = args.motor
    print(f"[i] 型号号 reg3={model_id} → motor={motor}; Kp(reg50)={kp_eff}")

    if args.current_abort is None:
        # HD-1910 mode 4 的 reg69 不是可信电流: 起步瞬态恒读 ≈2.1A(317 LSB),
        # 单条内峰值可达 4.06A, 均 > 5.1V/3.75Ω≈1.36A 的物理堵转上限
        # (hd1910_servo_notes.md §6.5 T5 / §6.6⑤, 2026-09-10 又一次实测 2.03A 误中止)。
        # → HD 默认关闭电流中止, 安全链改由 温度 + 30s 看门狗 + 位置误差卡死检测
        #   + 3A 物理保险丝 + 人在场 承担; HLS 保持原 1.0A 阈值。
        current_abort_a = 0.0 if motor == "hd1910" else CURRENT_LIMIT_A
    else:
        current_abort_a = float(args.current_abort)
    jam_err_rad = None if not args.jam_err_rad else float(args.jam_err_rad)
    print(f"[i] 安全链: 温度>{TEMP_ABORT_C:.0f}°C / 看门狗 {WATCHDOG_S:.0f}s / "
          f"电流中止={current_abort_a:.2f}A"
          f"{' (关闭)' if not current_abort_a else ''} / "
          f"卡死检测={'关闭' if jam_err_rad is None else f'{jam_err_rad:.2f}rad×{args.jam_hold_s:.1f}s'}")

    safety = Safety(ser, pid, original_limit_lsb, current_abort_a=current_abort_a,
                    jam_err_rad=jam_err_rad, jam_hold_s=args.jam_hold_s,
                    hard_limit_rad=hard_limit_rad)

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

    # 型号/有效增益已在上面(早于 Safety)读取: reg3 决定电流中止阈值。

    # 零位(垂下)
    zeros = [r16(56) for _ in range(30)]
    zeros = [z for z in zeros if z is not None]
    if not zeros:
        print("[!] 读不到舵机位置(reg56 无应答), 检查 ID/接线后重试。")
        return 1
    q_zero = float(np.median(zeros))
    print(f"[i] 台架舵机 ID={pid}, 零位 q_zero={q_zero:.1f} LSB(摆杆垂直向下)")
    # HD-1910(mode4) 位置误差不做 ±2048 回绕: q_zero 不能落在编码器 0/4095 附近,
    # 否则目标会撞角度限位(实测 4094→338 走 -3755 LSB 长路径; 目标被钳位 [0,4095])。
    # 安全带按所选轨迹的实测振幅计算(默认四轨迹峰值为 ±1147 LSB);
    # 超出时先尝试 reg31(位置偏移)自动居中到 2048, 失败才要求物理重装摆臂。
    belt_lo, belt_hi, belt_amp = zero_belt([trajmap[n] for n in trajs])
    print(f"[i] 轨迹峰值 ±{belt_amp} LSB → q_zero 安全带 [{belt_lo}, {belt_hi}]")
    reg31_offset_lsb = 0
    if not (belt_lo <= q_zero <= belt_hi):
        ok, msg, r31_val = center_zero_reg31(ser, pid)
        if not ok:
            print(f"[!] reg31 自动居中失败: {msg}")
            print(f"[!] HD-1910 mode4 控制环实测不使用 reg31(只偏置读数), "
                  "软件居中不可行; reg31 已恢复原值 → 必须物理对准。")
            # 动态建议: 把静止读数 q_zero 移到 2048 附近所需的旋转
            d_shift = int(round(q_zero - 2048.0))
            if abs(d_shift) > 2048:
                d_shift -= 4096
            alpha = abs(d_shift) * 360.0 / 4096.0
            teeth = abs(d_shift) / 163.84
            print(f"[!] 建议(当前 q_zero={q_zero:.0f} → 目标 2048, Δ={abs(d_shift)} LSB):")
            print(f"    ① 松开两个 C 夹, 把舵机本体绕输出轴转约 {alpha:.1f}°"
                  "(读数向 2048 靠拢的方向; 1°≈11.38 LSB, 可连续微调);")
            print(f"    ② 或拔下舵盘转约 {teeth:.1f} 齿装回(每齿 163.84 LSB, "
                  f"推荐 {teeth:.0f} 齿)。")
            print(f"    调完只读 reg56 确认落在 1900~2200(理想 2048±100), "
                  "手动 ±90° 无碰撞复查后重跑。")
            return 1
        reg31_offset_lsb = r31_val
        print(f"[i] {msg}; 重新测量零位...")
        zeros = [r16(56) for _ in range(30)]
        zeros = [z for z in zeros if z is not None]
        q_zero = float(np.median(zeros))
        print(f"[i] 台架舵机 ID={pid}, 零位 q_zero={q_zero:.1f} LSB(摆杆垂直向下)")
        if not (belt_lo <= q_zero <= belt_hi):
            print(f"[!] 自动居中后读数仍在安全带 [{belt_lo}, {belt_hi}] 外, "
                  "请重新安装摆臂后重试。")
            return 1

    # 符号检测
    write_regs(ser, pid, 40, [1])
    time.sleep(0.1)
    goal = int(round(q_zero + 0.2 / LSB_RAD))
    goal = min(4095, max(0, goal))
    write_regs(ser, pid, 42, [goal & 0xFF, (goal >> 8) & 0xFF])
    time.sleep(0.5)
    now = r16(56)
    if now is None:
        write_regs(ser, pid, 40, [0])
        print("[!] 符号检测读不回位置(reg56 无应答), 中止。")
        return 1
    d = int(((now - q_zero + 2048) % 4096) - 2048)
    if abs(d) < 20:
        write_regs(ser, pid, 40, [0])
        print(f"[!] 符号检测失败: 命令 +0.2rad 后 Δ{d:+.0f} LSB, 舵机未动"
              "(检查写序/运行模式/限流/机械卡死)。")
        return 1
    sign = 1.0 if d >= 0 else -1.0
    print(f"[i] 符号检测: +0.2rad → Δ{d:+.0f} LSB → sign={sign:+.0f}")
    write_regs(ser, pid, 42, [int(q_zero) & 0xFF, (int(q_zero) >> 8) & 0xFF])
    time.sleep(0.4)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"id": pid, "q_zero_lsb": q_zero, "sign": sign, "motor": motor,
                "kp": kp_eff, "reg31_offset_lsb": reg31_offset_lsb,
                "geometry": dict(arm_mass=args.arm_mass, arm_length=args.arm_length,
                                 hub_mass=args.hub_mass, hub_radius=args.hub_radius),
                "limit_A": args.limit_a, "hard_limit_deg": args.hard_limit_deg,
                "amp_max_deg": args.amp_max_deg,
                "traj_scale": {n: trajmap[n].k for n in trajs},
                "traj_peak_deg": {n: math.degrees(traj_peak_rad(trajmap[n])) for n in trajs},
                "current_abort_A": current_abort_a, "jam_err_rad": jam_err_rad,
                "tip_cases": {}, "records": []}

    n_rec = 0
    total = len(tip_masses) * len(trajs) * args.reps
    for m_tip in tip_masses:
        # Each mass case requires a physical weight change. Stop torque and wait
        # for an explicit operator acknowledgement before starting that block.
        write_regs(ser, pid, 46, [0, 0])
        write_regs(ser, pid, 40, [0])
        write_regs(ser, pid, 55, [0])
        peak_deg = math.degrees(max(traj_peak_rad(trajmap[n]) for n in trajs))
        lim_txt = (f"，运行时硬限 {args.hard_limit_deg:.0f}°" if hard_limit_rad else "")
        print(f"\n[换砝码] 请安装端部总质量 {m_tip * 1000:.1f} g，锁紧 M6 螺母，"
              f"手动检查 ±{peak_deg:.0f}° 无碰撞{lim_txt}。")
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
        # 档位起始也先主动稳位(上一个档/换砝码后摆杆可能仍在摆动), 否则本档第一条
        # 记录仍从余摆中途起跑。safety.reset() 先重置看门狗 —— START 等待可能很久。
        safety.reset()
        st_s, st_q = settle_at_zero(ser, pid, q_zero, r16, safety)
        print(f"[i] 起跑前稳位 {st_s:.2f}s, 残余 {st_q:+.4f} rad")
        for tname in trajs:
            traj = trajmap[tname]
            for rep in range(args.reps):
                safety.reset()
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
                    glb = int(round(q_zero + sign * angle / LSB_RAD))
                    glb = min(4095, max(0, glb))  # 钳位而非 %4096: 越界即安全带失效, 不绕回
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
                    safety.check(current_A=cur, temp_C=tmp,
                                 pos_err_rad=(q - qg) if enable else None, pos_rad=q)
                    time.sleep(max(0.0, dt_loop - (time.time() - t_start - t_now)))
                st_s, st_q = settle_at_zero(ser, pid, q_zero, r16, safety)
                if abs(st_q) > 0.05:
                    print(f"[稳位残余 {st_q:+.3f}rad/{st_s:.1f}s] ", end="", flush=True)
                if len(entries) < 50:
                    print("FAIL(样本过少), 跳过")
                    continue
                dt_med = float(np.median([entries[i]["t"] - entries[i - 1]["t"]
                                          for i in range(1, len(entries))]))
                vin_samples = [e["vin"] for e in entries if e["vin"] is not None]
                vin_measured = float(np.median(vin_samples)) if vin_samples else 12.5
                log = {
                    "motor": motor, "kp": kp_eff, "vin": vin_measured, "dt": dt_med,
                    "mass": eq["m_eq_kg"], "arm_mass": 0.0, "length": eq["L_eq_m"],
                    "trajectory": tname, "tip_mass_kg": m_tip, "rep": rep,
                    "traj_scale": trajmap[tname].k,
                    "traj_peak_deg": math.degrees(traj_peak_rad(trajmap[tname])),
                    "amp_max_deg": args.amp_max_deg,
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
