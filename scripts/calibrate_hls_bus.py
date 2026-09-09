#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HLS-2909(FT-HLS 固件 3.45, 型号号 6922)总线标定实验 —— 无需示波器/摆锤。

利用固件自带反馈做"软件标定"(理论依据见 docs/rl_servo_analysis.md §6):
  60 Present_Load   = 固件输出占空比 (0.1%/LSB, BIT10 方向)   → error_gain / max_pwm
  69 Present_Current= 电机相电流 (6.5mA/LSB, BIT15 方向)      → kt / R / 摩擦
  58 Present_Velocity= 当前速度 (0.732RPM/LSB, BIT15 方向)    → 速度
  62 Present_Voltage = 供电电压 (0.1V/LSB)                   → vin

原理(每条都源自 BAM 公式, 与摩擦无关):
  电学方程:  I = (duty·vin − kt·dq)/R
  ⇒ I = vin·duty/R − (kt/R)·dq    → 对数据做二元线性回归 → R = 1/a, kt = b/a
  摩擦(准静态, 恒速段):  τ_fric(ω) = kt·⟨I⟩ − ⟨τ_grav⟩   (整圈平均后重力项≈0)
  固件 P 环:  duty = (q_err/8)·kp·error_gain  (Kp 32 已读) → error_gain = 8·斜率/kp

实验(3 个; 均为"写寄存器+读数", 舵机会动, 已在台架裸机上授权):
  sweep:    恒速模式(33=1)下 46 号目标速度阶梯 ±10..250 LSB, 恒速段采样 (56,58,60,62,69,63)
  error-gain: 位置模式(33=0)下 42 号目标位置小阶跃 (±10/±30/±100/±300 LSB),
             瞬态采样 (56,60) → (误差, 占空比) 斜率
  temp:     只读温度/电压/电流(安全监控)

用法(系统 python 3.12, 自带 pyserial):
  /usr/bin/python3 scripts/calibrate_hls_bus.py --port /dev/ttyACM0 --id 10 sweep
  /usr/bin/python3 scripts/calibrate_hls_bus.py --port /dev/ttyACM0 --id 23 error-gain

输出: 打印拟合结果 + 保存 hls2909_calibration/<ts>/ 下 JSON(含每点原始数据)。
安全: 每步之间恢复零速位/原位; 温度 > 60°C 立即中止并关闭扭矩。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_hls_registers import (  # noqa: E402  (系统 python 只有 pyserial, 不依赖仓库 venv)
    RPM_TO_RAD_S,
    read_regs,
    sm11,
    sm16,
    u8,
    u16_le,
    write_regs,
)

# ── 常量(来自 docs/feetech_hls_memtable.md / 实测) ────────────────────────────
KPFW = 32            # EPROM 21 实测 Kp(Kp 1/8 固件缩放)
KDFW = 32            # EPROM 22 实测 Kd(Kd 1/4 缩放)
RESOLUTION = 4096    # 编码器分辨率/圈
LSB_TO_RAD = 2 * np.pi / RESOLUTION
V_LSB = 0.1          # 62 号 [V/LSB]
I_LSB = 0.0065       # 69 号 [A/LSB]
DUTY_LSB = 0.001     # 60 号 [1/LSB]
VEL_LSB_RAD_S = RPM_TO_RAD_S * 0.732     # 58/46 号 [rad/s per LSB]
TEMP_LIMIT_C = 60.0


def sm16_enc(v: int) -> list[int]:
    """16 位符号幅值编码(BIT15 方向)。"""
    mag = abs(int(v)) & 0x7FFF
    if v < 0:
        mag |= 0x8000
    return [mag & 0xFF, (mag >> 8) & 0xFF]


def read_sm16(ser, pid: int, addr: int):
    r = read_regs(ser, pid, addr, 2)
    return None if r is None else sm16(r, 0)


def read_sm11(ser, pid: int, addr: int):
    r = read_regs(ser, pid, addr, 2)
    return None if r is None else sm11(r, 0)


def read_u8(ser, pid: int, addr: int):
    r = read_regs(ser, pid, addr, 1)
    return None if r is None else r[0]


def read_state(ser, pid: int) -> dict:
    """一次采样: 位置/速度/占空比/电压/电流/温度。返回 None 读失败。"""
    pos = u16_le(read_regs(ser, pid, 56, 2) or b"\x00\x00", 0)
    vel = read_sm16(ser, pid, 58)
    duty = read_sm11(ser, pid, 60)
    vin = read_u8(ser, pid, 62)
    cur = read_sm16(ser, pid, 69)
    tmp = read_u8(ser, pid, 63)
    return {
        "pos_lsb": pos, "vel_lsb": vel, "duty_lsb": duty,
        "vin_L": vin, "cur_lsb": cur, "temp_C": tmp,
        "vel_rad_s": None if vel is None else vel * VEL_LSB_RAD_S,
        "duty": None if duty is None else duty * DUTY_LSB,
        "vin_V": None if vin is None else vin * V_LSB,
        "cur_A": None if cur is None else cur * I_LSB,
    }


def check_temp(ser, pid: int) -> None:
    t = read_u8(ser, pid, 63)
    if t is not None and t > TEMP_LIMIT_C:
        write_regs(ser, pid, 40, [0])  # 扭矩关
        raise SystemExit(f"[安全中止] 温度 {t}°C > {TEMP_LIMIT_C}°C")


def set_torque(ser, pid: int, on: bool) -> None:
    """Torque_Enable(40) + Lock(55) 必须一起写(lerobot 驱动同款):
    只写 40 时固件不输出 PWM(实测踩过)。"""
    write_regs(ser, pid, 40, [1 if on else 0])
    write_regs(ser, pid, 55, [1 if on else 0])


def set_mode(ser, pid: int, mode: int) -> None:
    write_regs(ser, pid, 33, [mode])


def sample_steady(ser, pid: int, n: int = 25, sleep_s: float = 0.006) -> list[dict]:
    out = []
    for _ in range(n):
        s = read_state(ser, pid)
        if s is not None and s["cur_lsb"] is not None:
            out.append(s)
        time.sleep(sleep_s)
    return out


def mean_ok(vals: list) -> float | None:
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


# ── 过载安全防护(所有实验强制生效) ─────────────────────────────────────────
TEMP_ABORT_C = 55.0        # 立即断扭矩的温度阈值(HLS 上限 ~80°C, 留裕量)
CURRENT_LIMIT_A = 1.2      # 持续电流阈值(额定连续 0.2A; 堵转保护 1.95A 太热)
CURRENT_BURST_S = 0.5      # 电流超限容忍时间(超时即断扭矩)
STALL_RECOVERY_S = 0.5     # 撞挡块后的"停滞确认→回收"时间上限
WATCHDOG_S = 90.0          # 单次实验总时长上限


class SafetyAbort(RuntimeError):
    pass


class Safety:
    """每个采样点前/后检查; 任何越限立即执行紧急停机序列并抛 SafetyAbort。"""

    def __init__(self, ser, pid: int):
        self.ser, self.pid = ser, pid
        self.t0 = time.time()
        self._over_current_since = None
        self.stall_since = None

    def read_temp(self):
        t = read_u8(self.ser, self.pid, 63)
        return None if t is None else float(t)

    def read_cur(self):
        c = read_sm16(self.ser, self.pid, 69)
        return None if c is None else float(c) * I_LSB

    def emergency(self, reason: str):
        """紧急停机: 扭矩关 + Lock 关 + 速度/目标回零(尽力而为)。"""
        try:
            write_regs(self.ser, self.pid, 46, [0, 0])
            write_regs(self.ser, self.pid, 40, [0])
            write_regs(self.ser, self.pid, 55, [0])
        except Exception:
            pass
        raise SafetyAbort(f"[安全中止] {reason}")

    def check(self, now: float | None = None, strict: bool = True) -> None:
        now = now if now is not None else time.time()
        if now - self.t0 > WATCHDOG_S:
            self.emergency(f"总时长超 {WATCHDOG_S:.0f}s")
        t = self.read_temp()
        if t is not None and t > TEMP_ABORT_C:
            self.emergency(f"温度 {t:.0f}°C > {TEMP_ABORT_C:.0f}°C")
        c = self.read_cur()
        if c is None:
            return
        # 严格档(数据采集): 1.2A×0.5s; 探测档(撞挡块不可避免): 1.9A×1.2s
        lim = CURRENT_LIMIT_A if strict else 1.9
        burst = CURRENT_BURST_S if strict else 1.2
        if abs(c) > lim:
            if self._over_current_since is None:
                self._over_current_since = now
            elif now - self._over_current_since > burst:
                self.emergency(f"持续电流 {c:+.2f}A > {lim:.1f}A 超过 {burst:.0f}s")
        else:
            self._over_current_since = None

    def stall_guard(self, moved_lsb: int) -> None:
        """撞挡块/停滞检测: 位置不动但没到目标 → 记开始时间, 超时回收。"""
        now = time.time()
        if moved_lsb >= 3:
            self.stall_since = None
            return
        if self.stall_since is None:
            self.stall_since = now
        elif now - self.stall_since > STALL_RECOVERY_S:
            self.emergency(f"输出轴停滞(挡块?)超过 {STALL_RECOVERY_S:.0f}s")


def probe_range(ser, pid: int, safety: Safety) -> tuple[int, int]:
    """探测输出轴自由行程两端(300 LSB 小步, 方向确定; 停滞 0.5s 即停)。
    返回 (lo, hi) 绝对位置 LSB。"""
    STEP = 300

    def cur_pos() -> int:
        r = read_regs(ser, pid, 56, 2)
        return u16_le(r or b"\x00\x00", 0)

    def probe_dir(dirn: int) -> int:
        pos = cur_pos()
        hit = False
        for _ in range(24):                   # 最多 24 步 × 0.3s
            safety.check(strict=False)
            goal = (pos + dirn * STEP) % RESOLUTION   # 300 < 2048 → 方向确定
            write_regs(ser, pid, 42, [goal & 0xFF, (goal >> 8) & 0xFF])
            time.sleep(0.3)
            now_pos = cur_pos()
            moved = (now_pos - pos) % RESOLUTION
            if (dirn > 0 and moved > RESOLUTION // 2) or (dirn < 0 and moved < RESOLUTION // 2):
                moved = 0                      # 反方向(异常)视为没动
            if moved < 20:
                safety.stall_guard(0)          # 挡块: 0.5s 内回收
                hit = True
                break
            pos = now_pos
        # 回收: 目标=当前位置(err=0), 低速待命
        write_regs(ser, pid, 46, [0, 0])
        write_regs(ser, pid, 42, [pos & 0xFF, (pos >> 8) & 0xFF])
        time.sleep(0.15)
        safety.stall_since = None
        if not hit:
            safety.emergency(f"探测 {dirn:+d} 方向 {24*STEP} LSB 未遇挡块(行程异常)")
        return pos

    set_torque(ser, pid, True)
    h1 = probe_dir(+1)
    safety.check(strict=False)
    l1 = probe_dir(-1)
    safety.check(strict=False)
    lo, hi = min(l1, h1), max(l1, h1)
    print(f"[probe] 自由行程 {lo}..{hi}  ({(hi-lo)*LSB_TO_RAD*57.3:.0f}°)")
    return lo, hi


# ── 实验 1: 限程往返扫速 → kt / R / 摩擦曲线 ────────────────────────────────
def cmd_sweep(ser, pid: int, out_dir: Path) -> None:
    """台架输出轴行程受限(实测 ID23 ≈ 51°): 先探测自由行程两端, 再在其间
    往返恒速采集 —— 正反向按位置配对 → 重力项相消, 只剩摩擦分量。
    数据点离挡块留 150 LSB 余量(不撞), 探边界时靠停滞检测限时回收。"""
    safety = Safety(ser, pid)
    print(f"[sweep] 舵机 ID={pid}: 探测自由行程 → 限程往返扫速 (防护:"
          f" {TEMP_ABORT_C:.0f}°C / {CURRENT_LIMIT_A:.1f}A×{CURRENT_BURST_S:.0f}s / 看门狗 {WATCHDOG_S:.0f}s)")
    set_torque(ser, pid, False); time.sleep(0.3)
    mode = read_u8(ser, pid, 33)
    if mode != 0:
        set_mode(ser, pid, 0); time.sleep(0.3)
    write_regs(ser, pid, 41, [254])
    write_regs(ser, pid, 46, [0xFF, 0x7F])   # 全速
    set_torque(ser, pid, True); time.sleep(0.2)

    lo, hi = probe_range(ser, pid, safety)
    margin = 150
    lo_s, hi_s = lo + margin, hi - margin
    if hi_s - lo_s < 200:
        write_regs(ser, pid, 40, [0]); write_regs(ser, pid, 55, [0])
        print(f"[sweep] 自由行程过小 ({hi-lo} LSB ≈ {(hi-lo)*LSB_TO_RAD*57.3:.0f}°), 放弃扫速: 需要 ≥200 LSB")
        return
    print(f"[sweep] 采样区间 {lo_s}..{hi_s}")

    def cur_pos() -> int:
        r = read_regs(ser, pid, 56, 2)
        return u16_le(r or b"\x00\x00", 0)

    # ── 往返扫速(端点在余量内, 不撞挡块) ──
    speeds = [10, 20, 40, 80, 120, 160, 200, 250]
    results = []          # 每拍原始样本 (θ, ω, I, duty, V, T)
    for spd in speeds:
        for dirn in (+1, -1):
            safety.check()
            start_pos = cur_pos()
            stop = hi_s if dirn > 0 else lo_s
            write_regs(ser, pid, 46, sm16_enc(spd * dirn))
            time.sleep(0.06)                  # 爬坡开始
            write_regs(ser, pid, 42, [stop & 0xFF, (stop >> 8) & 0xFF])
            # 恒速段密集采样(每次读前 safety.check; 读到挡块边缘即停)
            samples = []
            t_run = time.time()
            while time.time() - t_run < 1.2:
                s = read_state(ser, pid)
                if s is None:
                    continue
                samples.append(s)
                safety.check()
                if dirn > 0 and s["pos_lsb"] >= hi_s - 10:
                    break
                if dirn < 0 and s["pos_lsb"] <= lo_s + 10:
                    break
            # 回收: 停在当前位(不起扭矩对抗挡块), 短暂冷却
            now_pos = cur_pos()
            write_regs(ser, pid, 46, [0, 0])
            write_regs(ser, pid, 42, [now_pos & 0xFF, (now_pos >> 8) & 0xFF])
            time.sleep(0.25)
            safety.check()
            moved = abs(now_pos - start_pos)
            print(f"  {dirn:+d}×{spd:>3} LSB → Δ{moved:>4} LSB  样本 {len(samples)}"
                  f"  ω={mean_ok([s['vel_rad_s'] for s in samples]):+.3f} rad/s"
                  f"  I={mean_ok([s['cur_A'] for s in samples]):+.3f} A"
                  f"  T={mean_ok([s['temp_C'] for s in samples]):.0f}°C")
            results.append({"cmd_lsb": spd * dirn, "samples": [
                {"pos": s["pos_lsb"], "vel": s["vel_rad_s"], "cur": s["cur_A"],
                 "duty": s["duty"], "vin": s["vin_V"], "temp": s["temp_C"]}
                for s in samples]})

    set_torque(ser, pid, False)
    write_regs(ser, pid, 55, [0])

    # ── 拟合 ──
    X, y = [], []
    for r in results:
        for s in r["samples"]:
            if s["vel"] is None or s["cur"] is None or s["duty"] is None or s["vin"] is None:
                continue
            X.append([s["vin"] * s["duty"], -s["vel"]])
            y.append(s["cur"])
    if len(X) < 20:
        print("[sweep] 有效样本不足, 跳过拟合")
        return
    X, y = np.array(X), np.array(y)
    (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
    if a <= 0 or abs(b) < 1e-6:
        print(f"[sweep] 电学回归异常 (a={a:.4f}, b={b:.4f}), 检查数据")
        return
    R_fit = 1.0 / a
    kt_fit = b * R_fit
    yhat = X @ np.array([a, b])
    r2 = 1 - np.sum((y - yhat) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)

    # 摩擦: 按位置分桶, 正反向配对 → 重力相消
    bins = {}
    for r in results:
        up = r["cmd_lsb"] > 0
        for s in r["samples"]:
            if s["vel"] is None or s["cur"] is None:
                continue
            key = s["pos"] // 32
            bins.setdefault(key, {"fwd": [], "rev": []})
            (bins[key]["fwd"] if up else bins[key]["rev"]).append((abs(s["vel"]), kt_fit * s["cur"]))
    fric_pts = []
    for key, d in bins.items():
        if not d["fwd"] or not d["rev"]:
            continue
        for w1, t1 in d["fwd"]:
            for w2, t2 in d["rev"]:
                fric_pts.append((abs(w1 + w2) / 2, (t1 - t2) / 2))
    if fric_pts:
        w2_arr = np.array([p[0] for p in fric_pts])
        t2_arr = np.abs(np.array([p[1] for p in fric_pts]))
        (coul, visc), *_ = np.linalg.lstsq(
            np.column_stack([np.ones_like(w2_arr), w2_arr]), t2_arr, rcond=None)
    else:
        coul, visc = float("nan"), float("nan")

    out = {
        "summary": {
            "kp_fw": KPFW, "kd_fw": KDFW,
            "kt_Nm_per_A": float(kt_fit), "R_ohm": float(R_fit),
            "fit_r2": float(r2),
            "friction_coulomb_Nm": float(coul),
            "friction_viscous_Nm_s_per_rad": float(visc),
            "free_range_lsb": [lo, hi], "n_runs": len(results),
        },
        "runs": results,
        "friction_points": [{"abs_vel": float(w), "tau": float(t)} for w, t in fric_pts],
    }
    p = out_dir / "sweep.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[sweep] 电学回归: R={R_fit:.3f} Ω  kt={kt_fit:.3f} Nm/A  (R²={r2:.4f})")
    print(f"[sweep] 摩擦: Coulomb={coul:.4f} Nm  viscous={visc:.4f} Nm·s/rad")
    print(f"[sweep] 已保存 {p}")


# ── 实验 2: 位置模式误差→占空比 → error_gain ─────────────────────────────────
def cmd_error_gain(ser, pid: int, out_dir: Path) -> None:
    """在 free-range 内做小阶跃: duty = err·kp·error_gain/8 → error_gain。
    (台架行程 ~51° < 饱和所需误差, max_pwm 在本台架不可辨识 —— 记为受限。)
    """
    safety = Safety(ser, pid)
    print(f"[error-gain] 舵机 ID={pid}: 探测限程 → 误差阶梯标定 (防护同 sweep)")
    set_torque(ser, pid, False); time.sleep(0.3)
    mode = read_u8(ser, pid, 33)
    if mode != 0:
        set_mode(ser, pid, 0); time.sleep(0.3)
    write_regs(ser, pid, 41, [254])
    write_regs(ser, pid, 46, [0xFF, 0x7F])
    set_torque(ser, pid, True); time.sleep(0.2)

    lo, hi = probe_range(ser, pid, safety)
    margin = 150
    lo_s, hi_s = lo + margin, hi - margin
    pos0 = u16_le(read_regs(ser, pid, 56, 2) or b"\x00\x00", 0)
    print(f"[error-gain] 采样区间 {lo_s}..{hi_s}, 当前位 {pos0}")

    def clamp_goal(goal: int) -> int:
        return max(lo_s, min(hi_s, goal))

    pairs_up, pairs_down = [], []
    duty_max = 0.0
    for delta in [10, 30, 100, 300]:
        for sign in (+1, -1):
            safety.check()
            goal = clamp_goal(pos0 + sign * delta)
            if goal == pos0:
                print(f"  Δ={sign * delta}: 已到限程端, 跳过")
                continue
            # 回位轮询: 确保上一笔动完(位置回到 pos0 ±5)再发新指令
            for _ in range(20):
                time.sleep(0.05)
                p_now = u16_le(read_regs(ser, pid, 56, 2) or b"\x00\x00", 0)
                if abs(((p_now - pos0 + RESOLUTION // 2) % RESOLUTION) - RESOLUTION // 2) <= 5:
                    break
            write_regs(ser, pid, 42, [goal & 0xFF, (goal >> 8) & 0xFF])
            # 只取"下发后前 3 拍"(dq≈0 → D 项≈0, duty≈P 项 = err·kp·error_gain/8)
            t0 = time.time()
            n_early = 0
            while time.time() - t0 < 0.12 and n_early < 4:
                time.sleep(0.002)
                pos = u16_le(read_regs(ser, pid, 56, 2) or b"\x00\x00", 0)
                vel = read_sm16(ser, pid, 58)
                duty_lsb = read_sm11(ser, pid, 60)
                if duty_lsb is None:
                    continue
                if vel is not None and abs(vel) > 2:
                    continue                      # 已经跑起来 → D 项入镜, 丢
                err = (goal - pos + RESOLUTION // 2) % RESOLUTION - RESOLUTION // 2
                if sign * err <= 0:
                    continue
                duty = duty_lsb * DUTY_LSB
                if abs(duty) > abs(duty_max):
                    duty_max = duty
                (pairs_up if sign > 0 else pairs_down).append((err * LSB_TO_RAD, duty))
                n_early += 1
                safety.check(strict=False)
            write_regs(ser, pid, 42, [pos0 & 0xFF, (pos0 >> 8) & 0xFF])
            time.sleep(0.3)
            safety.check()
            print(f"  Δ={sign * delta:>4} LSB({sign*delta*LSB_TO_RAD*57.3:+.1f}°):"
                  f" 早段样本 {len(pairs_up if sign > 0 else pairs_down)} 点")

    set_torque(ser, pid, False)
    write_regs(ser, pid, 55, [0])

    def fit_slope(pairs):
        e = np.array([p[0] for p in pairs]); d = np.array([p[1] for p in pairs])
        m = np.abs(d) < abs(duty_max) * 0.98     # 只取未饱和区
        if m.sum() < 10 or e[m].ptp() < 1e-4:
            return float("nan"), float("nan")
        (slope, offset), *_ = np.linalg.lstsq(np.column_stack([e[m], np.ones(m.sum())]), d[m], rcond=None)
        return float(slope), float(offset)

    s_up, b_up = fit_slope(pairs_up)
    s_dn, b_dn = fit_slope(pairs_down)
    s_all, b_all = fit_slope(pairs_up + pairs_down)
    slope = float(np.mean([s_up, s_dn])) if np.isfinite(s_up) and np.isfinite(s_dn) else s_all
    # duty = err·kp·error_gain/8 → error_gain = 8·slope/kp
    error_gain = 8 * slope / KPFW

    out = {
        "summary": {
            "kp_fw": KPFW,
            "slope_duty_per_rad": slope, "offset_duty": b_all,
            "error_gain": error_gain,
            "max_pwm_measured": abs(duty_max),
            "n_pairs": len(pairs_up) + len(pairs_down),
        },
        "pairs_up": pairs_up, "pairs_down": pairs_down,
    }
    p = out_dir / "error_gain.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[error-gain] slope={slope:.4f} duty/rad → error_gain={error_gain:.4f} (kp={KPFW}, /8)")
    print(f"[error-gain] max_pwm(实测饱和)={abs(duty_max):.3f}")
    print(f"[error-gain] 已保存 {p}")


def cmd_reset(ser, pid: int) -> None:
    set_torque(ser, pid, False)
    write_regs(ser, pid, 46, [0, 0])
    time.sleep(0.2)
    print(f"[reset] ID={pid} 扭矩已关, 速度归零")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--id", type=int, default=10)
    ap.add_argument("cmd", choices=["sweep", "error-gain", "reset"])
    args = ap.parse_args()

    import serial
    ser = serial.Serial(args.port, 1_000_000, timeout=0.3)
    ser.reset_input_buffer()
    print(f"[i] {args.port} @1Mbps 打开, 舵机 ID={args.id}")

    out_dir = Path("hls2909_calibration") / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[i] 输出目录 {out_dir.resolve()}")

    try:
        if args.cmd == "sweep":
            cmd_sweep(ser, args.id, out_dir)
        elif args.cmd == "error-gain":
            cmd_error_gain(ser, args.id, out_dir)
        else:
            cmd_reset(ser, args.id)
    finally:
        try:
            write_regs(ser, args.id, 40, [0])
            write_regs(ser, args.id, 46, [0, 0])
        except Exception:
            pass
        ser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
