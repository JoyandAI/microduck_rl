#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HD-1910-C001 全项特性测试（一次跑完，无需电流表）。

覆盖
----
  T1 kt       空载恒速法测力矩常数 ke≡kt（**需空载**）        R² + 斜率/截距
  T2 控制周期  阶跃时高频读 reg60，看占空比的更新粒度(reg81=10ms 待验证)  ms
  T3 回差      小步阶梯正反行程，比较同一目标下的位置差        LSB / 度
  T4 静摩擦    目标极缓上爬，记"开始动"的目标偏移与占空比      LSB / duty
  T5 D 项      (kp,kd) 组合下的阶跃响应：过冲/上升/整定      **需带载**(有惯量)
  T6 静置卸载  保持 60s，看是否漂移/卸载，随后测响应           —
  T7 热        保持 ~30% 占空比 120s，记温升与跟随变化         **需带载**
  T9 总线延迟  50 次写+读往返，报 p50/p95                     ms

自动判定载荷: 保持 +600LSB 的稳态占空比 >0.08 → 判为"带载(摆杆)"，否则"空载"。
带载/空载决定 T1 与 T5/T7 各自是否有效（脚本会打印实际执行清单）。

安全（与 record_pendulum_bench.py 同款）
--------------------------------------
临时限流 0.975A(reg44，结束恢复)；温度>50°C 中止；每步看门狗；目标越界(距 q0 超过
amp 上限)中止；HD 电流中止默认关闭(reg69 不可信)；结束断扭并恢复 reg44/reg46/reg50/reg51。

用法
----
    /usr/bin/python3 scripts/measure_hd1910_full.py --port /dev/ttyACM0 --id 1 \\
      --out hd1910_calibration/full_characterization
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
LSB_DEG = 360.0 / 4096.0
TEMP_ABORT_C = 50.0
WATCHDOG_S = 240.0          # 单测试看门狗(热测试最长)
AMP_LIMIT = 900             # 相对 q0 的最大偏移 [LSB]


class Bench:
    def __init__(self, port, pid, limit_a):
        import serial
        self.ser = serial.Serial(port, 1_000_000, timeout=0.1)
        self.pid = pid
        if not ping(self.ser, pid):
            raise RuntimeError(f"舵机 ID={pid} 无响应")
        r = read_regs(self.ser, pid, 44, 2)
        self.orig44 = u16_le(r, 0) if r else 300
        lim = int(round(limit_a / 0.0065))
        write_regs(self.ser, pid, 44, [lim & 0xFF, (lim >> 8) & 0xFF])
        self.orig50 = self.r(50, 1)
        self.orig51 = self.r(51, 1)
        # 配置写序: 46=0 → 46=max（已验证可动写序 #1）
        write_regs(self.ser, pid, 46, [0, 0])
        write_regs(self.ser, pid, 40, [0])
        write_regs(self.ser, pid, 46, [0xFF, 0x7F])
        time.sleep(0.4)
        self.q0 = float(np.median([v for v in (self.r(56, 2) for _ in range(30)) if v is not None]))
        self.t_start = time.time()
        self.amp = int(min(AMP_LIMIT, self.q0 - 200, 4095 - 200 - self.q0))

    def r(self, addr, n=2):
        b = read_regs(self.ser, self.pid, addr, n)
        if b is None:
            return None
        return u16_le(b, 0) if n == 2 else b[0]

    def duty(self):
        b = read_regs(self.ser, self.pid, 60, 2)
        return None if b is None else sm11(b, 0) * 0.001

    def cur(self):
        b = read_regs(self.ser, self.pid, 69, 2)
        return None if b is None else sm16(b, 0) * 0.0065

    def goal(self, g):
        g = int(max(0, min(4095, g)))
        write_regs(self.ser, self.pid, 42, [g & 0xFF, (g >> 8) & 0xFF])
        write_regs(self.ser, self.pid, 40, [1])
        return g

    def sample(self, t_ref):
        return dict(t=time.time() - t_ref, pos=self.r(56, 2), duty=self.duty(),
                    vin=(lambda v: None if v is None else v * 0.1)(self.r(62, 1)),
                    temp=self.r(63, 1), current_A=self.cur())

    def guard(self, pos, temp, tstamp):
        if temp is not None and temp > TEMP_ABORT_C:
            raise RuntimeError(f"温度 {temp}°C > {TEMP_ABORT_C}°C")
        if time.time() - tstamp > WATCHDOG_S:
            raise RuntimeError(f"看门狗 {WATCHDOG_S:.0f}s")
        if pos is not None and abs(pos - self.q0) > self.amp + 150:
            raise RuntimeError(f"目标越界 pos={pos} q0={self.q0:.0f}")

    def hold(self, g, dur, dt=0.006, fast=False):
        """保持目标 g 共 dur 秒，按 dt 采样（fast=True 时只读位置+占空比，采样更快）。"""
        t_ref = time.time()
        out = []
        while time.time() - t_ref < dur:
            self.goal(g)
            s = self.sample(t_ref) if not fast else dict(t=time.time() - t_ref, pos=self.r(56, 2), duty=self.duty())
            out.append(s)
            self.guard(s.get("pos"), s.get("temp"), t_ref)
            time.sleep(max(0.0, dt - (time.time() - t_ref - s["t"])))
        return out

    def slew(self, g_from, g_to, secs, dt=0.006):
        """按真实时间线性插值到 g_to（避免循环开销导致速度偏低）。"""
        t_ref = time.time()
        out = []
        while True:
            t = time.time() - t_ref
            f = min(1.0, t / secs)
            g = self.goal(g_from + (g_to - g_from) * f)
            s = self.sample(t_ref)
            s["target"] = g
            out.append(s)
            self.guard(s["pos"], s["temp"], t_ref)
            if f >= 1.0:
                break
            time.sleep(max(0.0, dt - (time.time() - t_ref - t)))
        return out

    def cleanup(self):
        try:
            write_regs(self.ser, self.pid, 40, [0])
            write_regs(self.ser, self.pid, 46, [0, 0])
            if self.orig50 is not None:
                write_regs(self.ser, self.pid, 50, [self.orig50])
            if self.orig51 is not None:
                write_regs(self.ser, self.pid, 51, [self.orig51])
            write_regs(self.ser, self.pid, 44, [self.orig44 & 0xFF, self.orig44 >> 8])
        except Exception:
            pass


# ---------------------------------------------------------------- tests
def detect_load(b: Bench):
    """保持 +600 LSB 看稳态占空比 → 判断轴上有没有摆杆/砝码。"""
    off = min(600, b.amp)
    s = b.hold(b.q0 + off, 0.8)
    d = [abs(x["duty"]) for x in s if x["duty"] is not None]
    held = float(np.mean(d[-max(3, len(d) // 3):])) if d else float("nan")
    return held, off


def t1_kt(b: Bench, speeds=(0.8, 1.2, 1.8, 2.6, 3.6, 5.0)):
    """空载恒速法: |duty|·vin = ke·|ω| + c  ⇒ 斜率 = ke ≡ kt。"""
    legs = []
    for w in speeds:
        for d in (+1, -1):
            for a, c in ((b.q0, b.q0 + d * b.amp), (b.q0 + d * b.amp, b.q0)):
                legs.append((w, d, a, c))
    recs = []
    for (w, d, g0, g1) in legs:
        dur = abs(g1 - g0) * LSB_RAD / w
        s = b.slew(g0, g1, dur)
        recs.append(dict(w_cmd=d * w, samples=s))
    rows = []
    for r in recs:
        s = r["samples"]
        s = s[int(len(s) * 0.25):int(len(s) * 0.75)]
        s = [x for x in s if x["pos"] is not None and x["duty"] is not None and x["vin"] is not None]
        if len(s) < 6:
            continue
        t = np.array([x["t"] for x in s]); q = np.array([x["pos"] for x in s]) * LSB_RAD
        rows.append(dict(w=float(np.polyfit(t, q, 1)[0]), duty=float(np.mean([abs(x["duty"]) for x in s])),
                         vin=float(np.median([x["vin"] for x in s])), n=len(s)))
    if len(rows) < 4:
        return dict(error="有效腿数不足")
    W = np.array([abs(r["w"]) for r in rows]); Y = np.array([r["duty"] * r["vin"] for r in rows])
    (k, c), *_ = np.linalg.lstsq(np.c_[W, np.ones(len(W))], Y, rcond=None)
    r2 = 1 - ((Y - (k * W + c)) ** 2).sum() / max(((Y - Y.mean()) ** 2).sum(), 1e-12)
    return dict(ke_kt=k, intercept=c, r2=float(r2), n_legs=len(rows),
                legs=[{kk: vv for kk, vv in r.items()} for r in rows])


def t2_control_period(b: Bench):
    """阶跃时高频读 reg60: 占空比只可能在控制更新时刻变化 → 变化间隔 = 控制周期。"""
    b.hold(b.q0, 0.4)
    t_ref = time.time()
    g = b.goal(b.q0 + 250)
    samples = []
    while time.time() - t_ref < 0.6:
        samples.append((time.time() - t_ref, b.duty()))
    b.goal(b.q0)
    time.sleep(0.5)
    # 游程分析: 占空比保持同一数值的连续样本时长 = 报告粒度(若固件以 10ms 更新, 应≈10ms)
    runs, cur, t0r = [], 0, None
    for (t, d) in samples:
        if d is None:
            continue
        if t0r is None:
            t0r = t
        cur += 1
        if d != samples[[i for i, x in enumerate(samples) if x[0] == t][0]][1]:
            pass
    # 简化: 用相邻样本差判断"新值开始"
    starts = []
    for i in range(1, len(samples)):
        (t0, d0), (t1, d1) = samples[i - 1], samples[i]
        if d0 is None or d1 is None:
            continue
        if abs(d1 - d0) > 1e-9:
            starts.append(t1)
    iv = np.diff(starts) if len(starts) > 2 else np.array([])
    runs_ms = None
    if len(starts) > 2:
        runs_ms = float(np.median(np.diff(starts)) * 1e3)
    return dict(n_changes=len(starts), change_interval_median_ms=runs_ms,
                change_interval_p25_ms=float(np.percentile(iv, 25) * 1e3) if len(iv) else None,
                raw_n=len(samples), samples_head=[[round(t, 4), d] for t, d in samples[:40]],
                note="若 reg60 只在固件控制更新时变化, 变化间隔中位数≈控制周期(reg81=10ms 待验证)")


def t3_backlash(b: Bench):
    """小步阶梯正反行程。"""
    step, n, settle = 40, 4, 0.35
    up, dn = [], []
    for i in range(n + 1):
        b.hold(b.q0 + i * step, settle)
        up.append(b.r(56, 2))
    for i in range(n, -1, -1):
        b.hold(b.q0 + i * step, settle)
        dn.append(b.r(56, 2))
    gaps = [u - d for u, d in zip(up, dn) if u is not None and d is not None]
    return dict(step_lsb=step, up=up, down=dn,
                hysteresis_lsb=float(np.mean(gaps)) if gaps else None,
                hysteresis_deg=float(np.mean(gaps) * LSB_DEG) if gaps else None)


def t4_stiction(b: Bench):
    """目标极缓上爬，找"开始动"的偏移与占空比（含死区 reg26/27=1LSB）。"""
    b.hold(b.q0, 0.5)
    t_ref = time.time()
    start = b.r(56, 2)
    onset = None
    duty_at = None
    for k in range(1, 401):                     # 最多 +200 LSB
        b.goal(b.q0 + k * 0.5)
        pos = b.r(56, 2)
        d = b.duty()
        if pos is not None and start is not None and abs(pos - start) >= 2 and onset is None:
            onset, duty_at = k * 0.5, d
            break
        time.sleep(0.05)
    b.hold(b.q0, 0.4)
    return dict(onset_offset_lsb=onset, onset_duty=None if duty_at is None else abs(duty_at),
                elapsed_s=round(time.time() - t_ref, 2))


def t5_dterm(b: Bench):
    """(kp,kd) 组合下的阶跃响应 —— 需要带载(有惯量)才能看出阻尼差异。"""
    combos = [(32, 16), (32, 40), (32, 100), (16, 40), (64, 40)]
    out = []
    for kp, kd in combos:
        write_regs(b.ser, b.pid, 50, [kp & 0xFF])
        write_regs(b.ser, b.pid, 51, [kd & 0xFF])
        time.sleep(0.2)
        r50, r51 = b.r(50, 1), b.r(51, 1)
        b.hold(b.q0, 0.4)
        s = b.hold(b.q0 + 150, 1.2, dt=0.005, fast=True)
        pos = np.array([x["pos"] for x in s if x["pos"] is not None], dtype=float)
        t = np.array([x["t"] for x in s if x["pos"] is not None])
        if len(pos) > 5:
            tgt = b.q0 + 150
            overshoot = float((pos.max() - tgt) * LSB_DEG)
            rise = float(t[np.argmax(pos >= (b.q0 + 0.9 * 150))] * 1e3) if (pos >= b.q0 + 0.9 * 150).any() else None
            ss = float(np.mean(pos[-8:]))
            out.append(dict(kp=kp, kd=kd, readback=(r50, r51), overshoot_deg=overshoot,
                            rise_90_ms=rise, steady_err_lsb=float(tgt - ss),
                            n=len(pos), pos_head=[round(float(p), 1) for p in pos[:12]]))
        b.hold(b.q0, 0.3)
    write_regs(b.ser, b.pid, 50, [b.orig50 or 32])
    write_regs(b.ser, b.pid, 51, [b.orig51 or 40])
    return dict(combos=out, restored=(b.r(50, 1), b.r(51, 1)))


def t6_unload(b: Bench, hold_s=60.0):
    """静置保持: 是否漂移/卸载（reg19 bit2 语义未确认）。"""
    g = b.q0 + 400
    b.hold(g, 0.8)                       # 先让它到位(否则"drift"其实只是阶跃响应)
    t_ref = time.time()
    series = []
    while time.time() - t_ref < hold_s:
        b.goal(g)
        p, d, t = b.r(56, 2), b.duty(), b.r(63, 1)
        series.append(dict(t=round(time.time() - t_ref, 2), pos=p,
                           duty=None if d is None else d, temp=t, reg40=b.r(40, 1)))
        time.sleep(0.5)
    pos = [x["pos"] for x in series if x["pos"] is not None]
    resp = b.hold(g + 100, 0.6, fast=True)
    ok = any(x["pos"] is not None and abs(x["pos"] - g) > 30 for x in resp)
    b.hold(b.q0, 0.4)
    return dict(hold_s=hold_s, pos_first=pos[0] if pos else None, pos_last=pos[-1] if pos else None,
                drift_lsb=(pos[-1] - pos[0]) if len(pos) > 1 else None,
                duty_first=series[0]["duty"], duty_last=series[-1]["duty"],
                temp_first=series[0]["temp"], temp_last=series[-1]["temp"],
                responsive_after_hold=bool(ok))


def t7_thermal(b: Bench, hold_s=120.0):
    """保持 ~30% 占空比，记温升与跟随/占空比漂移（需带载）。"""
    g = b.q0 + 700 if abs(700) < b.amp else b.q0 + b.amp
    t_ref = time.time()
    series = []
    while time.time() - t_ref < hold_s:
        b.goal(g)
        p, d, t = b.r(56, 2), b.duty(), b.r(63, 1)
        series.append(dict(t=round(time.time() - t_ref, 2), pos=p,
                           duty=None if d is None else d, temp=t, vin=b.r(62, 1)))
        if t is not None and t > TEMP_ABORT_C:
            break
        time.sleep(1.0)
    temps = [x["temp"] for x in series if x["temp"] is not None]
    dus = [abs(x["duty"]) for x in series if x["duty"] is not None]
    return dict(hold_s=hold_s, target=g, temp_start=temps[0] if temps else None,
                temp_max=max(temps) if temps else None, temp_end=temps[-1] if temps else None,
                duty_mean=float(np.mean(dus)) if dus else None,
                duty_drift=float(dus[-1] - dus[0]) if len(dus) > 1 else None, n=len(series))


def t9_latency(b: Bench, n=50):
    """写目标+读位置 的往返时间分布（参考 command_delay）。"""
    ts = []
    for i in range(n):
        t0 = time.perf_counter()
        b.goal(b.q0 + (10 if i % 2 else -10))
        b.r(56, 2)
        ts.append((time.perf_counter() - t0) * 1e3)
    b.hold(b.q0, 0.3)
    ts = np.array(ts)
    return dict(n=n, p50_ms=float(np.median(ts)), p95_ms=float(np.percentile(ts, 95)),
                min_ms=float(ts.min()), max_ms=float(ts.max()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--id", type=int, default=1)
    ap.add_argument("--limit-a", type=float, default=0.975)
    ap.add_argument("--out", default="hd1910_calibration/full_characterization")
    ap.add_argument("--skip", default="", help="跳过的测试, 逗号分隔(如 T7,T5)")
    ap.add_argument("--thermal-s", type=float, default=120.0)
    ap.add_argument("--unload-s", type=float, default=60.0)
    args = ap.parse_args()

    skip = {s.strip().upper() for s in args.skip.split(",") if s.strip()}
    b = Bench(args.port, args.id, args.limit_a)
    atexit.register(b.cleanup)
    print(f"[i] q0={b.q0:.0f} LSB ({b.q0*LSB_DEG:.1f}°)  可用幅值 amp={b.amp} LSB "
          f"({b.amp*LSB_DEG:.1f}°)  模式 reg33={b.r(33, 1)} Kp={b.r(50,1)} Kd={b.r(51,1)}")
    if b.amp < 250:
        print("[!] 可用幅值太小(零点太靠边)，请把舵盘/机体转一下再跑。")
        return 1

    held_duty, off = detect_load(b)
    loaded = bool(held_duty is not None and held_duty > 0.08)
    print(f"[i] 载荷判定: 保持 +{off} LSB 稳态 |duty|={held_duty:.3f} → "
          f"{'**带载(摆杆/砝码在轴上)**' if loaded else '**空载(仅舵盘)**'}")

    plan = [("T9", "总线延迟", t9_latency, {}),
            ("T2", "控制周期", t2_control_period, {}),
            ("T3", "回差/迟滞", t3_backlash, {}),
            ("T4", "静摩擦/启动", t4_stiction, {})]
    plan.append(("T1", "kt 空载恒速", t1_kt, {}) if not loaded else
                ("T5", "D 项扫 (kp,kd)", t5_dterm, {}))
    plan.append(("T6", "静置卸载", t6_unload, dict(hold_s=args.unload_s)))
    if loaded:
        plan.append(("T7", "热/降额", t7_thermal, dict(hold_s=args.thermal_s)))
    else:
        print("[i] 空载: 跳过 T5(D 项需惯量) 与 T7(热测试需负载电流) —— "
              "要测这两项请把摆杆装回去再跑一次本脚本。")

    res = {"meta": dict(port=args.port, id=args.id, q0_lsb=b.q0, amp_lsb=b.amp,
                        loaded=loaded, held_duty=held_duty, time=time.strftime("%Y-%m-%d %H:%M:%S"),
                        reg3=b.r(3, 2), mode=b.r(33, 1), kp=b.r(50, 1), kd=b.r(51, 1),
                        vin_V=(lambda v: None if v is None else v * 0.1)(b.r(62, 1)),
                        temp_C=b.r(63, 1)), "tests": {}}

    for tid, name, fn, kw in plan:
        if tid in skip:
            print(f"\n--- {tid} {name}: 跳过 ---")
            continue
        print(f"\n--- {tid} {name} ---", flush=True)
        t0 = time.time()
        try:
            r = fn(b, **kw)
            r["elapsed_s"] = round(time.time() - t0, 1)
            res["tests"][tid] = dict(name=name, ok=True, **r)
            print(f"    ✓ {json.dumps({k: v for k, v in r.items() if k not in ('legs', 'combos', 'up', 'down', 'pos_head')}, ensure_ascii=False, default=str)[:300]}")
        except Exception as e:                                  # 单项失败不中断
            res["tests"][tid] = dict(name=name, ok=False, error=f"{type(e).__name__}: {e}")
            print(f"    ✗ {type(e).__name__}: {e}")
            try:
                b.hold(b.q0, 0.5)
            except Exception:
                pass

    b.cleanup()
    atexit.unregister(b.cleanup)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fp = out / "hd1910_full_characterization.json"
    fp.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[done] 全部结果 → {fp.resolve()}  (扭矩已关, reg44/50/51/46 已恢复)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
