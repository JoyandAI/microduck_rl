#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 T5 实测阶跃波形反解固件 D 的**有效缩放**，并给出部署档 `kd` 的取值。

背景：模型控制律是
    duty = (q_t − q)·kp·error_gain/p_scale + (−dq)·kd·error_gain/d_scale
其中 P 项缩放由拟合确定（error_gain=0.163，与 reg50 一起给出实测的环刚度），
但 **D 项缩放（d_scale≈1/4 只是沿用 HLS 惯例，从未用真机数据验过）**。T5 提供了
(kp,kd) 五组带载阶跃的实测过冲/上升时间 —— 足够反解。

做法：用已辨识的 m5 被控对象做**闭环**仿真，扫"有效 D 系数"
    D_eff := kd·error_gain/d_scale   [占空比 / (rad/s)]
使仿真的过冲等于实测过冲，从而得到每个固件 kd 对应的 D_eff；
再看 D_eff 是否与 kd_fw 成正比（正比 ⇒ 线性 D 项），比例系数就是固件缩放。

用法：
    PYTHONPATH=vendor/bam .venv/bin/python scripts/calib_hd1910_dterm.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LSB_RAD = 2 * np.pi / 4096


def simulate_step(model, sim_mod, mass, length, arm_mass, vin, kp_fw, D_eff,
                  q0, step_rad, dt=1e-4, T=0.6):
    """闭环仿真一次阶跃，返回 (t, q) 轨迹（q 以 q0 为基准，rad）。"""
    eg = model.error_gain.value
    d_scale = model.actuator.d_scale
    # 让 D 的占空比系数等于目标 D_eff
    model.kd.value = D_eff * d_scale / eg
    log = {"mass": mass, "arm_mass": arm_mass, "length": length,
           "kp": kp_fw, "vin": vin, "dt": dt}
    model.actuator.load_log(log)
    sim = sim_mod.Simulator(model)
    sim.reset(q0, 0.0)
    q_t = q0 + step_rad
    ts, qs = [], []
    for _ in range(int(T / dt)):
        u = model.actuator.compute_control(q_t, sim.q, sim.dq, dt)
        sim.step(u, True, dt)
        ts.append(sim.t)
        qs.append(sim.q)
    return np.asarray(ts), np.asarray(qs) - q0


def metrics(t, q, step_rad):
    over = float((q.max() - step_rad) / LSB_RAD)              # 过冲 [LSB]
    thr = 0.9 * step_rad
    idx = np.argmax(q >= thr) if (q >= thr).any() else None
    rise = float(t[idx] * 1e3) if idx is not None else None   # 上升 90% [ms]
    return over, rise


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t5-json", default="hd1910_calibration/full_75_loaded_t5/hd1910_full_characterization.json")
    ap.add_argument("--model", default="hd1910_calibration/fit75v2/fit_m5_75v2.json")
    ap.add_argument("--out", default="hd1910_calibration/fit75v2/dterm_calibration.json")
    ap.add_argument("--dt", type=float, default=1e-4)
    args = ap.parse_args()

    from bam.model import load_model
    from bam import simulate as sim_mod

    t5 = json.loads(Path(args.t5_json).read_text(encoding="utf-8"))
    meta = t5["meta"]
    q0 = float(meta["q0_lsb"])
    vin = float(meta.get("vin_V") or 7.7)
    combos = [c for c in t5["tests"]["T5"]["combos"] if c.get("series")]
    print(f"[i] T5: q0={q0:.0f} LSB, vin={vin} V, {len(combos)} 组带波形")

    # 150 g + 15 cm 的等效单摆参数（与该日志一致）
    mass, length, arm_mass = 0.15922, 0.15329, 0.0
    step_lsb = 150
    step_rad = step_lsb * LSB_RAD

    model = load_model(json_file=args.model)
    eg = model.error_gain.value
    d_scale = model.actuator.d_scale
    kd_deployed = float(model.kd.value)          # 循环里会被改写，先记下来
    D_now = kd_deployed * eg / d_scale
    print(f"[i] 现部署档: kd={model.kd.value} eg={eg} d_scale={d_scale} "
          f"⇒ D_eff={D_now:.5f} duty/(rad/s)\n")

    grid = np.array([0.0, 0.002, 0.005, 0.008, 0.011, 0.0141, 0.018, 0.025, 0.035, 0.05, 0.08, 0.12])


    print(f"{'kp_fw':>6s} {'kd_fw':>6s} {'实测过冲':>9s} {'实测上升':>9s} "
          f"{'D@过冲匹配':>11s} {'D@上升匹配':>11s} {'仿真上升@D过冲':>13s}")
    res = []
    for c in combos:
        kp_fw, kd_fw = float(c["kp"]), float(c["kd"])
        meas_over = c["overshoot_deg"] / np.degrees(LSB_RAD)   # deg → LSB
        meas_rise = float(c["rise_90_ms"])
        scan = []
        for D in grid:
            t_s, q_s = simulate_step(model, sim_mod, mass, length, arm_mass, vin,
                                     kp_fw, D, q0, step_rad, dt=args.dt)
            scan.append((D, *metrics(t_s, q_s, step_rad)))
        ov = np.array([s[1] for s in scan])
        ri = np.array([s[2] for s in scan])
        Ds = np.array([s[0] for s in scan])
        # 过冲随 D 单调下降；上升时间随 D 单调上升 → 各自插值找匹配点
        o = np.argsort(ov)
        D_over = float(np.interp(meas_over, ov[o], Ds[o]))
        o2 = np.argsort(ri)
        D_rise = float(np.interp(meas_rise, ri[o2], Ds[o2]))
        t_s, q_s = simulate_step(model, sim_mod, mass, length, arm_mass, vin,
                                 kp_fw, D_over, q0, step_rad, dt=args.dt)
        _, rise_sim = metrics(t_s, q_s, step_rad)
        res.append(dict(kp_fw=kp_fw, kd_fw=kd_fw, meas_overshoot_deg=c["overshoot_deg"],
                        meas_rise_ms=meas_rise, D_eff_from_overshoot=D_over,
                        D_eff_from_rise=D_rise, sim_rise_ms=rise_sim))
        print(f"{kp_fw:6.0f} {kd_fw:6.0f} {c['overshoot_deg']:+9.3f} "
              f"{meas_rise:9.1f} {D_over:11.5f} {D_rise:11.5f} {rise_sim:13.1f}")

    # 取 kp_fw=32 的三点做线性关系（D_eff = s·kd_fw + b）
    sel = [r for r in res if r["kp_fw"] == 32.0]
    fit_info = None
    if len(sel) >= 2:
        kd_arr = np.array([r["kd_fw"] for r in sel])
        kd_deployed, D_now2 = None, None

        def lin(key):
            D_arr = np.array([r[key] for r in sel])
            A = np.vstack([kd_arr, np.ones_like(kd_arr)]).T
            sl, ic = np.linalg.lstsq(A, D_arr, rcond=None)[0]
            pr = A @ np.array([sl, ic])
            ss_res = float(((D_arr - pr) ** 2).sum())
            ss_tot = float(((D_arr - D_arr.mean()) ** 2).sum())
            to_kd = lambda D: D * d_scale / eg
            return sl, ic, (1 - ss_res / ss_tot if ss_tot > 0 else float("nan")), to_kd(sl * 40 + ic)

        s_ov, i_ov, r2_ov, kd_ov = lin("D_eff_from_overshoot")
        s_ri, i_ri, r2_ri, kd_ri = lin("D_eff_from_rise")
        fit_info = dict(slope_from_overshoot=s_ov, intercept_from_overshoot=i_ov, r2_ov=r2_ov,
                        kd_from_overshoot=kd_ov, kd_from_rise=kd_ri,
                        r2_from_rise=r2_ri, slope_from_rise=s_ri, intercept_from_rise=i_ri)
        print(f"\n[i] 现部署 kd={model.kd.value:.4f}（D_eff={D_now:.5f}）")
        print(f"[i] 按**过冲**匹配 ⇒ kd_fw=40 等效 D_eff={s_ov*40+i_ov:.5f} ⇒ 模型应取 "
              f"kd={kd_ov:.4f}（R²={r2_ov:.3f}）")
        print(f"[i] 按**上升时间**匹配 ⇒ kd_fw=40 等效 D_eff={s_ri*40+i_ri:.5f} ⇒ 模型应取 "
              f"kd={kd_ri:.4f}（R²={r2_ri:.3f}）")
        print("[i] 两个指标给出不同 D ⇒ 模型缺一个快瞬态滞后项（真机 D 作用在"
              "滤波/量化后的速度上）；部署值取两者之间属折中，见文档。")

    Path(args.out).write_text(json.dumps(
        {"source": args.t5_json, "model": args.model, "q0_lsb": q0, "vin_V": vin,
         "D_now": D_now, "deployed_kd": model.kd.value, "combos": res,
         "fit": fit_info},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[i] → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
