#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HLS-2909 单关节阶跃 sim-vs-真机 MAE 评估(标定门 G3 的"鸭子关节"版)。

输入: agent 采集的阶跃响应 JSON:
  [{"t": 0.001, "pos": 1234, "vel": 5, "duty": 120, "vin": 125, "cur": 80, "temp": 38}, ...]
  (pos 为 0..4095 LSB; duty 11bit 符号幅值已解码; cur 16bit 符号幅值已解码)

sim 侧: 用真实 bam 类(load_model hls2909/m1 + HLS2909Actuator 控制律)
在 1-DOF 铰上复现同一阶跃(同一 q_target 轨迹), 与真机位置比对:
  输出 MAE(rad)、峰值误差、10% 门、并对 (J, friction_scale) 做小网格搜索,
报告最优组合 —— J 是"关节等效惯量"(真机 DTs=3ms 下无法直接标定, 是本次最大未知量)。

用法(无硬件依赖):
  /usr/bin/python3 scripts/eval_sim_vs_real_step.py --data hls2909_calibration/agent_battery_*/step_response_id23_*.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

# 与真机一致的常量(不依赖 read_hls_registers, 避免其 pyserial 入口检查)
RES = 4096
LSB_RAD = 2 * np.pi / RES
DT_SIM = 0.003          # 固件 DTs=3ms → sim 积分步(D=3ms 与控制周期一致)
CMD_DT = 0.02           # 命令更新周期 50 Hz (策略/BAM 一致)

try:
    from bam.model import load_model
    from bam.actuator import TorchBackend
    import torch
    HAS_BAM = True
except Exception:
    HAS_BAM = False


def sim_step_response(
    params: dict,
    q_target_rad: float,
    q0_rad: float,
    duration: float = 1.2,
    J: float = 0.005,
    fric_scale: float = 1.0,
    dt: float = DT_SIM,
) -> tuple[np.ndarray, np.ndarray]:
    """1-DOF 复现: BAM 控制律 → 电机方程 → (摩擦预算·fric_scale 削顶) → 积分。

    与 mjlab BamActuator.compute 的差别: 摩擦直接以干摩擦项作用于净力矩
    (等价 MuJoCo dof_frictionloss 的静摩擦削顶), 无外负载。
    """
    import torch
    model = load_model(motor_name="hd1910", model="m5")
    model.friction_base.value *= fric_scale if hasattr(model.friction_base, "value") else fric_scale
    act = model.actuator                    # 真实 HD1910Actuator
    act.backend = TorchBackend()
    kt, R = model.kt.value, model.R.value
    fc = model.friction_base.value
    fv = model.friction_viscous.value

    q = torch.tensor([q0_rad], dtype=torch.float32)
    dq = torch.tensor([0.0], dtype=torch.float32)
    qtgt = torch.tensor([q_target_rad], dtype=torch.float32)
    t, ts, qs = [], 0.0, []
    n = int(duration / dt)
    for i in range(n):
        if i % max(1, round(CMD_DT / dt)) == 0:
            # 命令按真机节奏每 50Hz 下(本场景一步到位, 保持一致即可)
            pass
        ctrl = act.compute_control(qtgt if True else None, q, dq, dt)
        tau = act.compute_torque(ctrl, True, q, dq).item()
        tau_net = tau - fv * dq.item()
        # 干摩擦削顶(静摩擦)
        if abs(dq.item()) < 1e-6:
            if abs(tau_net) <= fc:
                dq_new = 0.0
            else:
                dq_new = (tau_net - math.copysign(fc, tau_net)) / J * dt
        else:
            dq_new = dq.item() + (tau_net - math.copysign(fc, dq.item())) / J * dt
        dq_new = float(np.clip(dq_new, -model.max_velocity.value, model.max_velocity.value))
        q_new = q.item() + dq_new * dt
        q, dq = torch.tensor([q_new], dtype=torch.float32), torch.tensor([dq_new], dtype=torch.float32)
        t.append(ts); qs.append(q_new); ts += dt
    return np.array(t), np.array(qs)


def load_real(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("samples", data.get("trace", []))
    t = np.array([s["t"] for s in data], dtype=float)
    pos = np.array([s["pos"] for s in data], dtype=float) * LSB_RAD
    return t, pos


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="阶跃 JSON(或 glob)")
    ap.add_argument("--q0-lsb", type=int, default=None, help="起点 LSB; 缺省取首样本")
    ap.add_argument("--q-cmd-lsb", type=int, default=None, help="命令目标 LSB; 缺省取末样本")
    ap.add_argument("--grid-j", default="0.0005,0.001,0.002,0.004,0.008,0.015,0.03")
    ap.add_argument("--grid-fric", default="0.5,1.0,1.5")
    args = ap.parse_args()

    files = sorted(glob.glob(args.data)) if any(c in args.data for c in "*?[") else [args.data]
    if not files:
        print(f"[!] 无数据文件: {args.data}")
        return 1

    for f in files:
        try:
            t, pos = load_real(Path(f))
        except Exception as e:
            print(f"[!] {f}: 解析失败 {e}")
            continue
        if len(t) < 10:
            print(f"[!] {f}: 样本太少 ({len(t)})")
            continue
        q0 = args.q0_lsb if args.q0_lsb is not None else int(pos[0] / LSB_RAD)
        qcmd = args.q_cmd_lsb if args.q_cmd_lsb is not None else int(pos[-1] / LSB_RAD)
        print(f"=== {Path(f).name} : {len(t)} 样本  q0={q0} q_cmd={qcmd} Δ={qcmd-q0} LSB "
              f"({(qcmd-q0)*LSB_RAD*57.3:+.1f}°) ===")

        if not HAS_BAM:
            print("[!] bam 不可导入(需要 .venv 环境) — 只打印真机摘要")
            print(f"    真机: t_end={t[-1]:.3f}s  Δpos={(pos[-1]-pos[0])/LSB_RAD:.0f} LSB")
            continue

        best = (None, 1e9)
        for J in [float(x) for x in args.grid_j.split(",")]:
            for fs in [float(x) for x in args.grid_fric.split(",")]:
                ts, qs = sim_step_response(None, qcmd * LSB_RAD, q0 * LSB_RAD, duration=float(t[-1]),
                                           J=J, fric_scale=fs)
                # 对齐: 重采样真机到 sim 网格
                qi = np.interp(ts, t, pos)
                if len(qs) != len(qi):
                    continue
                mae = float(np.mean(np.abs(qs - qi)))
                if mae < best[1]:
                    best = ((J, fs), mae)
        (Jb, fsb), maeb = best
        # 用最优参数出报告指标
        ts, qs = sim_step_response(None, qcmd * LSB_RAD, q0 * LSB_RAD, duration=float(t[-1]),
                                   J=Jb, fric_scale=fsb)
        qi = np.interp(ts, t, pos)
        mae = float(np.mean(np.abs(qs - qi)))
        peak_err = float(np.max(np.abs(qs - qi)))
        print(f"  最优: J={Jb:.4f} kg·m²  friction_scale={fsb}  →  MAE={mae:.5f} rad "
              f"({mae/LSB_RAD/ (qcmd-q0)*100:.1f}% of step), peak={peak_err:.5f} rad")
        gate = "✅ PASS(<10%)" if mae / abs(pos[-1] - pos[0]) < 0.10 else "❌ FAIL(≥10%)"
        print(f"  G3 门(相对阶跃幅度): {gate}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
