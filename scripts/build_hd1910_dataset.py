#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""整理 HD1910 腿摆标定数据 → 可复现的 train/validation 划分 + 清单。

背景（⚠️ 必读）：
  `record_pendulum_bench.py:479` 原用 ``r16(50)`` 读 Kp，但 reg50=Kp、reg51=Kd 是
  两个独立的 8 位寄存器，于是 ``log["kp"] = 32 + 40*256 = 10272``。而
  ``bam/feetech/actuator.py:213`` 是 ``self.kp = log["kp"]``（日志值直接覆盖模型），
  导致模型 P 增益 = 10272 x error_gain = 实测真值(5.18) 的 323 倍，控制器全程 bang-bang。
  本脚本输出 ``kp`` 已修正为低字节(真实 Kp)的数据集。

用法:
    python3 build_dataset.py \
        --src /home/joyandai/microduck_rl/hd1910_calibration/bench_combined72_processed \
        --out /home/joyandai/bam/hd1910_fit/dataset
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


def kp_fixed(raw: int) -> int:
    """把 log["kp"] 修正为真实 Kp（低字节）。"""
    return int(raw) & 0xFF


def audit(d: dict) -> dict:
    T = d["telemetry"]
    q = np.array([x["position"] for x in T])
    u = np.array([x["duty"] for x in T])
    te = np.array([x["torque_enable"] for x in T])
    sp = np.array([x["speed"] for x in T])
    return {
        "n_samples": len(T),
        "torque_on_frac": round(float(te.mean()), 4),
        "duty_abs_max": round(float(np.abs(u).max()), 4),
        "duty_sat_frac": round(float(np.mean(np.abs(u) >= 0.95)), 5),
        "deg_min": round(float(np.degrees(q.min())), 2),
        "deg_max": round(float(np.degrees(q.max())), 2),
        "speed_abs_max": round(float(np.abs(sp).max()), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/joyandai/microduck_rl/hd1910_calibration/bench_combined72_processed")
    ap.add_argument("--out", default="/home/joyandai/bam/hd1910_fit/dataset")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    train_d, val_d = out / "train", out / "val"
    for p in (train_d, val_d):
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True)

    logs = {f.name: json.loads(f.read_text()) for f in sorted(src.glob("*.json"))
            if f.name != "manifest.json"}

    groups: dict[str, list[str]] = defaultdict(list)
    for n in logs:
        groups[n.rsplit("_rep", 1)[0]].append(n)

    entries, n_tr, n_va = [], 0, 0
    for key in sorted(groups):
        names = sorted(groups[key])
        val_names = {names[-1]} if len(names) >= 2 else set()
        for name in names:
            d = dict(logs[name])
            raw_kp = int(d["kp"])
            d["kp"] = kp_fixed(raw_kp)
            dst = val_d if name in val_names else train_d
            json.dump(d, open(dst / name, "w"))
            is_val = name in val_names
            if is_val:
                n_va += 1
            else:
                n_tr += 1
            entries.append({
                "name": name, "split": "val" if is_val else "train",
                "arm": "arm10" if name.startswith("arm10_") else "arm15",
                "trajectory": d["trajectory"], "rep": d["rep"],
                "tip_mass_kg": d["tip_mass_kg"], "length_m": d["length"],
                "mass_kg": d["mass"], "arm_mass_kg": d["arm_mass"],
                "vin": round(float(d["vin"]), 3),
                "kp_log_raw": raw_kp, "kp_used": d["kp"],
                "traj_scale": d["traj_scale"], **audit(d),
            })

    manifest = {
        "source": str(src),
        "n_total": len(entries), "n_train": n_tr, "n_val": n_va,
        "split_rule": "每组(轨迹_配重_臂长)的最后一次重复 rep2 留作独立验证",
        "kp_fix": "log['kp'] = raw & 0xFF；raw = kp + kd*256（reg50/51 被当成 16 位读的 bug）",
        "kp_raw_values": sorted({e["kp_log_raw"] for e in entries}),
        "kp_used_values": sorted({e["kp_used"] for e in entries}),
        "arms": sorted({e["arm"] for e in entries}),
        "trajectories": sorted({e["trajectory"] for e in entries}),
        "tip_masses": sorted({e["tip_mass_kg"] for e in entries}),
        "vin_values": sorted({e["vin"] for e in entries}),
        "integrity": {
            "any_duty_saturated_over_50pct": any(e["duty_sat_frac"] > 0.5 for e in entries),
            "max_abs_deg": max(max(abs(e["deg_min"]), abs(e["deg_max"])) for e in entries),
            "hard_limit_deg": "正 +96.7 / 负 -88.4（hd1910_servo_notes.md）",
        },
        "logs": entries,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"[i] {len(entries)} 条 → train {n_tr} / val {n_va}")
    print(f"[i] kp 修正: {manifest['kp_raw_values']} → {manifest['kp_used_values']}")
    print(f"[i] 最大摆角 {manifest['integrity']['max_abs_deg']}°（限位 {manifest['integrity']['hard_limit_deg']}）")
    print(f"[i] manifest: {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
