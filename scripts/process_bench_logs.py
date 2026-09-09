#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""摆锤 log 后处理 —— 重采样到均匀 dt 网格(bam 回放按固定 dt 步进)。

BAM 的 rollout 假设 log["dt"] 恒定: 我们的记录是"时间戳变间隔"(总线抖动),
直接拟合会有 dt 失配。本脚本把每条 log 的 position/speed/goal_position/
torque_enable 线性插值到均匀网格(默认 5 ms), 剔除 NaN/空段, 输出
processed/ 供 fit_leg_pendulum.py 使用(对 raw 目录不动)。

用法:
    /usr/bin/python3 scripts/process_bench_logs.py \
      --in hls2909_calibration/bench --out hls2909_calibration/bench_processed \
      --dt 0.005
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def process_log(log: dict, dt: float) -> dict | None:
    entries = log.get("entries", [])
    if len(entries) < 50:
        return None
    t = np.array([e["t"] for e in entries]) if "t" in entries[0] else \
        np.arange(len(entries)) * ((entries[-1]["position"] - 0) * 0 + log["dt"])
    # 若没有 t 字段, 用累计 dt(按 log dt)
    keys = ["position", "speed", "goal_position", "torque_enable"]
    # 变间隔样本 → 均匀网格
    t = t.astype(float)
    if t[0] > t[-1] or np.ptp(t) < 1e-6:
        return None
    n = int(round((t[-1] - t[0]) / dt)) + 1
    tg = np.linspace(t[0], t[-1], n)
    out = {"entries": []}
    for k in log:
        if k != "entries":
            out[k] = log[k]
    out["dt"] = dt
    ok_entries = []
    for k in keys:
        if k not in entries[0]:
            continue
        v = np.array([e[k] for e in entries], dtype=float)
        vg = np.interp(tg, t, v)
        for i, vi in enumerate(vg):
            if i >= len(ok_entries):
                ok_entries.append({})
            ok_entries[i][k] = float(vi)
    for i, e in enumerate(ok_entries):
        ok_entries[i]["torque_enable"] = bool(e.get("torque_enable", False))
    out["entries"] = ok_entries
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", dest="out_dir", required=True)
    ap.add_argument("--dt", type=float, default=0.005)
    args = ap.parse_args()

    src, dst = Path(args.in_dir), Path(args.out_dir)
    dst.mkdir(parents=True, exist_ok=True)
    n_ok = n_skip = 0
    for f in sorted(src.glob("*.json")):
        if f.name == "manifest.json":
            continue
        try:
            log = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        pr = process_log(log, args.dt)
        if pr is None:
            n_skip += 1
            print(f"skip {f.name} (样本不足/时间异常)")
            continue
        (dst / f.name).write_text(json.dumps(pr, indent=2), encoding="utf-8")
        n_ok += 1
    print(f"[done] {n_ok} 条重采样 → {dst} (skip {n_skip})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
