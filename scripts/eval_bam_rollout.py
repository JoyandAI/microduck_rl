#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BAM 摆锤拟合结果评估：逐模型 × 逐组 × 逐相位的位置 MAE。

评估口径（两种都给，便于与历史数字对齐）：
  - `reset 0.5 s`：每 0.5 s 把仿真状态重新同步到真机 —— 与 bam.fit compute_score
    同口径，衡量**局部**模型保真度；
  - `full horizon`：全程不同步，模型自己闭环跑完整条轨迹 —— 衡量**部署**可靠性
    （策略仿真里没有任何 resync）。

相位拆分：torque_enable=True（上扭段）/ False（断扭自由落体段）。

用法：
    PYTHONPATH=vendor/bam .venv/bin/python scripts/eval_bam_rollout.py \
        --fit-dir hd1910_calibration/fit75v2 \
        --train-dir hd1910_calibration/bench75_train48_v2 \
        --val-dir hd1910_calibration/bench75_val24_v2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

TIP_RE = re.compile(r"_tip([\d.]+)kg")


def load_logs(d: str) -> dict[str, dict]:
    out = {}
    for f in sorted(Path(d).glob("*.json")):
        if f.name == "manifest.json":
            continue
        out[f.name] = json.loads(f.read_text(encoding="utf-8"))
    return out


def group_of(name: str, log: dict) -> str:
    arm = "10cm" if name.startswith("arm10_") else "15cm"
    tip = float(TIP_RE.search(name).group(1)) * 1000
    traj = name.split("_tip")[0].replace("arm10_", "")
    return f"{arm}/{tip:.0f}g/{traj}"


def rollout_err(model, log: dict, reset_period: float | None) -> dict:
    from bam.simulate import Simulator

    sim = Simulator(model)
    res = sim.rollout_log(log, reset_period=reset_period, simulate_control=True)
    pred = np.asarray(res[0], dtype=float)
    real = np.array([e["position"] for e in log["entries"]], dtype=float)
    te = np.array([bool(e["torque_enable"]) for e in log["entries"]])
    n = min(len(pred), len(real))
    err = np.abs(pred[:n] - real[:n])
    out = {"mae": float(err.mean()), "n": int(n)}
    for tag, mask in (("on", te[:n]), ("off", ~te[:n])):
        out[f"mae_{tag}"] = float(err[mask].mean()) if mask.sum() >= 5 else None
    return out


def eval_dir(model, logs: dict, reset_period: float | None) -> dict:
    per_log = {name: rollout_err(model, log, reset_period) for name, log in logs.items()}
    overall = float(np.mean([v["mae"] for v in per_log.values()]))
    by_group: dict[str, float] = {}
    for name, v in per_log.items():
        g = group_of(name, logs[name])
        by_group.setdefault(g, []).append(v["mae"])
    phases = {}
    for tag in ("on", "off"):
        vals = [v[f"mae_{tag}"] for v in per_log.values() if v[f"mae_{tag}"] is not None]
        phases[tag] = float(np.mean(vals)) if vals else None
    return {
        "overall": overall,
        "by_group": {k: float(np.mean(v)) for k, v in sorted(by_group.items())},
        "worst_group": max((float(np.mean(v)), k) for k, v in by_group.items()),
        "phase": phases,
        "per_log": per_log,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-dir", required=True)
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--pattern", default="fit_m*_*.json")
    ap.add_argument("--out", default=None, help="报告前缀（默认 <fit-dir>/mae_report）")
    args = ap.parse_args()

    from bam.model import load_model

    fit_dir = Path(args.fit_dir)
    models = sorted(fit_dir.glob(args.pattern))
    if not models:
        print(f"[err] {fit_dir} 下没有匹配 {args.pattern} 的拟合结果")
        return 1
    train, val = load_logs(args.train_dir), load_logs(args.val_dir)
    print(f"[i] 模型 {len(models)} 个; 训练 {len(train)} 条; 验证 {len(val)} 条\n")

    report = {}
    for mp in models:
        tag = mp.stem
        model = load_model(json_file=str(mp))
        r = {
            "model_file": str(mp),
            "train": eval_dir(model, train, 0.5),
            "val": eval_dir(model, val, 0.5),
            "val_full": eval_dir(model, val, None),
        }
        report[tag] = r
        print(f"== {tag} ==")
        print(f"   训练(reset0.5) {r['train']['overall']:.5f} | "
              f"验证(reset0.5) {r['val']['overall']:.5f} | "
              f"验证(全程) {r['val_full']['overall']:.5f}")
        w, wk = r["val"]["worst_group"]
        print(f"   验证最差组 {w:.5f} ({wk}); 相位 on/off "
              f"{r['val']['phase']['on']:.5f}/{r['val']['phase']['off']:.5f}")
        for g, v in r["val"]["by_group"].items():
            print(f"     {g:34s} {v:.5f}")

    # 汇总表（按验证 MAE 排序）
    print("\n| 模型 | 训练 | 验证(reset0.5) | 验证(全程) | 最差组 | 组数 |")
    print("|---|---:|---:|---:|---:|---:|")
    rows = sorted(report.items(), key=lambda kv: kv[1]["val"]["overall"])
    for tag, r in rows:
        w, wk = r["val"]["worst_group"]
        print(f"| {tag} | {r['train']['overall']:.5f} | {r['val']['overall']:.5f} | "
              f"{r['val_full']['overall']:.5f} | {w:.5f} ({wk.split('/')[-1]}) | "
              f"{len(r['val']['by_group'])} |")

    out = Path(args.out) if args.out else fit_dir / "mae_report"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[i] 明细 → {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
