#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""腿摆标定 —— bam.fit 包装 + MAE 选档报告(你执行)。

对 record_leg_pendulum.py 录出的 log 目录跑 bam.fit(optuna/CMA-ES, 已随
.venv 可用), 对 m1/m3/m6 各拟合一次, 然后逐 log 回放算 sim-vs-真机位置 MAE,
输出 mae_report.md —— <10% 过门, 选较小者作为落地参数。

用法:
    .venv/bin/python scripts/fit_leg_pendulum.py \
      --logdir hls2909_calibration/pendulum \
      --actuator hls2909 --models m1 m6 \
      --trials 20000 --out hls2909_calibration/fit

说明: bam.fit 内部 = simulate.Simulator(Pendulum) + HLS2909Actuator 控制律 +
CMA-ES 优化(compute_score = 每 log 位置 MAE 均值)。fitted 参数在 --out 下
fit_mN.json; 需要 optuna(已装入 .venv)。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def build_params(logdir) -> dict:
    """直接把 log 目录 => dict(不依赖 bam 内部)。"""
    out = {}
    for f in sorted(Path(logdir).glob("*.json")):
        if f.name == "manifest.json":
            continue
        out[f.name] = json.loads(f.read_text(encoding="utf-8"))
    return out


def run_fit(args, model: str, logdir: str) -> Path:
    """子进程跑 bam.fit(与仓库 vendor 包一致)。"""
    out_json = Path(args.out) / f"fit_{model}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "bam.fit",
           "--logdir", logdir,
           "--actuator", args.actuator,
           "--model", model,
           "--method", "cmaes",
           "--trials", str(args.trials),
           "--workers", "1",
           "--output", str(out_json)]
    print("[fit] " + " ".join(cmd))
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        print(f"[fit] {model} 失败(返回 {r.returncode}), 跳过 MAE")
        return out_json
    return out_json


def evaluate(model_path: Path, logs: dict, reset_period: float = 0.5) -> float:
    """逐 log 回放求位置 MAE(与 bam.fit 的 compute_score 一致口径)。"""
    from bam.model import load_model
    from bam.simulate import Simulator

    model = load_model(json_file=str(model_path))
    sim = Simulator(model)
    maes = []
    for name, log in logs.items():
        res = sim.rollout_log(log, reset_period=reset_period, simulate_control=True)
        pos_pred = np.array(res[0])
        pos_real = np.array([e["position"] for e in log["entries"]])
        n = min(len(pos_pred), len(pos_real))
        maes.append(float(np.mean(np.abs(pos_pred[:n] - pos_real[:n]))))
    return float(np.mean(maes))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--actuator", default="hls2909")
    ap.add_argument("--models", nargs="+", default=["m1", "m6"])
    ap.add_argument("--trials", type=int, default=20000)
    ap.add_argument("--out", default="hls2909_calibration/fit")
    args = ap.parse_args()
    args.out = str(Path(args.out))
    Path(args.out).mkdir(parents=True, exist_ok=True)

    logs = build_params(args.logdir)
    print(f"[i] log 数 = {len(logs)}")
    if not logs:
        print("[!] 空 logdir, 先跑 record_leg_pendulum.py")
        return 1

    # 每个质量/轨迹的最后一次重复作为独立验证集，避免用拟合数据给自己打分。
    groups = {}
    for name in logs:
        key = name.rsplit("_rep", 1)[0]
        groups.setdefault(key, []).append(name)
    validation_names = {sorted(names)[-1] for names in groups.values() if len(names) >= 2}
    train_names = set(logs) - validation_names
    if not validation_names:
        print("[!] 每组只有 1 条数据，无法切独立验证集；请用 --reps >= 2 采集")
        return 1

    report = []
    with tempfile.TemporaryDirectory(prefix="hls2909_fit_") as tmp:
        fit_dir = Path(tmp)
        for name in train_names:
            shutil.copy2(Path(args.logdir) / name, fit_dir / name)
        train_logs = {k: logs[k] for k in sorted(train_names)}
        validation_logs = {k: logs[k] for k in sorted(validation_names)}
        print(f"[i] 分组: 拟合 {len(train_logs)} 条, 独立验证 {len(validation_logs)} 条")

        for mdl in args.models:
            print(f"\n===== 拟合 {mdl} =====")
            out_json = run_fit(args, mdl, str(fit_dir))
            if not out_json.exists():
                continue
            try:
                train_mae = evaluate(out_json, train_logs, reset_period=0.5)
                validation_mae = evaluate(out_json, validation_logs, reset_period=0.5)
            except Exception as e:
                print(f"[eval] {mdl} 评估失败: {e}")
                continue
            print(f"[eval] {mdl}: train={train_mae:.5f} rad, validation={validation_mae:.5f} rad")
            report.append({"model": mdl, "train_mae": train_mae,
                           "validation_mae": validation_mae, "params": str(out_json)})

    # 报告
    md = ["# 腿摆标定 MAE 报告\n",
          f"- log 目录: {args.logdir}({len(logs)} 条), 拟合输出: {args.out}\n",
          f"- 分组: 拟合 {len(train_names)} 条, 独立验证 {len(validation_names)} 条(每组最后一次重复)\n",
          "- 口径: bam.simulate 回放, 每 0.5s 与真机轨迹同步, 位置 MAE 平均\n",
          "- 门: **独立验证 MAE < 0.157 rad**且 train/validation 无明显分叉\n\n",
          "| 模型 | 拟合 MAE(rad) | 验证 MAE(rad) | 判定 |\n|---|---:|---:|---|\n"]
    for r in report:
        gate = "PASS" if r["validation_mae"] < 0.1 * np.pi / 2 else "FAIL"
        md.append(f"| {r['model']} | {r['train_mae']:.5f} | "
                  f"{r['validation_mae']:.5f} | {gate} |\n")
    md.append("\n## 建议\n- 取 PASS 中 MAE 最小档落地到 vendor/bam/bam/params/hls2909/mN.json;\n"
              "- 若全 FAIL: 检查装配(机身固定/零位/砝码 r_f)、挂载点测量, 或增大 trials;\n"
              "- 外置磁编 v1(高于 150Hz 采样)可再降 MAE。\n")
    rep = Path(args.out) / "mae_report.md"
    rep.write_text("".join(md), encoding="utf-8")
    print(f"\n[report] {rep.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
