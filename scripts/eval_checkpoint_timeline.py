#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐检查点时间线评估：看行为**什么时候**出现，而不是只在训练结束看一次。

动机：走路"零命令站定"和坐起立"能不能起来"都是**后期才成形**的行为
（实测：走路 8250 迭代还只是"基本站定"，11999 才干净）。等 7.5 h 跑完才发现
方向不对太贵；本脚本把每个 `model_*.pt` 导出 ONNX 再评测一遍，给出时间线。

- 走路（`--kind walk`）：零命令 `gait_forensics`（步频 / resets / 倾角 / 偏航漂移）
- 坐起立（`--kind sitstand`）：`eval_sitstand_head`（各相位稳态 z / tilt）

用法：
    .venv/bin/python scripts/eval_checkpoint_timeline.py \
        --run-dir logs/rsl_rl/velocity/Mjlab-Velocity-Flat-MicroDuck_r2 --kind walk
    .venv/bin/python scripts/eval_checkpoint_timeline.py \
        --run-dir logs/rsl_rl/microduck_sitstand/Mjlab-SitStand-Flat-MicroDuck_r2 \
        --kind sitstand --every 2

注意：**会占用 GPU**（每个检查点要建一次环境 + 12 s rollout），训练在跑时请少查几个。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from queue_walk_sitstand_report import eval_sitstand, eval_walk_zero_cmd  # noqa: E402

PY = ".venv/bin/python"
PARAMS = "vendor/bam/bam/params/hd1910/m5.json"


def checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for p in run_dir.glob("model_*.pt"):
        m = re.search(r"model_(\d+)\.pt$", p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--task", default=None, help="默认按 run 目录名猜")
    ap.add_argument("--kind", choices=("walk", "sitstand"), default="walk")
    ap.add_argument("--every", type=int, default=1, help="每 N 个检查点评一次")
    ap.add_argument("--seconds", type=float, default=12.0, help="走路零命令 rollout 时长")
    ap.add_argument("--tmp", default="/tmp/ckpt_timeline")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"[err] 没有目录 {run_dir}")
        return 1
    task = args.task
    if task is None:
        task = ("Mjlab-Velocity-Flat-MicroDuck" if args.kind == "walk"
                else "Mjlab-SitStand-Flat-MicroDuck")
    ckpts = checkpoints(run_dir)
    if not ckpts:
        print(f"[i] {run_dir} 还没有 model_*.pt")
        return 0
    ckpts = ckpts[:: max(1, args.every)]
    tmp = Path(args.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    print(f"[i] {run_dir.name}: {len(ckpts)} 个检查点（{ckpts[0][0]} … {ckpts[-1][0]}），task={task}")

    rows = []
    for it, pt in ckpts:
        onnx = tmp / f"{run_dir.name}_model_{it}.onnx"
        if not onnx.exists():
            r = subprocess.run(
                # export.py 的第一个参数是**位置参数**(task id)，其余才是 ExportConfig
                [PY, "scripts/export.py", task,
                 "--checkpoint-file", str(pt), "--onnx-file", str(onnx)],
                capture_output=True, text=True)
            if r.returncode != 0 or not onnx.exists():
                print(f"  [{it}] 导出失败: {(r.stderr or r.stdout).strip().splitlines()[-1:]} ")
                continue
        if args.kind == "walk":
            j = tmp / f"zc_{run_dir.name}_{it}.json"
            subprocess.run(
                [PY, "scripts/gait_forensics.py", "--onnx", str(onnx), "--task", task,
                 "--lin-vel-x", "0", "--lin-vel-y", "0", "--ang-vel-z", "0",
                 "--seconds", str(args.seconds), "--json-out", str(j),
                 "--tag", f"iter {it} zero-cmd", "--params-json", PARAMS],
                capture_output=True, text=True)
            v = eval_walk_zero_cmd(str(j))
            rows.append({"iter": it, **v})
            print(f"  [{it:>6}] 步频 {v.get('gait_freq_hz'):.3f} Hz  resets={v.get('resets')}  "
                  f"tilt_max={v.get('tilt_max_deg'):.1f}°  离地={v.get('foot_clearance_mm'):.1f}mm  "
                  f"偏航漂移={v.get('yaw_drift_deg'):+.0f}°  ⇒ {'站定' if v.get('ok') else '未达标'}")
        else:
            log = tmp / f"ss_{run_dir.name}_{it}.log"
            with log.open("w") as fh:
                subprocess.run(
                    [PY, "scripts/eval_sitstand_head.py", "--onnx", str(onnx), "--task", task,
                     "--episode-length-s", "60", "--params-json", PARAMS],
                    stdout=fh, stderr=subprocess.STDOUT)
            v = eval_sitstand(str(log))
            rows.append({"iter": it, **v})
            print(f"  [{it:>6}] 末段 STAND z={v.get('last_stand_z_m')} tilt={v.get('last_stand_tilt_deg')} "
                  f"坐姿 z={v.get('min_sit_z_m')} ⇒ {v.get('why')}")

    if rows:
        out = Path(args.out) if args.out else tmp / f"timeline_{run_dir.name}.json"
        out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[i] → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
