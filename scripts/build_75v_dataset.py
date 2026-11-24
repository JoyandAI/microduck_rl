#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""7.5 V BAM 摆锤数据集构建（可复现，2026-09-15）。

把 6 个台架采集目录合并成规范数据集，修掉两处已知的数据缺陷：

1. **15 cm 的 50 g / 100 g 目录名与实际砝码相反**（实测证据：目录 "50" 顶部平台
   |duty| 0.0757 / 平衡误差 0.73°，目录 "100" 仅 0.0350 / 0.28°；重载侧误差更大
   ⇒ "50" 里装的是 100 g）。因此**互换这两个目录的物理元数据**（mass / length /
   tip_mass_kg），文件名按真实砝码重写。
2. **15 cm / 150 g 的旧批次 `bench_wegiht150_7.5V` 不可用**（该批 duty/B 与周期都
   与 150 g+15 cm 的几何不符），用重采批次 `bench_wegiht150_7.5V_rerun2` 替换；
   重采批在同一口径下位于载荷排序顶端（|duty| 0.1363 / 误差 1.46°，比 100 g 组
   高 1.8×/2.0×）⇒ 确为 150 g + 15 cm。

输出（`hd1910_calibration/` 下）：
  - `bench75_all_v2/`            72 条合并原始（未重采样，保留 entries+telemetry）
  - `bench75_all_v2_processed/`  重采样到 dt=0.005（bam.fit 可用格式）
  - `bench75_train48_v2/`        rep0+rep1 = 48 条训练
  - `bench75_val24_v2/`          rep2 = 24 条独立验证

用法：
    /usr/bin/python3 scripts/build_75v_dataset.py            # 写入
    /usr/bin/python3 scripts/build_75v_dataset.py --check    # 只体检不写
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from process_bench_logs import process_log  # noqa: E402  复用同一重采样口径

CAL = Path(__file__).resolve().parents[1] / "hd1910_calibration"

#: (源目录, 真实砝码 kg, 物理元数据取自哪个目录(None=自身), 输出文件名前缀)
SOURCES: list[tuple[str, float, str | None, str]] = [
    # 15 cm 杆（标签换位：目录名与真实砝码相反）
    ("bench_wegiht50_7.5V", 0.1024, "bench_wegiht100_7.5V", ""),
    ("bench_wegiht100_7.5V", 0.0524, "bench_wegiht50_7.5V", ""),
    # 15 cm / 150 g：重采批替换旧批
    ("bench_wegiht150_7.5V_rerun2", 0.1524, None, ""),
    # 10 cm 杆（标签经同口径载荷排序验证正确）
    ("bench_wegiht50_arm10cm_7.5V", 0.0524, None, "arm10_"),
    ("bench_wegiht100_arm10cm_7.5V", 0.1024, None, "arm10_"),
    ("bench_wegiht150_arm10cm_7.5V", 0.1524, None, "arm10_"),
]

META_KEYS = ("mass", "length", "arm_mass", "hub_mass", "hub_radius")
SPLIT_RE = re.compile(r"^(?P<traj>.+)_tip(?P<tip>[\d.]+)kg_rep(?P<rep>\d+)\.json$")


def out_name(traj: str, tip: float, rep: int, prefix: str) -> str:
    return f"{prefix}{traj}_tip{tip}kg_rep{rep}.json"


def build_merged(dst: Path, check_only: bool) -> list[dict]:
    """合并 6 个源目录 → dst，返回写入记录清单。"""
    written: list[dict] = []
    for src_dir, tip, meta_dir, prefix in SOURCES:
        src = CAL / src_dir
        if not src.is_dir():
            raise SystemExit(f"[err] 源目录不存在: {src}")
        meta_src = CAL / meta_dir if meta_dir else src
        for f in sorted(src.glob("*.json")):
            m = SPLIT_RE.match(f.name)
            if not m:  # 跳过 manifest.json 等
                continue
            traj, rep = m.group("traj"), int(m.group("rep"))
            log = json.loads(f.read_text(encoding="utf-8"))
            # 物理元数据（质量/等效摆长）以"真实砝码对应目录"的同名记录为准
            mp = meta_src / out_name(traj, tip, rep, prefix)
            if meta_dir and not mp.exists():
                raise SystemExit(f"[err] 元数据模板缺失: {mp}")
            meta = json.loads(mp.read_text(encoding="utf-8")) if meta_dir else log
            log = dict(log)
            for k in META_KEYS:
                if k in meta:
                    log[k] = meta[k]
            log["tip_mass_kg"] = tip
            log["rep"] = rep
            log["trajectory"] = traj
            log["source"] = f"{src_dir}/{f.name}"
            rec = {
                "file": out_name(traj, tip, rep, prefix),
                "traj": traj, "rep": rep, "tip": tip,
                "mass": log.get("mass"), "length": log.get("length"),
                "n": len(log.get("entries", [])),
            }
            written.append(rec)
            if not check_only:
                dst.mkdir(parents=True, exist_ok=True)
                (dst / rec["file"]).write_text(
                    json.dumps(log, indent=2), encoding="utf-8")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-prefix", default="bench75")
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--check", action="store_true", help="只体检不写文件")
    args = ap.parse_args()

    merged = CAL / f"{args.out_prefix}_all_v2"
    processed = CAL / f"{args.out_prefix}_all_v2_processed"
    train = CAL / f"{args.out_prefix}_train48_v2"
    val = CAL / f"{args.out_prefix}_val24_v2"

    recs = build_merged(merged, args.check)

    # 体检表：每组条数 / 等效质量 / 等效摆长 / B_max = m·g·L
    groups: dict[tuple[float, float], list[dict]] = {}
    for r in recs:
        key = (round(r["length"], 3), round(r["tip"], 4))
        groups.setdefault(key, []).append(r)
    print(f"{'杆长(m)':>8s} {'砝码(kg)':>9s} {'条数':>4s} {'等效质量':>9s} "
          f"{'等效摆长':>9s} {'B_max(N·m)':>10s}")
    for (L, tip), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        mass = rs[0]["mass"]
        print(f"{L:8.3f} {tip:9.4f} {len(rs):4d} {mass:9.5f} {L:9.5f} "
              f"{mass * 9.81 * L:10.4f}")
    n_rep = {k: sum(1 for r in recs if r["rep"] == k) for k in (0, 1, 2)}
    print(f"[i] 合并 {len(recs)} 条; rep{ n_rep } (rep2 → 独立验证)")
    if args.check:
        return 0

    # 重采样到 dt（bam.fit 输入口径）
    processed.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for f in sorted(merged.glob("*.json")):
        pr = process_log(json.loads(f.read_text(encoding="utf-8")), args.dt)
        if pr is None:
            print(f"[warn] {f.name} 样本不足，跳过")
            continue
        (processed / f.name).write_text(json.dumps(pr, indent=2), encoding="utf-8")
        n_ok += 1
    print(f"[i] 重采样 → {processed.name} ({n_ok} 条, dt={args.dt})")

    # 切分：rep2 = 独立验证，rep0/1 = 训练
    for d in (train, val):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.json"):
            old.unlink()
    n_tr = n_va = 0
    for f in sorted(processed.glob("*.json")):
        m = SPLIT_RE.match(f.name)
        if not m:
            continue
        rep = int(m.group("rep"))
        dst = val if rep == 2 else train
        (dst / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
        n_tr += rep != 2
        n_va += rep == 2
    print(f"[i] 切分 → {train.name} {n_tr} 条 / {val.name} {n_va} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
