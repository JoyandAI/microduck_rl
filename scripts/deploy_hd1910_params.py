#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 BAM 拟合结果落地：写参数 → 三处同步 → 更新全参数档案（含 md5 校验）。

落地位置（三处必须 md5 一致）：
  参数主副本  bam/bam/params/hd1910/m{1..6}.json
  运行时副本  microduck_rl/vendor/bam/bam/params/hd1910/m{1..6}.json   ← RL venv 实际加载的
  档案主档    microduck_rl/hd1910_calibration/params/hd1910_characterization.json
  档案副本    bam/bam/params/hd1910/characterization.json + vendor 同名文件

用法：
    PYTHONPATH=vendor/bam .venv/bin/python scripts/deploy_hd1910_params.py \
        --fit-dir hd1910_calibration/fit75v2 \
        --mae-report hd1910_calibration/fit75v2/mae_report.json \
        --dataset v2 --deployed m5 [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]          # microduck_rl
BAM = ROOT.parent / "bam"                           # /home/joyandai/bam
PARAM_DIRS = [BAM / "bam" / "params" / "hd1910",
              ROOT / "vendor" / "bam" / "bam" / "params" / "hd1910"]
CHAR_PATHS = [ROOT / "hd1910_calibration" / "params" / "hd1910_characterization.json",
              BAM / "bam" / "params" / "hd1910" / "characterization.json",
              ROOT / "vendor" / "bam" / "bam" / "params" / "hd1910" / "characterization.json"]

DATASET_TEXT = {
    "v2": ("72 条 @7.5V (2 杆长 10/15cm × 3 砝码 50/100/150g × 4 轨迹 × 3 次) —— "
           "**v2 数据集**：15cm 的 50/100g 目录名与实际砝码互换已修正(按对方元数据重写 mass/length/"
           "tip_mass + 文件名)，15cm/150g 用重采批次 bench_wegiht150_7.5V_rerun2 替换旧可疑批次；"
           "训练 48 / 独立验证 24 (rep2)"),
}


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-dir", required=True)
    ap.add_argument("--mae-report", default=None)
    ap.add_argument("--t5-json", default="hd1910_calibration/full_75_loaded/hd1910_full_characterization.json")
    ap.add_argument("--dataset", default="v2")
    ap.add_argument("--deployed", default="m5")
    ap.add_argument("--pinned", default="kt=0.692(实测) R=3.75 error_gain=0.163 kd=0.346 "
                                       "max_current=3.25 max_velocity=12.0423 max_acceleration=500.0")
    ap.add_argument("--trials", type=int, default=20000)
    ap.add_argument("--kd", type=float, default=None,
                    help="覆盖 kd（如 T5 阶跃标定值 0.5）；写回参数文件并在档案里记来源")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    fit_dir = ROOT / args.fit_dir
    fits = {}
    for m in ("m1", "m2", "m3", "m4", "m5", "m6"):
        cands = sorted(fit_dir.glob(f"fit_{m}_*.json"))
        if not cands:
            print(f"[err] 缺少 {m} 的拟合结果于 {fit_dir}")
            return 1
        fits[m] = json.loads(cands[0].read_text(encoding="utf-8"))
        if args.kd is not None:
            fits[m]["kd"] = float(args.kd)
            fits[m]["kd_source"] = ("T5 阶跃扫频反解 (150g+15cm, kp=32): 过冲匹配 D_eff=0.0205@kd_fw=40 "
                                    "⇒ kd≈0.50; 上升时间另给更强 D(不可信, 见 dterm_calibration.json)")
        print(f"[i] {m}: {cands[0].name}"
              + (f" (kd→{args.kd})" if args.kd is not None else ""))

    # ---- 1) 写参数三处 ----
    for d in PARAM_DIRS:
        for m, p in fits.items():
            dst = d / f"{m}.json"
            if not args.dry_run:
                dst.write_text(json.dumps(p, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")

    # ---- 2) 档案更新 ----
    report = json.loads((ROOT / args.mae_report).read_text()) if args.mae_report else {}
    t5 = None
    t5p = ROOT / args.t5_json
    if t5p.exists():
        t5 = json.loads(t5p.read_text(encoding="utf-8"))

    def mae_of(tag: str, key: str = "val", sub: str = "overall") -> float | None:
        for k, v in report.items():
            if k.endswith(tag) or f"_{tag}_" in k:
                return round(float(v[key][sub]), 5)
        return None

    val = {m: mae_of(m) for m in fits}
    train = {m: mae_of(m, "train") for m in fits}
    full = {m: mae_of(m, "val_full") for m in fits}

    def freefall(tag: str) -> float | None:
        for k, v in report.items():
            if k.endswith(tag) or f"_{tag}_" in k:
                xs = [x["mae_off"] for x in v["val"]["per_log"].values() if x["mae_off"] is not None]
                return round(float(np.mean(xs)), 5) if xs else None
        return None

    # 部署档：默认按 --deployed，但若并列(极差<0.0005)则保留现网默认
    cands = {m: v for m, v in val.items() if v is not None}
    best = min(cands, key=cands.get) if cands else args.deployed
    spread = (max(cands.values()) - min(cands.values())) if cands else float("nan")

    char = json.loads(CHAR_PATHS[0].read_text(encoding="utf-8"))
    char["generated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bm = char.setdefault("bam_models", {})
    bm["dataset"] = DATASET_TEXT.get(args.dataset, args.dataset)
    bm["pinned"] = f"{args.pinned} (CMA-ES {args.trials} trial)"
    bm["dataset_v1_mislabeled"] = ("旧 7.5V 结果(0.00789–0.00936)基于标签错位的数据集，已被本批取代；"
                                   "原始数据保留在 bench75_all_mislabeled*，拟合保留在 fit75/。")
    for key, table in (("validation_mae_rad", val), ("train_mae_rad", train),
                       ("validation_full_horizon_mae_rad", full),
                       ("freefall_mae_rad", {m: freefall(m) for m in fits})):
        bm[key] = {m: v for m, v in table.items() if v is not None}
    bm["validation_spread_rad"] = round(float(spread), 5)
    bm["deployed_model"] = args.deployed
    bm["deployed_reason"] = (
        f"独立验证 MAE 最优 {best}={cands.get(best)}，极差 {spread:.5f} rad；"
        + ("m4/m5/m6 并列(<0.0005 rad) ⇒ 保持现网默认 m5（物理上最完整，无需改配置）"
           if spread < 0.0005 or args.deployed == best else
           f"最优档 {best} 领先超过并列阈值 ⇒ 部署 {args.deployed}"))
    bm["deployed_params"] = fits[args.deployed]
    if t5:
        tests = t5.get("tests", {})
        if "T5" in tests:
            bm["dterm_hardware"] = {
                "note": "T5 (kp,kd) 阶跃扫频，带载 150g+15cm，150 LSB 阶跃；D 项影响明确（过冲随 kd 单调下降）",
                "combos": [{k: v for k, v in c.items() if k != "series" and k != "pos_head"}
                           for c in tests["T5"].get("combos", [])],
            }
        if "T7" in tests:
            t7 = tests["T7"]
            bm["thermal_hardware"] = {
                "note": "T7 静持 120 s（150g+15cm @7.5V）：温升与占空比漂移；模型无热参数，仅影响长跑预算",
                "temp_start_C": t7.get("temp_start"), "temp_max_C": t7.get("temp_max"),
                "duty_mean": t7.get("duty_mean"), "hold_s": t7.get("hold_s"),
                "target_lsb": t7.get("target"),
            }
    art = char.setdefault("artifacts", {})
    art["fits"] = f"{args.fit_dir}/fit_m{{1..6}}_{Path(args.fit_dir).name.split('_')[-1]}.json (钉死口径, v2 数据集)"
    art["data"] = ("hd1910_calibration/bench75_all_v2(合并) → bench75_all_v2_processed(重采样 dt=5ms) → "
                   "bench75_train48_v2 / bench75_val24_v2")
    art["eval"] = "scripts/eval_bam_rollout.py（逐模型×逐组×逐相位，含 reset0.5s 与全程两种口径）"
    pend = [x for x in char.get("pending_tests", [])
            if not x.startswith(("T5", "T7"))]
    char["pending_tests"] = pend

    if not args.dry_run:
        CHAR_PATHS[0].write_text(json.dumps(char, indent=2, ensure_ascii=False) + "\n",
                                 encoding="utf-8")
        for p in CHAR_PATHS[1:]:
            shutil.copy2(CHAR_PATHS[0], p)

    # ---- 3) md5 校验 ----
    print("\n[md5 校验]")
    ok = True
    for name in [f"{m}.json" for m in fits] + ["characterization.json"]:
        hs = []
        for d in PARAM_DIRS:
            f = d / name
            hs.append(md5(f) if f.exists() else "MISSING")
        same = len(set(hs)) == 1
        ok &= same
        print(f"  {name:26s} {'一致 ✓' if same else '不一致 ✗'}  {hs[0][:12]}")
    c0 = md5(CHAR_PATHS[0])
    for p in CHAR_PATHS[1:]:
        ok &= (md5(p) == c0)
    print(f"  characterization.json 三处一致 ✓ ({c0[:12]})" if ok else "  [warn] 有文件不一致")

    print("\n[落地结果]")
    print(f"  数据集: {bm['dataset'][:60]}...")
    print(f"  验证 MAE: {bm.get('validation_mae_rad')}")
    print(f"  部署档: {args.deployed} ({bm['deployed_reason'][:80]})")
    print(f"  kt={fits[args.deployed].get('kt')} R={fits[args.deployed].get('R')} "
          f"eg={fits[args.deployed].get('error_gain')} kd={fits[args.deployed].get('kd')}")
    if args.dry_run:
        print("\n[i] --dry-run：未写任何文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
