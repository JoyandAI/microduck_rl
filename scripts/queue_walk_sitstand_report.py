#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""walk+sitstand 队列的自动判定报告（训练结束后跑）。

读 `logs/queue_walk_sitstand/` 下的 results.tsv 与评测产物，给出**判据化**的结论，
而不是让人再去看日志：

走路（用户要求：**静止时不许原地踏步**）
  ① 零命令站定：`resets==0` 且 步频 < 0.4 Hz 且 `tilt_max < 30°`
     —— 参考基线：11999 迭代策略 = 0.083 Hz / resets=0 / tilt_max 5.5°（合格）
  ② 0.2 m/s 步态：`resets==0` 且 步频 ∈ [0.5, 3.0] Hz（真的在迈步）
  ③ 参数指纹一致性：fp_before == fp_after，否则该 run 判 CONTAMINATED（09-10 事故守卫）

坐下起立
  从 `eval_sitstand_head.py` 的稳态表里取各相位 z 与 tilt：
  最后一段 STAND 的 z ≥ 0.10 m 且 tilt < 45° ⇒ 起立成功（坐姿 z 明显更低才说明真在坐）。

用法：
    .venv/bin/python scripts/queue_walk_sitstand_report.py            # 读默认队列目录
    .venv/bin/python scripts/queue_walk_sitstand_report.py --out logs/queue_walk_sitstand
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

# 【判据设计（2026-09-15 实测修正）】步频**不能**做零命令站定的硬判据：
# 同一 11999 检查点重复 4 次得 0.083 / 0.167 / 8.333 / 8.333 Hz —— 站着不动时
# 足底接触抖动会让频谱估计器输出 8.33 Hz（= 采样混叠），是噪声不是踏步。
# 干净的判别量是**离地高度**与**姿态**：踏步策略离地 19.4 mm，站定策略 0.5–4.2 mm
# （差 4–40×，且不随采样抖动）。步频只作信息量打印。
WALK_ZERO_MAX_CLEARANCE_MM = 8.0
WALK_ZERO_MAX_TILT_DEG = 15.0
# 【动作变化率】区分"真踏步"与"站着小幅抖"的第二判据。实测标定（2026-09-15，同一评测口径）：
#   站定：11999 四次重复 = 0.014 / 0.024 / 0.024 / 0.052   （上界 0.052）
#   踏步：旧 run 8000 = 0.064，新 run 1000 = 0.095          （下界 0.064）
# 0.06 落在两带之间；两个判据同时成立才算站定（保守）。
WALK_ZERO_MAX_ACTION_RATE = 0.06
# 【"resets" 不等于摔倒（2026-09-16 修正）】速度任务 episode_length_s = 20 s：评测跑满 20 s
# 必然产生一次**超时重置**，而它和"摔倒"在计数上无法区分。实测混淆案例：新 walk 策略零命令
# 20 s 评测 resets=1，但 tilt_max 仅 10.4°、z=118.6 mm（站得好好的）—— 那次是超时。
# 判据改为：**摔倒 = resets>0 且 tilt_max>45°**；评测时长默认取 18 s(<20) 以避开超时重置。
WALK_FALL_TILT_DEG = 45.0
WALK_ZERO_MIN_SAMPLE_S = 12.0
WALK_GAIT_FREQ_RANGE = (0.5, 3.0)
SITSTAND_MIN_STAND_Z_M = 0.10
SITSTAND_MAX_TILT_DEG = 45.0


def _load_json(p: str) -> dict | None:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def eval_walk_zero_cmd(path: str) -> dict:
    d = _load_json(path)
    if not d:
        return {"ok": False, "why": f"缺文件/解析失败: {path}"}
    g, t, a = d.get("gait", {}), d.get("trunk", {}), d.get("actions", {})
    freq = g.get("gait_freq_hz")
    clear = (g.get("foot_clearance_body_z_m") or 0) * 1e3
    tilt_max = t.get("tilt_max_deg")
    resets = d.get("resets")
    rate = a.get("mean_abs_rate")
    fell = bool(resets) and (tilt_max is not None and tilt_max > WALK_FALL_TILT_DEG)
    ok = ((not fell) and clear < WALK_ZERO_MAX_CLEARANCE_MM
          and (rate is not None and rate < WALK_ZERO_MAX_ACTION_RATE)
          and tilt_max is not None and tilt_max < WALK_ZERO_MAX_TILT_DEG)
    return {
        "ok": ok, "resets": resets, "gait_freq_hz": freq,
        "foot_clearance_mm": clear,
        "tilt_mean_deg": t.get("tilt_mean_deg"), "tilt_max_deg": tilt_max,
        "trunk_z_mm": (t.get("z_mean") or 0) * 1e3, "yaw_drift_deg": t.get("yaw_drift_deg"),
        "action_rate": a.get("mean_abs_rate"),
        "fell": fell,
        "why": ("站定合格" if ok else
                f"不合格: 摔倒={fell}(resets={resets}) 离地={clear:.1f}mm "
                f"(<{WALK_ZERO_MAX_CLEARANCE_MM}) 动作率={rate} "
                f"(<{WALK_ZERO_MAX_ACTION_RATE}) tilt_max={tilt_max}° "
                f"(<{WALK_ZERO_MAX_TILT_DEG})"),
    }


def eval_walk_gait(path: str) -> dict:
    d = _load_json(path)
    if not d:
        return {"ok": False, "why": f"缺文件/解析失败: {path}"}
    g, t = d.get("gait", {}), d.get("trunk", {})
    freq, resets = g.get("gait_freq_hz"), d.get("resets")
    lo, hi = WALK_GAIT_FREQ_RANGE
    tilt_max = t.get("tilt_max_deg")
    fell = bool(resets) and (tilt_max is not None and tilt_max > WALK_FALL_TILT_DEG)
    ok = ((not fell) and freq is not None and lo <= freq <= hi)
    return {"ok": ok, "resets": resets, "fell": fell, "tilt_max_deg": tilt_max,
            "gait_freq_hz": freq,
            "foot_clearance_mm": (g.get("foot_clearance_body_z_m") or 0) * 1e3,
            "tilt_mean_deg": t.get("tilt_mean_deg"), "yaw_drift_deg": t.get("yaw_drift_deg"),
            "why": ("步态合格" if ok else
                    f"不合格: 摔倒={fell}(resets={resets}) 步频={freq} Hz")}


_STEADY_ROW = re.compile(r"^\s*(?P<label>[A-Za-z_][A-Za-z0-9_ ]*?)\s+"
                         r"(?P<z>-?\d+\.\d+)\s+(?P<tilt>-?\d+\.\d+)\s+(?P<neck>-?\d+\.\d+)")


def eval_sitstand(path: str) -> dict:
    """从稳态表里取各相位 z/tilt；判据看**最后一段 STAND**。"""
    try:
        txt = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {"ok": False, "why": f"缺文件: {path}"}
    rows, in_steady = {}, False
    for line in txt.splitlines():
        if "STEADY STATE" in line:
            in_steady = True
            continue
        if not in_steady:
            continue
        if line.strip().startswith("---") or not line.strip():
            continue
        m = _STEADY_ROW.match(line)
        if m and "t[s]" not in line:
            rows.setdefault(m.group("label").strip(), []).append(
                (float(m.group("z")), float(m.group("tilt")), float(m.group("neck"))))
        if "timeline" in line:
            break
    if not rows:
        return {"ok": False, "why": "稳态表为空（评测未产出/格式变化）"}
    labels = list(rows)
    stand_labels = [k for k in labels if "STAND" in k.upper()]
    sit_labels = [k for k in labels if "SIT" in k.upper()]
    last_stand = rows[stand_labels[-1]][-1] if stand_labels else None
    sit_z = min((v[0] for k in sit_labels for v in rows[k]), default=None)
    ok = bool(last_stand and last_stand[0] >= SITSTAND_MIN_STAND_Z_M
              and abs(last_stand[1]) < SITSTAND_MAX_TILT_DEG)
    # 头下垂是文档记录的既有缺陷（`docs/policy_failure_analysis.md`：策略把头当第三条腿
    # 撑地），不构成本次判据的失败，但必须显式提示，别让它藏在"起立成功"后面。
    necks = [v[2] for k in labels for v in [rows[k][-1]] if len(rows[k][-1]) > 2]
    droop = min(necks) if necks else None
    return {"ok": ok, "phases": {k: rows[k][-1] for k in labels},
            "last_stand_z_m": last_stand[0] if last_stand else None,
            "last_stand_tilt_deg": last_stand[1] if last_stand else None,
            "min_sit_z_m": sit_z,
            "neck_rel_min_rad": droop,
            "neck_warning": (None if droop is None else
                             ("头下垂(颈 pitch 相对 HOME %.2f rad) —— 既有'头三脚架'问题，"
                              "非本次判据" % droop) if droop < -0.5 else "头姿正常"),
            "why": ("起立成功" if ok else
                    f"未起立/姿态差: last STAND z={last_stand} (阈值 z≥{SITSTAND_MIN_STAND_Z_M} m, "
                    f"|tilt|<{SITSTAND_MAX_TILT_DEG}°)")}


def repeated_zero_cmd(out: Path, onnx: str, tag: str, repeats: int, seconds: float,
                      task: str, params_json: str) -> dict:
    """重复跑零命令评测并取中位数。

    单样本噪声实测极大：同一 11999 检查点重复 4 次，步频 0.083→8.333 Hz、
    偏航漂移 −101°→+0.7°。判据取"每次都必须过"（保守），并报中位数与极差。
    """
    vals = []
    for i in range(repeats):
        j = out / f"{tag}_zc_r{i}.json"
        subprocess.run(
            [".venv/bin/python", "scripts/gait_forensics.py", "--onnx", onnx,
             "--task", task, "--lin-vel-x", "0", "--lin-vel-y", "0",
             "--ang-vel-z", "0", "--seconds", str(seconds),
             "--json-out", str(j), "--tag", f"{tag} r{i}",
             "--params-json", params_json],
            capture_output=True, text=True)
        v = eval_walk_zero_cmd(str(j))
        if v.get("gait_freq_hz") is not None:
            vals.append(v)
    if not vals:
        return {"ok": False, "why": "重复评测全部失败"}
    med = lambda k: sorted(v[k] for v in vals)[len(vals) // 2]
    ok = all(v["ok"] for v in vals)
    return {
        "ok": ok, "n": len(vals), "per_run_ok": [v["ok"] for v in vals],
        "foot_clearance_mm_median": med("foot_clearance_mm"),
        "foot_clearance_mm_max": max(v["foot_clearance_mm"] for v in vals),
        "tilt_max_deg_median": med("tilt_max_deg"),
        "gait_freq_hz_median": med("gait_freq_hz"),
        "yaw_drift_deg_median": med("yaw_drift_deg"),
        "yaw_drift_deg_range": [min(v["yaw_drift_deg"] for v in vals),
                                max(v["yaw_drift_deg"] for v in vals)],
        "why": (f"{len(vals)} 次重复全过（离地中位 {med('foot_clearance_mm'):.1f} / "
                f"最大 {max(v['foot_clearance_mm'] for v in vals):.1f} mm；"
                f"偏航漂移中位 {med('yaw_drift_deg'):+.0f}°，范围 "
                f"{min(v['yaw_drift_deg'] for v in vals):+.0f}…"
                f"{max(v['yaw_drift_deg'] for v in vals):+.0f}°）" if ok else
                f"{sum(v['ok'] for v in vals)}/{len(vals)} 次通过："
                f"{[v['why'] for v in vals if not v['ok']][:1]}"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/queue_walk_sitstand")
    ap.add_argument("--repeats", type=int, default=1,
                    help=">1 时本脚本自己重跑零命令评测 N 次（单样本噪声太大，最终验收建议 3–5）")
    ap.add_argument("--seconds", type=float, default=18.0,
                    help="评测时长；速度任务 episode=20 s，用 18 可避开超时重置（见 WALK_FALL_TILT_DEG 注释）")
    ap.add_argument("--task", default="Mjlab-Velocity-Flat-MicroDuck")
    ap.add_argument("--params-json", default="vendor/bam/bam/params/hd1910/m5.json")
    args = ap.parse_args()
    out = Path(args.out)
    tsv = out / "results.tsv"
    if not tsv.exists():
        print(f"[err] 没有 {tsv}")
        return 1
    rows = list(csv.DictReader(tsv.open(), delimiter="\t"))
    if not rows:
        print("[i] results.tsv 还没有数据行（训练仍在进行）")
        return 0

    print(f"{'run':<10}{'rc':>4}{'指纹一致':>10}{'判据':>10}  说明")
    print("-" * 100)
    verdicts = {}
    for r in rows:
        name, rc = r["name"], r.get("rc", "?")
        fp_ok = r.get("fp_before") == r.get("fp_after")
        if not fp_ok:
            verdicts[name] = {"ok": False, "why": "参数指纹在训练期间变化 ⇒ CONTAMINATED"}
        elif name == "walk":
            if args.repeats > 1 and r.get("onnx", "-") not in ("-", ""):
                v0 = repeated_zero_cmd(out, r["onnx"], name, args.repeats, args.seconds,
                                       args.task, args.params_json)
            else:
                v0 = eval_walk_zero_cmd(r.get("eval_zero_cmd", ""))
            vg = eval_walk_gait(r.get("eval_walk02", ""))
            verdicts[name] = {"ok": v0["ok"] and vg["ok"], "zero_cmd": v0, "gait": vg,
                              "why": f"零命令: {v0['why']} | 0.2m/s: {vg['why']}"}
        elif name == "sitstand":
            vs = eval_sitstand(r.get("eval_sitstand", ""))
            verdicts[name] = vs
        else:
            verdicts[name] = {"ok": False, "why": "未知 run"}
        v = verdicts[name]
        print(f"{name:<10}{str(rc):>4}{'✓' if fp_ok else '✗':>10}"
              f"{'PASS' if v.get('ok') else 'FAIL':>10}  {v.get('why','')[:70]}")

    print("\n--- 明细 ---")
    for name, v in verdicts.items():
        print(f"[{name}]")
        for k, val in v.items():
            if k == "why":
                continue
            print(f"  {k}: {val}")

    rep = out / "report.json"
    rep.write_text(json.dumps(verdicts, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[i] → {rep}")
    all_ok = all(v.get("ok") for v in verdicts.values())
    print(f"[{'PASS' if all_ok else 'FAIL'}] 全部判据{'通过' if all_ok else '未全部通过'}")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
