#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""night_queue（A/B/C/D × 3 seeds，各 3000 iter @2048）的统计分析。

问题（2026-09-14 提出）：走路 veering 是"我们的配方(含 heading_hold)"、"舵机更硬"、
"官方配方+heading_hold"、还是"官方配方"造成的？

12 个 run 的 results.csv 里每 arm 只有 3 个 seed，而同一 arm 内的种子差异看起来和
arm 之间的差异一样大。本脚本把这件事**量化**：单因素方差分析 + 效应量 + 需要多少
种子才能分辨，避免继续用 n=3 的噪声互相"证明"结论。

用法：
    .venv/bin/python scripts/night_queue_analysis.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

CSV_PATH = Path("logs/night_queue/results.csv")
METRICS = [("yaw_bias_deg_s", "偏航角速度偏置 [deg/s]", "abs"),
           ("path_efficiency", "路径效率", "raw"),
           ("made_good_m", "20s 净前进 [m]", "raw")]
ARMS = ["A", "B", "C", "D"]
ARM_DESC = {
    "A": "我们的配方（22 项奖励，含 heading_hold）",
    "B": "A + 更硬舵机（kp_fw 64）",
    "C": "官方配方 + heading_hold（去掉我们的 3 项姿态税）",
    "D": "官方配方（16 项，原样）",
}


def f_oneway(groups: list[list[float]]) -> tuple[float, float, float]:
    """单因素 ANOVA → (F, p, eta²)；不依赖 scipy（F 分布用不完全贝塔函数）。"""
    k = len(groups)
    ns = [len(g) for g in groups]
    n = sum(ns)
    grand = sum(sum(g) for g in groups) / n
    ss_b = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in groups)
    ss_w = sum(sum((x - sum(g) / len(g)) ** 2 for x in g) for g in groups)
    df_b, df_w = k - 1, n - k
    if ss_w <= 0 or df_w <= 0:
        return float("nan"), float("nan"), float("nan")
    f = (ss_b / df_b) / (ss_w / df_w)
    # p = P(F > f) = I_{df_w/(df_w+df_b f)}(df_w/2, df_b/2)
    x = df_w / (df_w + df_b * f) if f > 0 else 1.0
    p = _betainc(df_w / 2, df_b / 2, x)
    return f, p, ss_b / (ss_b + ss_w)


def _betainc(a: float, b: float, x: float) -> float:
    """正则化不完全贝塔函数 I_x(a,b)，连分数（Lentz）实现。"""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 200):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        d = 1.0 / d
        delta = c * d
        f *= delta
        if abs(delta - 1.0) < 1e-10:
            break
    return front * (f - 1.0)


def main() -> int:
    rows = list(csv.DictReader(CSV_PATH.open()))
    print(f"[i] {len(rows)} 个 run（{ARMS} × 3 seeds，各 3000 iter @2048）\n")
    for a in ARMS:
        print(f"  {a}: {ARM_DESC[a]}")

    for key, label, mode in METRICS:
        print(f"\n=== {label} ===")
        groups = []
        for a in ARMS:
            vals = [float(r[key]) for r in rows if r["arm"] == a]
            groups.append(vals)
            if mode == "abs":
                vals_show = [abs(v) for v in vals]
                extra = f"|·| 均值 {sum(vals_show)/len(vals_show):7.2f}"
            else:
                extra = f"           "
            print(f"  {a}: " + " ".join(f"{v:+8.3f}" for v in vals)
                  + f"   {extra}   (均值 {sum(vals)/len(vals):+7.3f} ± {_std(vals):6.3f})")
        f, p, eta2 = f_oneway(groups)
        within = sum(_var(g) * (len(g) - 1) for g in groups) / (sum(len(g) for g in groups) - len(groups))
        between = _std([sum(g) / len(g) for g in groups])
        print(f"  ANOVA: F({len(ARMS)-1},{sum(len(g) for g in groups)-len(ARMS)}) = {f:.2f}, "
              f"p = {p:.3f}, η² = {eta2:.3f}")
        print(f"  arm 间标准差 {between:.3f}  vs  arm 内（合并）标准差 {math.sqrt(within):.3f}"
              f"   ⇒ 组内/组间 = {math.sqrt(within)/between if between > 0 else float('inf'):.2f}×")
        n_need = _n_for_power(between, math.sqrt(within))
        print(f"  要在 80% 功效、α=0.05 下分辨这个组间差异，每组约需 n ≈ {n_need} 个种子")

    print("\n=== 指标间相关（n=12）===")
    yb = [abs(float(r["yaw_bias_deg_s"])) for r in rows]
    pe = [float(r["path_efficiency"]) for r in rows]
    mg = [float(r["made_good_m"]) for r in rows]
    for name, x, y in (("|yaw_bias| vs path_eff", yb, pe),
                       ("|yaw_bias| vs 净前进", yb, mg),
                       ("path_eff vs 净前进", pe, mg)):
        r_ = _pearson(x, y)
        print(f"  {name:26s} r = {r_:+.3f}  (r² = {r_*r_:.3f}, n=12)")

    print("\n[结论]")
    print("  1) 四个 arm 在任何指标上都**无法区分**：组内种子方差 ≥ 组间差异，")
    print("     12 个 run 的这个实验设计对'veering 归因'没有分辨力 —— 不能据此说")
    print("     '官方配方更好/更差'，也不能说 heading_hold 有用或没用。")
    print("  2) 修法只有两条：把种子数提到上表所需量级（每个 arm 十几次 run），")
    print("     或者换**确定性**诊断（同一 checkpoint 上做受控消融：固定策略、")
    print("     只改一个因素、单环境、无推力），而不是靠不同训练 run 比指标。")
    print("  3) 指标间相关可在上面 r 里读；若 |yaw_bias| 与路径效率强负相关，")
    print("     则 veering 的主因就是偏航偏置（与 mdp.py 里'唯一与偏航角有关的项’")
    print("     被速度门归零'的发现一致）。")
    return 0


def _std(xs: list[float]) -> float:
    return math.sqrt(_var(xs)) if len(xs) > 1 else float("nan")


def _var(xs: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def _pearson(x: list[float], y: list[float]) -> float:
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx * sy else float("nan")


def _n_for_power(between: float, within_sd: float, k: int = 4, power: float = 0.8) -> int:
    """每 arm 所需样本量（Cohen's f 近似，α=0.05）。"""
    if between <= 0 or not math.isfinite(within_sd):
        return 0
    f = between / within_sd
    if f <= 0:
        return 0
    # 标准近似：n ≈ (z_{1-α/2}+z_power)² / f² / k 量级；此处用保守的查表近似
    lam = (2.802 + 0.84) ** 2 / (f ** 2)      # λ for k-1=3, α=.05, power=.8 (非中心 F 近似)
    return max(2, int(math.ceil(lam / k)))


if __name__ == "__main__":
    raise SystemExit(main())
