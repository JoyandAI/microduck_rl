#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成腿摆标定 runbook 的示意图(原理/装配/轨迹)。可重现:
    .venv/bin/python scripts/make_leg_pendulum_figs.py
输出: docs/leg_pendulum_figs/fig1_physics.png, fig2_setup.png, fig3_trajectories.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent.parent / "docs" / "leg_pendulum_figs"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ── Fig 1: 物理原理(腿摆 → 等效摆锤) ─────────────────────────────────────────
def fig1() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # --- 左: 腿摆侧视示意 ---
    ax1.set_xlim(-0.5, 3.6); ax1.set_ylim(-3.3, 1.1); ax1.set_aspect("equal")
    ax1.axis("off")
    # 桌面
    ax1.plot([-0.4, 3.5], [0.0, 0.0], color="k", lw=3)
    ax1.text(3.3, 0.15, "桌面边缘", ha="right", fontsize=11)
    # 机身(俯趴) + 压块
    ax1.add_patch(FancyBboxPatch((0.3, 0.15), 1.9, 0.45, boxstyle="round,pad=0.04",
                                 fc="#e8e8e8", ec="k", lw=1.2))
    ax1.add_patch(Rectangle((0.7, 0.62), 1.0, 0.3, fc="#9ec9e8", ec="k", lw=1.2))
    ax1.text(1.2, 0.78, "压块 ≥5kg", ha="center", fontsize=10)
    ax1.text(1.15, 0.42, "鸭身(刚性固定)", ha="center", fontsize=10)
    ax1.add_patch(Circle((0.75, -0.05), 0.14, fc="#d9d9d9", ec="k", lw=1.2))
    ax1.text(0.75, 0.22, "髋轴(hip)\n轴⊥纸面", ha="center", fontsize=9)
    # 腿(下垂, 摆角 θ)
    th = np.deg2rad(28)
    L = 2.5
    ax1.plot([0.75, 0.75 + L * np.sin(th)], [-0.05, -0.05 - L * np.cos(th)],
             color="#c00000", lw=4, solid_capstyle="round")
    ax1.text(0.75 + L * np.sin(th) / 2 + 0.15, -0.9, "腿(刚性)", color="#c00000", fontsize=10)
    # 垂直参考线 + θ
    ax1.plot([0.75, 0.75], [-0.05, -0.05 - L * np.cos(th) * 1.35], "k--", lw=0.8)
    ax1.add_patch(Arc((0.75, -0.05), 1.5, 1.5, theta1=-90 + 0, theta2=-90 + np.rad2deg(th),
                      color="k", lw=1.4))
    ax1.text(1.02, -0.62, "θ(从垂下)", fontsize=10)
    # COM
    r = 0.42
    com = (0.75 + r * np.sin(th), -0.05 - r * np.cos(th))
    ax1.plot(*com, "o", ms=9, color="k")
    ax1.annotate("m_leg·r_com\n(质心, MJCF 已知)", com, xytext=(1.75, -2.6),
                 arrowprops=dict(arrowstyle="->", color="k"), fontsize=10)
    # 足端砝码
    rf = 2.2
    foot = (0.75 + rf * np.sin(th), -0.05 - rf * np.cos(th))
    ax1.add_patch(Rectangle((foot[0] - 0.16, foot[1] - 0.3), 0.32, 0.3,
                            fc="#8fd18f", ec="k", lw=1.2))
    ax1.text(foot[0], foot[1] - 0.45, "砝码 m_f 100/200g", ha="center", fontsize=10)
    ax1.annotate("挂点距髋轴 r_f(卡尺实测)", foot, xytext=(2.15, -0.2),
                 arrowprops=dict(arrowstyle="->", color="k"), fontsize=10)
    # 重力与扭矩公式
    for x, y in [(1.1, -2.3), (0.75 + rf * np.sin(th), -1.9)]:
        ax1.add_patch(FancyArrowPatch((x, y), (x, y - 0.28), arrowstyle="-|>", color="#555", lw=1.4))
    ax1.text(1.15, -3.15, "重力矩  B(θ) = (m_leg·r_com + m_f·r_f)·g·sin θ\n"
                         "绕轴惯量  M = m_leg·r_com² + m_f·r_f²  (精确值取 MJCF)",
             fontsize=11, ha="center", color="#333")

    # --- 右: 等效变换 ---
    ax2.set_xlim(-0.2, 3.6); ax2.set_ylim(-2.6, 1.4); ax2.axis("off")
    ax2.text(1.6, 1.2, "等效到标准 Pendulum(不改 bam)", ha="center", fontsize=13, weight="bold")
    ax2.text(0.9, 0.55, "腿摆\n(M, B_max)", ha="center", fontsize=12)
    ax2.text(2.6, 0.55, "Pendulum\n(m_eq, L_eq)", ha="center", fontsize=12)
    ax2.add_patch(FancyArrowPatch((1.35, 0.55), (2.2, 0.55), arrowstyle="-|>", lw=2, color="#c00000"))
    ax2.text(1.77, 0.78, "等价", ha="center", fontsize=10, color="#c00000")
    ax2.text(1.77, -0.12, "L_eq = M·g / B_max\nm_eq = B_max²/(M·g²)\narm_mass = 0",
             ha="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.4", fc="#fff3c4", ec="k"))
    ax2.text(1.6, -1.4, "两刚体动力学在 (M, B(θ)) 下精确一致 —— 直接写进\n"
                        "每条 log 的 mass/length 字段即可", ha="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_physics.png", dpi=140)

    print("fig1 done")


# ── Fig 2: 装配图(俯视 + 安全区) ─────────────────────────────────────────────
def fig2() -> None:
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set_xlim(-1, 12); ax.set_ylim(-1, 8); ax.set_aspect("equal"); ax.axis("off")
    # 桌子
    ax.add_patch(Rectangle((-0.9, -0.9), 10.4, 8.6, fc="#f2ede4", ec="k", lw=2))
    ax.text(0.2, 7.5, "桌面(俯视)", fontsize=12)
    ax.plot([9.7, 9.7], [-0.9, 7.7], color="k", lw=3)   # 桌沿
    ax.text(9.85, 7.3, "桌沿", rotation=90, va="top", fontsize=11)
    # 鸭身(俯视)
    ax.add_patch(FancyBboxPatch((2.6, 2.6), 4.6, 2.6, boxstyle="round,pad=0.1",
                                fc="#e8e8e8", ec="k", lw=1.4))
    ax.add_patch(Circle((7.0, 3.9), 0.55, fc="#ddd", ec="k", lw=1.2))   # 头
    ax.text(7.0, 3.9, "头", ha="center", fontsize=10)
    ax.text(4.9, 4.9, "鸭身", ha="center", fontsize=13)
    # 压块
    ax.add_patch(Rectangle((3.6, 3.1), 2.2, 1.6, fc="#9ec9e8", ec="k", lw=1.4))
    ax.text(4.7, 3.9, "压块 ≥5kg", ha="center", fontsize=11)
    # 两条腿
    for y0, b in ((2.2, "左腿(=测试)"), (5.4, "右腿")):
        ax.plot([4.6, 5.6], [y0, y0 - 1.6], color="#c00000", lw=3.5, solid_capstyle="round")
        ax.text(5.75, y0 - 1.5, b, fontsize=10, color="#c00000")
    # 摆动弧(左腿)
    th = np.linspace(-75, 75, 100)
    xx = 4.6 + 2.1 * np.sin(np.deg2rad(th)); yy = 0.6 - 2.1 * np.cos(np.deg2rad(th))
    ax.plot(xx, yy, "#999", lw=1.2, ls="--")
    ax.text(3.4, -0.5, "摆动平面±90°(无碰撞)", fontsize=10, color="#555")
    # 砝码
    ax.add_patch(Rectangle((5.45, 0.0), 0.5, 0.5, fc="#8fd18f", ec="k", lw=1.2))
    ax.text(5.7, -0.25, "砝码", fontsize=10)
    # 安全区
    ax.add_patch(Rectangle((1.0, -0.85), 8.4, 1.5, fc="#ffd9d9", ec="#c00000", lw=1.6, alpha=0.6))
    ax.text(5.3, 0.35, "⚠ 摆动区: 手勿入 / 头勿探", fontsize=12, color="#c00000", ha="center")
    # 线缆
    ax.plot([0.0, 0.0], [0.3, 3.0], color="#444", lw=2)
    ax.plot([0.0, 2.6], [3.0, 4.0], color="#444", lw=2)
    ax.text(-0.5, 4.3, "12V 电源 + USB-TTL(线固定)\n红: 电源  黑: GND  白: 信号", fontsize=10)
    # 装配顺序编号
    steps = [(4.7, 6.4, "① 压牢鸭身"), (4.7, 5.9, "② 左腿悬空垂出桌沿"),
             (5.85, 1.35, "③ 足端挂砝码"), (0.2, 1.0, "④ 线缆固定"), (5.3, 0.05, "⑤ 全程人在场")]
    for x, y, s in steps:
        ax.text(x, y, s, fontsize=10.5, color="k",
                bbox=dict(boxstyle="round,pad=0.25", fc="#fff", ec="#888", lw=0.8))
    fig.tight_layout()
    fig.savefig(OUT / "fig2_setup.png", dpi=140)
    print("fig2 done")


# ── Fig 3: 激励轨迹 ─────────────────────────────────────────────────────────
def fig3() -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    t = np.linspace(0, 6, 1200)
    # lift_and_drop
    kf = [[0, 0, 0], [2, -np.pi / 2, 0]]
    from scipy.interpolate import CubicSpline
    # bam 的 cubic_interpolate 是每段 Hermite; 用近似(端点导数=0)即可示意
    def cubic(kf, tt):
        for k in range(len(kf) - 1):
            t0, x0, d0 = kf[k]; t1, x1, d1 = kf[k + 1]
            if t0 <= tt <= t1:
                h = t1 - t0; s = (tt - t0) / h
                return (x0 * (2 * s ** 3 - 3 * s ** 2 + 1) + x1 * (-2 * s ** 3 + 3 * s ** 2)
                        + (d0 * h) * (s ** 3 - 2 * s ** 2 + s) + (d1 * h) * (s ** 3 - s ** 2))
        return kf[-1][1]
    y1 = np.array([cubic(kf, x) for x in t])
    enable = t < 2.0
    axes[0].plot(t, y1, lw=2, color="#c00000")
    axes[0].fill_between(t, -np.pi / 2, np.pi / 2, where=enable, color="#ffe0e0", alpha=0.5)
    axes[0].text(1.0, 0.9, "t<2s: 扭矩ON 抬到 −90°\n随后 扭矩OFF → 自由落体(背驱/Stribeck)",
                 fontsize=10)
    axes[0].set_ylabel("θ [rad]\n(lift_and_drop)", fontsize=11)
    # sin_time_square
    axes[1].plot(t, np.sin(t ** 2), lw=2, color="#1f77b4")
    axes[1].text(3.6, 1.05, "sin(t²): 一次跑完全速度谱(主推荐)", fontsize=10)
    axes[1].set_ylabel("θ [rad]\n(sin_time_square)", fontsize=11)
    # up_and_down
    kf2 = [[0, 0, 0], [3, np.pi / 2, 0], [6, 0.8 * np.pi / 2, 0]]
    y3 = np.array([cubic(kf2, x) for x in t])
    axes[2].plot(t, y3, lw=2, color="#2ca02c")
    axes[2].text(3.9, 1.35, "0→90°→72°: 静态摩擦/负载相关低速段", fontsize=10)
    axes[2].set_ylabel("θ [rad]\n(up_and_down)", fontsize=11)
    axes[2].set_xlabel("t [s](轨迹 6s; pendulum 帧 0=垂直向下)", fontsize=11)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_trajectories.png", dpi=140)
    print("fig3 done")


if __name__ == "__main__":
    fig1(); fig2(); fig3()
    print("ALL ->", OUT.resolve())
