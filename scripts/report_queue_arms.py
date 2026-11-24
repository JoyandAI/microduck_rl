#!/usr/bin/env python
"""Cross-arm report for the overnight/extension queues (2 sources, 1 row per arm).

Source A — training-time final metrics (2048 envs, the TRAINING command mix).
Source B — 64-env frozen vx = 0.2 straightness eval (each JSON names its own ONNX,
           so an arm can never be mis-attributed to the wrong run).

Metric discipline (AGENTS.md):
  * `net_yaw_deg` is a 20 s random walk — quote the MEDIAN |.| and the signed
    mean separately (the signed mean cancels when the walk is symmetric).
  * never quote a ratio below half the window: `frac_envs_full_window` gates it.
  * `Episode_Reward/<term>` is the WEIGHTED value; compare only arms whose
    curriculum end-weights match (printed in the header).
"""
from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

TRAIN_KEYS = [
    "Episode_Metrics/yaw_dc_abs",
    "Episode_Metrics/yaw_dc_signed",
    "Episode_Reward/flight",
    "Episode_Reward/yaw_dc",
    "Episode_Reward/head_motion",
    "Episode_Reward/head_droop",
    "Episode_Reward/head_forward",
    "Episode_Reward/head_pose_tracking",
    "Episode_Termination/fell_over",
    "Metrics/twist/error_vel_xy",
    "Metrics/twist/error_vel_yaw",
    "Metrics/air_time_mean",
    "Metrics/peak_height_mean",
]

# arm -> (train log, vx0.2 json)
ARMS: dict[str, tuple[str | None, str | None]] = {
    # ── official recipe, our hardware (the alignment sweep) ──────────────────
    "OFFICIAL align_official": ("logs/queue_align/align_official_train.log", "logs/queue_align/align_official_walk.json"),
    "align_full": ("logs/queue_align/align_full_train.log", "logs/queue_align/align_full_walk.json"),
    "align_fwd": ("logs/queue_align/align_fwd_train.log", "logs/queue_align/align_fwd_walk.json"),
    # ── our recipe + head block, no flight penalty (the 3-seed baseline) ─────
    "HEAD  ours_head s42": ("logs/queue_ours/ours_head_train.log", "logs/queue_ours/ours_head_walk.json"),
    "HEAD  ours_head_s43": ("logs/queue_overnight/ours_head_s43_train.log", "logs/queue_overnight/ours_head_s43_vx0.2.json"),
    # ── our recipe + head + flight penalty, r1 (routeA) vs r2 (overnight) ────
    "FLIGHT s42 r1": ("logs/queue_routeA/ours_flight_s42_train.log", "logs/queue_routeA/ours_flight_s42_vx0.2.json"),
    "FLIGHT s42 r2": ("logs/queue_overnight/ours_flight_s42_train.log", "logs/queue_overnight/ours_flight_s42_vx0.2.json"),
    "FLIGHT s43 r1": ("logs/queue_routeA/ours_flight_s43_train.log", "logs/queue_routeA/ours_flight_s43_vx0.2.json"),
    "FLIGHT s43 r2": ("logs/queue_overnight/ours_flight_s43_train.log", "logs/queue_overnight/ours_flight_s43_vx0.2.json"),
    "FLIGHT s44 r1": ("logs/queue_routeA/ours_flight_s44_train.log", "logs/queue_routeA/ours_flight_s44_vx0.2.json"),
    "FLIGHT s44 r2": ("logs/queue_overnight/ours_flight_s44_train.log", "logs/queue_overnight/ours_flight_s44_vx0.2.json"),
    # ── turn-first curriculum, our recipe WITHOUT head terms ────────────────
    "TURN ours s42": ("logs/queue_overnight/turn_ours_s42_train.log", "logs/queue_overnight/turn_ours_s42_vx0.2.json"),
    "TURN ours s43": ("logs/queue_overnight/turn_ours_s43_train.log", "logs/queue_overnight/turn_ours_s43_vx0.2.json"),
    "TURN official s42": ("logs/queue_overnight/turn_official_s42_train.log", "logs/queue_overnight/turn_official_s42_vx0.2.json"),
    "TURN official s43": ("logs/queue_overnight/turn_official_s43_train.log", "logs/queue_overnight/turn_official_s43_vx0.2.json"),
}


def final_train_metrics(path: Path) -> dict[str, float]:
    """Last value seen for each key over the whole log (the final metric block)."""
    out: dict[str, float] = {}
    key_re = re.compile(r"^\s*([A-Za-z_][\w/]*):\s*(-?[\d.]+)\s*$")
    for line in path.read_text(errors="replace").splitlines():
        m = key_re.match(line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def straight_metrics(path: Path) -> dict[str, float]:
    d = json.loads(path.read_text())
    per = d.get("per_env") or []
    if not per:
        return {}
    ny = [abs(e["net_yaw_deg"]) for e in per if e.get("net_yaw_deg") is not None]
    sny = [e["net_yaw_deg"] for e in per if e.get("net_yaw_deg") is not None]
    out = {
        "envs": len(per),
        "window_s": d.get("median_window_s"),
        "full_frac": d.get("frac_envs_full_window"),
        "falls": d.get("falls"),
        "resets": d.get("n_envs_with_resets"),
        "med_abs_net_yaw": st.median(ny) if ny else float("nan"),
        "mean_signed_net_yaw": st.mean(sny) if sny else float("nan"),
        "med_abs_yaw_dc": st.median([abs(e["yaw_dc_body_deg_s"]) for e in per
                                     if e.get("yaw_dc_body_deg_s") is not None]),
        "med_path_eff": st.median([e["path_efficiency"] for e in per
                                   if e.get("path_efficiency") is not None]),
        "med_speed_along": st.median([e["speed_along_mps"] for e in per
                                      if e.get("speed_along_mps") is not None]),
        "med_achieved_vx": st.median([e["achieved_vx_mean"] for e in per
                                      if e.get("achieved_vx_mean") is not None]),
        "med_tilt": st.median([e["tilt_mean_deg"] for e in per
                               if e.get("tilt_mean_deg") is not None]),
        "med_hipyawL_limit_frac": st.median([e.get("hip_yaw_L_at_limit_frac", 0.0) for e in per]),
    }
    return out


def main() -> int:
    print("=" * 132)
    print("A. 训练期末指标（2048 env，训练命令分布）—— Episode_Reward/* 是乘过权重的值")
    print("=" * 132)
    hdr = f"{'arm':22s} {'yaw_dc_abs':>10s} {'dc_signed':>9s} {'flight':>8s} {'yaw_dc':>8s} {'head_mot':>9s} {'head_drop':>9s} {'head_fwd':>8s} {'fell_over':>9s} {'err_vxy':>7s} {'err_wz':>7s} {'air_t':>6s} {'pk_h':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for label, (tlog, _) in ARMS.items():
        p = REPO / tlog if tlog else None
        if not p or not p.exists():
            print(f"{label:22s}  (缺训练日志)")
            continue
        m = final_train_metrics(p)
        g = lambda k: m.get(k, float("nan"))  # noqa: E731
        print(
            f"{label:22s} {g('Episode_Metrics/yaw_dc_abs'):>10.4f} {g('Episode_Metrics/yaw_dc_signed'):>9.4f} "
            f"{g('Episode_Reward/flight'):>8.4f} {g('Episode_Reward/yaw_dc'):>8.4f} "
            f"{g('Episode_Reward/head_motion'):>9.4f} {g('Episode_Reward/head_droop'):>9.4f} "
            f"{g('Episode_Reward/head_forward'):>8.4f} {g('Episode_Termination/fell_over'):>9.3f} "
            f"{g('Metrics/twist/error_vel_xy'):>7.3f} {g('Metrics/twist/error_vel_yaw'):>7.3f} "
            f"{g('Metrics/air_time_mean'):>6.3f} {g('Metrics/peak_height_mean'):>6.3f}"
        )

    print()
    print("=" * 132)
    print("B. 64 env 冻结 vx=0.2，20 s —— 直行/漂移（每个 JSON 自带 ONNX 出处）")
    print("=" * 132)
    hdr = f"{'arm':22s} {'med|netYaw|':>11s} {'mean_signed':>11s} {'med|yaw_dc|':>11s} {'med_pathEff':>11s} {'med_speed':>9s} {'med_vx':>7s} {'tilt':>6s} {'hipsYawLim':>10s} {'falls':>5s} {'resets':>6s} {'full%':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for label, (_, jpath) in ARMS.items():
        p = REPO / jpath if jpath else None
        if not p or not p.exists():
            print(f"{label:22s}  (缺评估 JSON)")
            continue
        s = straight_metrics(p)
        print(
            f"{label:22s} {s['med_abs_net_yaw']:>11.1f} {s['mean_signed_net_yaw']:>11.1f} "
            f"{s['med_abs_yaw_dc']:>11.2f} {s['med_path_eff']:>11.3f} {s['med_speed_along']:>9.4f} "
            f"{s['med_achieved_vx']:>7.3f} {s['med_tilt']:>6.1f} {s['med_hipyawL_limit_frac']:>10.3f} "
            f"{str(s['falls']):>5s} {str(s['resets']):>6s} {100 * (s['full_frac'] or 0):>5.0f}%"
        )
    print()
    print("=" * 132)
    print("C. 转向通道（64 env，20 s，冻结命令）—— gain = 实测 yaw rate / 命令 yaw rate")
    print("   基准：官方 released ONNX 57.6 %；本轮之前 ours HEAD 11.3 %；35 % 是验收线")
    print("=" * 132)
    TURN = {
        "TURN ours s42": "turn_ours_s42",
        "TURN ours s43": "turn_ours_s43",
        "TURN official s42": "turn_official_s42",
        "TURN official s43": "turn_official_s43",
    }
    hdr = f"{'arm':20s} {'wz=+0.6':>9s} {'wz=-0.6':>9s} {'wz=0':>8s} {'gain%':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for label, stem in TURN.items():
        vals = {}
        for tag in ("0.6", "-0.6", "0.0"):
            p = REPO / f"logs/queue_overnight/{stem}_wz{tag}.json"
            if p.exists():
                vals[tag] = json.loads(p.read_text())["aggregate"]["yaw_rate_bias_deg_s"]["mean"]
        if "0.6" in vals and "-0.6" in vals:
            gain = 100.0 * (vals["0.6"] - vals["-0.6"]) / 2.0 / 34.377
            print(f"{label:20s} {vals['0.6']:>9.2f} {vals['-0.6']:>9.2f} "
                  f"{vals.get('0.0', float('nan')):>8.2f} {gain:>6.1f}%")

    print()
    print("=" * 132)
    print("D. 同一 config + 同一 seed 训两遍（r1 = queue_routeA, r2 = queue_overnight）")
    print("   → 这是本管线的可重复性下限，任何小于它的臂间差异都不算数")
    print("=" * 132)
    print(f"{'arm':16s} {'yaw_dc_abs r1':>13s} {'r2':>8s} {'Δ':>7s} | {'med|netYaw| r1':>14s} {'r2':>8s} {'Δ':>7s} | {'pathEff r1':>10s} {'r2':>8s}")
    for stem in ("ours_flight_s42", "ours_flight_s43", "ours_flight_s44"):
        r1t = final_train_metrics(REPO / f"logs/queue_routeA/{stem}_train.log")
        r2t = final_train_metrics(REPO / f"logs/queue_overnight/{stem}_train.log")
        r1s = straight_metrics(REPO / f"logs/queue_routeA/{stem}_vx0.2.json")
        r2s = straight_metrics(REPO / f"logs/queue_overnight/{stem}_vx0.2.json")
        d1 = r2t.get("Episode_Metrics/yaw_dc_abs", float("nan")) - r1t.get("Episode_Metrics/yaw_dc_abs", float("nan"))
        d2 = r2s["med_abs_net_yaw"] - r1s["med_abs_net_yaw"]
        print(f"{stem:16s} {r1t.get('Episode_Metrics/yaw_dc_abs', float('nan')):>13.4f} "
              f"{r2t.get('Episode_Metrics/yaw_dc_abs', float('nan')):>8.4f} {d1:>+7.4f} | "
              f"{r1s['med_abs_net_yaw']:>14.1f} {r2s['med_abs_net_yaw']:>8.1f} {d2:>+7.1f} | "
              f"{r1s['med_path_eff']:>10.3f} {r2s['med_path_eff']:>8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
