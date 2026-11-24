#!/usr/bin/env python
"""Head-block evidence across arms — from two sources, never mixed.

A. training-time (2048 envs, training command mix): `Episode_Reward/<term>` is the
   WEIGHTED value, so divide by the curriculum end-weight to get the raw metric and
   make arms comparable. Prints the weights too, so an uneven schedule is visible.
B. `gait_forensics.py` probe (1 env, 12 s, frozen cmd): absolute values from a
   single-env probe are NOT quotable (AGENTS.md: same checkpoint varies 20-44x
   between draws) — used here only for head-SHAPE comparison, and the DR draw is
   not pinned, so treat as ordering-only.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

HEAD_TERMS = {
    "head_motion": "Curriculum/head_stability_weight",
    "head_droop": "Curriculum/head_droop_weight",
    "head_forward": "Curriculum/head_forward_weight",
}

ARMS = {
    "ours_head s42 (base)": "logs/queue_ours/ours_head_train.log",
    "ours_head_s43 (base)": "logs/queue_overnight/ours_head_s43_train.log",
    "ours_flight s42 r2": "logs/queue_overnight/ours_flight_s42_train.log",
    "ours_flight s43 r2": "logs/queue_overnight/ours_flight_s43_train.log",
    "ours_flight s44 r2": "logs/queue_overnight/ours_flight_s44_train.log",
    "turn_ours s42 (NO head)": "logs/queue_overnight/turn_ours_s42_train.log",
}

GAITS = {
    "ours_head s42": "logs/queue_overnight/ours_head_gait.json",
    "ours_head_s43": "logs/queue_overnight/ours_head_s43_gait.json",
    "ours_flight s42": "logs/queue_overnight/ours_flight_s42_gait.json",
    "ours_flight s43": "logs/queue_overnight/ours_flight_s43_gait.json",
    "ours_flight s44": "logs/queue_overnight/ours_flight_s44_gait.json",
    "turn_ours s42 (NO head)": "logs/queue_overnight/turn_ours_s42_gait.json",
    "turn_official s42": "logs/queue_overnight/turn_official_s42_gait.json",
}


def metrics(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    key_re = re.compile(r"^\s*([A-Za-z_][\w/]*):\s*(-?[\d.]+)\s*$")
    for line in path.read_text(errors="replace").splitlines():
        m = key_re.match(line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def main() -> int:
    print("A. 训练期头部代价（raw = 记录值 / 课程末权重；越接近 0 越好）")
    hdr = f"{'arm':26s} {'w_mot':>6s} {'raw_motion':>11s} {'w_drop':>6s} {'raw_droop':>10s} {'raw_fwd':>9s} {'head_track':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for label, rel in ARMS.items():
        p = REPO / rel
        if not p.exists():
            print(f"{label:26s}  (缺日志)")
            continue
        m = metrics(p)
        vals = {}
        for term, wkey in HEAD_TERMS.items():
            w = m.get(wkey)
            v = m.get(f"Episode_Reward/{term}")
            vals[term] = (w, (v / w) if (w not in (None, 0.0) and v is not None) else None)
        wm, rm = vals["head_motion"]
        wd, rd = vals["head_droop"]
        _, rf = vals["head_forward"]
        print(f"{label:26s} {wm if wm is not None else float('nan'):>6.3f} "
              f"{rm if rm is not None else float('nan'):>11.4f} "
              f"{wd if wd is not None else float('nan'):>6.2f} "
              f"{rd if rd is not None else float('nan'):>10.4f} "
              f"{rf if rf is not None else float('nan'):>9.5f} "
              f"{m.get('Episode_Reward/head_pose_tracking', float('nan')):>10.4f}")

    print()
    print("B. gait 探针（1 env / 12 s，单一 DR draw —— 只看排序，绝对数不可引用）")
    hdr = ("  " + f"{'arm':26s} {'head_pitch_osc':>14s} {'head_roll_osc':>13s} {'pitch_dc':>9s} "
           f"{'rel_trunk_std':>13s} {'dev_angle':>10s} {'tilt_mean':>10s} {'z_mean':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for label, rel in GAITS.items():
        p = REPO / rel
        if not p.exists():
            print(f"  {label:26s}  (缺)")
            continue
        d = json.loads(p.read_text())
        h, t = d.get("head", {}), d.get("trunk", {})
        print(f"  {label:26s} {h.get('pitch_osc_std_deg', float('nan')):>14.2f} "
              f"{h.get('roll_osc_std_deg', float('nan')):>13.2f} "
              f"{h.get('pitch_dc_deg', float('nan')):>9.2f} "
              f"{h.get('rel_trunk_angle_std_deg', float('nan')):>13.3f} "
              f"{h.get('dev_angle_mean_deg', float('nan')):>10.2f} "
              f"{t.get('tilt_mean_deg', float('nan')):>10.2f} "
              f"{t.get('z_mean', float('nan')):>8.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
