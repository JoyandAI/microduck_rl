#!/usr/bin/env python
"""One table for the 2026-09-20/21 runs: what each arm actually achieved.

Sources, all self-identifying so an arm can never be attributed to the wrong run:
  A. the arm's own training log (final metric block)
  B. the P64 frozen-command straightness JSON (vx=0.2) — the JSON names its ONNX
  C. the zero-command standstill JSON (gait_forensics at vx=0)

Metric discipline (AGENTS.md): `net_yaw_deg` is a 20 s random walk, so report the
median |.| AND the fraction of envs that actually walked, and never quote a ratio
whose window collapsed (`frac_envs_full_window` is printed).
"""
from __future__ import annotations

import glob
import json
import os
import re
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

ARMS = {
    # ── references ───────────────────────────────────────────────────────────
    "align_official (official verbatim)": ("logs/queue_align", "align_official", "logs/rsl_rl/velocity/align_official"),
    "align_full (official+head)":         ("logs/queue_align", "align_full", "logs/rsl_rl/velocity/align_full"),
    "align_fwd  (official+head)":         ("logs/queue_align", "align_fwd", "logs/rsl_rl/velocity/align_fwd"),
    "ck7000 (old recipe)":                (None, None, "logs/rsl_rl/velocity/2026-09-16_18-29-18_velocity"),
    # ── deviant-recipe family (the previous queues) ──────────────────────────
    "ours_head s42":  ("logs/queue_overnight", "ours_head", "logs/rsl_rl/velocity/ours_head"),
    "ours_head_s43":  ("logs/queue_overnight", "ours_head_s43", "logs/rsl_rl/velocity/ours_head_s43"),
    "ours_head_s44":  ("logs/queue_overnight", "ours_head_s44", "logs/rsl_rl/velocity/ours_head_s44"),
    "ours_flight s42 (r1)": ("logs/queue_routeA", "ours_flight_s42", "logs/rsl_rl/velocity/ours_flight_s42"),
    "ours_flight s43 (r1)": ("logs/queue_routeA", "ours_flight_s43", "logs/rsl_rl/velocity/ours_flight_s43"),
    "ours_flight s44 (r1)": ("logs/queue_routeA", "ours_flight_s44", "logs/rsl_rl/velocity/ours_flight_s44"),
    "ours_stance s42":  ("logs/queue_overnight", "ours_stance_s42", "logs/rsl_rl/velocity/ours_stance_s42"),
    "ours_ztrack s42":  ("logs/queue_overnight", "ours_ztrack_s42", "logs/rsl_rl/velocity/ours_ztrack_s42"),
    "ours_flight_stance s42": ("logs/queue_overnight", "ours_flight_stance_s42", "logs/rsl_rl/velocity/ours_flight_stance_s42"),
    "ours_yawdc s42":   ("logs/queue_overnight", "ours_yawdc_s42", "logs/rsl_rl/velocity/ours_yawdc_s42"),
    # ── family B (official + head, 12000 it) ────────────────────────────────
    "famB_head_flight s42": ("logs/queue_family_b", "famB_head_flight_s42", "logs/rsl_rl/velocity/famB_head_flight_s42"),
    "famB_head_flight s43": ("logs/queue_family_b", "famB_head_flight_s43", "logs/rsl_rl/velocity/famB_head_flight_s43"),
    "famB_head_flight s44": ("logs/queue_family_b", "famB_head_flight_s44", "logs/rsl_rl/velocity/famB_head_flight_s44"),
}


def train_metrics(run_dir: str) -> dict[str, float]:
    """Final metric block of the newest *_train.log next to the run dir's queue."""
    for out in ("logs/queue_family_b", "logs/queue_overnight", "logs/queue_routeA",
                "logs/queue_align", "logs/queue_ours"):
        p = REPO / out / f"{Path(run_dir).name}_train.log"
        if p.exists():
            m = {}
            for line in p.read_text(errors="replace").splitlines():
                mm = re.match(r"^\s*([A-Za-z_][\w/]*):\s*(-?[\d.]+)\s*$", line)
                if mm:
                    m[mm.group(1)] = float(mm.group(2))
            return m
    return {}


def straight(out_dir: str | None, stem: str | None) -> dict:
    if not out_dir:
        return {}
    for name in (f"{stem}_vx0.2.json", f"{stem}_walk.json"):
        p = REPO / out_dir / name
        if p.exists():
            d = json.loads(p.read_text())
            per = [e for e in d.get("per_env", []) if "net_yaw_deg" in e]
            if not per:
                return {"empty": True, "window": d.get("median_window_s")}
            w = [e for e in per if e["speed_along_mps"] > 0.02]
            return {
                "n": len(per), "nwalk": len(w),
                "net": st.median([abs(e["net_yaw_deg"]) for e in per]),
                "net_walk": st.median([abs(e["net_yaw_deg"]) for e in w]) if w else float("nan"),
                "sp": st.median([e["speed_along_mps"] for e in per]),
                "pe": st.median([e["path_efficiency"] for e in per]),
                "falls": d.get("falls"), "window": d.get("median_window_s"),
                "full": d.get("frac_envs_full_window"),
                "dc": st.median([abs(e["yaw_dc_body_deg_s"]) for e in per]),
                "hiplim": st.median([e.get("hip_yaw_L_at_limit_frac", 0.0) for e in per]),
            }
    return {}


def stand(out_dir: str | None, stem: str | None) -> dict:
    if not out_dir:
        return {}
    p = REPO / out_dir / f"{stem}_stand.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    t, g = d.get("trunk", {}), d.get("gait", {})
    clr = g.get("foot_clearance_body_z_m")
    return {"resets": d.get("resets"), "tilt": t.get("tilt_max_deg"),
            "clr": clr * 1000 if isinstance(clr, (int, float)) else float("nan")}


def main() -> int:
    print("A. 各臂最终成绩（训练 12000/9000 it；直行 = 64 env / 20 s / vx=0.2 / DR 开）")
    hdr = (f"{'臂':30s} {'iters':>5s} {'走动':>5s} {'med|Δψ|':>8s} {'走动臂':>7s} {'速度':>7s} "
           f"{'pathEff':>7s} {'摔':>3s} {'窗口':>5s} | {'站立reset':>9s} {'站倾角':>7s} {'站离地':>7s}")
    print(hdr); print("-" * len(hdr))
    for label, (out, stem, run) in ARMS.items():
        tm = train_metrics(run)
        s = straight(out, stem)
        sd = stand(out, stem)
        it = int(tm.get("Learning iteration", 0)) if "Learning iteration" in tm else 0
        f = lambda v, fmt="{:.1f}": (fmt.format(v) if isinstance(v, (int, float)) and v == v else "-")  # noqa: E731
        print(f"{label:30s} {it:>5d} {str(s.get('nwalk','-'))+'/64':>5s} "
              f"{f(s.get('net')):>8s} {f(s.get('net_walk')):>7s} {f(s.get('sp'),'{:.4f}'):>7s} "
              f"{f(s.get('pe'),'{:.3f}'):>7s} {str(s.get('falls','-')):>3s} {f(s.get('window')):>5s} | "
              f"{str(sd.get('resets','-')):>9s} {f(sd.get('tilt')):>7s} {f(sd.get('clr'),'{:.2f}'):>7s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
