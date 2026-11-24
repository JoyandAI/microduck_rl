#!/usr/bin/env python3
"""Morning report for the overnight A/B queue (scripts/night_queue.sh).

Reads logs/night_queue/results.csv, aggregates per arm, and — most importantly —
prints the count of runs per arm so a half-finished queue is obvious.

The scientific point: measured on 2026-09-14, the run-to-run spread of
`path_efficiency` at a FIXED config and FIXED seed is as large as every effect we
are chasing (same cfg+seed, 3500 vs 12000 iterations gave 0.569 vs 0.124; within
one run, checkpoints ranged 0.043..0.489). So an arm mean over 3 runs still has a
standard error of order 0.15 — read the whole distribution, not just the mean.

Usage:
    uv run python scripts/night_queue_report.py
    uv run python scripts/night_queue_report.py --sweep night_A_s42   # also sweep
                                                                     # that run's
                                                                     # checkpoints
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

ARMS = {
    "A": "our recipe (22 terms, kp_fw=32)",
    "B": "A + stiffer servo in sim (kp_fw=64)",
    "C": "official recipe + heading_hold (19 terms)",
    "D": "official recipe exactly (16 terms)",
}
RESULTS = Path("logs/night_queue/results.csv")
OUT = Path("logs/night_queue")


def load() -> list[dict]:
    if not RESULTS.exists():
        sys.exit(f"no results yet: {RESULTS}")
    rows = []
    with RESULTS.open() as f:
        for r in csv.DictReader(f):
            if not r.get("job"):
                continue
            try:
                r["path_efficiency"] = float(r["path_efficiency"])
                r["yaw_bias_deg_s"] = float(r["yaw_bias_deg_s"])
                r["made_good_m"] = float(r["made_good_m"])
            except (TypeError, ValueError):
                r["path_efficiency"] = None
            rows.append(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=None,
                    help="also sweep every checkpoint of this job id (e.g. night_A_s42)")
    ap.add_argument("--eval-envs", type=int, default=64)
    args = ap.parse_args()

    rows = load()
    print(f"{len(rows)} finished runs\n")

    print(f"{'arm':<4} {'n':>2} {'eff mean':>9} {'eff min':>8} {'eff max':>8} "
          f"{'yaw-bias mean':>14} {'best run':>14}")
    print("-" * 72)
    for arm, desc in ARMS.items():
        rr = [r for r in rows if r["arm"] == arm and r["path_efficiency"] is not None]
        if not rr:
            print(f"{arm:<4} {0:>2}   (no finished runs yet)")
            continue
        eff = [r["path_efficiency"] for r in rr]
        yaw = [r["yaw_bias_deg_s"] for r in rr]
        best = max(rr, key=lambda r: r["path_efficiency"])
        print(f"{arm:<4} {len(rr):>2} {st.mean(eff):>9.3f} {min(eff):>8.3f} "
              f"{max(eff):>8.3f} {st.mean(yaw):>+14.2f} {best['job']:>14}")
    print()
    for arm, desc in ARMS.items():
        print(f"  {arm}: {desc}")

    # best overall
    good = [r for r in rows if r["path_efficiency"] is not None]
    if good:
        b = max(good, key=lambda r: r["path_efficiency"])
        print(f"\nBEST OVERALL: {b['job']}  eff={b['path_efficiency']:.3f}  "
              f"yaw={b['yaw_bias_deg_s']:+.2f} deg/s")
        print(f"  onnx: {b['onnx']}")
        print("\nReference points (64 envs, seed 0, vx=0.2, DR on):")
        print("  official alpha_walking (XL330) : eff 0.784, yaw -0.34 deg/s")
        print("  walk_3500 (no heading_hold)    : eff 0.569, yaw +0.41 deg/s")
        print("  acceptance bar                 : eff >= 0.65 and |yaw| <= 1.0 deg/s")

    if args.sweep:
        sweep(args.sweep, args.eval_envs)
    return 0


def sweep(job: str, eval_envs: int) -> None:
    """Export + evaluate every checkpoint of a run (checkpoint selection is a
    real lever: the official ships `BEST_alpha_walking_rough.onnx`)."""
    arm, seed = job.rsplit("_s", 1)
    exp = f"night_{arm}_s{seed}"
    dirs = sorted(glob.glob(f"logs/rsl_rl/{exp}/*/"))
    if not dirs:
        sys.exit(f"no run dir for {job} (looked in logs/rsl_rl/{exp}/)")
    rundir = Path(dirs[-1])
    cks = sorted(rundir.glob("model_*.pt"),
                 key=lambda p: int(p.stem.split("_")[1]))
    print(f"\n=== checkpoint sweep: {job}  ({len(cks)} checkpoints in {rundir}) ===")
    tmp = Path("/tmp/cksweep_night")
    tmp.mkdir(exist_ok=True)
    for ck in cks:
        it = ck.stem.split("_")[1]
        onnx = tmp / f"{job}_{it}.onnx"
        if not onnx.exists():
            subprocess.run(
                ["uv", "run", "python", "scripts/export.py",
                 "Mjlab-Velocity-Flat-MicroDuck", "--checkpoint-file", str(ck),
                 "--onnx-file", str(onnx)],
                check=False, capture_output=True,
                env={"WANDB_MODE": "offline", "PATH": "/usr/bin:/bin",
                     "HOME": str(Path.home())})
        if not onnx.exists():
            print(f"  ck{it:>5}: export failed")
            continue
        jf = tmp / f"{job}_{it}.json"
        subprocess.run(
            ["uv", "run", "python", "scripts/eval_straightness.py",
             "--onnx", str(onnx), "--lin-vel-x", "0.2", "--seconds", "20",
             "--num-envs", str(eval_envs), "--seed", "0", "--json", str(jf)],
            check=False, capture_output=True)
        try:
            a = json.loads(jf.read_text())["aggregate"]
            print(f"  ck{it:>5}: eff {a['path_efficiency']['mean']:.3f}   "
                  f"yaw-bias {a['yaw_rate_bias_deg_s']['mean']:+.2f} deg/s")
        except Exception as e:  # noqa: BLE001
            print(f"  ck{it:>5}: eval failed ({e})")


if __name__ == "__main__":
    raise SystemExit(main())
