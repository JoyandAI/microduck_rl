#!/usr/bin/env python3
"""Aggregate the Q1 straightness matrix into a verdict table.

Reads every `logs/q1_matrix/*.json` written by `scripts/q1_straightness_matrix.sh`
and reports, per arm:

  * the per-seed mean of each metric (mean +/- sd ACROSS SEEDS — that is the
    run-to-run uncertainty a single measurement hides), and
  * a pooled per-environment Welch t-test between arms at the same command,
    using all N_envs x N_seeds draws as the sample.

Why both: this repo has already been burned twice by drawing a conclusion from
one rollout (agent_B's single-env "+279.9 deg" official yaw that the 64-env
measurement does not reproduce) and by ignoring arm-internal seed spread (the
4-arm night queue, where within-arm sd exceeded the between-arm difference).

Usage:
    uv run python scripts/q1_matrix_report.py
    uv run python scripts/q1_matrix_report.py --dir logs/q1_matrix --metric path_efficiency
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ARM_DESC = {
    "O_xl": "official alpha_walking @ ITS hardware (XL330 model + XL330 act)",
    "O_hd": "official alpha_walking @ OUR hardware (HD-1910 model + act, 7.5 V)",
    "U_hd": "our walk_hd1910_12000  @ OUR hardware",
    "U_xl": "our walk_hd1910_12000  @ OFFICIAL hardware (XL330)",
}

# metric -> (header, format, higher_is_better)
METRICS = [
    ("yaw_rate_bias_deg_s", "yaw bias deg/s", "{:+8.3f}", False),
    ("yaw_dc_body_deg_s", "yaw DC deg/s", "{:+8.3f}", False),
    ("yaw_ac_lock_deg_s", "yaw AC deg/s", "{:+8.3f}", None),
    ("turning_radius_m", "turn R m", "{:9.2f}", True),
    ("path_efficiency", "pathEff", "{:8.3f}", True),
    ("path_efficiency_5s", "pathEff5s", "{:8.3f}", True),
    ("total_path_m", "path m", "{:8.2f}", True),
    ("speed_along_mps", "made-good m/s", "{:8.3f}", True),
    ("net_yaw_deg", "net yaw deg", "{:+9.1f}", False),
    ("foot_pair_torsion_deg", "torsion deg", "{:+8.2f}", None),
]


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Welch t statistic and a normal-approx two-sided p (no scipy dependency)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan"), float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / na + vb / nb)
    if se <= 0:
        return float("nan"), float("nan")
    t = (a.mean() - b.mean()) / se
    # two-sided p from the normal approximation (n is in the hundreds here)
    p = math.erfc(abs(t) / math.sqrt(2.0))
    return float(t), float(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="logs/q1_matrix")
    ap.add_argument("--metric", default=None,
                    help="plot one metric instead of the full table")
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.dir) / "*.json")))
    if not files:
        raise SystemExit(f"no *.json in {args.dir} — run q1_straightness_matrix.sh first")

    # runs[(arm, vx, nodr)] = list of (seed, summary_dict)
    runs: dict[tuple, list] = defaultdict(list)
    for f in files:
        d = json.loads(Path(f).read_text())
        if "aggregate" not in d:
            continue
        name = Path(f).stem                      # e.g. U_hd_vx0.2_s3
        arm, _, rest = name.partition("_vx")
        arm = name.split("_vx")[0]
        vx = float(rest.split("_s")[0])
        seed = int(rest.split("_s")[1].split("_")[0])
        nodr = rest.endswith("_nodr")
        runs[(arm, vx, nodr)].append((seed, d))

    keys = sorted(runs.keys())
    print(f"=== Q1 straightness matrix — {len(files)} run files, "
          f"{sum(len(v) for v in runs.values())} parsed ===")
    for (arm, vx, nodr) in keys:
        print(f"  {arm:5s} vx={vx:<4g} DR={'OFF' if nodr else 'on '}  "
              f"{len(runs[(arm, vx, nodr)])} seeds  ({ARM_DESC.get(arm, arm)})")

    for (arm, vx, nodr) in keys:
        rs = runs[(arm, vx, nodr)]
        nenv = rs[0][1].get("num_envs", 0)
        print(f"\n--- {arm}  vx={vx:g}  DR={'OFF' if nodr else 'ON'}  "
              f"({len(rs)} seeds x {nenv} envs) ---")
        hdr = f"{'metric':<16}{'seed-mean':>11}{'seed-sd':>10}{'pooled mean':>13}"
        print(hdr)
        print("-" * len(hdr))
        for met, label, fmt, _ in METRICS:
            if args.metric and met != args.metric:
                continue
            per_seed = []
            pooled = []
            for _, d in rs:
                agg = d.get("aggregate", {}).get(met)
                if agg is None:
                    continue
                per_seed.append(agg["mean"])
                pooled += [p[met] for p in d["per_env"]
                           if not p.get("insufficient_data") and met in p]
            if not per_seed:
                continue
            pm = np.array(per_seed, dtype=float)
            pl = np.array(pooled, dtype=float)
            print(f"{label:<16}{pm.mean():>11.3f}{pm.std(ddof=1) if len(pm) > 1 else 0:>10.3f}"
                  f"{pl.mean():>13.4f}")
        # ---- survival + POPULATION turning radius -------------------------
        # `turning_radius_m` is a per-env RATIO, so averaging it across envs is
        # inflated by any env whose |wz| happens to be ~0 (one such env sends the
        # mean to infinity-ish). The number to QUOTE is the population ratio,
        # built from the aggregate speed and the aggregate yaw rate. Both are
        # printed so the difference is visible rather than hidden.
        def _win(d):
            if "median_window_s" in d:
                return d["median_window_s"], d.get("survived_window", False)
            # JSONs written before that field existed: fall back to the recorded
            # step counts. Without this fallback an old run reports "did not
            # survive" even when it walked the full window.
            w = float(np.median([p.get("steps", 0) * 0.02 for p in d["per_env"]]))
            return w, w >= 10.0

        ws = [_win(d) for _, d in rs]
        spd = float(np.mean([d["aggregate"]["total_path_m"]["mean"]
                             for _, d in rs])) / 20.0
        yaw = float(np.mean([d["aggregate"]["yaw_dc_body_deg_s"]["mean"]
                             for _, d in rs]))
        rpop = spd / abs(np.radians(yaw)) if abs(yaw) > 1e-9 else float("inf")
        nsurv = sum(1 for w in ws if w[1])
        print(f"  survival          : median window {np.median([w[0] for w in ws]):.2f} s, "
              f"{nsurv}/{len(ws)} seeds survived")
        print(f"  turn radius (pop) : {rpop:.2f} m   "
              f"(quote THIS; the per-env mean above is a ratio average)")
        if nsurv < len(ws):
            print("  [!] some seeds DID NOT SURVIVE the window — their ratio metrics "
                  "are short-arc artifacts, not straightness")

        nr = [d.get("n_envs_with_resets", 0) for _, d in rs]
        if any(nr):
            print(f"  [!] resets seen mid-rollout: {nr} envs across seeds")

    # ---- between-arm tests at the walking command, DR on -------------------
    print("\n=== between-arm Welch t-tests (pooled per-env, vx=0.2, DR on) ===")
    pool: dict[str, dict[str, np.ndarray]] = {}
    for (arm, vx, nodr) in keys:
        if abs(vx - 0.2) > 1e-9 or nodr:
            continue
        vals: dict[str, list] = defaultdict(list)
        for _, d in runs[(arm, vx, nodr)]:
            for p in d["per_env"]:
                if p.get("insufficient_data"):
                    continue
                for met, *_ in METRICS:
                    if met in p:
                        vals[met].append(p[met])
        pool[arm] = {k: np.array(v, dtype=float) for k, v in vals.items()}

    arms = [a for a in ("O_xl", "O_hd", "U_hd", "U_xl") if a in pool]
    for met, label, _, _ in METRICS:
        line = f"{label:<16}"
        for a in arms:
            if met in pool[a]:
                line += f"{a}={pool[a][met].mean():+9.3f}  "
        print(line)
    print()
    hdr = f"{'metric':<16}" + "".join(f"{a+' vs '+arms[0]:>18}" for a in arms[1:])
    print(hdr)
    print("-" * len(hdr))
    for met, label, _, _ in METRICS:
        if met not in pool.get(arms[0], {}):
            continue
        line = f"{label:<16}"
        for a in arms[1:]:
            if met not in pool[a]:
                line += f"{'n/a':>18}"
                continue
            t, p = welch(pool[a][met], pool[arms[0]][met])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            line += f"{t:>+10.2f} p={p:<5.1e}{sig:<3}"[:18].ljust(18)
        print(line)
    print("\n  ns = not significant at p<0.05. 'U_hd vs O_xl' is the headline")
    print("  comparison: our policy on our hardware vs the official policy on its own.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
