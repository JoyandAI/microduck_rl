#!/usr/bin/env python3
"""Held-case battery: how a detector behaves through one long, forced pick-up per scenario.

Each scenario is one closed-loop `pickup_datagen.py --detector ... --force-pickup-at 2` run in
which every env is picked up at t = 2 s and held to the end, with HandCfg overrides that make the
case (grip the head, turn the duck upside down, spin it 180°, ...). Reported per scenario:

  paused%    share of held time, from 1 s after the lift, that the robot spends paused
  latency    lift → first pause (p50 / p90)
  resumes    false resumes per env while held (each one = the policy running in the hand)

  uv run scripts/pickup_battery.py --detector logs/pickup/nocur/model.pt --tag v1
"""

import argparse
import os
import subprocess
import sys

import numpy as np

DT = 0.02
COMMON = "drop_prob=0.0;shake_prob=0.0"
SCENARIOS = {
    # the case that already works on the robot — the control
    "trunk, casual tilt": "head_grip_prob=0.0;orient_prob=0.0;yaw_turn_prob=0.0;tilt_buckets=((20.0,1.0),)",
    "HEAD grip, casual tilt": "head_grip_prob=1.0;orient_prob=0.0;yaw_turn_prob=0.0;tilt_buckets=((20.0,1.0),)",
    "trunk, upside down (pitch 180)": "head_grip_prob=0.0;orient_prob=1.0;orient_pitch=(2.9,3.14);orient_roll=(-0.1,0.1);yaw_turn_prob=0.0;tilt_buckets=((10.0,1.0),)",
    "trunk, on its side (roll 90)": "head_grip_prob=0.0;orient_prob=1.0;orient_pitch=(-0.1,0.1);orient_roll=(1.45,1.57);yaw_turn_prob=0.0;tilt_buckets=((10.0,1.0),)",
    "trunk, spun 180 about vertical": "head_grip_prob=0.0;orient_prob=0.0;yaw_turn_prob=1.0;yaw_turn=(2.9,3.14);yaw_rate=(0.0,0.0);tilt_buckets=((10.0,1.0),)",
    "HEAD grip, upside down": "head_grip_prob=1.0;orient_prob=1.0;orient_pitch=(2.9,3.14);orient_roll=(-0.1,0.1);yaw_turn_prob=0.0;tilt_buckets=((10.0,1.0),)",
}


def analyze(path, pick_t=2.0):
    d = np.load(path)
    held, paused, sr = d["held"], d["paused"], d["since_reset"]
    T, N = held.shape
    t0 = int(pick_t / DT)
    cover, lat, res, n = [], [], [], 0
    for e in range(N):
        if (sr[t0:, e] < sr[t0, e]).any():  # episode reset mid-scenario
            continue
        h = held[t0:, e]
        if not h[int(1.0 / DT):].any():
            continue
        n += 1
        p = paused[t0:, e]
        on = np.argmax(h)
        first = np.nonzero(p[on:])[0]
        lat.append(first[0] * DT if len(first) else np.inf)
        win = slice(on + int(1.0 / DT), None)
        cover.append(p[win][h[win]].mean())
        r = np.nonzero(p[on:-1] & ~p[on + 1:] & h[on + 1:])[0]
        res.append(len(r))
    lat = np.array(lat)
    if n == 0:
        return 0, np.nan, np.nan, np.nan, np.nan, np.nan
    return n, np.mean(cover), np.percentile(lat, 50), np.percentile(lat, 90), np.mean(res), np.mean(np.isinf(lat))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--only", default=None, help="substring of a scenario name")
    args = ap.parse_args()
    rows = []
    for name, hand in SCENARIOS.items():
        if args.only and args.only not in name:
            continue
        slug = "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")
        out = f"data/pickup/battery_{args.tag}_{slug}.npz"
        if not os.path.exists(out):
            subprocess.run([sys.executable, "scripts/pickup_datagen.py", "--detector", args.detector, "--out", out,
                            "--num-envs", str(args.num_envs), "--seconds", str(args.seconds), "--seed", "200",
                            "--force-pickup-at", "2.0", "--hand", f"{COMMON};{hand}"],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        rows.append((name, *analyze(out)))
    print(f"\n[{args.tag}] {'scenario':34s} {'envs':>5s} {'paused%':>8s} {'lat p50':>8s} {'lat p90':>8s} {'resumes/env':>12s} {'never':>6s}")
    for name, n, cov, l50, l90, res, never in rows:
        print(f"[{args.tag}] {name:34s} {n:5d} {100 * cov:7.1f}% {l50:7.2f}s {l90:7.2f}s {res:12.2f} {100 * never:5.1f}%")


if __name__ == "__main__":
    main()
