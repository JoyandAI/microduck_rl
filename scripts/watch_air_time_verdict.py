#!/usr/bin/env python3
"""Watch the posture-fix run and print the iteration-6000 verdict.

Why this exists. On 2026-09-11 the walking policy was found to farm `air_time`
by marching on the spot (command non-zero is enough; nothing checks that the
body moved). `mdp.feet_air_time_progress_gated` closes that hole and is wired
behind `AIR_TIME_PROGRESS_GATE_V_REF`, but turning it on costs a restart, so the
decision was deferred to iteration 6000. A decision deferred to "when I look at
it" is a decision that gets forgotten — this script makes the checkpoint
self-announcing.

It extracts, from the training log, the two terms that separate the two
hypotheses, and prints a verdict:

  * `track_linear_velocity` still climbing  -> the gait is just late; leave it.
  * `track_linear_velocity` stalled while
    `air_time` is high                       -> stepping without translating;
                                                enable the gate and restart.

It also compares against the PREVIOUS run at the same iteration, because "behind"
only means something relative to the run that worked.

Usage:
    uv run python scripts/watch_air_time_verdict.py \
        --log train_velocity_posture_fix.log \
        --ref-log train_velocity_hd1910.log \
        --at 6000
    # block until iteration 6000 exists (polls every 60 s):
    uv run python scripts/watch_air_time_verdict.py --log ... --wait
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

METRICS = {
    "track_linear_velocity": "Episode_Reward/track_linear_velocity",
    "air_time": "Episode_Reward/air_time",
    "pose": "Episode_Reward/pose",
    "err_vel_xy": "Metrics/twist/error_vel_xy",
    "err_vel_yaw": "Metrics/twist/error_vel_yaw",
}

# Term ceilings, for reading "% of max".
MAXIMA = {"track_linear_velocity": 2.0, "air_time": 3.0, "pose": 1.0}


def parse_log(path: Path) -> tuple[dict[int, dict[str, float]], int]:
    """Return {iteration: {metric: value}} plus the highest iteration seen."""
    txt = path.read_text(errors="ignore")
    blocks = re.split(r"Learning iteration (\d+)/\d+", txt)
    out: dict[int, dict[str, float]] = {}
    last = 0
    for i in range(1, len(blocks) - 1, 2):
        it = int(blocks[i])
        last = max(last, it)
        body = blocks[i + 1]
        row = {}
        for name, tag in METRICS.items():
            m = re.search(rf"{re.escape(tag)}:\s*(-?[\d.]+)", body)
            if m:
                row[name] = float(m.group(1))
        if row:
            out[it] = row
    return out, last


def nearest(data: dict[int, dict[str, float]], it: int) -> dict[str, float] | None:
    if not data:
        return None
    key = min(data, key=lambda k: abs(k - it))
    return data[key]


def fmt(row: dict[str, float] | None, name: str, as_pct: bool = False) -> str:
    if not row or name not in row:
        return "   n/a"
    v = row[name]
    if as_pct and name in MAXIMA:
        return f"{v:6.3f}({100 * v / MAXIMA[name]:3.0f}%)"
    return f"{v:6.3f}      "


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--ref-log", default=None, help="the earlier run that walked")
    ap.add_argument("--at", type=int, default=6000)
    ap.add_argument("--interval", type=float, default=15.0)
    ap.add_argument("--wait", action="store_true", help="poll until --at is reached")
    ap.add_argument("--timeout-min", type=float, default=180.0)
    args = ap.parse_args()

    log = Path(args.log)
    deadline = time.time() + args.timeout_min * 60
    while True:
        data, last = parse_log(log) if log.exists() else ({}, 0)
        if not args.wait or last >= args.at or time.time() > deadline:
            break
        print(f"[watch] at iteration {last}/{args.at} ...", flush=True)
        time.sleep(args.interval)

    if last < args.at:
        print(f"[watch] log only reached iteration {last}; reporting anyway")

    now = nearest(data, args.at)
    ref = None
    if args.ref_log and Path(args.ref_log).exists():
        ref_data, _ = parse_log(Path(args.ref_log))
        ref = nearest(ref_data, args.at)

    print("=" * 88)
    print(f"AIR-TIME VERDICT  @ iteration {args.at}   (log reached {last})")
    print("=" * 88)
    print(f"  {'metric':<26}{'this run':>16}{'prev run':>12}   reading")
    print("  " + "-" * 84)
    for name in ("track_linear_velocity", "air_time", "pose", "err_vel_xy", "err_vel_yaw"):
        pct = name in MAXIMA
        note = f"{100 * now[name] / MAXIMA[name]:.0f}% of max" if (pct and now and name in now) else ""
        print(f"  {name:<26}{fmt(now, name, pct):>16}{fmt(ref, name, pct):>12}   {note}")

    # ---- verdict -----------------------------------------------------------
    print("  " + "-" * 84)
    if not now:
        print("  no metrics parsed yet")
        return 2

    # Trend over a MEANINGFUL span: a 12-log window is only ~12 iterations out
    # of 15000 and reads "flat" for any run. Compare against ~1000 iterations
    # back, which is a quarter of the curriculum's ramp.
    its = sorted(data)
    span = 1000
    window = [i for i in its if i >= args.at - span]
    recent = [data[i].get("track_linear_velocity") for i in window]
    recent = [v for v in recent if v is not None]
    span_actual = (window[-1] - window[0]) if len(window) >= 2 else 0
    climbing = len(recent) >= 4 and (recent[-1] - recent[0]) > 0.08

    tl = now.get("track_linear_velocity", 0.0)
    at = now.get("air_time", 0.0)
    frac = tl / MAXIMA["track_linear_velocity"]
    at_frac = at / MAXIMA["air_time"]

    print(f"  track_linear_velocity = {tl:.3f} ({frac*100:.0f}% of max), "
          f"air_time = {at:.3f} ({at_frac*100:.0f}% of max)")
    delta = (recent[-1] - recent[0]) if len(recent) >= 2 else float("nan")
    print(f"  trend over the last {span_actual} iterations: "
          f"{'STILL CLIMBING' if climbing else 'FLAT'} (delta {delta:+.3f})")
    print()

    if frac >= 0.70:
        print("  VERDICT: ON TRACK — velocity tracking is in the healthy range;")
        print("           do NOT touch the reward. Let the run finish.")
        return 0
    # Still climbing with a real slope -> do not restart a run that is working.
    # Restarting costs an hour; a late gait is not a broken one.
    if climbing and delta > 0.10:
        print("  VERDICT: LATE BUT CLIMBING — the gait is still maturing, so the")
        print(f"           marching is most likely an immature gait, not a farm.")
        print("           Do NOT restart. Re-check at 9000 (or when it plateaus).")
        return 0
    if at_frac > 0.28 and frac < 0.55:
        print("  VERDICT: **STEPPING WITHOUT TRANSLATING** (air_time high, tracking low,")
        print("           and no longer improving).")
        print("           Action: set AIR_TIME_PROGRESS_GATE_V_REF = 0.15 in")
        print("           microduck_velocity_env_cfg.py and restart the run.")
        print("           (The gated reward is implemented and unit-tested; see")
        print("            mdp.feet_air_time_progress_gated.)")
        return 1
    print("  VERDICT: AMBIGUOUS — tracking is low but air_time is not high either.")
    print("           Inspect a video/playback before changing any weight.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
