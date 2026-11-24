#!/usr/bin/env python3
"""PASS/FAIL a walking policy against the 2026-09-11 posture acceptance table.

The table lives in `docs/posture_defect_analysis.md` §7. It exists because the
two defects this fixes ("legs splay apart", "the head wobbles badly") are
VISUAL: a policy can improve its reward and still look wrong. Every row below
is a number that was measured on both the broken local policy and the official
`alpha_walking.onnx`, so "pass" means "in the official's regime", not just
"different from before".

Runs `gait_forensics.py` twice (walking at vx=0.2 and standing at vx=0) and
reads the JSON summaries, so it needs no extra instrumentation.

Usage:
    uv run python scripts/check_posture_acceptance.py --onnx walk_fixed.onnx
    uv run python scripts/check_posture_acceptance.py --onnx x.onnx --keep-json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (label, key-path into the summary, comparison, threshold, official reference,
#  measured-on-the-broken-policy)
#   "ge" = value must be >= threshold, "le" = <= threshold
ROWS = [
    # --- safety / capability: the fix must not cost the gait ---
    ("falls in 20 s", ("resets",), "le", 0, "0", "0"),
    ("trunk z (walk) [mm]", ("trunk", "z_mean"), "ge", 0.100, "119.8", "91.1"),
    # --- the leg defect ---
    ("foot sep (walk) [mm] vs nominal 83.6", ("gait", "foot_sep_body_y_m", "mean"), "ge", 0.078, "83.0", "70.3"),
    ("foot sep (stand) [mm]", ("stand", "gait", "foot_sep_body_y_m", "mean"), "ge", 0.062, "84.7", "31.2"),
    ("hip_yaw dev L [deg]", ("splay", "toe_out_deg", "left_abs"), "le", 8.0, "4.6", "16.0"),
    ("hip_yaw dev R [deg]", ("splay", "toe_out_deg", "right_abs"), "le", 8.0, "5.5", "16.5"),
    ("abduction [deg]", ("splay", "abduction_deg", "mean_abs"), "le", 3.0, "0.4", "5.8"),
    ("hip_yaw parked at limit (walk) [%]",
     ("limits", "left_hip_yaw", "frac_within_1deg"), "le", 0.05, "~0", "0.48"),
    ("hip_yaw raw |a| (walk)", ("actions", "left_hip_yaw", "mean_abs"), "le", 1.5, "n/a", "1.33"),
    ("hip_yaw raw max|a| (stand)", ("stand", "actions", "left_hip_yaw", "max_abs"), "le", 1.5, "n/a", "10.27"),
    # --- the head defect ---
    ("head nod vs trunk std (walk) [deg]",
     ("head", "rel_trunk_pitch_std_deg"), "le", 6.0, "1.7", "21.4"),
    ("head nod vs trunk p2p (walk) [deg]",
     ("head", "rel_trunk_pitch_p2p_deg"), "le", 20.0, "6.9", "68.9"),
    ("neck gait-frequency amplitude [deg]", ("gait", "neck_amp_deg"), "le", 3.0, "0.75", "16.4"),
    ("head world pitch std (walk) [deg]", ("head", "pitch_std_deg"), "le", 6.0, "n/a", "11.5"),
    # --- actuator health ---
    ("hip_joint torque/duty headroom proxy: mean|a| over 14 joints",
     ("mean_abs_action",), "le", 0.60, "0.160", "0.395"),
]

# Standing-specific extra rows (the crouch/twist shows up worst at cmd 0).
STAND_ROWS = [
    ("stance twist: hip_yaw dev L [deg]", ("splay", "toe_out_deg", "left_abs"), "le", 8.0, "0.78", "30.4"),
    ("stance trunk z [mm]", ("trunk", "z_mean"), "ge", 0.100, "n/a", "91.2"),
]


def dig(obj, path):
    cur = obj
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def run_forensics(onnx: str, out_json: Path, vx: float, seconds: float, tag: str) -> dict:
    cmd = [
        sys.executable, str(HERE / "gait_forensics.py"),
        "--onnx", onnx, "--lin-vel-x", str(vx),
        "--warmup", "2", "--seconds", str(seconds),
        "--tag", tag, "--json-out", str(out_json),
    ]
    print(f"[i] {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"gait_forensics failed for vx={vx}")
    return json.loads(out_json.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--keep-json", action="store_true",
                    help="write the raw summaries next to the onnx")
    args = ap.parse_args()

    onnx = str(Path(args.onnx).resolve())
    tmpdir = Path(tempfile.mkdtemp(prefix="posture_accept_"))
    if args.keep_json:
        tmpdir = Path(onnx).parent / (Path(onnx).stem + "_accept")
        tmpdir.mkdir(exist_ok=True)

    walk = run_forensics(onnx, tmpdir / "walk.json", 0.2, args.seconds, "walk vx=0.2")
    stand = run_forensics(onnx, tmpdir / "stand.json", 0.0, min(args.seconds, 12.0), "stand vx=0")

    walk["mean_abs_action"] = sum(
        v["mean_abs"] for k, v in walk["actions"].items() if isinstance(v, dict)
    ) / max(1, sum(1 for v in walk["actions"].values() if isinstance(v, dict)))
    stand["mean_abs_action"] = sum(
        v["mean_abs"] for k, v in stand["actions"].items() if isinstance(v, dict)
    ) / max(1, sum(1 for v in stand["actions"].values() if isinstance(v, dict)))

    rows = list(ROWS) + [(lbl, ("stand",) + p, c, t, o, m) for lbl, p, c, t, o, m in STAND_ROWS]

    n_pass = 0
    failures = []
    print("\n" + "=" * 100)
    print(f"POSTURE ACCEPTANCE  {Path(onnx).name}")
    print("=" * 100)
    print(f"  {'criterion':<52} {'value':>11} {'target':>10} {'official':>10}  {'before':>9}  verdict")
    print("  " + "-" * 96)
    for label, path, cmp_, thr, official, before in rows:
        # "stand" as the first element is the scope marker; otherwise the whole
        # path is relative to the walking summary.
        if path[0] == "stand":
            src, rest = stand, path[1:]
        else:
            src, rest = walk, path
        val = dig(src, rest) if rest else None
        if val is None:
            print(f"  {label:<52} {'n/a':>11} {'':>10} {official:>10}  {before:>9}  SKIP")
            continue
        if cmp_ == "ge":
            ok = val >= thr
            tstr = f">={thr}"
        else:
            ok = val <= thr
            tstr = f"<={thr}"
        disp = f"{val:.4f}" if isinstance(val, float) and abs(val) < 10 else f"{val:.1f}"
        tdisp = f"{thr:.3f}" if abs(thr) < 10 else f"{thr:.1f}"
        print(f"  {label:<52} {disp:>11} {tdisp:>10} {official:>10}  {before:>9}  "
              f"{'PASS' if ok else '**FAIL**'}")
        if ok:
            n_pass += 1
        else:
            failures.append(label)

    total = len(rows)
    print("  " + "-" * 96)
    print(f"  {n_pass}/{total} passed")
    if failures:
        print("\n  FAILED (in the order they should be attacked):")
        for f in failures:
            print(f"    - {f}")
    print(f"\n  summaries: {tmpdir}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
