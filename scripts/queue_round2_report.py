#!/usr/bin/env python3
"""Verdict table for queue round 2, plus a full checkpoint sweep.

Two jobs, both about not fooling ourselves:

1. VERDICT. Print each trained arm against the numeric acceptance table agreed
   before the queue ran, with the official policy and the PRE-FIX policy as
   reference rows in the same protocol (64 envs, seed 0, 20 s, frozen commands,
   DR on). A number is only interpretable next to those two.

2. SWEEP. Rank EVERY saved checkpoint of each walk run, not just the last one.
   This is not optional polish: within the previous single run the checkpoints
   ranged 0.043..0.489 path efficiency, and the official deliverable IS a
   selected checkpoint (`BEST_alpha_walking_rough.onnx`). Picking the best
   checkpoint is part of shipping a policy, so the queue saves every 1000 iters.

Ranking is on the METRIC THAT MATTERS, in this order:
    |yaw DC rate| <= 1.0 deg/s   (the acceptance bar; the official sits at 0.23)
    then path_efficiency >= 0.65  (documented definition: along / TOTAL path)
and any checkpoint whose gait speed is under 40 % of the command is rejected —
"A straight line while barely walking" is not a pass (the official policy is
slow in sim too: 0.08 m/s at a 0.2 command).

Usage:
    uv run python scripts/queue_round2_report.py
    uv run python scripts/queue_round2_report.py --sweep
    uv run python scripts/queue_round2_report.py --sweep --runs walk_sym,walk_yawdc
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "logs/queue_round2"
TMP = Path("/tmp/queue_round2_sweep")
EVAL_ENVS = 64
SWEEP_SECONDS = 20

# Reference rows measured in the SAME protocol (see logs/q1_matrix/).
# Reference rows measured in the SAME protocol: 64 envs x 5 seeds, vx=0.2, DR on,
# 20 s, frozen commands (logs/q1_matrix/, aggregated by scripts/q1_matrix_report.py).
#
# ⚠️ These must stay on the SAME protocol and the SAME definition as the arm rows.
# A first version quoted the official's numbers from a SINGLE-SEED PILOT
# (-0.23 deg/s, turning radius 60.5 m) against arm rows from 5 seeds — i.e. it
# compared a lucky draw with a mean, in a report whose whole point is that such
# comparisons are how this repo got misled. Turning radius is the POPULATION
# value (aggregate speed / |aggregate yaw rate|), not the mean of per-env ratios
# (which reads 11.5 m for the official and 2.59 m for ours, inflated by envs
# whose |wz| is ~0).
REFS = {
    "official alpha_walking @ XL330/737 g (its own hw)":
        {"yaw_rate_bias_deg_s": -0.58, "path_efficiency": 0.311,
         "turning_radius_m": 9.14, "total_path_m": 1.86},
    "our walk_hd1910_12000 (PRE-FIX) @ HD-1910/779 g":
        {"yaw_rate_bias_deg_s": -12.18, "path_efficiency": 0.063,
         "turning_radius_m": 1.38, "total_path_m": 5.86},
    "ck7000 (checkpoint pick, 5 seeds) @ HD-1910/779 g":
        {"yaw_rate_bias_deg_s": -0.62, "path_efficiency": 0.353,
         "turning_radius_m": 24.1, "total_path_m": 5.22},
}

ACCEPT_YAW_DEG_S = 1.0      # ~4x the official's 0.23 deg/s: generous, but 12x better than pre-fix
# Turning-radius bar. The official's POPULATION radius is 9.14 m, pre-fix ours is
# 1.38 m, ck7000 is 24.1 m. Set at 5.0 m: comfortably above the pre-fix failure
# and below the reference -- a bar the reference itself cannot clear is not a bar.
# (The first draft said 10 m, taken from the official's MEAN-OF-RATIOS value of
# 11.5 m. That definition is inflated by envs with |wz| ~ 0, and it put the bar
# ABOVE the reference's true value, so the reference row printed `fail`. Same
# mistake as the path-efficiency bar earlier; the integration test caught it.)
ACCEPT_TURN_RADIUS_M = 5.0
ACCEPT_EFF = 0.30            # REFERENCE PARITY, not aspiration -- see note below
MIN_SPEED_FRACTION = 0.30    # so "stand still and score straight" cannot pass

# WHY THE BARS ARE SET HERE AND NOT AT pathEff >= 0.65 AS FIRST DRAFTED.
# The 0.65 figure came from docs/forensics/走歪归因报告.md §0, which measured the
# official policy at **vx = 0.4 with DR OFF over 16 seeds** and got 0.842. Under
# the protocol used here (vx = 0.2, DR ON, 64 envs) the SAME official policy
# scores only 0.311 -- because at this command it walks at 0.081 m/s (40 % of the
# command) and wanders, so path efficiency is dominated by speed rather than by
# crookedness. A bar the reference itself cannot clear is not a bar. So:
#   * the primary acceptance is the YAW metrics (|yawDC|, turnR) -- those are
#     protocol-robust: the official sits at -0.23 deg/s / 60 m at every command
#     and window tried, and pre-fix at -12.08 deg/s / 3.2 m;
#   * path_efficiency is reported at REFERENCE PARITY (0.30) as a floor, with the
#     reference value always printed on the row above so it can never be read as
#     an absolute standard;
#   * a speed floor keeps "walk the straightest line by not walking" out.
# The vx = 0.4 / DR-off protocol is reproduced separately
# (VX="0.4" bash scripts/q1_straightness_matrix.sh) for comparability with §0.


def _agg(js: Path) -> dict | None:
    try:
        d = json.loads(Path(js).read_text())
    except Exception:
        return None
    return d.get("aggregate")


def _val(a: dict, key: str) -> float:
    """Accept both an aggregate dict ({'mean':..}) and a plain float (REFS rows)."""
    v = a[key]
    return float(v["mean"]) if isinstance(v, dict) else float(v)


def _row(label: str, a: dict | None, seconds: float = SWEEP_SECONDS) -> str:
    if not a:
        return f"  {label:<52}{'(no data)':>14}"
    yaw = _val(a, "yaw_rate_bias_deg_s")
    eff = _val(a, "path_efficiency")
    rad = _val(a, "turning_radius_m")
    spd = _val(a, "total_path_m") / max(seconds, 1e-6)
    ok = (abs(yaw) <= ACCEPT_YAW_DEG_S and rad >= ACCEPT_TURN_RADIUS_M
          and eff >= ACCEPT_EFF and spd >= MIN_SPEED_FRACTION * 0.2)
    flag = "PASS" if ok else ("slow" if spd < MIN_SPEED_FRACTION * 0.2 else "fail")
    return (f"  {label:<52}{yaw:>+9.2f}{eff:>10.3f}{rad:>11.2f}{spd:>9.3f}   {flag}")


def _discover_arms() -> dict[str, dict[str, Path]]:
    """arm -> {"walk02": path, "zerocmd": path, "sitstand": path}, from results.tsv.

    Do NOT discover arms by filename glob. The first version globbed
    `walk*_walk02.json`, which silently DROPPED every arm whose name does not
    start with "walk" — including `turn_first` and `turnfirst_yawdc`, i.e. exactly
    the arms that target the user's turn-on-command requirement. A verdict table
    that quietly omits the decisive arm is worse than no table.
    """
    out: dict[str, dict[str, Path]] = {}
    tsv = OUT / "results.tsv"
    if not tsv.exists():
        return out
    for line in tsv.read_text().splitlines()[1:]:
        f = line.split("\t")
        if len(f) < 13:
            continue
        arm = f[0]
        d: dict[str, Path] = {}
        for key, idx in (("walk02", 10), ("zerocmd", 11), ("sitstand", 12)):
            v = f[idx].strip()
            if v and v != "-":
                p = ROOT / v if not v.startswith("/") else Path(v)
                if p.exists():
                    d[key] = p
        if d:
            out[arm] = d
    return out


def verdict() -> int:
    arms = _discover_arms()
    hdr = (f"  {'arm / reference':<52}{'yawDC':>9}{'pathEff':>10}"
           f"{'turnR m':>11}{'speed':>9}")
    print("=== Q1 walking: acceptance (64 envs, seed 0, 20 s, frozen cmds, DR on) ===")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for label, a in REFS.items():
        print(_row(label, a))
    print()
    if not arms:
        print("  (no arms with results yet — results.tsv empty or no evals done)")
    for arm in sorted(arms):
        js = arms[arm].get("walk02")
        print(_row(f"{arm} (POST-FIX, final ckpt)", _agg(js) if js else None))
    print()
    print(f"  acceptance: |yawDC| <= {ACCEPT_YAW_DEG_S} deg/s  AND  turnR >= "
          f"{ACCEPT_TURN_RADIUS_M} m  AND  pathEff >= {ACCEPT_EFF} (reference parity)"
          f"  AND  speed >= {MIN_SPEED_FRACTION}x command")
    print("  NOTE pathEff is along / TOTAL ARC LENGTH (the definition used in")
    print("       docs/forensics/走歪归因报告.md). Over a 20 s window a circling")
    print("       policy scores ~0 even while walking briskly; `turnR m` is the")
    print("       loop-free reading of the same defect.")

    print("\n=== Q1 walking: zero command (the standing-drift judge) ===")
    zc = [(a, d["zerocmd"]) for a, d in sorted(arms.items()) if "zerocmd" in d]
    if not zc:
        print("  (no data)")
    for arm, js in zc:
        a = _agg(js)
        if not a:
            print(f"  {arm:<52}(no data)"); continue
        print(f"  {arm:<52} yawDC {a['yaw_rate_bias_deg_s']['mean']:+.3f} deg/s   "
              f"net yaw {a['net_yaw_deg']['mean']:+.1f} deg   "
              f"gait speed {a['total_path_m']['mean']/SWEEP_SECONDS:.3f} m/s   "
              f"tilt {a['tilt_mean_deg']['mean']:.2f} deg")

    print("\n=== Q2 sit-stand: head-tripod rest ===")
    sit = [(a, d["sitstand"]) for a, d in sorted(arms.items()) if "sitstand" in d]
    any_sit = bool(sit)
    for arm, log in sit:
        rows = _parse_sitstand(log)
        print(f"  {arm}:")
        if not rows:
            print("     (could not parse; see the log)")
            continue
        print(f"     {'phase':<14}{'z_m':>8}{'tilt':>8}{'headF_N':>9}{'head_z_mm':>10}"
              f"{'neck_rel':>10}   verdict")
        # The SIT half must be checked too. The first version of this verdict only
        # looked at STAND phases, so a policy that had LOST the ability to sit
        # would still print PASS -- and that is not hypothetical: measured on
        # sitstand_headfix, the SIT segment's lowest z was 0.104 at ck3000,
        # 0.080 at ck5000 and 0.094 at ck6000 against a 0.060 target, i.e. the
        # sit was degrading while every STAND metric stayed green.
        for phase, z, tilt, neck, headf, headz in rows:
            v = "-"
            if phase.endswith("_SIT"):
                v = "PASS" if z <= 0.065 else "fail"
            if phase.startswith("seg") and "STAND" in phase:
                # The acceptance table judges the STAND rest on PHYSICAL quantities
                # -- head-ground LOAD and head HEIGHT -- not on the neck joint angle.
                # The fix lifts the head and removes the load while the neck STILL
                # parks near its hard stop, so a neck-angle criterion reports "fail"
                # on a repaired policy; that exact contradiction was in this script's
                # first version. Keep the two questions separate and judge the
                # physical one; report the joint angle for information.
                import math as _m
                if _m.isnan(headf):
                    v = "PASS*" if (tilt < 20.0 and neck > -0.10 and z > 0.10) else "fail*"
                else:
                    ok = (tilt < 20.0 and z > 0.10 and headf < 0.5 and headz > 0.14)
                    v = "PASS" if ok else "fail"
            mark = "  <- the acceptance phase" if "STAND" in phase else ""
            hs = "  n/a" if headf != headf else f"{headf:9.3f}{headz*1000:10.1f}"
            print(f"     {phase:<14}{z:>8.3f}{tilt:>8.1f}{hs}{neck:>10.3f}   {v}{mark}")
    if not any_sit:
        print("  (no data)")
    print("  acceptance (PHYSICAL): STAND tilt < 20 deg, z >= 0.10 m, "
          "head-ground force < 0.5 N,")
    print("              head body height > 140 mm, AND SIT segment reaches z <= 0.065 m.")
    print("              PRE-FIX: 35.0 deg, 3.28 N, 88 mm (and it DID sit).")
    print("  ⚠️ `eval_sitstand_head.py` is UNSEEDED: each run draws different DR, so a")
    print("     single PASS/FAIL is one sample. Repeat before believing a difference.")
    print("              `*` = log predates the headF/head_z columns; fell back to the")
    print("              neck-angle criterion (which the fix does NOT satisfy -- the neck")
    print("              stays near its stop while the head carries no load).")
    return 0


def _parse_sitstand(log: Path) -> list[tuple]:
    """Pull the STEADY-STATE block out of eval_sitstand_head.py's stdout.

    Parsed POSITIONALLY (first four whitespace fields), not by a fixed-column
    regex. A fixed regex broke the moment two columns were appended to that
    script's output (`headF_N`, `head_z_mm`): every row stopped matching, the
    parser returned [], and the Q2 section would have printed "could not parse"
    while looking like a normal empty result. Positional parsing is immune to
    added trailing columns, which is the only kind of change this table gets.
    """
    txt = log.read_text(errors="ignore")
    if "STEADY STATE ONLY" not in txt:
        return []
    tail = txt.split("STEADY STATE ONLY", 1)[1]
    out: list[tuple[str, float, float, float]] = []
    for line in tail.splitlines():
        parts = line.split()
        if len(parts) < 5 or not re.fullmatch(r"seg\d+_(?:STAND|SIT)", parts[0]):
            continue
        try:
            z, tilt, neck = (float(parts[1]), float(parts[2]), float(parts[3]))
        except ValueError:
            continue          # header / separator / a future non-numeric column
        # The PHYSICAL columns (`headF_N`, `head_z_mm`) were appended to that
        # script's output for the 2026-09-16 fix, so they are optional here:
        # a log from before the change simply reports nan and the verdict falls
        # back to the joint-angle criterion.
        # Column order in that table is
        #   phase z tilt neck_rel head_rel neck_abs hcmd_neck err headF_N head_z_mm
        # i.e. the physical columns are parts[8] and parts[9] -- NOT parts[7]/[8]
        # (parts[7] is `err`). Parsing them by an off-by-one index is exactly the
        # kind of silent mis-read that yields a plausible number from the wrong
        # column, so the test asserts on known fixture values.
        headf = headz = float("nan")
        if len(parts) >= 10:
            try:
                headf, headz = float(parts[8]), float(parts[9]) / 1000.0
            except ValueError:
                pass
        out.append((parts[0], z, tilt, neck, headf, headz))
    return out


def _run_dirs_from_results() -> dict[str, Path]:
    """arm name -> run directory, straight out of results.tsv (column 4).

    Do NOT reconstruct run directories from the arm name: `MICRODUCK_RUN_DIR_NAME`
    is only honoured when `MICRODUCK_TASK_NAMED_RUNS=1` is ALSO set, so a queue
    that sets only the name lands in mjlab's default timestamped directories --
    which is exactly what happened on 2026-09-16. The queue records the real path
    it parsed from the training log; read that instead of guessing.
    """
    out: dict[str, Path] = {}
    tsv = OUT / "results.tsv"
    if not tsv.exists():
        return out
    for line in tsv.read_text().splitlines()[1:]:
        f = line.split("\t")
        if len(f) < 5:
            continue
        name, rd = f[0], f[4]
        if rd and rd not in ("-", "smoke-failed"):
            q = ROOT / rd if not rd.startswith("/") else Path(rd)
            if q.is_dir():
                out[name] = q
    return out


def sweep(runs: list[str]) -> int:
    """Rank EVERY checkpoint of each walk arm -- not just the last one.

    Not optional polish: within the previous single run the checkpoints ranged
    0.043..0.489 path efficiency (and the yaw bias wandered +7.7 -> -3.65 ->
    +12.7 -> -0.75 -> -11.95 deg/s), and the official deliverable IS a selected
    checkpoint (`BEST_alpha_walking_rough.onnx`). Picking the best checkpoint is
    part of shipping a policy, which is why the queues save every 1000 iters.

    Ranking is on the metric that matters, in order:
        |yaw DC| <= 1.0 deg/s AND turnR >= 10 m AND pathEff >= 0.30
        AND gait speed >= 30 % of the command
    then |yawDC| ascending, then turning radius descending.
    A checkpoint whose gait speed is under the floor is REJECTED: "a straight
    line while barely walking" is not a pass (the official policy is slow in sim
    too -- 0.08 m/s at a 0.2 command -- but it does walk).

    ⚠️ THIS FUNCTION WAS ONCE DELETED BY A CARELESS SLICE-EDIT and nothing noticed,
    because `--sweep` is only exercised at the very end of a queue. A test now
    invokes every CLI flag against an empty sandbox so a missing callee fails in
    CI rather than at 03:00.
    """
    TMP.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    known = _run_dirs_from_results()
    for run in runs:
        rd = known.get(run)
        if rd is None:
            print(f"[!] {run}: no usable run_dir in results.tsv "
                  f"(known arms: {sorted(known)})", file=sys.stderr)
            continue
        cks = sorted(rd.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        print(f"== {run}: {len(cks)} checkpoints in {rd}")
        for ck in cks:
            it = int(ck.stem.split("_")[1])
            if it < 1000:
                continue
            raw = TMP / f"{run}_{it}.onnx"
            dyn = TMP / f"{run}_{it}_dyn.onnx"
            js = TMP / f"{run}_{it}.json"
            if not dyn.exists():
                subprocess.run([str(ROOT / ".venv/bin/python"), "scripts/export.py",
                                "Mjlab-Velocity-Flat-MicroDuck",
                                "--checkpoint-file", str(ck), "--onnx-file", str(raw)],
                               cwd=ROOT, capture_output=True)
                if not raw.exists():
                    print(f"   ck{it}: export FAILED"); continue
                subprocess.run([str(ROOT / ".venv/bin/python"),
                                "scripts/make_dynamic_batch_onnx.py",
                                "--onnx", str(raw), "--write", str(dyn),
                                "--check-batch", "32"],
                               cwd=ROOT, capture_output=True)
                if not dyn.exists():
                    print(f"   ck{it}: dynamic-batch rewrite FAILED"); continue
            if not js.exists():
                subprocess.run([str(ROOT / ".venv/bin/python"),
                                "scripts/eval_straightness.py", "--onnx", str(dyn),
                                "--lin-vel-x", "0.2", "--num-envs", str(EVAL_ENVS),
                                "--seed", "0", "--seconds", str(SWEEP_SECONDS),
                                "--freeze-pose-cmd", "--json", str(js)],
                               cwd=ROOT, capture_output=True)
            a = _agg(js)
            if not a:
                print(f"   ck{it}: eval FAILED"); continue
            yaw = _val(a, "yaw_rate_bias_deg_s")
            eff = _val(a, "path_efficiency")
            rad = _val(a, "turning_radius_m")
            spd = _val(a, "total_path_m") / SWEEP_SECONDS
            ok = (abs(yaw) <= ACCEPT_YAW_DEG_S and rad >= ACCEPT_TURN_RADIUS_M
                  and eff >= ACCEPT_EFF and spd >= MIN_SPEED_FRACTION * 0.2)
            results.append({"run": run, "ckpt": it, "yaw_dc": yaw,
                            "path_efficiency": eff, "turning_radius_m": rad,
                            "gait_speed_mps": spd, "accept": bool(ok),
                            "onnx": str(dyn)})
            print(f"   ck{it:<6} yawDC {yaw:+7.2f}  eff {eff:6.3f}  "
                  f"R {rad:8.2f} m  speed {spd:.3f}  {'PASS' if ok else ''}")

    results.sort(key=lambda r: (not r["accept"], abs(r["yaw_dc"]),
                                -r["turning_radius_m"], -r["path_efficiency"]))
    (OUT / "sweep_ranking.json").write_text(json.dumps(results, indent=1))
    print(f"\n[i] {len(results)} checkpoints ranked -> {OUT / 'sweep_ranking.json'}")
    passed = [r for r in results if r["accept"]]
    if passed:
        print("=== checkpoints meeting the acceptance bar (best first) ===")
        for r in passed[:10]:
            print(f"  {r['run']:<18} ck{r['ckpt']:<6} yawDC {r['yaw_dc']:+6.2f}  "
                  f"eff {r['path_efficiency']:.3f}  R {r['turning_radius_m']:.2f} m")
        print(f"\n  deliverable candidate: {passed[0]['onnx']}")
        print("  ⚠️ 'straight' is only HALF the acceptance: run --steer on the winner.")
    else:
        print("[!] NO checkpoint met the bar. Report what the checkpoints actually")
        print("    do (that is the result), do not ship the least-bad one silently.")
    return 0


def steer_check(runs: list[str], envs: int = 64, seconds: int = 12) -> int:
    """Does the policy actually PRODUCE the yaw rate it is commanded?

    This is a separate acceptance axis from "walks straight", and it was missed
    until 2026-09-16: the delivered `walk_hd1910_12000` walks straight enough but
    the measured yaw rate barely moves when the command sweeps +-0.6 rad/s
    (+-34 deg/s) — a response under 2 %, with the WRONG sign at a standstill. The
    official policy turns fine (+18.5 deg/s for a +34 deg/s command), so this is
    our defect, not the task's. A walking policy that cannot be steered is not
    deployable even if its straight line is beautiful.
    """
    TMP.mkdir(parents=True, exist_ok=True)
    known = _run_dirs_from_results()
    onnx_of: dict[str, Path] = {}
    for arm, rd in known.items():
        c = sorted(rd.glob("*.onnx"))
        if c:
            onnx_of[arm] = c[-1]
    # the currently delivered artefact, always, as the reference row
    ref = ROOT / "walk_hd1910_12000.onnx"
    if ref.exists():
        onnx_of["(delivered) walk_hd1910_12000"] = ref

    print("=== Q1 steering: does the command actually produce yaw? ===")
    print(f"  {'policy':<36}{'wz=0':>10}{'wz=+0.6':>10}{'response':>10}   verdict")
    for arm, onnx in onnx_of.items():
        outs = {}
        for wz, tag in ((0.0, "z"), (0.6, "p")):
            js = TMP / f"steer_{arm.replace('/','_')}_{tag}.json"
            if not js.exists():
                subprocess.run([str(ROOT / ".venv/bin/python"),
                                "scripts/eval_straightness.py", "--onnx", str(onnx),
                                "--lin-vel-x", "0.2", "--ang-vel-z", str(wz),
                                "--num-envs", str(envs), "--seed", "0",
                                "--seconds", str(seconds), "--freeze-pose-cmd",
                                "--json", str(js)], cwd=ROOT, capture_output=True)
            a = _agg(js)
            if a:
                outs[tag] = _val(a, "yaw_dc_body_deg_s")
        if len(outs) < 2:
            print(f"  {arm:<36}{'(no data)':>30}")
            continue
        resp = outs["p"] - outs["z"]
        # the command is 0.6 rad/s = 34.4 deg/s; require at least a 10 % response
        ok = resp >= 0.1 * 34.38
        print(f"  {arm:<36}{outs['z']:>+10.2f}{outs['p']:>+10.2f}{resp:>+10.2f}   "
              f"{'PASS' if ok else 'FAIL - cannot steer'}")
    print(f"\n  reference: official alpha_walking @ its own hw responds "
          f"+18.5 deg/s to a +0.6 rad/s command (54 % tracking).")
    print(f"  acceptance: response >= 3.4 deg/s (10 %). A policy below this is")
    print(f"  effectively unsteerable on the robot even if it walks a straight line.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--steer", action="store_true",
                    help="also measure whether the arms actually respond to a yaw command")
    ap.add_argument("--runs", default="walk_sym,walk_yawdc,turn_first,turnfirst_yawdc")
    args = ap.parse_args()
    rc = verdict()
    if args.steer:
        print()
        rc |= steer_check([r for r in args.runs.split(",") if r])
    if args.sweep:
        print()
        rc |= sweep([r for r in args.runs.split(",") if r])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
