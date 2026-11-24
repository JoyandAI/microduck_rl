#!/usr/bin/env python3
"""Sweep EVERY checkpoint of the 12 overnight runs and rank them.

Rationale: the overnight A/B showed the four arms are statistically
indistinguishable (all pairwise Welch t < 0.7) while WITHIN a single run the
checkpoints ranged 0.043..0.489. So the deliverable is not "which recipe" but
"which checkpoint" — the official ships `BEST_alpha_walking_rough.onnx` for
exactly this reason.
"""
from __future__ import annotations
import glob, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "logs/night_queue"
TMP = Path("/tmp/sweep_all_night"); TMP.mkdir(exist_ok=True)
EVAL_ENVS = int(os.environ.get("EVAL_ENVS", "32"))

runs = sorted(glob.glob(str(ROOT / "logs/rsl_rl/night_*/*/")))
print(f"{len(runs)} runs", flush=True)
best = []
for rd in runs:
    rd = Path(rd)
    job = rd.parts[-2]
    for ck in sorted(rd.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1])):
        it = int(ck.stem.split("_")[1])
        if it < 1000:          # early checkpoints are never competitive
            continue
        onnx = TMP / f"{job}_{it}.onnx"
        if not onnx.exists():
            subprocess.run([str(ROOT / ".venv/bin/python"), "scripts/export.py",
                            "Mjlab-Velocity-Flat-MicroDuck",
                            "--checkpoint-file", str(ck), "--onnx-file", str(onnx)],
                           cwd=ROOT, capture_output=True,
                           env={**os.environ, "WANDB_MODE": "offline"})
        if not onnx.exists():
            print(f"{job} ck{it}: export FAILED", flush=True); continue
        jf = TMP / f"{job}_{it}.json"
        subprocess.run([str(ROOT / ".venv/bin/python"), "scripts/eval_straightness.py",
                        "--onnx", str(onnx), "--lin-vel-x", "0.2", "--seconds", "20",
                        "--num-envs", str(EVAL_ENVS), "--seed", "0", "--json", str(jf)],
                       cwd=ROOT, capture_output=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
        try:
            a = json.loads(jf.read_text())["aggregate"]
            eff = a["path_efficiency"]["mean"]; yaw = a["yaw_rate_bias_deg_s"]["mean"]
            best.append((eff, yaw, job, it, str(onnx)))
            print(f"{job} ck{it:>5}: eff {eff:+.4f}  yaw {yaw:+7.2f}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{job} ck{it}: eval FAILED ({e})", flush=True)

best.sort(reverse=True)
print("\n=== TOP 15 ===", flush=True)
for eff, yaw, job, it, onnx in best[:15]:
    print(f"  eff {eff:+.4f}  yaw {yaw:+7.2f}  {job} ck{it}", flush=True)
(OUT / "sweep_ranking.json").write_text(json.dumps(
    [{"eff": e, "yaw": y, "job": j, "ckpt": i, "onnx": o} for e, y, j, i, o in best], indent=1))
print(f"\nwritten {OUT/'sweep_ranking.json'}", flush=True)
