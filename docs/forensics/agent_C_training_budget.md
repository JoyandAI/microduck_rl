# Agent C — Was the walking policy undertrained, and would more training fix the posture defects?

Scope: `logs/rsl_rl/velocity/2026-09-10_18-43-53_velocity/` (run "velocity", `walk_hd1910_15000.onnx`).
Method: read the full 15 000-iteration training history from **two independent sources**, fit the
late-run trends, check the curriculum boundaries, then run the decisive experiment — export **five**
checkpoints spanning the run to ONNX and measure the posture in the faithful BAM env.

Artifacts (all under `docs/forensics/artifacts/`):
`tfevents_scalars.csv` (11.6 MB, 15 000 rows x 44 scalars), `training_curves.png`,
`checkpoint_trajectory.{json,csv}`, `curriculum_boundary_windows.csv`, `saturation_fits.json`,
`ckpt/model_{2000,5000,8000,11000,14999}.onnx` + `ckpt/gf_ckpt_*_12s.{json,csv}`.
**No tracked source file outside `docs/forensics/` was modified; no training run was launched.**

---

## Verdict

**MIXED — but dominated by *converged-but-wrong-equilibrium*, not undertraining.**

The three sub-defects behave completely differently over training:

| defect | behaviour over 13 000 iterations | diagnosis |
|---|---|---|
| leg **splay** (abduction / foot separation) | **monotonically improving**, 2.3x better (10.43° → 4.62°) | converging, not budget-limited |
| leg **toe-out duck stance** (hip_yaw +16.5°) | L goes 19.1° → 17.2°, R goes 10.6° → 16.3°; the \|L−R\| gap collapses 8.5° → 0.86° | **converged to a common non-zero attractor — equilibrium** |
| leg **L/R asymmetry** (hip_pitch/knee) | still shrinking fast at the end (23.4°, 22.0° at iter 14999) | **genuinely undertrained** |
| **neck wobble** | flat within a ±15% measurement band, or very slightly worse (neck_amp 14.4 → 18.0, d_neck_pitch p2p 55→62) | **equilibrium / not the thing being priced** |

**Single strongest piece of evidence** — the toe-out convergence in the checkpoint table below:
both hip_yaw joints *start 8.5° apart and meet at the same wrong value*. The left leg moves
19.1° → 17.2° and the right 10.6° → 16.3°, while \|L−R\| falls 8.53° → 4.69° → 2.34° → 1.45° → 0.86°
(monotone, R² = 0.85). A budget-limited optimiser would walk both joints toward HOME (0°); instead
the two legs are being driven to a *common* non-zero value. That is the signature of a reward
equilibrium, not of an unfinished search. It is corroborated in hard numbers: `Episode_Reward/pose`
(the only term that could pull the legs back to HOME) is worth **+0.0154 out of ~106 total reward =
0.014 % of the reward mass**, and its fitted asymptote (0.01345) sits *below* its own recent
5 000-iteration averages (0.0114/0.0122/0.0129/0.0118/0.0154) — i.e. the term is statistically flat
and has no upward headroom left to claim.

Secondary headline: the estimates of remaining budget are wrong in the *optimistic* direction.
The yaw-tracking error curve is a saturating exponential, not a linear ramp, and its fitted
asymptote lies **above** the target that the previous analysis assumed was reachable (see
"Budget comparison" and "What would more training buy?").

---

## Run facts

| | |
|---|---|
| run dir | `logs/rsl_rl/velocity/2026-09-10_18-43-53_velocity/` |
| iterations | **15 000 / 15 000** (`max_iterations: 15000` in `params/agent.yaml`) — completed |
| `num_steps_per_env` | 24 |
| `num_envs` | **2 048** (`params/env.yaml: scene.num_envs`) |
| env-steps | 15000 x 2048 x 24 = **737.3 M** |
| wall clock | `model_0.pt` 2026-09-10 18:44:01 → `model_14999.pt` 23:14:38 = **4 h 30 m 37 s**; log reports `Iteration time: 1.04s / Time elapsed: 04:25:25` |
| throughput | ~47 300 fps (`Perf/total_fps`, flat all run) |
| git | commit `316b364860d2458aca88572c20f6985164fa6715`, branch `feat/feetech-hls2909`; diff snapshot in `<run>/git/microduck_rl.diff` |
| checkpoint list | **66 files, every 250 iterations**: `model_0, model_250, ... model_14750` plus the final odd one named `model_14999.pt` (the run stops at 14999, not 15000) |
| ONNX in run dir | `2026-09-10_18-43-53_velocity.onnx` (metadata `run_path=2026-09-10_18-43-53_velocity`) |
| policy ONNX used here | `walk_hd1910_15000.onnx`; its ONNX metadata says `run_path=logs/rsl_rl/velocity/2026-09-10_18-43-53_velocity/model_14999.pt` — **verified to be this run's final checkpoint**, and its initialiser weights are **bit-identical** (max abs diff 0.0) to a fresh export of `model_14999.pt` |

Task id: `Mjlab-Velocity-Flat-MicroDuck` (confirmed by the run's `BamActuator model='m5'` banner and
by the absence of a terrain curriculum in `env.yaml`; `gait_forensics.py`'s default task).

---

## Training curves

**Both sources read successfully and agree to float32 rounding.** Two paths exist and I used both:

1. `uv run python` + `tensorboard.backend.event_processing.event_accumulator` on
   `logs/rsl_rl/velocity/2026-09-10_18-43-53_velocity/events.out.tfevents.1789037037.joyandai.553071.0`
   (42.5 MB) → 44 scalar tags x 15 000 rows.
2. The wandb offline store: `wandb/offline-run-20260910_184357-cnrsrpwp/run-cnrsrpwp.wandb`
   (109 MB, **15 000 history records**, steps 0…14999), read with
   `wandb.sdk.internal.datastore.DataStore` + `wandb_internal_pb2.Record`. Note the WAL item field is
   `HistoryItem.nested_key` (a repeated field), **not** `key` — that detail cost one iteration to find.
   `scripts/wandb_utils.py` is **not** usable for this: it only wraps `wandb.Api()` against the
   online project `pollen-robotics/mjlab_microduck` and `find_latest_run()`/`print_run_info()` have no
   offline path (`WANDB_MODE=offline`, not logged in → `api.runs()` cannot resolve the local run).

Cross-check (wandb vs tfevents, identical to ~1e-7):

| scalar | step 0 | step 1000 | step 14999 |
|---|---|---|---|
| `Train/mean_reward` | 0.9521824 | 76.16833 | 107.80321 |
| `Episode_Reward/head_pose_tracking` | 0.02261974 | 1.5632886 | 1.6770496 |
| `Metrics/twist/error_vel_yaw` | 0.05762258 | 4.5587902 | 2.5656948 |
| `Policy/mean_std` | 0.5008380 | 0.3038925 | 0.1265952 |
| `Episode_Reward/pose` | 0.00505202 | 0.0102173 | 0.0175444 |

Values at the requested iterations (raw logged sample, not windowed):

| metric | 1000 | 3000 | 6000 | 10000 | 15000 | trend at end |
|---|---|---|---|---|---|---|
| `Train/mean_reward` | 76.17 | 81.40 | 94.44 | 101.11 | 107.80 | **still rising** (+1.29/1000 it over [12000,15000), R²=0.33) |
| `Train/mean_episode_length` | 918.8 | 932.0 | 988.4 | 972.0 | 982.9 | **flat** (max 1000; no trend, R²=0.015) |
| `Episode_Reward/track_linear_velocity` | 0.847 | 0.923 | 1.256 | 1.329 | 1.296 | **flat** (slope +0.0084/1000, R²=0.09) |
| `Episode_Reward/track_angular_velocity` | 0.041 | 0.080 | 0.131 | 0.184 | 0.227 | rising (+0.0063/1000, R²=0.41) |
| `Episode_Reward/upright` | 1.408 | 1.338 | 1.692 | 1.716 | 1.709 | **flat** (R²=0.04) |
| **`Episode_Reward/pose`** (leg pose) | 0.0102 | 0.0093 | 0.0114 | 0.0177 | 0.0175 | **flat/oscillating** (+0.00028/1000, R²=0.04) |
| **`Episode_Reward/head_pose_tracking`** | 1.5633 | 1.3681 | 1.6693 | 1.6838 | 1.6770 | **flat** (+0.0028/1000, R²=0.016) |
| **`Episode_Reward/head_pose_bias`** (≤0 by rule) | −0.0440 | −0.2355 | −0.2703 | −0.2676 | −0.2675 | **flat** (R²=0.013) |
| `Episode_Reward/action_rate_l2` | −1.0974 | −0.7931 | −0.7944 | −0.6674 | −0.5824 | rising *but weight-confounded* (see note) |
| `Episode_Reward/body_ang_vel` | −0.1775 | −0.1213 | −0.1100 | −0.0916 | −0.0838 | mildly improving |
| `Episode_Reward/angular_momentum` | −0.0000 | −0.0000 | −0.0000 | −0.0000 | −0.0000 | floor (weight −0.02, negligible) |
| `Episode_Reward/air_time` | 1.3059 | 1.1047 | 1.3302 | 1.2903 | 1.4456 | **flat↑** (R²=0.004) |
| `Episode_Reward/self_collisions` | −0.0125 | −0.0201 | −0.0139 | −0.0109 | −0.0102 | improving |
| `Episode_Reward/dof_pos_limits` | −0.0328 | −0.0592 | −0.0704 | −0.0883 | −0.0766 | mildly worsening/flat (knee sits nearer its limit) |
| `Episode_Reward/body_pose_tracking` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | weight 0 by design (obs slot kept alive) |
| `Episode_Metrics/mean_action_acc` | 0.5727 | 0.3241 | 0.2943 | 0.2699 | 0.2470 | still falling (R²=0.46) |
| `Metrics/twist/error_vel_xy` | 0.5497 | 0.4285 | 0.4551 | 0.4028 | 0.4043 | still falling slowly |
| `Metrics/twist/error_vel_yaw` | 4.5588 | 3.1616 | 3.2964 | 2.7621 | 2.5657 | still falling, **saturating** |
| `Policy/mean_std` | 0.3039 | 0.1745 | 0.1569 | 0.1402 | 0.1266 | still falling (R²=0.41) |
| `Loss/entropy` | 3.033 | −4.787 | −6.360 | −8.016 | −9.583 | still falling |
| `Loss/value` | 0.1031 | 0.1045 | 0.0555 | 0.0725 | 0.0348 | still falling |
| `Episode_Termination/fell_over` | 0.4583 | 0.6667 | 0.0833 | 0.0833 | 0.0417 | ~flat, near zero |
| `Episode_Termination/time_out` | 1.7083 | 2.0417 | 2.1667 | 2.0833 | 2.1667 | flat |

Windowed (5 000-iteration windows) view of the terms that govern the two defects:

| term | [3000,5000) | [5000,8000) | [8000,10000) | [10000,12000) | [12000,14000) | [14000,15000) |
|---|---|---|---|---|---|---|
| `head_pose_tracking` | 1.5966 | 1.6324 | 1.6618 | 1.6810 | 1.6833 | 1.6919 |
| `head_pose_bias` | −0.2843 | −0.2827 | −0.2655 | −0.2616 | −0.2577 | −0.2597 |
| `pose` | 0.0103 | 0.0114 | 0.0122 | 0.0129 | 0.0118 | 0.0154 |
| `action_rate_l2` | −0.8667 | −0.7784 | −0.7030 | −0.6578 | −0.6483 | −0.6140 |
| `Mean action acc` | 0.3123 | 0.2928 | 0.2766 | 0.2656 | 0.2630 | 0.2543 |

**Interpretation.**
* The two posture-governing terms are **flat from ~iteration 2 500 onward**: `head_pose_tracking`
  moves +0.6 % over iterations 10 000 → 15 000, `pose` moves +0.4 % (and oscillates),
  `head_pose_bias` is constant to 4 decimal places. Neither has an upward slope that a longer run
  would extend.
* The total reward *is* still rising (+1.3/1000 it at the end), but the rise is concentrated in the
  terms that are **confounded with exploration-noise decay**: `Policy/mean_std` is still falling
  (−0.0028/1000, R²=0.45), `Loss/entropy` is still falling, and the matching gains are in
  `action_rate_l2` (+0.0173/1000) and `mean_action_acc` (−0.0028/1000) — i.e. a shrinking action
  distribution, not a better gait. `track_angular_velocity` (+0.006/1000) is the only genuinely
  task-related late riser, and it is the one metric with a saturated, badly-tracking target (§ below).
* `action_rate_l2` caveat: its *weight* ramps −0.1 → −1.0 over iterations 500–1500
  (`Curriculum/action_rate_weight`), so the first 1 500 iterations of that curve are a weight
  artefact, not a behaviour change. Only from iteration 1 500 does the curve read as behaviour.
* Every `Episode_Reward/<penalty>` is ≤ 0 for the whole run — verified for all nine penalty terms
  (`action_rate_l2`, `angular_momentum`, `body_ang_vel`, `dof_pos_limits`, `foot_clearance`,
  `foot_slip`, `foot_swing_height`, `head_pose_bias`, `self_collisions`): max ≤ 0.0, so AGENTS.md's
  "no double-negated penalty farmed by the policy" rule holds for this run.

![training curves](artifacts/training_curves.png)

---

## Curriculum boundary check

Curriculum steps are **env steps** = `iteration x 24`, so the `step = 500*k` boundaries land on the
**last env step of logged iteration 499, 999, 1499, 1999**. This is directly visible in the logged
curriculum values, e.g. `Curriculum/head_pose_range` reads 0.0758 at row 499
= (23 x 0.07 + 0.21)/24, 0.2217 at row 999 = (23 x 0.21 + 0.49)/24, 0.5075 at row 1499
= (23 x 0.49 + 0.91)/24. Transitions measured from the log:

| curriculum | 0 | 500 | 750 | 1000 | 1250 | 1500 | 2000 |
|---|---|---|---|---|---|---|---|
| `head_pose_range` (rad) | 0.07 | 0.21 | — | 0.49 | — | 0.91 | **1.40** |
| `head_pose_bias_weight` | 0.0 | (1.0 at 600) | — | 2.0 | — | 3.0 | 3.0 |
| `action_rate_weight` | −0.1 | −0.2 | −0.4 | −0.6 | −0.8 | −1.0 | −1.0 |
| `standing_envs` | 0.02 | 0.05 | 0.10 | 0.15 | — | 0.20 | 0.25 |
| `com_range` (m) | 0.003 | 0.005 | — | 0.010 | — | 0.015 | 0.015 |
| `head_com_range` (m) | 0.003 | 0.005 | — | 0.010 | 0.010 | 0.010 | 0.010 |
| `body_pose_range` | 0.05 (constant — body_pose disabled in this env) | | | | | | |

**Steps DOWN exactly at boundaries: yes — at the 1500 and 2000 boundaries.** 250-iteration windows
(`curriculum_boundary_windows.csv`):

| metric | [1250,1500) | [1500,1750) | [1750,2000) | **[2000,2250)** | [2250,2500) | [2500,2750) |
|---|---|---|---|---|---|---|
| `Train/mean_reward` | 82.69 | **78.72** | 81.48 | **74.94** | 75.89 | 78.23 |
| `head_pose_tracking` | 1.6674 | 1.5878 | 1.6212 | **1.5247** | 1.5212 | 1.5388 |
| `air_time` | 1.3831 | 1.2998 | 1.2991 | **1.2024** | 1.1990 | 1.1974 |
| `mean_episode_length` | 944.0 | 925.1 | 933.0 | 921.7 | 928.6 | 936.6 |
| `upright` | 1.5321 | 1.5107 | 1.5344 | 1.4934 | 1.5079 | 1.5289 |
| `error_vel_xy` (lower better) | 0.5609 | 0.5324 | 0.5221 | 0.5134 | 0.5069 | 0.4958 |

* **iteration 2000** (`head_pose_range` 0.91 → 1.40 rad = 52° → 80°, `standing_envs` 0.20 → 0.25):
  mean reward **−8.0 %**, `head_pose_tracking` −6.0 %, `air_time` −7.4 %, episode length −1.2 %.
* **iteration 1500** (`head_pose_range` 0.49 → 0.91 rad = 28° → 52°, `head_pose_bias_weight`
  2.96 → 3.0): mean reward **−4.8 %**, `head_pose_tracking` −4.8 %, `air_time` −6.0 %.
* The 500 and 1000 boundaries show **no** step-down (mean reward 36.0 → 42.7 and 61.5 → 69.8, both up).
* Both dips are **transient**: reward returns to the pre-boundary level within ~250–1000 iterations
  (peak 82.69 → trough 74.94 → 78.23 at [2500,2750) → 86.8 average over [3000,5000), i.e. *above*
  the pre-2000 level).

So AGENTS.md's diagnostic fires, but weakly: the pacing is not *broken* (the policy recovers and
overshoots the pre-boundary level), it is *aggressive*. The 2 000-iteration head-pose range widening
cost roughly one full episode-length worth of reward and ~500–1000 iterations of re-consolidation —
which is 3–7 % of a 15 000-iteration budget, not a show-stopper. Worth reporting because it is the
only place in the run where a metric steps down for a *config* reason; it is **not** the cause of the
posture defects (the defects are still there 13 000 iterations later).

---

## Budget comparison

Common unit: **env-steps = iterations x num_envs x 24**. Also given as iteration-equivalents at 4096 envs.

| source | iterations | envs | env-steps | == iters @4096 envs | this run as % of it |
|---|---|---|---|---|---|
| AGENTS.md — "simple episodic tricks" | 1 000 | 4 096 | 98.3 M | 1 000 | 750 % |
| **AGENTS.md — "gaits and curriculum-heavy recovery"** | **4 000** | 4 096 | **393.2 M** | 4 000 | **187 %** |
| **AGENTS.md — same, upper end** | **6 000** | 4 096 | **589.8 M** | 6 000 | **125 %** |
| `microduck_velocity_env_cfg.py` `MicroduckRlCfg.max_iterations` (comment: "走路类任务 4000–6000 迭代") | 50 000 | 4 096 | 4 915 M | 50 000 | 15.0 % |
| **this run** | **15 000** | **2 048** | **737.3 M** | 7 500 | 100 % |

Two readings, and both matter:

1. **Measured against the operational guidance, the run is over budget.** 737 M env-steps is
   1.87x the 4 000-iteration mark and 1.25x the top of the recommended 4 000–6 000 band. Repo
   history confirms that band is the operative one, not the nominal ceiling: the previous
   walking runs in this repo were launched with `max_iterations: 6000`
   (`logs/rsl_rl/velocity/2026-09-03_17-02-23_hls2909-1` at 2 048 envs,
   `logs/rsl_rl/velocity/2026-09-07_13-49-41_hls2909-full` at 4 096 envs = 590 M env-steps).
   This run is the longest walking run in the repo's history.
2. **Measured against the cfg's own `max_iterations = 50_000`, it is only 15 % done.** If the author's
   intent really was 50 000 iterations, this run stopped 3.3x early (in iteration terms) — but the
   comment on that very line says the budget reference for walking is 4 000–6 000, so 50 000 is a
   safety ceiling against a runaway, not a target.

**Official pre-trained policies: no training metadata exists.** `/home/joyandai/microduck/policies/README.md`
documents provenance only as file names copied from `apirrone/microduck_runtime` @ `5f3b314`
(`alpha_walking.onnx` ← `BEST_alpha_walking_rough.onnx`, etc.). I read the ONNX metadata of
`alpha_walking.onnx` and `alpha_stand.onnx`: they carry `joint_names`, `default_joint_pos`,
`command_names`, `observation_names`, `action_scale` — and **`run_path = None`**. There is no
iteration count, env count or training-run id anywhere in them, and a repo-wide grep of
`/home/joyandai/microduck/` for budget/provenance keywords returns nothing relevant. **The upstream
budget for the "official look" is therefore unverifiable from local evidence** — the only budget
numbers that exist locally are the ones in AGENTS.md and the cfg (both of which point at 4 000–6 000,
i.e. *below* this run).

---

## Checkpoint trajectory experiment

Five checkpoints exported with the mandatory normalizer-baking path, then evaluated in the faithful
BAM env. Exact commands (identical for every checkpoint, `UV_CACHE_DIR=/tmp/uv-cache`):

```bash
RUN=logs/rsl_rl/velocity/2026-09-10_18-43-53_velocity
for IT in 2000 5000 8000 11000 14999; do
  uv run --no-sync python scripts/export.py Mjlab-Velocity-Flat-MicroDuck \
      --checkpoint-file $RUN/model_${IT}.pt \
      --onnx-file docs/forensics/artifacts/ckpt/model_${IT}.onnx
  uv run --no-sync python scripts/gait_forensics.py \
      --onnx docs/forensics/artifacts/ckpt/model_${IT}.onnx \
      --task Mjlab-Velocity-Flat-MicroDuck --lin-vel-x 0.2 --seconds 12 \
      --tag ckpt_${IT} \
      --csv  docs/forensics/artifacts/ckpt/gf_ckpt_${IT}_12s.csv \
      --json-out docs/forensics/artifacts/ckpt/gf_ckpt_${IT}_12s.json
done
# + 1 repeatability run of model_14999.onnx (identical file) to size the noise floor
```

Export worked directly from the `.pt` (`scripts/export.py --checkpoint-file`), so no rsl_rl-side
harness was needed. Observation normalisation is baked by `runner.export_policy_to_onnx`; I verified
the exported final checkpoint is **weight-bit-identical** to `walk_hd1910_15000.onnx`
(max abs initialiser diff = 0.0), which confirms both that the export path is faithful and that
`walk_hd1910_15000.onnx` really is this run's `model_14999.pt`.

Every checkpoint walked 12 s at vx = 0.2 m/s with **0 resets** (no falls) — so all five rows are
comparable, none is a fall artefact.

### Measured posture per checkpoint

| metric | 2000 | 5000 | 8000 | 11000 | 14999 | trend |
|---|---|---|---|---|---|---|
| **SPLAY** abduction mean (deg) | **10.43** | 7.56 | 6.38 | 6.21 | **4.62** | monotone ↓, R²=0.89 |
| **SPLAY** foot separation in trunk y (mm) | **89.9** | 76.5 | 76.5 | 73.4 | **68.7** | monotone ↓, R²=0.82 |
| **SPLAY** toe-out left hip_yaw (deg) | 19.10 | 20.58 | 18.39 | 17.63 | 17.18 | ↓ toward 16.5 |
| **SPLAY** toe-out right hip_yaw (deg) | **10.57** | 15.89 | 16.05 | 16.18 | **16.32** | ↑ toward 16.5 |
| **SPLAY** toe-out \|L−R\| gap (deg) | **8.53** | 4.69 | 2.34 | 1.45 | **0.86** | monotone ↓, R²=0.85 |
| leg hip_roll dev L (deg) | 11.20 | 7.67 | 3.87 | 7.06 | 5.42 | ↓ noisy |
| leg hip_roll dev R (deg) | −9.66 | −7.53 | −8.93 | −5.37 | −3.73 | ↓ (toward HOME −5.0) |
| leg hip_roll \|L−R\| asym (deg) | 20.86 | 15.20 | 12.79 | 12.43 | 9.16 | monotone ↓, R²=0.89 |
| leg hip_pitch dev L (deg) | 0.56 | −10.20 | −5.29 | −16.58 | **−26.19** | ↓ monotone toward R |
| leg hip_pitch dev R (deg) | −54.49 | −52.31 | −53.37 | −55.23 | −49.56 | flat |
| leg hip_pitch \|L−R\| asym (deg) | **55.04** | 42.11 | 48.08 | 38.65 | **23.37** | ↓, R²=0.82 |
| leg knee dev L (deg) | 53.17 | 45.97 | 52.69 | 46.37 | **30.42** | ↓ monotone toward R |
| leg knee dev R (deg) | −4.60 | −3.05 | 6.49 | 4.04 | 8.39 | flat-ish |
| leg knee \|L−R\| asym (deg) | **57.78** | 49.02 | 46.20 | 42.33 | **22.03** | monotone ↓, R²=0.91 |
| trunk tilt mean (deg) | **6.36** | 4.89 | 5.19 | 5.25 | **3.11** | ↓ (noisy, R²=0.72) |
| trunk roll std (deg) | 3.88 | 2.96 | 2.47 | 1.49 | 1.95 | ↓ |
| trunk z mean (m) | 0.095 | 0.098 | 0.093 | 0.092 | 0.099 | flat |
| gait frequency (Hz) | 2.083 | 2.083 | 2.167 | 2.167 | 2.167 | flat |
| **HEAD** neck_amp at gait freq (deg) | 14.42 | 15.38 | 16.75 | 18.07 | 18.04 | ↑ noisy |
| **HEAD** d_neck_pitch std (deg) | **14.11** | 16.60 | 14.64 | 15.83 | **17.41** | flat/noisy |
| **HEAD** d_neck_pitch p2p (deg) | 55.44 | 59.15 | 59.17 | 55.54 | 62.29 | flat |
| **HEAD** d_head_pitch p2p (deg) | 34.91 | 29.09 | **22.11** | 30.51 | 27.37 | non-monotone (min @8000) |
| **HEAD** d_head_roll std (deg) | 3.13 | 3.54 | 3.57 | 4.87 | 3.00 | flat |
| **HEAD** world pitch_osc_std (deg) | 12.67 | 17.91 | 6.38 | 7.53 | 18.71 | unusable (see noise) |
| trunk yaw drift over 12 s (deg) | +174.3 | −145.0 | −93.6 | −130.6 | +66.4 | random, all large |
| resets / falls | 0 | 0 | 0 | 0 | 0 | — |

### Noise floor — read this before believing any single difference

The same three evaluations of the **bit-identical** final ONNX give a run-to-run spread of:

| metric | run A | run B | run C (repeat of B) | spread |
|---|---|---|---|---|
| trunk tilt mean (deg) | 3.16 | 3.11 | **5.94** | **69 %** |
| SPLAY abduction mean (deg) | 4.79 | 4.62 | **6.07** | **28 %** |
| HEAD neck_amp (deg) | 18.90 | 18.04 | **16.28** | **15 %** |
| HEAD world pitch_osc_std (deg) | 19.59 | 18.71 | **3.17** | **119 %** |
| trunk yaw drift (deg) | −137 | +66 | −114 | 331 % |

The single-env eval is a one-sample draw of the domain-randomisation distribution, so it is
**multi-modal / high-variance**: the same weights can land in a visibly different gait mode
(trunk tilt 3.1° vs 5.9°, world head pitch oscillation 18.7° vs 3.2°). Consequences:

* **Trustworthy** (trend >> noise): every SPLAY row except the right toe-out taken alone — the
  abduction trend is 2.3x (130 %) against a 28 % band and is monotone over 5 points; the foot-separation
  trend (31 %) is large against a 5 % band; the three \|L−R\| asymmetry trends are monotone with
  R² 0.82–0.91 and shrink by 56–62 %.
* **Not trustworthy**: `HEAD world pitch_osc_std`, all `head.world.*` and `head.rel_trunk_*` scalars
  (they also inherit the trunk-yaw-drift contamination, since the head orientation is referred to a
  home attitude captured at t=0 while the robot turns up to 174° during the window), and the head
  trend in general. The `neck_amp_deg` rise of 14.4 → 18.0 sits **entirely inside** the 15 % repeat
  band.
* **Therefore the honest head statement is: flat.** Joint-space `d_neck_pitch` p2p is 55–62° at every
  checkpoint and `d_neck_pitch` std is 14.1–17.4°: there is no resolvable trend across 13 000
  iterations of training. The wobble was as large at iteration 2 000 as at 15 000.
* `trunk tilt_mean` (69 % band) is also marginal — read the SPLAY rows, not the tilt row.

**Answer to the decisive question:** *no mid-run checkpoint had a better stance that later
degraded.* The leg splay was **monotonically getting better** for the whole run; the head wobble was
**constant**. So the leg-splay part is "would improve with more iterations" (slowly), and the head
part is neither — it is stationary. The one defect with a steep live slope at the end is the
**L/R leg asymmetry**.

---

## What would more training buy?

Extrapolations are from the fitted trends (`saturation_fits.json`, `checkpoint_trajectory.json`);
"another 5 000 iterations" = +246 M env-steps = +67 % over this run's total.

**1. Fitted saturating reward/error curves (exponential, `value = c + A·exp(-t/tau)`, fitted on the
200-step-smoothed series from iteration 1 500):**

| curve | value @15000 | fitted asymptote `c` | `tau` (iters) | prediction @30000 | prediction @50000 |
|---|---|---|---|---|---|
| `Train/mean_reward` | 106.48 | **112.93** | 7 735 | 112.00 | 112.86 |
| `Episode_Reward/head_pose_tracking` | 1.6986 | **1.7492** | 10 060 | 1.7378 | 1.7476 |
| `Episode_Reward/pose` (leg pose) | 0.01312 | **0.01345** | 4 648 | 0.01344 | 0.01345 |
| `Metrics/twist/error_vel_xy` | 0.4153 | **0.4071** | 4 998 | 0.4075 | 0.4071 |
| `Metrics/twist/error_vel_yaw` | 2.6129 | **2.3035** | 8 500 | 2.3565 | 2.3085 |

Goodness of fit R² = 0.97 (yaw), 0.97 (xy), 0.97 (reward), 0.84 (head_pose_tracking).

**2. Refutation of the "~12 200 more iterations to close a 20 % yaw gap" estimate.**

I reproduced their arithmetic and then broke its premise. A linear fit over iterations
[12 000, 15 000) gives `error_vel_yaw` slope **−0.042 rad/s per 1 000 iters**, and
(2.566 − 0.8 x 2.566) / 0.042 x 1 000 = **12 220 iterations** — so the number is arithmetically
right, but the model is wrong:

* The curve is **not linear**; it is a saturating exponential with **`tau ≈ 8 500` iterations and an
  asymptote of 2.3035 rad/s** (R² = 0.97). The linear slope over the last 3 000 iterations is itself
  2.4x smaller than over [8 000, 10 000) (−0.094) — the deceleration is the signal.
* "Closing a 20 % gap" from 2.566 rad/s means reaching **2.053 rad/s**, which is **below the fitted
  asymptote**. Under the measured dynamics that target is *unreachable at any iteration count*:
  even at iteration 50 000 the model predicts 2.309 rad/s.
* Even the generous linear reading fails: at −0.042/1000 the floor is never hit because the slope
  keeps collapsing.

So: the limit is not the budget, and it is not reachable by budget. The reward-weight diagnosis
(`track_angular_velocity` weight 2.0, Gaussian `std = 0.707` against a command range of only
±1.0 rad/s, measured mean error 2.57 rad/s — the Gaussian is essentially flat over the error the
policy actually produces) is the correct direction. **The `wz` command range is ±1.0 rad/s, so the
steady-state yaw error (2.57 rad/s) is ~2.6x the largest command the robot is ever asked for**: the
policy is not "not trying hard enough", it is at the optimum of a reward whose tracking term has no
gradient in the region it lives in.

**3. What a further 5 000 iterations would actually buy (checkpoint-trend extrapolation, late window
[8000, 14999):**

| metric | now (14999) | slope /1000 it | predicted @ +5000 | verdict |
|---|---|---|---|---|
| leg hip_pitch \|L−R\| asym | 23.4° | −3.55 | **~5.6°** | **big real gain** |
| leg knee \|L−R\| asym | 22.0° | −3.54 | **~4.3°** | **big real gain** |
| leg hip_roll \|L−R\| asym | 9.2° | −0.54 | ~6.5° | modest gain |
| SPLAY abduction | 4.62° | −0.26 | ~3.3° | modest gain |
| foot separation | 68.7 mm | — | ~60 mm | modest gain |
| trunk tilt | 3.1° | (whole-run −0.20) | ~2.1° | modest, inside 69 % noise |
| **toe-out (both legs)** | **16.3–17.2°** | **L −0.21 / R +0.36** | **~16.5°** | **no gain — converged attractor** |
| **neck_amp** | 18.0° | **+0.17** | ~18.9° | **no gain — if anything worse** |
| **d_neck_pitch p2p** | 62.3° | flat | flat | no gain |
| `head_pose_tracking` reward | 1.6986 | — | ≤ 1.7378 (asymptote 1.7492) | **+3.0 % ceiling, ever** |
| `pose` reward | 0.0131 | — | ≤ 0.01345 | already saturated |
| `Train/mean_reward` | 106.5 | — | ≤ 112.9 | +6 % ceiling, ever |

---

## Alternative hypothesis: reward equilibrium, not budget

I only have to answer the empirical half (Agent A owns the reward math); the trajectory evidence:

* **Is the stance at a local optimum of the reward?** Yes, by the convergence argument: two hip_yaw
  joints starting 8.5° apart both move to +16.5° and stop. Under any monotone "get closer to HOME"
  pressure they would not do that. Corroboration in reward units: `pose` = +0.0154 out of a total of
  ~106 per-episode reward, i.e. **0.014 % of the reward mass**, and the fitted asymptote for that
  term (0.01345) is *below* its recent average — moving the legs back toward HOME is worth
  ≈ 0.015 reward, which is 1/7000th of the episode return. Compare `upright` (1.73, 1.63 %) and
  `air_time` (1.43, 1.35 %): the leg pose term is 100x smaller than the terms around it. Nothing in
  the objective pays for closing the duck stance.
* **Is the neck oscillation priced at all?** Per AGENTS.md the design intent is that it is
  *deliberately* not: "a 38 %-of-body-mass head MUST oscillate while walking; a tight instantaneous
  head-tracking std taxed walking so hard the policy stood still. Price only the escapable part —
  e.g. L1 on a 1 s EMA charges DC bias and lets oscillation cancel." And that is exactly what the
  data shows: `head_pose_bias` (the 1 s-EMA DC penalty, weight ramped to 3.0) settled at −0.26 and
  has been **constant to ±0.004 since iteration 2 500**, while the AC oscillation stayed put. The
  AC part is unpriced *by construction*, and the policy has correctly stopped optimising it.
  Corroborating: `action_rate_l2` and `mean_action_acc` both improved strongly over the run
  (mean |action acceleration| 1.23 → 0.25), so the neck wobble is not action jitter that a
  smoothness term would catch — it is a required whole-body motion at the 2.17 Hz gait frequency.
* **The one place where more training demonstrably helps is the L/R asymmetry** — and that is the
  part of defect (a) that a longer run really would fix (predicted 23.4° → ~5.6°). So the answer is
  not "budget is irrelevant"; it is "budget is irrelevant for the posture *targets*, and relevant for
  the *symmetry* the optimiser is still resolving".

---

## Confidence and open questions

**Confidence:**
* *High* — the training-history extraction (two independent sources agreeing to 1e-7; 44 scalars x
  15 000 rows; `tfevents_scalars.csv`). Run facts (iterations, envs, wall clock, 66 checkpoints) are
  read directly from the run's own params and log.
* *High* — the SPLAY trajectory (abduction 2.3x monotone; foot separation 31 % monotone; toe-out
  \|L−R\| gap 8.53° → 0.86° monotone): all far outside the measured repeat band, R² 0.82–0.91.
* *High* — the reward-mass arithmetic (`pose` = 0.014 % of episode return) and the flatness of
  `head_pose_tracking` / `head_pose_bias` / `pose` from iteration 2 500 onward (R² 0.013–0.04).
* *High* — the refutation of the "12 200 iterations" estimate: the fitted asymptote (2.3035) sits
  above the target (2.053), so no iteration count reaches it (R² = 0.97).
* *Medium* — the *magnitude* of what more training buys on the leg asymmetry (23.4° → ~5.6°): it is a
  linear extrapolation of a monotone 5-point trend with R² = 0.82, inside a noisy eval. Direction is
  safe, magnitude is ±2x.
* *Low / no claim* — the head wobble's absolute numbers and any statement that it "got worse":
  the 15–119 % repeat spread on the same weights makes every head scalar individually unreliable, and
  the world-frame head metrics are additionally contaminated by the 66–174° trunk yaw drift inside the
  12 s window. **The defensible claim is "flat across 13 000 iterations", not "worse".**

**Open questions / what would settle it:**
1. **Multi-env, fixed-seed evals.** The single biggest weakness here is the eval noise. The same
   policy scoring 3.1° and 5.9° trunk tilt in two runs means all posture claims should be re-run with
   `eval_onnx_bam.py`'s harness at, say, 64 envs x 3 s with a pinned seed and averaged, before any
   posture number is treated as a deployment fact. That is ~10 minutes of GPU and would upgrade most
   *medium* rows above to *high*.
2. **The yaw-tracking asymptote is measured on a stochastic policy.** `Metrics/twist/error_vel_yaw`
   is logged from the training rollouts (mean_std 0.127), so part of the 2.30 rad/s floor is
   exploration noise. A deterministic re-measurement of the *same* commands would separate
   "policy bias" from "exploration noise" — this matters because it decides whether the yaw defect is
   a reward-shape problem (weight 2.0, std 0.707 vs ±1.0 command range) or partly a noise artefact.
3. **Is +16.5° hip_yaw actually the cheapest stance?** The convergence argument proves it is an
   attractor of *this* objective, not that it is mechanically necessary. The distinguishing test:
   freeze hip_yaw at 0 in the sim and measure the cost in `upright` + `air_time` + `track_linear_velocity`.
   If the zero-toe-out stance costs ~0 reward, the attractor is pure under-specification; if it costs
   real reward, the duck stance is buying stability and the reward is right to allow it.
4. **Was 2 048 envs the right call?** Halving `num_envs` halves the per-iteration signal quality at the
   same iteration count; the run compensated with 2.5x the reference iteration count to reach the same
   env-step budget. Nothing in the curves suggests this hurt (fps was flat at 47 k), but a 4 096-env /
   6 000-iteration control at 590 M env-steps would be the clean A/B on "same budget, different shape".
5. **Curriculum pacing at 1 500 and 2 000.** The step-downs are real and reproducible in the log;
   what is unresolved is whether spacing those two stages further apart (e.g. 1 500 → 2 500) leaves the
   policy at a better head-pose equilibrium at the same wall clock. That is a training experiment
   (out of scope here: no training runs were launched).
