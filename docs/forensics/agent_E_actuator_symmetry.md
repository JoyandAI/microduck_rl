# Agent E — Actuator / servo-identification forensics: can the HD-1910 data explain a *systematic veer*?

Scope: `vendor/bam/bam/params/hd1910/*`, `hd1910_calibration/*`, the BAM actuator code, the
real-servo register dumps. NOT scope: reward, curriculum, model structure (agents A–D).
All analysis static (file reads + pure-python/numpy). No GPU, no Warp sim was run.

---

## VERDICT — **低 (low confidence)** that the servo/actuator data is the cause of the systematic veering

**Strongest single piece of evidence:** the training pipeline feeds **one** parameter file to
**all 14 joints**, and the BAM friction law that consumes it is **exactly mirror-symmetric and
exactly sign-symmetric**: I evaluated the deployed m5 budget at the walking operating point and
`fl(+dq) − fl(−dq) = 0.0 N·m` *identically* (frictionloss depends on `|dq|` only), and under the
mirror transform `(q,dq,τ) → (−q,−dq,−τ)` the budget and the control law are invariant
(`vendor/bam/bam/mjlab.py:502-515`, `:736`; `vendor/bam/bam/model.py:143-160`). There is also **no
per-joint DR** anywhere: `friction_scale`/`kp_scale`/`kd_scale` are all `(num_envs, 1)`
(`src/mjlab_microduck/actuator/friction_dr_bam.py:34,47`,
`vendor/bam/bam/mjlab.py:352-355`), and `ENABLE_KP_RANDOMIZATION = ENABLE_KD_RANDOMIZATION = False`
(`src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py:190-191`). **A single, symmetric
parameter set applied identically to every joint cannot, by itself, generate a LEFT ≠ RIGHT
yaw bias.**

This is consistent with the measurement you supplied: **the veer reproduces on the OFFICIAL
XL330 hardware** (our recipe, `kp_fw=200`, XL330 m6 params: yaw bias −4.6 ± 1.8 °/s, path
efficiency 0.679), while the official policy in that same env reaches 0.883. The XL330 and
HD-1910 parameter sets differ in *every* physical channel (kt 0.346 vs 0.692, R 2.49 vs 3.75,
friction 3–6× different, kp 200 vs 32, vin 6.5–8.2 vs 4.75–5.25) — if the veer survives that total
parameter swap with the same sign and roughly the same magnitude, **the actuator data is not the
dominant term.** I do **not** point the other way; there is no evidence in the servo data that
would overturn your XL330 result.

What the actuator data *can* still do (secondary, non-dominant): shift where the gait's
limit cycle sits, and — on the **real robot** — leave the policy blind to per-unit servo
variation, because per-unit variation is neither measured nor randomized. See §6/§7.

---

## 1. Is one parameter file applied to all 14 joints? — **YES, one file, no per-side/per-joint variation**

| Evidence | Value |
|---|---|
| `src/mjlab_microduck/robot/microduck_constants.py:127-135` | `_BAM_ACTUATOR_KWARGS = dict(motor_name="hd1910", model="m5", target_names_expr=(r"^(?!passive_).*",), …)` — **one** actuator cfg, **one** regex matching every servo joint |
| `…/microduck_constants.py:167` | `actuators = FrictionDRBamActuatorCfg(**_BAM_ACTUATOR_KWARGS)` — all robot cfgs (`:193-221`, `:230+`) reuse the same object |
| `vendor/bam/bam/mjlab.py:217` | `self._bam_model: Model = load_model(cfg._resolved_json_path)` — **one** Model instance; `BamActuatorCfg.__post_init__` resolves a single JSON path (`:150-161`) |
| `logs/rsl_rl/velocity/2026-09-10_18-43-53_velocity/params/env.yaml:127-153` | the run's *recorded* config: exactly one actuator, `motor_name: hd1910`, `model: m5`, `json_path: null`, `kp_fw: 32.0`, `friction_scale_range: null` |
| `src/mjlab_microduck/actuator/friction_dr_bam.py:47` | `base * fs` with `fs` shape `(N,1)` → **per-env**, broadcast over joints; friction DR can never differ L vs R within an environment |
| `vendor/bam/bam/mjlab.py:352-355, 456-459, 701` | `kp_scale`/`kd_scale` are `(num_envs, 1)` → same |
| `src/mjlab_microduck/tasks/mdp.py:3365-3393` | `randomize_bam_friction` samples `scale_range` per **env**, not per joint |

**Consequence:** the actuator model is a *pure, exactly mirror-symmetric* function; it contributes
**no** LEFT/RIGHT asymmetry to the sim. So the sim-side asymmetry must come from the model
geometry/inertia/contacts, from DR that is asymmetric by construction (e.g. CoM), or from the
**learned policy** — not from the servo identification. It *can* still be the source of a
dimensionless **rate-dependent** bias (§2/§4).

## 2. Is the friction model direction-dependent? — **NO, not in the sign-of-motion sense.** The name "directional" is about motor-vs-external torque *source*, not about `sign(dq)`

`vendor/bam/bam/model.py:143-160,163-206` and its mjlab twin `vendor/bam/bam/mjlab.py:496-531,736`:

```
stribeck_coeff = exp(-(|dq| / dtheta_stribeck)**alpha)          # mjlab.py:736  → |dq| only
fl = friction_base + sc*friction_stribeck
   + |ext*load_friction_external − mot*load_friction_motor|    # mjlab.py:502-506
   + sc*|ext*lf_external_stribeck − mot*lf_motor_stribeck|     # mjlab.py:509-515
```
`fl` is written into `dof_frictionloss` (`mjlab.py:587`), which MuJoCo clips into `[−fl, +fl]`
(`vendor/bam/bam/model.py:20-23`) — a **symmetric** dry-friction constraint.

### Quantified (pure-python evaluation of the exact formulas above)

Walking joint speeds actually used (measured sim joint velocities, `walk_angles.csv`, 8 s,
0.15 m/s, `np.gradient` over `time`): |dq| median per joint **0.08–1.83 rad/s**, p99 **0.3–3.4**,
max **0.3–3.5 rad/s** (knee p50 = 1.80/1.83, ankle p50 = 1.29/1.13, hip_pitch p50 = 0.58/0.81).
Bench excitation spans the same band: |speed| p50 = **0.461 rad/s** over all 72 logs
(`hd1910_calibration/bench_combined72_processed/*`, `dt=0.005`).

Friction budget, N·m, at |dq| = 0.5 rad/s (stall-torque reference = `vin_max·kt/R` =
**0.97 N·m** for the deployed HD-1910 m5; `train_velocity_official_align2.log:52`
`force_limit=±0.97Nm`):

| params | mot/ext (N·m) | driving | back-driven | Δ (that pair) | Δ / stall |
|---|---|---:|---:|---:|---:|
| **deployed** `hd1910/m5.json` | 0.05 / 0.05 | 0.0402 | 0.0478 | 0.0076 | 0.8 % |
| deployed | 0.10 / 0.10 | 0.0666 | 0.0819 | 0.0153 | 1.6 % |
| deployed | 0.20 / 0.20 | 0.1195 | 0.1500 | 0.0306 | 3.2 % |
| **trained-walk** `hd1910/m5_alt.json` | 0.10 / 0.10 | 0.0412 | 0.0670 | 0.0257 | 2.6 % |
| trained-walk | 0.20 / 0.20 | 0.0676 | 0.1190 | 0.0514 | 5.3 % |
| `fit_m5_72.json` (kd=40) | 0.20 / 0.20 | 0.0178 | 0.0443 | 0.0265 | 2.7 % |
| **XL330 m5 (official)** | 0.20 / 0.20 | 0.0373 | 0.0791 | 0.0418 | 4.3 % (of 1.13 N·m = 3.7 %) |

**Direct answer to the +/- asymmetry question: 0.000 N·m, i.e. 0 % of stall torque.**
`fl(|dq|=+0.5) − fl(|dq|=−0.5) = 0.0` exactly (reproduce: the two calls differ only in the sign of
`dq`, which enters solely through `abs(dq)`). There is **no** persistent sign-dependent friction
term in this model at all.

What *does* exist is a **load-direction** asymmetry (driving vs back-driven gearbox): 0.008–0.051 N·m
= **0.8–5.3 % of stall torque** for HD-1910, 3.7 % for XL330. It is applied **identically to both
legs**, so it cannot make left ≠ right; at ≤5 % of stall torque it is also small compared with the
per-step torque swings of a 0.2 m/s gait. It is a legitimate mechanism for changing limit-cycle
*shape*, not for a limb bias.

**Rate dependence (Stribeck), deployed m5, mot=ext=0.10 N·m:**

| \|dq\| (rad/s) | 0.001 | 0.1 | 0.2 | 0.5 | 0.79 | 1.0 | 2.0 | 3.0 | 8.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fl (N·m) | 0.0783 | 0.0782 | 0.0777 | 0.0666 | 0.0386 | 0.0221 | 0.0156 | 0.0156 | 0.0156 |

Same table for XL330 m5: 0.0396 → 0.0380 (@0.2) → 0.0247 (@0.5+) — flatter, and with
`alpha=8.31/dtheta=0.259` vs HD-1910's `alpha=3.46/dtheta=0.790`
(`vendor/bam/bam/params/xl330/m5.json`, `.../hd1910/m5.json`). **The HD-1910 fit places the
Stribeck knee at 0.79 rad/s with a soft exponent, i.e. right in the middle of the walking speed
band** (p50 0.08–1.8 rad/s): at 0.2 rad/s the budget is 5× the high-speed value. That is a genuine
rate-dependent, symmetric bias on gait *shape* (friction-dominated at the low-speed parts of every
step) — but it is the same for both legs, and it is a lower-speed-richer behaviour that applies
equally to the XL330 policy that does not veer.

## 3. `q_offset` and `command_delay` — **both are INERT in RL training**

| Param | Value | Consumed where | Effect in training |
|---|---|---|---|
| `q_offset` | **0.008087982 rad = 0.463°** (`vendor/bam/bam/params/hd1910/m5.json:5`, `characterization.json:230`) | **only** the identification rig: `vendor/bam/bam/mjlab.py:823-825,984` (`Pendulum(...).build_spec(..., q_offset=…)`) and `vendor/bam/bam/simulate.py:75-86`, `mujoco.py:253` | **None.** `BamActuator` (the training actuator, `mjlab.py:172-750`) never reads `q_offset` — full grep: no other consumer exists. The training-time encoder error is instead `ENABLE_ENCODER_BIAS` (`microduck_velocity_env_cfg.py:196`), sampled **zero-centered per joint** → cannot create a systematic common offset, and cannot create an L/R bias either (it is zero-mean) |
| `command_delay` | **0.02534 s** (`m5.json:20`) | only `Simulator.rollout_logs(command_delay=True)` — an MAE-evaluation option (`mjlab.py:798-820,903-904`), used by `bam/mae.py:131` | **None directly.** Training models latency as `delay_min_lag=4 … delay_max_lag=7` sim substeps = 20–35 ms (`microduck_constants.py:147-165`; `env.yaml:135-139`), randomized per env |

So: the 0.46° common offset **cannot** bias the trained gait (it isn't in the sim), and the delay is
already realized as a randomized 20–35 ms lag. Note the offset *would* be asymmetric if it were
applied identically to mirrored joints (`q_L = −q_R` makes `+δ` appear as `−δ` mirrored) — so this
is a real hazard **only if** anyone ever wires `q_offset` into training; it is not wired today.

## 4. HD-1910 vs XL330 on the properties that matter for a *straight* gait

| Property | XL330 era (official) | HD-1910 (current) | Source |
|---|---|---|---|
| `kp_fw` | **200** (`git show 79a993a^:src/mjlab_microduck/robot/microduck_constants.py`, line 124) | **32** (`microduck_constants.py:136`) | 6.25× softer |
| control law | `duty = Δq·kp·eg` | `duty = Δq·kp·eg/p_scale + (−dq)·kd·eg/d_scale`, `eg=0.163`, `p_scale=1`, `d_scale=4` (`vendor/bam/bam/feetech/actuator.py:695-698`) | effective P slope = **5.216 duty/rad** for HD-1910 vs `200·0.1587`=31.7 duty/rad for XL330 (`vendor/bam/bam/dynamixel/actuator.py:93-95`: `kp=400, error_gain=XL330_COUNTS/2π`, with `kp_fw=200` override) → **~6× softer position stiffness in volts/rad** |
| peak torque | `kt/R·vin_max` = 0.346/2.494·8.2 = **1.13 N·m** | 0.692/3.75·5.25 = **0.97 N·m** | `vendor/bam/bam/mjlab.py:415`; printed in `train_velocity_hd1910.log:52` (`0.97`/`1.03` depending on kt) |
| `max_velocity` | n/a (firmware ramp) | 8.0282 rad/s, **only used if `use_rate_limiting=True`, which is False** (`feetech/actuator.py:469,527,659-685`) | mode-4 is pure PD |
| friction @|dq|=0.5, τ=0.1 | 0.0247 N·m | 0.0666 (deployed) / 0.0412 (trained) | §2 table |
| Stribeck knee | 0.259 rad/s, α=8.31 | **0.790 rad/s, α=3.46** | `xl330/m5.json`, `hd1910/m5.json` |
| delay | 3–6 lags | 4–7 lags (20–35 ms) | `microduck_constants.py:164-165` |
| backlash | modelled as ±1° hinges (`-Backlash-` tasks) | same wrapper | `microduck_constants.py:230+` |

**Is HD-1910 fundamentally more prone to asymmetric limit cycles / stick-slip at walking speeds?**
*Partially, yes, on paper:* it is 6× softer (longer settling, larger steady-state error under load:
`Δq = τ/(5.216·5 V)·…`), and its identified Stribeck knee (0.79 rad/s) sits inside the walking band,
whereas the XL330's knee (0.26 rad/s) is below most of it — so the HD-1910 joint is friction-shaping
its motion over more of each step. **But** this mechanism is symmetric between legs and would need an
additional L/R seed to produce veer; and measuring it against the actual counter-evidence, the XL330
policy **with our recipe also veers** while the official policy on the same XL330 params does not.
⇒ Stick-slip/soft-kp can plausibly change *how much* the gait wobbles (± yaw rate scatter), it is
**not** a viable explanation of a *systematic* bias. **Cannot distinguish** is the honest verdict
between "soft kp amplifies an existing asymmetry" and "no effect" without the sim A/B in §8.

## 5. Was the identification good enough for the D channel and the friction? — **D: no (documented). Friction asymmetry: also NOT individually identifiable — different fits disagree by 10×+**

Residuals that are actually reported:

| Fit | val MAE (rad) | pinned `kd` | source |
|---|---:|---:|---|
| `fit_m5_72` (72 logs, kd=40) | 0.03209 | 40 | `hd1910_calibration/fit_m5/mae_report.md` |
| `fit_m5_kpfix` (kp 10272→32 bug fixed) | 0.00823 | 0.346 | `hd1910_calibration/fit_kpfix/eval_raw_kpfix.json`, `mae_report_kpfix.md §3` |
| final CMA-ES batch, deployed | **0.0206** | 40 | `vendor/bam/bam/params/hd1910/characterization.json` (`validation_mae_rad.m5`) |
| deployed-config comment (re-measured) | **0.0254** | — | `microduck_constants.py:129-134` (states the 0.00823 figure is **not reproducible** under any `reset_period`) |

**D channel — NOT identifiable, by the repo's own statement.**
`characterization.json` `d_scale.note`: *"D 强度扫频(kd=0.74/5/160 ↔ D 系数 0.03/0.20/1.63/6.52)
在 72 条上独立验证 **MAE 极差仅 0.0014 rad** → 台架数据无法分辨 D 强度 … 要定论必须做硬件 T5"*;
`kd.note`: *"有效 D 增益未测(回归 D 系数≈0.03 vs 模型 1.66)"*; `pending_tests[0]` = T5.
The free fit then produced `kd = 192.676` (`characterization.json:232`), which turned the servo into
a brake at 0.127 rad/s — this is the known, already-diagnosed actuator bug
(`docs/policy_failure_analysis.md:56-98`, `docs/training_config_velocity_hd1910.md:11-24`).

**Friction asymmetry — same disease.** Four m5 fits from the same 72 logs disagree wildly in the
directional terms while their *totals* stay close (my evaluation, same formula, τ=ext=0.1, |dq|=0.5):

| m5 fit | `load_friction_motor` | `load_friction_external` | `lf_motor_stribeck` | `lf_external_stribeck` | `dtheta` / `alpha` | tot. fl @0.1/0.1 |
|---|---:|---:|---:|---:|---|---:|
| `m5_alt` (= **trained walk**) | 0.1940 | 0.1285 | 0.2557 | **4.8e-5** | 0.9995 / 1.963 | 0.0412 |
| `fit72/fit_m5_72` (kd=40) | 0.0550 | 0.0343 | 0.2837 | 0.2224 | 0.3028 / 1.322 | 0.0412 |
| `fit_kpfix/fit_m5_kpfix` | = m5_alt (identical file content) | | | | | 0.0412 |
| **deployed** `hd1910/m5.json` | **0.0164** | 0.0372 | **0.6971** | 0.0737 | 0.7896 / 3.465 | 0.0666 |

`load_friction_motor` spans **0.0164 → 0.194 (12×)**; `lf_external_stribeck` spans
**4.8e-5 → 0.2224 (~4600×)**; `q_offset` flips sign between fits (−0.0229 … +0.0200); the `_alt`
batch's `command_delay` is ~5e-6 s vs 0.025 s in the deployed file. The repo's own note confirms the
stribeck family is unsupported: *"Stribeck 项不被数据支持 … m4/m5/m6 的 Stribeck 项同样收敛到 ~0"*
(`mae_report_kpfix.md §5`) — yet the **deployed** file puts the *entire* load friction into the
Stribeck term (`lf_motor_stribeck = 0.697` vs `load_friction_motor = 0.016`). The residual budget is
what the data constrains; the parameter *split* between the Coulomb and Stribeck branches is **not
identifiable**, and the `/d_scale`-style scaling of the whole anisotropy is a fit-noise channel.

⇒ **The friction-asymmetry parameters are fit noise in the same sense as `kd`.** They are a hazard
for *where the gait sits*, not evidence about left vs right.

**Second, unrelated-to-veer identification defect worth recording:** the walk policy the user is
testing was trained with `kt = 0.7358` (datasheet) and the **superseded** friction batch
(`train_velocity_hd1910.log:52`: `kt=0.7358 … friction_base=0.0149 friction_viscous=0.0056`, which is
`m5_alt`), not with the later bench-measured `kt = 0.692` / `friction_base = 0.0136`
(`train_velocity_official_align2.log:52`). `kt` was later measured at 0.646–0.692, i.e. **6–12 % low**
(`characterization.json` `kt.evidence`: 2 runs / 48 legs, ke 0.673–0.699, R²=0.991–0.998, 93–95 % of
datasheet). A 6 % torque-constant error is a *uniform* gain error on all 14 joints — again no L/R
asymmetry.

**Documentation drift (do not quote the docs as ground truth):** `characterization.json`'s
`bam_models.deployed_params.kd = 192.676`, and `policy_failure_analysis.md:11-13` say the on-disk
`m5.json` contains `kd=192.68` — but the on-disk and committed file has **`kd = 0.346`**
(`vendor/bam/bam/params/hd1910/m5.json:17`, committed in `d0ae701`), i.e. the D-slope 0.0141 duty/(rad/s)
that matches the bench regression. The dedicated walk run (`2026-09-10_18-43-53`, → `walk_hd1910_15000.onnx`,
15000 iters) started 18:43 on 09-10, i.e. **before** the 18:51 param rewrite, and its `env.yaml`
records no parameter *values* at all (`policy_failure_analysis.md:168-182`) — the run→param binding is
inferred from the `[BamActuator]` print in the log, not recorded.

## 6. Does any bench data show a LEFT/RIGHT servo-pair mismatch? — **No such data exists**

- The whole HD-1910 identification set is **one single physical servo**, bench bus **ID = 1**
  (`docs/feetech_hd1910_pendulum_quickstart.md:23,82`; `characterization.json` `pending_tests`:
  *"多舵机一致性"* is still open). 72 logs are 2 arm lengths × 3 tip masses × 4 trajectories × 3 reps of
  that one unit (`hd1910_calibration/dataset_manifest.json`). **No paired L/R test was ever run.**
- The only multi-servo register dump in the repo is from the **previous (HLS-2909) swap**
  (`hls_registers_id{10,11,12,13,14,20,21,22,23,24,30,31,32,33,34}.json`, 2026-09-07). Tabulating
  the 14 units: `kp=32, kd=32, ki=0, reg24(min force)=90, reg26/27(dead zone)=1, reg28(Iprot)=300`
  **identical for all 14**; only `reg31` (electrical position offset) differs, in a mirrored
  left/right pattern (`+1896, +1705, +1886, +2010, +1119` vs `−32737, −30802, −30789, −31134,
  −31882, −30964, −31175, −31042, −30827`) — that is per-joint zero calibration, not a defect.
  Supply readback 124–126 (i.e. 12.4–12.6 V) with no side pattern.
- Therefore: **"is there a real L/R actuator mismatch on the robot?" cannot be answered from this
  repo's data.** It is unmeasured, not measured-and-absent.

## 7. Why this still matters even though it is not the dominant cause

1. **Per-unit variation is unmodelled and untrained.** 14 distinct servos, one benched. Per-unit
   friction/zero-offset spread is real on hobby servos and is exactly the kind of thing that a
   soft-kp (32) position loop converts into a standing/gait symmetry error. It is invisible in the
   sim because (a) one shared JSON, (b) all DR is per-env, not per-joint (§1).
2. **All the asymmetry-relevant DR is zero-centered** (encoder bias, IMU misalignment, pushes) —
   `AGENTS.md` itself states IMU DR "CANNOT compensate a systematic mounting bias".
   *Speculation, labelled:* if the real robot's yaw bias were actuator-driven, it would most likely
   be a per-servo zero/friction asymmetry — and nothing in training teaches tolerance for that.

## 8. Cheap experiments that WOULD settle the actuator question

**Needs hardware (cheap: ~1 h, no GPU):**
- **E1 — per-servo pair test (the decisive one).** Same command trajectory, same bench, run **both
  ankle servos (and both hip_rolls)** back-to-back; log the measured angle and `duty`/`current_A`
  telemetry like the 72-log set did (fields already exist: `bench_combined72_processed/*.json`
  `telemetry`). Report `MAE_L−R` in rad and the *signed* duty difference at the same operating point.
  Also read `reg31` (zero offset) and reg 21/22/50/51 for all 14 units. **If the L/R pairs differ by
  < ~0.1° and their duty traces are within a few %, per-unit actuator asymmetry is dead as the
  cause — and nothing else in the servo data can produce a systematic veer.**
- **E2 — T5 (already listed as pending):** step response at `kd = 16 / 40 / 100` on hardware, to fix
  `d_scale`; and a **static hold at the gait's low speeds** to check stiction onset under load
  (`stiction_onset_offset` was measured unloaded only, 5 LSB, `characterization.json`).

**Needs a GPU (sim A/B, no real robot):**
- **E3 — per-side actuator DR at 0 mean isn't enough; test a *symmetric* perturbation.** Re-evaluate
  the *same frozen-command* policy with per-joint (not per-env) friction/gain draws, and with a
  **fixed** 5 % left-vs-right friction or zero-offset mismatch injected on one side only. If yaw bias
  appears ~linearly with the injected mismatch, the transfer chain is actuator-sensitive; if the bias
  is already −4.6 °/s with zero injected mismatch (your XL330 result says it is), the actuator is
  not the seed.
- **E4 — parameter A/B:** re-run the 8 s frozen-command eval on `walk_hd1910_15000.onnx` under
  (a) trained params, (b) deployed `m5.json`, (c) `m5_alt`, (d) XL330 m6 — measure **signed yaw-rate
  bias and net yaw**, not just tracking error. This is the single measurement that tells us whether
  any friction sub-model choice moves the *bias* at all. Expected from §2: it cannot move a
  systematic bias, only the scatter — which is exactly what the experiment should falsify or confirm.

---

### One-line summary
One shared, exactly mirror-symmetric parameter set (sign-asymmetry of the friction budget = **0 N·m**;
load-direction asymmetry only 0.8–5.3 % of stall torque), `q_offset`/`command_delay` inert in
training, no per-joint DR, no multi-servo data — so the servo identification **cannot** create a
LEFT/RIGHT bias, and the D/friction-anisotropy channels that *are* fit noise (12×–4600× disagreement
between fits on the same 72 logs) are a gait-shape hazard, not a veer mechanism. **低 confidence**
that actuator/servo data explains the systematic veering; consistent with the veer reproducing under
the completely different XL330 parameter set.
