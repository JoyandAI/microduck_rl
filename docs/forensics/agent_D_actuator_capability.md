# Agent D — Is the observed posture defect physically avoidable, or is the policy at the hardware limit?

Scope: quantify what the Feetech HD-1910-C001 servos on the regulated **5.0 V** rail can
actually deliver, and decide for each reported defect whether it is a hardware limit or a
policy choice. Everything below is either read out of a file (with the line cited) or
measured/computed in this session; scripts and raw data are in
`docs/forensics/artifacts/`.

> **Revision note.** All line numbers were taken from the working tree as read at
> 2026-09-11 12:15–12:20. A concurrent agent was editing
> `src/mjlab_microduck/tasks/mdp.py` and `microduck_velocity_env_cfg.py` while this analysis
> ran: those two files changed *after* every measurement here, and no line-number citation in
> this document refers to them being frozen. Nothing this analysis depends on was modified —
> `vendor/bam/bam/params/hd1910/m5.json`, `hd1910_calibration/params_recommended_m5.json`,
> `src/mjlab_microduck/robot/microduck_constants.py`, `src/mjlab_microduck/robot/microduck/robot_walk.xml`
> and `walk_hd1910_15000.onnx` are all unmodified.

Base rollout for all measurements: `walk_hd1910_15000.onnx`, 12 s at 50 Hz, twist
`vx=+0.20`, head-pose command frozen at exactly zero, 1 env, faithful BAM/HD-1910 model
(`scripts/gait_forensics.py` → `artifacts/gf_walk_hd1910_15000_12s.{csv,json}`).
Measured: gait 2.17 Hz (not 2.25 — the brief's 2.25 Hz run is a slightly different
warm-up/seed of the same behavior), trunk z 95.5 mm, 0 falls.

---

## Verdict

| Defect | Verdict | Deciding number |
|---|---|---|
| `neck_pitch` oscillates std 15–16.5°, p2p 46.6–57.9°, at gait frequency, while the head-pose command is zero | **FREELY AVOIDABLE** (policy-commanded, *not* hardware-limited) | Clamp the neck+head actions to HOME in the same env/gait: `neck_pitch` std **16.51° → 1.75°** (p2p 57.9° → 5.81°), duty saturation **64.5% → 0%**, peak |τ| **0.996 → 0.424 N·m** (46% of the 0.923 N·m stall), robot still walks at **0.172/0.200 m/s**, 0 falls in 12 s |
| `head_pitch` oscillates std 5.9–7.5° | **FREELY AVOIDABLE** | Same clamp: std **7.45° → 0.61°** (p2p 33.8° → 2.07°); neck peak |τ| falls to 0.424 N·m (46% of stall), duty saturation to 0%; head attitude error vs HOME drops to **0.02°** (pitch) |
| `hip_yaw` "held" at +16° toe-out, ~5° abduction | **POSE: freely avoidable (and beneficial). The way it is produced is NOT: partially avoidable / actuator-saturated** | Static hold torque at any `hip_yaw` angle = **0.000000 N·m** (vertical axis) — the +16° pose is free. But the measured +16° is the *mean of a saturating oscillation*: command std 28.4°, command outside the joint range **52.3%** of the time, joint within 1° of a range limit **40.7%** of the time, duty saturation **87.5%**, mean |τ| **0.858 N·m = 93% of stall** |
| Whole-body actuator margin during walking | **AT THE LIMIT** | All 10 leg joints: duty saturation 3–87.5%, mean |τ| 0.345–0.858 N·m = **37–93% of the 0.923 N·m stall**; per-joint tracking error std 5.0–24.8° |

**Headline.** The neck wobble is **not** a hardware limit. Holding the 289 g head still at
HOME during a 2.4 Hz step costs **0.031 N·m = 3.4% of stall (29× margin)**, and the same
policy in the same gait does exactly that — to **1.75° std / 5.81° p2p** — the moment it is
no longer allowed to command the head. What the servo *cannot* do is follow the ≈19–21°,
2.17 Hz neck oscillation the policy actually commands: that needs a duty cycle of **1.03
(>100%)**, so the actuator saturates 64.5% of the time and lags its own command by
**68.5°**. The 289 g head therefore also nods **22.3° (p2p 86°)** relative to its own trunk,
against a *commanded* 22.9° — the policy asks for the full amount and gets it in amplitude,
late by 68.5°.

---

## Head load model

Source: `src/mjlab_microduck/robot/microduck/robot_walk.xml`, compiled by MuJoCo at the
HOME configuration (`artifacts/head_load_model.py` → `artifacts/head_load_model.json`).
Total articulated mass from the same compile: **779.243 g** (brief says 779.2 g ✔).

Kinematic chain (all bodies distal of a hinge rotate with it):

| MJCF lines | body | parent joint | mass |
|---|---|---|---|
| `179`, `181`, `182` | `neck` | `neck_pitch` (range −90…+60°) | 39.841 g |
| `192`, `194`, `195` | `neck_pitch` | `head_pitch` (±90°) | 8.720 g |
| `201`, `203`, `204` | `yaw_roll_motion` | `head_yaw` (±170°) | 51.600 g |
| `214`, `216`, `217` | `jaw_soft` | `head_roll` (±25°) | 188.766 g |

Composite load seen by each actuator (MuJoCo `xipos`/`body_inertia` propagated to the
hinge axis, evaluated at HOME):

| hinge | axis (world, HOME) | anchor (world) | moving mass | mass share | CoM lever ⊥ axis **R** | I about axis (bodies) | I about axis (BAM `armature`) | τ_g(θ) law | τ_g at HOME | τ_g max |
|---|---|---|---|---|---|---|---|---|---|---|
| `neck_pitch` | `(0,−1,0)` | (26.0, 14.5, **152.4**) mm | **288.93 g** | **37.08%** | **75.584 mm** | 2.0717e−3 kg·m² | 2.2431e−3 kg·m² | 0.21426·sin(θ−15.12°) | +0.01823 N·m | 0.21426 N·m |
| `head_pitch` | `(0,+1,0)` | (8.9, 14.5, 199.4) mm | 249.09 g | 31.97% | 38.234 mm | 6.4321e−4 | 2.2431e−3 | 0.09342·sin(φ−36.2°) | +0.02691 N·m | 0.09342 N·m |
| `head_yaw` | `(0,0,+1)` | (8.9, 0.0, 218.1) mm | 240.37 g | 30.85% | 11.412 mm | 3.7266e−4 | 2.2431e−3 | ≡ 0 (axis ∥ g) | 0 | 0 |

Detail (per body, at HOME, from `head_load_model.json → head_assembly_neck_pitch.per_body`):

| body | mass | CoM world (mm) | lever arm ⊥ neck axis | τ_g about neck axis | I about neck axis |
|---|---|---|---|---|---|
| `neck` | 39.841 g | (17.45, 0.33, 175.91) | 25.00 mm | +0.003342 N·m | 4.260e−5 |
| `neck_pitch` | 8.720 g | (8.90, 0.00, 211.07) | 61.09 mm | +0.001463 N·m | 3.332e−5 |
| `yaw_roll_motion` | 51.600 g | (8.73, −5.01, 235.91) | 85.25 mm | +0.008742 N·m | 3.836e−4 |
| `jaw_soft` | 188.766 g | (23.47, 1.20, 237.20) | 84.82 mm | +0.004679 N·m | 1.612e−3 |

Key geometric fact that drives everything: the head-assembly CoM sits **75.3 mm above**
the `neck_pitch` axis and only **6.43 mm behind** it at HOME, so the gravity torque at HOME
is tiny (+0.0182 N·m) while the *maximum* gravity torque over the joint range is
m·g·R = 0.28893·9.81·0.075584 = **0.2143 N·m**. Balance angle (`τ_g` = 0) is
**neck_pitch = 15.12°** (with `head_pitch` at HOME), i.e. HOME sits only 4.9° from
balance. `head_pitch` balance is at **36.2°** with HOME at 20°.

Joint ranges read from the compiled model (`head_load_model.json → joint_ranges_deg`):
`neck_pitch` −90…+60, `head_pitch` ±90, `head_yaw` ±170, `head_roll` ±25;
`left_hip_yaw` −25…+30, `right_hip_yaw` −30…+25, `hip_roll` ±22.

---

## Torque budget: head

**The actuator ceiling (verified, not assumed).** Installed set =
`hd1910_calibration/params_recommended_m5.json` (byte-identical to
`vendor/bam/bam/params/hd1910/m5.json` **and** `/home/joyandai/bam/bam/params/hd1910/m5.json`,
`diff` clean): `kt=0.692`, `R=3.75 Ω`, `error_gain=0.163`, `kd=0.346`, `armature=2.2431e−3`.
`microduck_constants.py:127-153` pins `kp_fw=32.0`, `vin_range=(4.75, 5.25)`, `vin_min=4.0`,
`motor_name="hd1910"`, `model="m5"`, `delay_min_lag=4`, `delay_max_lag=7`.
`vendor/bam/bam/feetech/actuator.py` `HD1910Actuator` defaults `p_scale=1.0`,
`d_scale=4.0`, `max_pwm=1.0`, `max_current=3.25 A`.

| quantity | value | source |
|---|---|---|
| stall torque @5.0 V | **0.9227 N·m** | kt·V/R = 0.692·5.0/3.75 |
| BAM `forcerange` (MuJoCo clamp) | **±0.9688 N·m** | `bam/mjlab.py:287` `force_limit = max(vin_range)·kt/R` |
| "~1.03 N·m peak" claimed earlier | **0.9688 N·m with the installed kt=0.692** — 1.03 N·m only if you use the *datasheet* kt=0.7358 (5.25·0.7358/3.75). The fitted/installed kt is 0.692, so the sim ceiling is **±0.969 N·m** | `docs/policy_failure_analysis.md:504` |
| no-load speed (back-EMF) | 7.225 rad/s | V/kt |
| electrical damping kt²/R | **0.12770 N·m·s/rad** | dominant velocity term |
| firmware P stiffness K = kt·V·kp·eg/R | **4.813 N·m/rad** (duty per rad error = kp·eg = 5.216) | law in `feetech/actuator.py` |
| firmware D + viscous damping | kt·V·kd·eg/(R·d_scale) = 0.01301, + 0.00807 → c_fw = 0.02108 | idem |
| total closed-loop damping c | 0.14877 N·m·s/rad | kt²/R + c_fw |
| effective inertia at `neck_pitch` | **4.3148e−3 kg·m²** (bodies 2.0717e−3 + armature 2.2431e−3) | MJCF + m5.json |

**1. Static gravity torque the neck must supply** (law τ = 0.21426·sin(θ−15.12°) N·m):

| attitude | Δ from HOME | τ_g | % of 0.9227 N·m stall | margin |
|---|---|---|---|---|
| θ = 0° (HOME −20°) | −20° | −0.0559 N·m | 6.1% | 16.5× |
| **θ = 20° = HOME** | 0 | **+0.0182 N·m** | **2.0%** | **50.7×** |
| θ = 40° (HOME +20°) | +20° | +0.0901 N·m | 9.8% | 10.2× |
| θ = 45.8° (+25.8°) | — | +0.1094 N·m | 11.9% | 8.4× |
| worst in range (θ ≈ −75°/ +105°) | — | 0.2143 N·m | 23.2% | 4.3× |

(Both legs' feet, standing: the whole-robot posture-hold demand is what
`docs/policy_failure_analysis.md:554` quotes as ≤0.037 N·m vs 0.92 N·m = 25×; consistent.)

**2. Dynamic torque to *hold the head still* at 2.17 Hz.** Measured trunk motion in the base
rollout (`neck_torque_budget.json → trunk`): z_std 9.63 mm, z_p2p 37.3 mm; **a_z amplitude
2.00 m/s² at 2.17 Hz** (σ = 2.19, p2p 10.81 m/s²) → the vertical-acceleration term is

  τ_az = −r_x·m·a_z, with r_x = 4.70 mm mean (75.58·sin(θ−15.12°)) ⇒ **0.0027 N·m** at the
  gait harmonic, 0.0431 N·m broadband max. Vertical trunk motion is almost irrelevant to
  the neck *because the CoM lever is only 4.7 mm horizontal at HOME*.

Trunk pitch σ = 2.23° (amplitude 2.48° at 2.17 Hz) ⇒ α_trunk = **7.59 rad/s²** ⇒ Euler term

  τ_euler = I_neck·α = 2.0717e−3 · 7.59 = **0.0157 N·m**.

Lateral acceleration contributes **exactly zero** about a y-axis hinge (u ∥ y ⇒
u·(r×F_y) = 0).

| quantity | value | % of 0.9227 N·m |
|---|---|---|
| hold at HOME, DC gravity | 0.0182 N·m | 2.0% |
| gait harmonic (a_z + Euler, in phase) | 0.0184 N·m | 2.0% |
| **total required to hold still** | **0.0366 N·m** (broadband max 0.112 N·m; **measured max in the clamp run 0.0314 N·m**) | **3.4–4.0%** |
| **margin ratio** | **0.9227 / 0.0366 = 25×** (29× against the measured 0.0314 N·m) | — |

**3. Torque to *follow* the commanded oscillation.** The measured commanded neck target
oscillates with **amplitude 21.4°** at 2.17 Hz ⇒ 5.09 rad/s peak, 69.3 rad/s² peak:

| term | value |
|---|---|
| I_eff·α_peak | 4.3148e−3 · 69.3 = 0.2988 N·m |
| damping c·ωA (of which back-EMF alone 0.1277·5.09 = 0.650 N·m) | 0.7568 N·m |
| **demanded motor torque** (quadrature sum 1.056 N·m; peak-of-sum 0.95 N·m) | **> 0.9227 N·m stall** |
| ⇒ duty needed at peak | **1.028 (>100%) → saturation** |
| measured in the rollout | duty mean 0.796, p95 1.000, **saturated 64.5% of samples**, mean |τ| 0.753 N·m, max 0.996 N·m; command→joint gain 1.016, **phase −68.5°** |

**4. Answers.**
*Can the neck hold the head statically?* **Yes** — 0.0182 N·m at HOME (50.7× margin); worst
attitude anywhere in the joint range 0.2143 N·m (4.3× margin); and at rest the BAM m5
friction budget alone (0.0259 N·m at this load: 0.01355 base + 0.01166 load-Stribeck +
0.00039 load + 0.00029 Stribeck) **exceeds** the HOME gravity torque, so the head can sit at
HOME with *zero* motor torque and zero error.
*Can it hold it during a step?* **Yes** — 25–29× margin (table above), confirmed by
measurement (§ Holding stiffness).
*Maximum angular acceleration the neck can impart*: α_max = (τ_stall − τ_friction)/I_eff =
(0.9227 − 0.0259)/4.3148e−3 = **208 rad/s²** (≈11 900 °/s²). (Self-limiting: the BAM m5
load-dependent friction grows with motor torque — 0.697·|τ_motor| — so the realised
α is smaller; the measured mean load-dependent friction budget at the neck was 0.527 N·m
in the baseline.)
*Bandwidth*: ω_n = √(K/I) = 33.40 rad/s = **5.32 Hz**, ζ = 0.516, −3 dB tracking bandwidth
**6.64 Hz**. Disturbance-to-error compliance |Δθ/Δτ| ≤ **0.234 rad/N·m at every frequency**
(minimum of the denominator, ≈ 3–4 Hz), i.e. the 0.0314 N·m step disturbance can push the
head by at most **0.42°** at *any* frequency, including the 2.17 Hz gait and well past it.

---

## Holding stiffness and expected tracking error

The actuator is not a stiff PD — it is a voltage law, but its *effective* static stiffness
is still high: **K = 4.813 N·m/rad** ⇒ static compliance **1/K = 0.2078 rad/N·m = 11.9°/N·m**.
Predicted steady-state error for a gravity torque τ (friction-free):

| τ | error |
|---|---|
| 0.0182 N·m (HOME) | **0.217°** |
| 0.0901 N·m (±20° band edge) | 1.07° |
| 0.2143 N·m (worst attitude) | 2.55° |
| 0.0366 N·m (step disturbance, hold) | **0.44°** |

Friction makes it better, not worse: the resting friction budget (0.0259 N·m) is larger than
the HOME gravity torque, so the achievable residual is **0**, and the oscillation component
is band-limited by the loop (≤0.42° above).

**Measured (decisive) — same policy, same env, same 12 s, only the head actions clamped to
zero (`artifacts/neck_hold_probe.py`, `--hold neck` / `--hold head`):**

| metric | baseline (policy free) | head held at HOME | head pitch held | all 4 head joints held |
|---|---|---|---|---|
| `neck_pitch` std / p2p | 16.51° / 57.9° | **1.75° / 5.81°** | — | **1.91° / 6.03°** |
| `head_pitch` std / p2p | 7.45° / 33.8° | — | **0.61° / 2.07°** | 0.61° / 2.07° |
| neck err vs the **applied target** (σ) | 18.7° (target = HOME+policy action) | **1.75°** (target = HOME) | — | 1.91° |
| neck duty mean / saturated | 0.796 / **64.5%** | **0.150 / 0%** | — | — |
| neck |τ| mean / max | 0.753 / 0.996 N·m | **0.154 / 0.424 N·m** | — |
| head pitch **relative to the trunk** (σ / p2p) | **22.3° / 86.0°** | — | — | **2.5° / 8.0°** |
| head roll / yaw relative to the trunk (σ) | — | — | — | **0.02° / 0.05°** |
| trunk z mean | 95.5 mm | 97.0 mm | — | **96.8 mm** |
| forward velocity | (not recorded in the baseline CSV) | — | — | **0.172 m/s of 0.200 cmd (86%)** |
| falls in 12 s | 0 | 0 | — | **0** |

(The clamped-head run's 86% velocity tracking is in line with the ~92% credited to this gait
in `docs/policy_failure_analysis.md:504`. The gait is also *steadier* without the head swing:
trunk z σ 9.63 → 5.24 mm, z p2p 37.3 → 23.6 mm, gait frequency 2.17 → 2.42 Hz.)

> **Column-label note (`scripts/gait_forensics.py`, do not edit — tracked).** The CSV columns
> `head_rel_trunk_pitch` / `head_rel_trunk_roll` are **swapped relative to the code comment at
> `scripts/gait_forensics.py:361-362`**. Verified by replaying the recorded joint angles into
> MuJoCo and comparing the rotation-vector components sample by sample: the column *named*
> `head_rel_trunk_pitch` equals `rr[0]` (rotation about trunk **x**) and `head_rel_trunk_roll`
> equals `rr[1]` (rotation about trunk **y**). Since the trunk frame is x=forward, y=left,
> z=up, the **pitch** of the head relative to the trunk is the column named
> `head_rel_trunk_roll` (σ = 22.30°, p2p 86.0°), and it equals Δ`head_pitch` − Δ`neck_pitch`
> exactly (checked at 7 samples, e.g. t=9.0 s: Δ_head=+8.9°, Δ_neck=−15.9° → +24.61°). The
> small column named `head_rel_trunk_pitch` (σ = 3.14°) is the roll residue from the
> `head_yaw`/`head_roll` joints. This does not change any script output, only the reading of
> those two names.

So the observed 15–16.5° neck wobble — which shows up as **22.3° (p2p 86°)** of head pitch
relative to the trunk, against a commanded 22.9° — is **8.8–9.4× larger than the actuator's
achievable residual (1.75°)**, and the same gait is preserved without it.

Why the residual is 1.75° rather than the predicted 0.44°: the model's Coulomb/Stribeck/
load-dependent friction plus the 20–35 ms command delay produce a limit-cycle-like jitter
(there is also a 4.8 Hz component in the frozen run). Either way it is ~2°, i.e. "within a
few degrees" — the criterion in the task statement.

The **68.5° command→joint phase lag at 2.17 Hz** decomposes as ≈27° from the linear
P-D plant (K − Iω² + jcω) + ≈21° from the 4–7 substep delay (20–35 ms) + the remainder from
friction — i.e. the servo is late, not weak.

---

## Leg stance: torque and support polygon

**`hip_yaw` is a vertical axis** — MJCF anchor (6.0, 17.5, 115.0) mm, world axis
`(0, 0, −1)`, and the computed gravity torque about it is **0.000000 N·m** for both legs at
HOME. Gravity therefore cannot load `hip_yaw` at *any* toe-out angle: **holding hip_yaw at
0° or +16° costs 0 N·m**. The only load is (a) inertia of the 141.05 g leg below the hinge
(I about the axis = 1.579e−4 kg·m²; +armature = 2.401e−3 kg·m²) and (b) horizontal contact
forces, which need a friction force at the sole to produce a yaw moment (vertical GRF gives
exactly zero about a vertical axis).

**Where the measured 0.86 N·m at `left_hip_yaw` comes from** — reconstructed with the exact
BAM law from the recorded action/q/dq trace (`leg_stance_support.py`):

| joint | mean q | range | command mean ± σ | command [min,max] | cmd outside range | q at limit | err σ | duty mean / sat | |τ| mean / p95 / max |
|---|---|---|---|---|---|---|---|---|---|
| `left_hip_yaw` | +16.41° | −25…+30 | +23.6 ± 28.4° | **[−28.5, +71.3]** | **52.3%** | **40.7%** | 24.79° | 0.943 / **87.5%** | **0.858** / 0.989 / 0.991 |
| `right_hip_yaw` | +16.68° | −30…+25 | +21.4 ± 17.4° | [−3.0, +54.4] | **43.0%** | **41.5%** | 12.84° | 0.725 / 52.8% | 0.656 / 0.978 / 0.980 |
| `left_hip_roll` | +1.31° | ±22 | +0.05 ± 6.33° | [−16.0, +17.4] | 0% | 0% | 6.56° | 0.459 / 10.3% | 0.429 / 0.955 / 0.968 |
| `right_hip_roll` | +1.74° | ±22 | +1.52 ± 4.46° | [−10.1, +12.7] | 0% | 0% | 5.01° | 0.372 / 3.0% | 0.345 / 0.840 / 0.943 |
| `left_hip_pitch` | −55.45° | ±90 | −53.4 ± 14.1° | [−89.1, −22.1] | 0% | 0% | 12.12° | 0.781 / 56.0% | 0.733 / 1.023 / 1.048 |
| `left_knee` | +29.60° | ±90 | +28.7 ± 16.6° | [−4.0, +61.4] | 0% | 0% | 13.63° | 0.724 / 51.5% | 0.688 / 1.031 / 1.051 |
| `left_ankle` | +70.28° | ±90 | +70.9 ± 7.14° | [+48.8, +90.2] | 0.2% | 0% | 8.35° | 0.530 / 23.2% | 0.500 / 1.024 / 1.056 |
| `right_hip_pitch` | −26.06° | ±90 | −23.3 ± 11.8° | [−42.3, +5.6] | 0% | 0% | 9.24° | 0.636 / 32.3% | 0.604 / 1.061 / 1.071 |
| `right_knee` | +7.22° | ±90 | +7.9 ± 21.9° | [−28.7, +58.2] | 0% | 0% | 17.79° | 0.854 / 70.0% | 0.804 / 1.007 / 1.026 |
| `right_ankle` | +22.29° | ±90 | +20.9 ± 12.4° | [−14.4, +45.2] | 0% | 0% | 12.68° | 0.721 / 44.3% | 0.696 / 1.054 / 1.066 |

**So the "+16° toe-out" is not a pose the policy decided to hold** — it is the mean of a
commanded yaw oscillation of σ = 28.4° spanning **−28.5° … +71.3°**, clipped by the
−25…+30° joint range (the right leg: σ = 17.4°, −3.0…+54.4° inside −30…+25°). The joint sits
within 1° of a hard stop **41%** of the time and the servo is at full duty **88%** of the time.
Closed-form check of that torque: driving ±25° (0.436 rad) at 2.17 Hz needs
I·α = 2.401e−3·81 = 0.194 N·m **plus** back-EMF c·ωA = 0.1277·5.93 = 0.757 N·m →
**0.95 N·m > 0.923 N·m stall ⇒ saturated**, matching the measured 87.5%.

**Sign conventions verified numerically** (`leg_stance_support.json → convention_check`):
`left_hip_roll` = −10/−5/0/+5° ⇒ foot separation 66.10/83.64/100.83/117.51 mm, i.e. the
**left** leg abducts with **increasing** `hip_roll` and the right with decreasing — matching
the brief. HOME (∓5°, "knees-in") is 83.6 mm. Decomposing the observed walking stance —
`hip_roll` = +1.31° (L) / +1.74° (R) absolute, i.e. deviations +6.32° / −3.27° from HOME —
into a symmetric abduction `0.5·(dev_L − dev_R) = +4.8°` and an antisymmetric lean
`0.5·(dev_L + dev_R) = +1.5°`, the **symmetric** part is 4.8° of abduction from HOME, i.e.
`hip_roll ≈ 0` on both legs, which is the stance used below (plus a small 1.5° lateral lean
that does not change the polygon's size).

**Support polygon** (convex hull of the two sole collision meshes, projected on the ground;
the same convention that reproduces the repo's earlier "54.7 mm" figure — this session
computes 54.0 mm at HOME):

| stance | area | lateral width | fore–aft length | foot sep (sites) | lateral margin | sagittal margin (front / back) | ankle lateral authority |
|---|---|---|---|---|---|---|---|
| **HOME** (`hip_roll` ∓5°) | 6707 mm² | 124.84 mm | **54.00 mm** | 83.64 mm | **62.39 mm** | **20.41** (33.59 / 20.41) | 0.9962 |
| **HOME + 5° abduction** (`hip_roll`=0) | 7562 mm² | **140.70** | 54.00 | 100.83 mm | **70.32** | 20.41 (33.59 / 20.41) | 1.0000 |
| **HOME + 5° adduction** (`hip_roll` ∓10°) | 5834 mm² | 108.66 | 54.00 | 66.10 mm | 54.30 | 20.41 | — |
| **HOME + 16° toe-out** | 6916 mm² | **133.28** | **73.71** | 81.76 mm | 65.75 | **30.60** (43.11 / 30.60) | 0.9576 |
| **HOME + 5° abd + 16° toe-out** (= observed) | 7769 mm² | **148.46** | **78.06** | 98.27 mm | **73.35** | **32.77** (45.28 / 32.77) | 0.9613 |

Read-out (all numbers from the table above):

* Pure **+16° toe-out** widens the *lateral* support by **+8.44 mm (+6.8% of 124.8 mm,
  +4.2 mm per side)** and lengthens the *fore–aft* support by **+19.71 mm (+36.5%)**,
  lifting the limiting sagittal margin from 20.41 → **30.60 mm (+50%)**. Area +209 mm² (+3.1%).
* Pure **+5° abduction** widens the lateral support by **+15.86 mm (+12.7%)** and does
  nothing fore–aft.
* Cost of toe-out in lateral-balance authority: the ankle hinge axis leaves the world y-axis
  by exactly 16° (`(0,0.996,−0.087) → (0.275,0.958,−0.087)`), so ankle roll torque authority
  is ×cos16° = **0.9613 (−3.9%)** — negligible. Torque cost otherwise **zero** (gravity
  torque about a vertical axis is identically 0).
* Independent confirmation on the *real* configurations (every 20 ms sample of the rollout
  replayed into MuJoCo, contact-restricted hull): zeroing `hip_yaw` reduces the sagittal
  margin by **6.43 mm** and the lateral margin by **4.03 mm** on average — the toe-out
  genuinely enlarges the polygon in the gait the policy actually produces.
* Caveat on the axis: **the binding axis is sagittal, not lateral.** At HOME the lateral
  margin (62.4 mm) is **3.1× the sagittal margin (20.4 mm)**, and the documented fall mode is
  the CoM running off the *front* edge (`docs/policy_failure_analysis.md:490-505`: 54.7 mm
  polygon, CoM overshoot +102…+105 mm forward, face-plant at pitch −44…−48°). So the toe-out
  is buying the margin that actually matters, and it is buying it for free.
* The toe-out is **not** a kinematic consequence of abduction: `hip_roll` is an x-axis hinge
  and `hip_yaw` a z-axis hinge; they are independent DOFs, and abduction alone leaves the
  fore–aft polygon at 54.00 mm.
* Sim caveat: the feet have `condim=3` (`microduck_constants.py:101`), i.e. **no torsional
  contact friction**, so twisting a loaded foot is free in sim but not on the real robot
  (a 41×54 mm rubber pad under ~3.8 N would need roughly 0.02 N·m of torsional friction to
  scrub — still ≪ stall, but it is not modelled).

**So: is +16° toe-out a *cheaper* pose?** In torque terms it is exactly neutral (0 N·m
gravity holding torque at any yaw) and in stability terms it is strictly better — +8.4 mm
lateral, +19.7 mm fore–aft, +10.2 mm on the limiting margin, −3.9% ankle roll authority.
The duck stance is a *good* choice. What is not free, and not a choice the hardware supports,
is the ±28° yaw oscillation the policy superimposes on it.

---

## Implications for the reward design

What the measurements say the reward should stop asking for, and what it may legitimately ask for:

1. **Stop pricing instantaneous head-pose error as if the oscillation were unescapable.**
   `head_pose_tracking` (weight 2.0, per-joint Gaussian std 0.5 rad) scores the
   instantaneous 4-joint error, and the in-repo rationale for adding the DC-only companion
   term (`head_pose_bias` → `head_pose_bias_penalty` in `mdp.py`) assumes "*a 280 g head (38% of robot mass) **must**
   oscillate while stepping*" (`microduck_velocity_env_cfg.py`, comment above the
   `head_pose_bias` term). **That premise is
   falsified by measurement**: with the head actions clamped, `neck_pitch` tracks to
   1.75° std (5.81° p2p) and the same gait carries the robot at 0.172/0.200 m/s with 0 falls.
   The escapable head error is **~2°**, not 15°. A tight instantaneous tolerance with
   std ≈ 0.1 rad (5.7°) is therefore *not* an unescapable tax — the earlier run 5yay13u4 that
   died from `fine_std=0.1` was killed by a **different** problem, and the correct next test
   is to price the instantaneous error with the *oscillation already suppressed by the same
   policy*, e.g. by freezing/clamping the head actions for part of the episode or by pricing
   a 0.2 s EMA (which the 5.3 Hz actuator can follow but a 2.17 Hz bang-bang cannot fake).
2. **Price the head command against the actuator's real ceiling.** The plant saturates when
   `|τ| > 0.923 N·m`, which for the neck means commanded neck speed above roughly
   `0.923/0.1277 ≈ 7 rad/s` is unattainable and any command demanding >5 rad/s peak eats
   >70% of the duty budget. A per-joint rate/shaping penalty on the *neck/head command*
   (or a command-range limit tied to `7.23/ω`) is legitimate and is not "blocking a needed
   motion" — the needed motion is 0.
3. **`action_rate_l2` is not doing this job.** Measured mean per-step |Δaction| = 0.0751
   (max 0.1281) while the neck duty alternates between ±1 — the regularizer is priced on the
   normalized action, not on what saturates. If smoothness is the goal, price the *duty*
   or `|Δ(command)|` at the joints that saturate (hip_yaw 87.5%, knee 70%, neck 64.5%).
4. **Keep the DC-bias term** (`head_pose_bias`): DC head-pose error is genuinely the
   escapable part and costs ~0.018 N·m to cancel — it should be cheap to satisfy, and it is.
5. **Add the standard limit-proximity penalty to `hip_yaw`.** The joint is within 1° of a
   hard stop **40.7% (L) / 41.5% (R)** of the time because 52%/43% of the commanded targets
   are outside the joint range. AGENTS.md already prescribes the fix (qpos-side
   limit-proximity penalty on the offending joints); command-side penalties cannot work here
   because the ctrlrange is intentionally wide.
6. **The +16° toe-out stance is legitimate and should not be taxed.** It costs 0 N·m of
   holding torque and buys +8.4 mm lateral / +19.7 mm fore–aft of support polygon, and the
   fore–aft axis is the binding one (20.4 mm vs 62.4 mm lateral). A reward that wants it gone
   should say *why* — nothing in the physics does.
7. **Whole-robot implication of the measured saturation.** With leg joints at 37–93% of stall
   on average and duty saturation 3–87.5%, the gait is running the servos at their 5 V limit.
   Any new "smoothness" tax that is active during skill discovery will therefore look like a
   torque limit rather than a style choice; introduce it after the gait consolidates (as the
   existing `action_rate_weight` curriculum already does).

---

## Confidence and open questions

**High confidence (measured, and the decisive claim does not depend on any model fit):**
* The head assembly is 288.93 g = 37.08% of a 779.243 g robot, lever 75.58 mm to the
  `neck_pitch` axis, I = 2.0717e−3 kg·m² — read straight out of the MJCF and confirmed by
  MuJoCo's own subtree mass propagation.
* Clamping the head actions collapses the neck oscillation from 16.5° to 1.75° std
  (5.81° p2p) with 0% duty saturation while the robot keeps walking (86% of command,
  0 falls). This is a direct A/B in the same environment; it does not depend on the friction
  model, the armature value, or my torque reconstruction.
* Gravity torque about `hip_yaw` is identically zero (vertical axis), the joint ranges, and
  the support-polygon geometry are exact MuJoCo computations.
* Actuator identity: kt=0.692, R=3.75, kp_fw=32, eg=0.163, kd=0.346, 5.0 V rail,
  stall 0.9227 N·m, forcerange ±0.9688 N·m — all traced to
  `params_recommended_m5.json` / `m5.json` / `microduck_constants.py:127-153` / `bam/mjlab.py:287`.

**Medium confidence (model-dependent):**
* The absolute duty/τ numbers per joint are reconstructed from the recorded 50 Hz ONNX action
  trace, so they ignore the 4–7-substep command delay and the sub-step duty evaluation;
  expect a few percent. The *ranking* (neck and hip_yaw saturated, hip_roll/ankle not with
  much headroom) is robust. The reconstructions slightly exceed the ±0.9688 N·m forcerange
  when back-EMF adds, so true applied peaks are clamped there.
* The BAM m5 **load-dependent** friction (0.697·|τ_motor|) dominates the neck budget in the
  baseline (mean budget 0.527 N·m). It was fitted on the pendulum bench; if the real gearbox
  is less lossy, tracking improves and the saturation verdict gets *softer*, never harder.
* The reflected `armature` 2.2431e−3 kg·m² is 52% of the neck's effective inertia and was
  identified on the bench. Halving it would raise ζ and shift f_n up, again only making the
  hold easier.
* The trend of "all 14 joints are near saturation" is measured, but I did not run an
  independent open-loop torque probe on hardware.

**Open questions / what would settle the remainder:**
1. *Why does the policy command the wobble at all?* The action trace proves it does
   (commanded std 16.58° vs actual 16.51°), but not the incentive. Two candidates: (a) the
   2.17 Hz gait is a resonant limit cycle of policy + 20–35 ms delay + friction, not a
   deliberate balance strategy; (b) head angular momentum is being traded for balance. Test:
   re-run training with the head actions clamped to HOME for a fraction of envs and see
   whether the free-head policy's gait degrades — 12 s of clamping already shows no
   degradation, so (b) looks weak, but it needs a trained A/B, not one rollout.
2. *Is the real servo's friction lower than m5?* The 68.5° phase lag is partly friction's
   describing-function lag. A bench sweep of neck tracking at 2.17 Hz on hardware would
   separate "friction" from "delay".
3. *Contact torsional friction.* Feet are `condim=3` in sim; the real foot resists twist.
   Add condim 4-6 to the foot geoms if foot-scrub cost is to be modelled — it would matter
   for the ±28° hip_yaw oscillation, not for the toe-out pose.
4. *Does the support-polygon gain survive contact compliance?* The polygon is computed from
   the rigid sole hull; the 1 mm-close contact patch is 45.6 × 34 mm (vs a 54.0 × 41.2 mm
   hull), so the fore–aft gain is probably ~13–15 mm rather than 19.7 mm. Direction and
   ranking are unaffected.

---

### Artifacts

| file | what |
|---|---|
| `artifacts/head_load_model.py` / `.json` | MJCF head masses, CoM, inertias, lever arms, gravity sweeps, joint ranges, leg static loads |
| `artifacts/neck_torque_budget.py` / `.json` | actuator ceiling, command→joint transfer function, duty/τ reconstruction, disturbance torque, bandwidth |
| `artifacts/neck_hold_probe.py` | the decisive A/B (head actions clamped) |
| `artifacts/hold_neck.csv`, `artifacts/hold_head_all.csv` | 12 s raw traces, head pitch held / all four head joints held |
| `artifacts/hold_compare.json` | baseline vs head-held torque/duty/disturbance comparison |
| `artifacts/leg_stance_support.py` / `.json` | sign-convention check, static support polygons, measured-trace joint torques |
| `artifacts/gf_walk_hd1910_15000_12s.csv` / `.json` | baseline 12 s rollout (all joints, actions, body poses) |
| `artifacts/make_figs.py`, `artifacts/neck_torque_budget.png` | (a) τ_g vs stall, (b) closed-loop Bode, (c) commanded vs actual vs clamped neck, (d) neck duty cycle |
