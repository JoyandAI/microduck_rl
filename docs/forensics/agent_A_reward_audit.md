# Agent A — reward-function forensics for `walk_hd1901_15000.onnx`

Scope: explain, **quantitatively from the reward code**, the two visible defects of the
trained walking policy — (A) the head/neck wobble and (B) the leg splay + 16° toe-out
(and the yaw drift that accompanies it).

Method: read every reward term in `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py`
and every function in `src/mjlab_microduck/tasks/mdp.py` / the vendored mjlab reward
library that they resolve to, then evaluate those functions **at the measured state**
supplied in the task (12 s re-sim, vx = 0.2 m/s, head_pose command ≡ 0) and against the
per-term scalars logged at iteration 14999. No file outside `docs/forensics/` was modified;
no training was started; all arithmetic is CPU-only.

Artifacts read (all read-only):

| What | Where |
|---|---|
| Env / reward recipe | `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py` (1419 lines) |
| Custom MDP functions | `src/mjlab_microduck/tasks/mdp.py` (7188 lines) |
| Stock reward forms | `.venv/.../mjlab/tasks/velocity/mdp/rewards.py`, `.venv/.../mjlab/envs/mdp/rewards.py` |
| Logged reward scalars, iter 14999 | `train_velocity_hd1910.log` (tail) = run `logs/rsl_rl/velocity/2026-09-10_18-43-53_velocity/` |
| Frozen cfg of that run | `logs/rsl_rl/velocity/2026-09-10_18-43-53_velocity/params/env.yaml`, `params/agent.yaml` |
| Robot geometry / masses / limits | `src/mjlab_microduck/robot/microduck/robot_walk.xml`, `.../microduck_constants.py` |
| Actuator stiffness | `src/mjlab_microduck/robot/microduck_constants.py:127-167`, `vendor/bam/bam/params/hd1910/characterization.json`, `m5.json` |

**Independent kinematic validation used throughout this report.** I re-derived forward
kinematics from `robot_walk.xml` and the sole meshes (`assets/sole_left.stl`,
`sole_right.stl`). Feeding the *measured mean* joint deviations into that chain reproduces
the audited trunk height: **92.3 mm vs the measured 92.8 mm (0.5 mm error)**; the HOME
stance puts the lowest sole point at z = −0.8 mm with the trunk at 120.8 mm (i.e. HOME is
exactly the designed rest pose). Every lever arm, axis and CoM number below comes from that
validated chain.

Two validity caveats, stated once and not used as excuses:
1. The run was trained against a working-tree `robot_walk.xml` that then differed from the
   current file. The current file totals **779.24 g** (matching the audit's 779 g); the
   version committed at the run's own commit hash `316b3648` totals **800.24 g** — a 2.7 %
   mass difference concentrated in the newly swapped HD-1910 servos. Ratios below are
   insensitive to it; absolute sag/moment numbers are ±3 %.
2. The HD-1910 BAM parameters were re-fitted at 18:51 on the run's start date (08 minutes
   after the run began); `params/env.yaml` does not record which parameter file the run
   consumed. The actuator numbers used here are the current, characterization-backed ones.

---

## Verdict

**Defect A — the ±15° head/neck wobble is *under-priced but largely unescapable*.**
`head_pose_tracking` (weight 2.0, per-joint Gaussian, std 0.5) charges **0.093 reward
units/step** for the measured four-joint oscillation — **1.72 %** of the 5.42/step total —
and the companion `head_pose_bias` EMA penalty charges only its DC part (0.019/step of the
oscillation leaks through the 1 s EMA, 0.070/step in total once the measured DC bias is
included), because that EMA attenuates the
2.25 Hz component to **7.15 %** of its amplitude. Nothing else prices head motion at all:
`body_ang_vel` is trunk-only and xy-only (`rewards.py:184-193`), `angular_momentum` at
weight −0.02 prices the head's oscillating angular momentum at **−1.9e-7/step** (i.e. zero),
and the only remaining channel is `action_rate_l2` at **0.0055/step**. So the wobble is
*rationally free*, but it is probably not a choice: the head is 37.1 % of the mass on a
75.7 mm lever, and the servo's identified P-stiffness is only **0.150 N·m/rad**
(`error_gain` 0.163 duty/rad × kt 0.692 × 5.0 V ÷ R 3.75, ki = 0) → the head's *inertial*
deflection is **7.6°/ (m/s²)** of base fore-aft acceleration, so the observed ±15° requires
only ~2 m/s², and the neck's natural frequency √(k_eff/I) = **1.79 Hz** from the servo spring
alone (**1.88 Hz** once the 0.0152 N·m/rad gravity term is added) sits right under the
2.25 Hz gait line (resonant amplification ~1.2–1.4 at ζ≈0.2–0.3). Verdict: **under-priced,
and the correct fix is not a bigger price on the oscillation** (the codebase already
documents that `fine_std=0.1` killed walking — `microduck_velocity_env_cfg.py:1116-1139`).

**Defect B1 — the 16° toe-out (hip_yaw) is *effectively unpriced* and is a real policy
choice.** The hip_yaw axis is vertical: the FK gravity moment about it is exactly
**0.00000 N·m**, so no amount of actuator sag or servo stiffness can produce a sustained
16° offset — it must be commanded. For that choice the reward charges **0.0042/step =
0.077 %** of the total (marginal price **0.00024 reward per degree**). Worse, the same
16° is what makes the robot walk in a circle: FK shows both hip_yaw joints share the world
axis (0,0,−1), so equal-sign values on both sides rotate *both* feet the same way in world
(**common-mode** yaw, not a symmetric duck stance): at +16° both sides the left sole's long
axis rotates to −16° (left foot toe-**in**) and the right sole's to −16° (right foot
toe-**out**) — the foot pair is simply pointed 16° off the trunk's travel direction. A
sustained lateral push-off component follows, and the resulting yaw drift costs only
**0.070/step (1.3 %)** because `rel_heading_envs = 0.0` (`:996`) means **no term prices
heading position at all** — only the instantaneous rate, against std 0.707 rad/s, where a
0.133 rad/s drift sits 0.19 σ from the target.

**Defect B2 — the ~5°/3.5° leg abduction (hip_roll) is *unachievable-as-written and
under-priced*, in that order.** The pose term's `hip_roll` std is **0.05 rad (2.86°)**,
i.e. *smaller than the actuator's own load-induced position error*: FK gives a static
gravity moment on the left hip_roll of **0.0320 N·m** at HOME → **12.2°** of P-loop sag at
0.150 N·m/rad, and the measured hip_roll wanders with **σ = 4.56°** (left). The observed
deviation is *outward* while the static sag is *inward*, so the policy is commanding an
outward offset — but the point stands that the reward is asking a 2.86° Gaussian to police
a quantity whose instantaneous value is set by load, not by pose. For the whole splay +
toe-out the reward charges only **0.0289/step = 0.53 %**, so even a perfect correction is
worth half a percent — i.e. **under-priced by ≥ 1 order of magnitude**, not unpriced.

---

## Reward inventory at iteration 15000

Final weights are from `train_velocity_hd1910.log` (iteration 14999/15000 block) and agree
with `params/env.yaml` after the two weight curricula are applied
(`Curriculum/action_rate_weight: -1.0000`, `Curriculum/head_pose_bias_weight: 3.0000`).
`Episode_Reward/<term>` = mean episode sum of `raw · weight · dt` ÷ `max_episode_length_s`
(20 s) — mjlab `managers/reward_manager.py:100-131`; with the observed 982.94-step episodes
that equals **0.983 × (per-step raw·weight)**, so the logged column and the "cost/step"
column below are comparable up to that 1.7 % factor.

| # | Term | Weight @ iter 15000 | Functional form (file:line) | What it prices | Logged `Episode_Reward` |
|---|---|---|---|---|---|
| 1 | `track_linear_velocity` | **+2.0** | `exp(-‖v_cmd,xy − v_xy‖²/std²)`, std = √0.1 = 0.316 (`rewards.py:27-44`) | fore/aft + lateral speed tracking | **+1.2962** |
| 2 | `track_angular_velocity` | **+2.0** | `exp(-[(ωz−ωz_cmd)² + ωx²+ωy²]/std²)`, std = √0.5 = 0.707 (`rewards.py:47-64`) | instantaneous yaw **rate** (never yaw *position*) | **+0.2271** |
| 3 | `upright` | **+2.0** | `exp(-‖proj_grav_xy‖²/std²)`, std = √0.05 (`rewards.py:67-110`), body = `trunk_base` (`:463`) | trunk tilt from world up | **+1.7092** |
| 4 | `pose` (= `variable_posture`) | **+1.0** | `exp(-mean_over_10_legs((q−q_home)²/std²))` (`rewards.py:466`), std speed-selected (`:450-460`), `walking_threshold = 0.01` (`:458`), legs only (`:455-457`) | leg posture vs HOME | **+0.0175** |
| 5 | `body_ang_vel` | **−0.05** | `Σ(ωx²+ωy²)` of the trunk **only** (`rewards.py:184-193`) | trunk pitch/roll thrash — **not yaw, not the head** | **−0.0838** |
| 6 | `angular_momentum` | **−0.02** | `‖L_root‖²` of the whole robot (`rewards.py:196-206`) | whole-body angular momentum — **dead in practice** | **−0.0000** |
| 7 | `dof_pos_limits` | **−1.0** | stock soft-limit violation; only fires outside the 0.9-scaled soft limit (`mjlab/envs/mdp/rewards.py:81-96`, `soft_joint_pos_limit_factor = 0.9` in `microduck_constants.py:199`) | joints parking on limits | **−0.0766** |
| 8 | `action_rate_l2` | **−1.0** (curriculum 0 → −1.0 by iter 1500, `:1205-1218`) | `Σ_14 (Δa)²` (`mjlab/envs/mdp/rewards.py:58-66`) | jitter; **the only term that touches neck motion** | **−0.5824** |
| 9 | `air_time` | **+3.0** | `Σ_feet 1[0.125 < air_time < 0.300]`, gated on ‖cmd‖>0.01 (`rewards.py:209-238`) | step timing / flight window | **+1.4456** |
| 10 | `foot_clearance` | **−2.0** | `Σ_feet \|h−0.02\|·‖v_foot,xy‖` while moving (`rewards.py:241-268`) | scraping / dragging | −0.0077 |
| 11 | `foot_swing_height` | **−0.25** | `Σ ((peak/0.02)−1)²` **at first contact only** (`rewards.py:271-323`) | swing-arc shape, once per landing | −0.0159 |
| 12 | `foot_slip` | **−0.1** (deliberately weak, `:491-496`) | `Σ_feet ‖v_foot,xy‖²·1[contact]` (`rewards.py:326-353`) | sliding while loaded | −0.0005 |
| 13 | `self_collisions` | **−1.0** | `Σ found` over the `trunk_base`-subtree sensor (`rewards.py:162-181`, sensor `:366-373`) | any self-contact, incl. leg↔leg | −0.0102 |
| 14 | `head_pose_tracking` | **+2.0** | `mean_4 exp(-(err/0.5)²)`, `err = (q − q_home) − cmd` (`mdp.py:5128-5192`) | instantaneous neck/head command error | **+1.6770** |
| 15 | `body_pose_tracking` | **0.0** (disabled by design, `:1098-1114`) | 6-axis Gaussian (`mdp.py:5320-5367`) | nothing — obs/command slot kept alive only; skipped entirely at weight 0 (`reward_manager.py:122-124`) | 0.0000 |
| 16 | `head_pose_bias` | **+3.0** (curriculum 0 → 1 → 2 → 3 by iter 1500, `:1347-1358`) | `−mean_4 \|EMA_{τ=1s}(err)\|`, self-negating, positive weight (`mdp.py:5229-5317`) | **DC-only** head/neck error | **−0.2675** |

`Σ Episode_Reward = 5.3280` → **total ≈ 5.42 reward units per 20 ms step** (cross-check:
`Mean reward 107.80` per episode ÷ (982.94 × 0.02 s) = 5.48). In PPO's own units (×dt) that
is 0.108/step; all percentages below are ratios, so the dt factor cancels.

**Curricula that change the above** (step = iteration × 24, `NUM_STEPS_PER_ENV = 24`; all are
step functions, `mdp.py:3442-3461` `reward_weight`, comparison is `>` so the stage applies
one step after its boundary):

| Curriculum | Term affected | Stages | **Final @ 15000** |
|---|---|---|---|
| `action_rate_weight` (`:1205-1218`) | `action_rate_l2` weight | −0.1 @ 0, −0.2 @ 500, −0.4 @ 750, −0.6 @ 1000, −0.8 @ 1250, −1.0 @ 1500 | **−1.0** (log ✓) |
| `head_pose_bias_weight` (`:1347-1358`) | `head_pose_bias` weight | 0 @ 0, +1 @ 600, +2 @ 1000, +3 @ 1500 | **+3.0** (log ✓) |
| `standing_envs` (`:1224-1237`) | `twist.rel_standing_envs` | 0.02 → 0.25 by iter 2000 | **0.25** (log ✓) |
| `head_pose_range` (`:1250-1263`) | head command ranges | 5 %→100 % of ±1.10/±1.10/±1.40/±0.31 rad by iter 2000 | **1.40** (log ✓) |
| `com_range` (`:1289-1311`) | trunk CoM DR range | 0.003 → 0.015 m | **0.015** |
| `head_com_range` (`:1316-1329`) | head CoM DR range | 0.003 → 0.010 m | **0.010** |
| `body_pose_range` (`:1269-1284`) | body command ranges | single stage | 0.05 |
| (no curriculum) | `command_vel` deleted (`:1337`), `terrain_levels` deleted on flat (`:1336`) | — | velocity command ranges are **fixed** ±0.4/±0.3/±1.0 (`:1004-1006`) |

### When do `std_standing` / `std_walking` apply?

The two dicts are only *parameters*; the switch is inside the stock reward class
`variable_posture` (`rewards.py:385-466`):

* `std_standing`, `std_walking`, `std_running` are each resolved against the **selected leg
  joint names** at init (`:409-427`, `resolve_matching_names_values` → order follows the
  joint list; every regex must match, else `ValueError`).
* At every step (`:446-460`) the mask is driven by the **command**, not by measured speed:
  `total_speed = ‖cmd_xy‖ + |cmd_wz|`; `standing_mask = total_speed < walking_threshold`;
  `walking_mask = walking_threshold ≤ total_speed < running_threshold`.
* The microduck cfg sets `walking_threshold = 0.01` and `running_threshold = 1.5`
  (`microduck_velocity_env_cfg.py:458`; `running_threshold` is left at the library default,
  absent from `params/env.yaml`). With `vx = 0.2, wz = 0` → `total_speed = 0.2 ≥ 0.01` →
  **`walking_mask = 1`, so `std_walking` is in force** for the audited run. No other code
  path mutates these dicts at runtime (grep over `src/`: only the three cfg files that
  *assign* them; the only runtime interaction is `backlash.py:74-90`, which edits the joint
  *selector*, not the stds).

Two consequences worth flagging: (i) the switch is a **hard discontinuity in the reward's
sharpness at a 0.01 m/s command** — the same crouched walking posture scores `exp(-3.85) =
0.021` under `std_walking` but `exp(-21.75) ≈ 4e-10` under `std_standing`: the term collapses
by 7 orders of magnitude at a threshold the policy can feel through its own command input; (ii)
`running_mask` needs `total_speed ≥ 1.5`, which requires ‖cmd_xy‖ = 0.5 *and* |cmd_wz| = 1.0
simultaneously — reachable only in a measure-zero corner (`std_running` is a copy of
`std_walking` anyway, `:444`).

---

## Cost of the head wobble

`head_pose_tracking` (`mdp.py:5128-5192`) with `std = 0.5`, `cmd ≡ 0`, and mean over the 4
joints `[neck_pitch, head_pitch, head_yaw, head_roll]` (`_NECK_JOINT_PATTERNS`, `mdp.py:123`):

    per_joint = exp(-(err/0.5)²),   err = (q − q_home) − cmd ;   reward = mean_4(per_joint)

**Which integration identity I used.** The measured neck motion is a *single spectral line*:
the audit reports `neck_pitch` swinging at 2.25 Hz with std 15.0° and p2p 46.6°. For a
harmonic oscillation of amplitude `A` the correct expectation is the **Bessel identity**

    E[exp(−(A cosθ/s)²)] = exp(−A²/2s²)·I₀(A²/2s²)

and for a Gaussian `x ~ N(0,σ²)` it is `1/√(1+2σ²/s²)` (the identity quoted in the brief).
I used the **Bessel form**, since a 2.25 Hz-dominated spectrum is sinusoidal, not Gaussian;
assuming Gaussianity with the same σ would **over-price by 58 %** (`neck_pitch`: 0.196 vs
0.124 per joint). Plugging the *mean* error in instead gives `exp(0) = 1` → **cost exactly
zero**, which is the structural blind spot of a Gaussian centred on the command.

| Joint | amplitude A (p2p/2) | z = A²/2s² | E[exp] | cost/joint | weighted /step (= w 2.0 ÷ 4) |
|---|---|---|---|---|---|
| `neck_pitch` | 15.0° | 0.1371 | 0.87600 | **0.12400** | **0.06200** |
| `head_pitch` | 3.9° | 0.0093 | 0.99080 | 0.00920 | 0.00460 |
| `head_yaw` | 8.0° | 0.0388 | 0.96231 | 0.03769 | 0.01885 |
| `head_roll` | 5.1° | 0.0160 | 0.98416 | 0.01584 | 0.00792 |
| **Σ (all four, oscillation only)** | | | | 0.18673 | **0.09337** |
| Σ (`neck_pitch` + `head_pitch` only, as literally asked) | | | | 0.13320 | **0.06660** |

Adding the measured **DC** means changes the term materially, because a Gaussian is
sensitive to the offset as well as the amplitude — for `m + A cosθ` the exact average is
`exp(−(m²+A²)/s²)·I₀(2mA/s²)`:

| Joint | DC mean | with-DC cost/joint | weighted /step |
|---|---|---|---|
| `neck_pitch` | +1.59° (head tipped slightly **back**, not drooping) | 0.24148 | 0.12074 |
| `head_pitch` | +0.83° | 0.01917 | 0.00958 |
| `head_yaw` | −2.04° | 0.07898 | 0.03949 |
| `head_roll` | −0.85° | 0.03238 | 0.01619 |
| **Σ** | | 0.37201 | **0.18601** |

So the instantaneous tracker charges **0.0934/step for the oscillation alone** and
**0.186/step** when the measured bias is included.

`head_pose_bias_penalty` (`mdp.py:5229-5317`) is the term the task asked to quantify as
"nearly free for an oscillation around zero". Exactly how free: the EMA is
`y ← (1−α)y + α·err` with `α = step_dt/tau_s = 0.02/1.0 = 0.02`, a first-order low-pass whose
gain at frequency f is `|H| = α/√(1 − 2(1−α)cos(2πf·dt) + (1−α)²)`. At f = 2.25 Hz,
`ω = 2π·2.25·0.02 = 0.28274 rad/step` → **|H| = 0.0715**, i.e. **92.8 % of the 2.25 Hz
amplitude is removed**. The residual `|EMA|` of a sinusoid averages `(2/π)·|H|·A`:

| Joint | EMA amplitude | mean\|EMA\| osc-only | mean\|EMA\| with DC |
|---|---|---|---|
| `neck_pitch` | 1.073° | 0.683° | 1.590° |
| `head_pitch` | 0.279° | 0.178° | 0.830° |
| `head_yaw` | 0.571° | 0.363° | 2.040° |
| `head_roll` | 0.367° | 0.234° | 0.850° |
| **mean over 4** | | 0.364° | 1.328° |
| **penalty (w = +3.0)** | | **−0.0191/step** | **−0.0695/step** |

**Total price of the observed head wobble:** `0.0934 + 0.0191 = 0.1125/step` (oscillation
only), or `0.1860 + 0.0695 = 0.2556/step` if the measured DC bias is charged to the same
defect. The training log's `−0.2675` for `head_pose_bias` corresponds to 0.272/step of DC
error — consistent with the training-time picture (random head commands, partial tracking),
not with the audited eval.

Not-priced-at-all channels for the same motion (all FK numbers from the validated chain):

| Channel | Form / weight | Head contribution | Verdict |
|---|---|---|---|
| `body_ang_vel` | trunk body only, xy only (`rewards.py:184-193`, cfg `:489`) | **excluded by construction** | head rotation unpriced |
| `angular_momentum` | `−0.02·‖L‖²` (`rewards.py:196-206`) | head I about the neck axis = 1.18e-3 kg·m², peak ω = 3.70 rad/s → rms \|L_head\| = **0.0031** kg·m²/s = **36 %** of the measured whole-body `Metrics/angular_momentum_mean = 0.0086` | **−1.9e-7/step ≈ 0** |
| `action_rate_l2` | `−1.0·Σ_14(Δa)²` | neck needs Δq = ω·dt = 0.074 rad/step → **−0.0055/step** | the only real price |
| `self_collisions`, `foot_*`, `air_time`, `upright`, `pose` | — | no head/neck argument at all (selectors are sites/bodies/leg-joint regex) | 0 |

---

## Cost of the leg splay / toe-out

The pose term is **not** a mean of per-joint Gaussians: it is a **single Gaussian of the mean
squared normalized error over the 10 leg joints** (`rewards.py:462-466`):

    pose = exp( − (1/10)·Σ_i (q_i − q_home,i)²/std_i² )

This is the single most important structural fact in this audit — it means every joint is
priced not absolutely but *relative to the total posture error*, and it dilutes any one
joint's tolerance by √10 ≈ 3.16.

At the measured mean posture (deviation from HOME in degrees, order
`L hip_yaw, L hip_roll, L hip_pitch, L knee, L ankle, R hip_yaw, R hip_roll, R hip_pitch,
R knee, R ankle`), with `std_walking = {yaw 0.3, roll 0.05, pitch 0.4, knee 0.4, ankle 0.25}`:

| Joint | dev | std | exp-contribution (dev/std)² |
|---|---|---|---|
| left_hip_yaw | +16.31° | 0.30 | 0.900 |
| left_hip_roll | +6.56° | 0.05 | **5.244** |
| left_hip_pitch | −27.22° | 0.40 | 1.411 |
| left_knee | +32.70° | 0.40 | 2.036 |
| left_ankle | +44.36° | 0.25 | **9.591** |
| right_hip_yaw | +16.15° | 0.30 | 0.883 |
| right_hip_roll | −3.55° | 0.05 | 1.536 |
| right_hip_pitch | −53.65° | 0.40 | **5.480** |
| right_knee | +4.34° | 0.40 | 0.036 |
| right_ankle | +48.28° | 0.25 | **11.361** |
| **Σ / mean** | | | **38.476 / 3.848** |

    pose = exp(−3.848) = 0.02133  →  ×weight 1.0  =  **0.0213/step**

That matches the logged `Episode_Reward/pose = 0.0175` (0.0178/step) to within the training
distribution — the pose term is **already 98 % saturated** at the policy's operating point.

**Counterfactuals (holding every other joint at its measured value):**

| Correction | pose after | gain (/step) | share of total |
|---|---|---|---|
| zero both `hip_roll` deviations (undo the 5°/3.5° splay) | 0.04202 | **+0.02069** | **0.38 %** |
| zero both `hip_yaw` deviations (undo the 16° toe-out) | 0.02549 | **+0.00416** | **0.077 %** |
| zero all four | 0.05022 | **+0.02889** | **0.53 %** |

**Marginal (local) prices at the operating point** — this is what PPO's advantage actually
sees, because `dR/dx = −R·2x/(N·std²)` and `R = 0.0213`:

| Joint | d(pose)/dθ | per degree |
|---|---|---|
| left_hip_roll | −0.195/rad | **−0.0034/step per degree** |
| right_hip_roll | +0.106/rad | +0.0018/step per degree (already past HOME→vertical) |
| left_hip_yaw | −0.0135/rad | **−0.00024/step per degree** |
| right_hip_yaw | −0.0134/rad | −0.00023/step per degree |

**Answering the "is std = 0.05 rad flat?" question properly.** Two different kinds of
flatness have to be separated:

1. *Per-joint* (if `hip_roll` were the only joint, i.e. form `exp(−(θ/0.05)²)`):
   θ = 0° → 1.000, **5° → 4.75e-2**, 10° → 5.1e-6, **15° → 1.24e-12**, 22° → 2.4e-26.
   Yes — beyond ~6° the per-joint form is numerically dead; 0.05 rad is a 2.86° tolerance,
   so 5° is already 1.75σ and 15° is 5.2σ.
2. *In the code as written* (mean inside a single exp over 10 joints), the same deviations
   give **0.737 at 5°** and **0.0645 at 15°** as the only deviating joint — deceptively
   non-flat. But at the real operating point the term's **value is 0.0213**, so the
   *absolute* price of a hip_roll degree collapses to **0.0034 reward units**: flat in the
   only currency that matters (advantage). The trap is therefore **not** "std too tight to
   reach" in the sense of a dead Gaussian — it is a **collective-saturation trap**: the
   ankle (+44°/+48°) and right hip_pitch (−54°) excursions that *walking itself requires*
   push the exponential 3.85 deep, and every other joint's error is then priced at
   `R × small`.

Supporting actuator-side evidence that `std = 0.05` is below the achievable floor:
`hip_roll`'s static gravity moment at HOME is **0.0320 N·m** (FK: left leg subtree 0.1058 kg,
CoM 30.8 mm outboard of the axis), and the identified P-loop stiffness is
**0.150 N·m/rad** with **ki = 0** (`characterization.json`: `error_gain 0.163` duty/rad,
`kp` reg 32, `ki` reg 0; kt 0.692, R 3.75, 5.0 V) → **12.2° of inherent sag** from the leg's
own weight alone, before ground-reaction loads. The joint then wanders with **σ = 4.56°**
(measured) ≈ 1.6 × the entire std. Note the measured deviation is *outward* while the static
sag is *inward*, so the policy is also commanding an offset — see "Is the defect
profitable?" for the decisive test.

---

## Scale: defect cost as % of total reward

Denominator: **T = 5.42 reward units/step** (iteration 14999 log, as derived above). An
independent reconstruction at the audited eval state — replacing the training-time values of
the terms that the given measurements determine (`upright` 1.888 from tilt 2.9°; `pose`
0.0213; `head_pose_tracking` 1.907 with cmd = 0; `head_pose_bias` −0.070; `body_ang_vel`
−0.030 from the 2.25 Hz trunk rates; `track_angular_velocity` 0.582 from the measured
0.133 rad/s drift plus the trunk rates; `action_rate_l2` −0.13 from mean|Δa| = 0.0755,
assuming Δa ≈ Gaussian) gives **6.8/step**; the percentages below move by ≤ 25 % between the
two denominators, so the conclusions are scale-insensitive.

| Item | cost/step | **% of total** | Equivalent phrasing |
|---|---|---|---|
| `neck_pitch` ±15° under `head_pose_tracking` | 0.0620 | **1.14 %** | 4.3 % of the `air_time` reward |
| `head_pitch` ±3.9° under `head_pose_tracking` | 0.0046 | 0.08 % | — |
| **all four head joints oscillating** | **0.0934** | **1.72 %** | 6.5 % of `air_time`; 5.5 % of `track_linear_velocity` |
| the oscillation seen by `head_pose_bias` (EMA) | 0.0191 | 0.35 % | the EMA removes 92.8 % of it |
| **head wobble, oscillation only, total** | **0.1125** | **2.08 %** | — |
| head wobble incl. the measured DC bias | 0.2556 | 4.71 % | 13 % of `head_pose_tracking`'s own maximum (2.0) |
| leg splay (both `hip_roll` at 5.0°/3.6°) | 0.0207 | **0.38 %** | 1.4 % of `air_time` |
| toe-out (both `hip_yaw` at 16°) | 0.0042 | **0.077 %** | **1/347 of `air_time`** |
| **splay + toe-out, fully corrected** | **0.0289** | **0.53 %** | — |
| *(for scale)* the yaw drift itself, via `track_angular_velocity` | 0.070 | 1.29 % | — |
| *(for scale)* `dof_pos_limits` already being paid (ankle on its soft limit) | 0.078 | 1.44 % | 2.7× the whole leg-splay price |
| *(for scale)* expected cost of falling (1.7 % of episodes end early) | 0.046 | 0.85 % | 1.6× the whole leg-splay price |

**Answers to the two headline questions.**
* Swinging the neck ±15° costs **1.14 %** of the total by itself, **1.7 %** for the whole
  four-joint wobble, **2.1 %** including the EMA term — i.e. the policy pays about **1 part
  in 50** for the defect the user is complaining about.
* Holding a 16° common-mode toe-out costs **0.077 %** — about **1 part in 1300**; even the
  full splay *and* toe-out correction together is only **0.53 %**.

---

## Bugs and traps found

**1. `pose`'s collective Gaussian (`rewards.py:466`) — the dominant defect in the recipe.**
`exp(−mean(err²/std²))` over 10 joints converts per-joint tolerances into a *posture-error
budget*. The per-joint effective tolerance is `√10 × std`, and once walking forces the ankle
(9.6 + 11.4 exp-units) and right hip_pitch (5.5) deep into the tail, the whole term reads
0.021 and every marginal per-degree price collapses (0.0002–0.0034/step). Consequence: the
term cannot police *any* joint in this regime, including the three it was written for.
Not a sign error, not a weight error — an aggregation error.

**2. `hip_roll` std = 0.05 rad is below the actuator's own load error (achievable-floor bug).**
`std_standing` and `std_walking` both use 0.05 (`:315`, `:326`) = 2.86°, whereas the FK
gravity moment on hip_roll (0.0320 N·m) at the identified P-stiffness (0.150 N·m/rad,
ki = 0) yields 12.2° of static sag, and the measured joint wanders with σ = 4.56°. The
"0.1→0.06→0.05" tuning history in the comment (`:315`) tightened a target that the hardware
does not hold. Note this argument does *not* apply to `hip_yaw`: its axis is vertical, the
gravity moment about it is exactly 0, and a 16° sustained offset therefore cannot be sag.

**3. `head_pose_bias_penalty` double gate — confirmed present in the code, confirmed NOT
wired for this task.** `mdp.py:5302` gates `err` (`err = err * gate`) and `mdp.py:5315-5316`
gates `out` (`out = out * gate`), so wherever a gate is passed the effective weight is
gate². In `microduck_velocity_env_cfg.py:1140-1144` the term is constructed with **only**
`{"command_name": "head_pose", "tau_s": 1.0}` — `gate_height_low` is absent, so
`mdp.py:5283` takes the `else` branch, `gate = None`, and neither multiply executes.
`params/env.yaml` confirms it (no gate keys). The gated users are the recovery tasks
(`microduck_standup_env_cfg.py:267+`, `microduck_velstand_env_cfg.py:184`,
`microduck_sitstand_env_cfg.py:292+`), and `tests/test_head_pose_bias.py:157-163` locks the
velocity env's ungated status. **The velocity env is unaffected by this defect.**

**4. `angular_momentum` at weight −0.02 is a dead term.** Logged `−0.0000` with
`Metrics/angular_momentum_mean = 0.0086` kg·m²/s → the penalty is `0.02 × 0.0086² = 1.5e-6`.
Meanwhile the head alone contributes a *rms* 0.0031 kg·m²/s (36 % of that mean). A term that
was presumably intended to stop limbs thrashing prices the head's momentum at zero. Same
family: `foot_slip` (−0.1) logs −0.0005, and because both are *squared* forms their gradient
vanishes near the operating point instead of acting as a constant penalty.

**5. `body_ang_vel` cannot see either defect.** `asset_cfg.body_names = ("trunk_base",)`
(`:489`) plus `ang_vel[:, :2]` (`rewards.py:192`) excludes the head *and* the yaw axis. The
trunk yaw drift (1.6 rad/12 s) and all neck/head rotation are outside it by construction.

**6. No heading term exists — `rel_heading_envs = 0.0` (`:996`).** `track_angular_velocity`
prices instantaneous rate only (`rewards.py:47-64`; its docstring advertises a heading mode
that the implementation does not contain). With std = 0.707 rad/s, the audited drift
(0.133 rad/s) is 0.19σ → **97.4 % of full credit retained** for a robot walking in a circle.
Training corroborates that yaw is essentially uncontrolled: `Metrics/twist/error_vel_yaw =
2.5657` rad/s at iteration 14999, i.e. the mean |ωz error| is 2.5× the entire commanded
range (±1.0). Note also the standing bucket resamples the command to exactly (0,0,0) for
25 % of envs (`standing_envs_curriculum`, final stage 0.25) — a large part of the experience
is spent at a command where the drift is not visible at all.

**7. Sign conventions — all clean in this env.** Every penalty in the iteration-14999 log is
≤ 0, satisfying the AGENTS.md rule: `body_ang_vel −0.0838`, `angular_momentum −0.0000`,
`dof_pos_limits −0.0766`, `action_rate_l2 −0.5824`, `foot_clearance −0.0077`,
`foot_swing_height −0.0159`, `foot_slip −0.0005`, `self_collisions −0.0102`,
`head_pose_bias −0.2675`. The two self-negating microduck functions carry positive weights
(`head_pose_bias` +3.0) as intended; `body_pose_tracking` at exactly 0.0 is skipped by the
manager (`reward_manager.py:122-124`), so it costs nothing and logs 0 by design.
No jackpot-style gating of a positive reward on a bad state exists in this env (the only
gates are command gates at `> 0.01`).

**8. Reward mass is ~6:1 against the penalties.** Positives: 1.296 + 0.227 + 1.709 + 0.018 +
1.446 + 1.677 = **6.37**; negatives: 0.084 + 0.000 + 0.077 + 0.582 + 0.008 + 0.016 + 0.001 +
0.010 + 0.268 = **1.04**. Any posture that buys a few percent of the positive
stack pays for itself many times over — the *structural* reason both defects survive.

**9. Housekeeping (not defects, but relevant to "why nothing stops it"):**
`feet_distance_penalty` (`mdp.py:5571`), `no_stepping_penalty` (`:5546`),
`gait_symmetry_penalty` (`:4843`) and `joint_pos_limit_proximity` (`:1989`) are implemented
in `mdp.py` but wired into **no part of the velocity recipe** (grep over the env cfgs). The
only lateral/posture instruments actually installed are `pose` (saturated) and `upright`.

---

## Is the defect profitable?

Measured/hypothesis split, with the decisive measurement for each.

### A. Head/neck wobble as a balance actuator — *hypothesis, mechanism quantified, sign of the net payoff unknown*

The physical authority is large enough to matter (all FK-derived, 779 g / 289 g model):

| Quantity | Value |
|---|---|
| head assembly mass | 0.28893 kg = **37.1 %** of 0.77924 kg |
| head CoM height above ground | 0.2279 m (whole-body CoM: 0.1439 m) |
| perpendicular lever from the head CoM to the `neck_pitch` axis | **0.0757 m** |
| head CoM excursion for ±15° | 19.6 mm amplitude |
| resulting whole-body CoM shift | **7.26 mm** (per side, fore-aft) |
| peak head CoM acceleration at 2.25 Hz, 19.6 mm | 3.92 m/s² → inertial force 1.13 N |
| **resulting ZMP shift** (`F·z/(m·g)`) | **33.7 mm** |
| for comparison | CoM randomization is capped at ±15 mm (`:1294-1304`), the heel is ~20 mm behind the ankle, the contact patch is 119 mm wide |
| head rms angular momentum about the neck axis | 0.0031 kg·m²/s = **36 %** of the measured whole-body mean |
| price of that momentum under `angular_momentum` | **−1.9e-7/step** |

So the head *can* move the support point by ~34 mm at the gait frequency and is charged
~0.11/step for doing it. Whether that is *net profitable* (as opposed to merely affordable)
cannot be read off the reward code — it depends on how much of `track_linear_velocity`,
`upright` and `air_time` the wobble buys back. There is one genuine reason to suspect it is
**not** a strategy at all: at 0.150 N·m/rad and I = 1.18e-3 kg·m² about the neck axis, the
neck's natural frequency is **1.79–1.88 Hz**, the actuator's inertia-driven deflection is
**7.6° per m/s²** of base acceleration, and 15° therefore needs only ~2 m/s² (0.2 g) at
2.25 Hz — entirely ordinary for this robot. That reading makes the wobble a *resonant,
soft-servo, largely unescapable* artifact rather than a choice.

**Decisive measurements (all available from the existing re-sim, no new training):**
1. Log the ONNX's 14 action channels alongside qpos (the raw action is already a delta from
   HOME — `use_default_offset=True`, `actions.py:218-219`). Compare the raw `neck_pitch`
   action with `qpos − HOME`. If the raw command is ~flat (near its HOME-relative mean) while
   the joint oscillates by the deflection predicted by `θ = m·d·a_base/k` (7.6° per m/s², in
   phase with the *negative* base acceleration), the wobble is passive; if the raw command
   carries the ±15° itself, it is commanded.
2. Cross-correlate `neck_pitch` with the base fore-aft acceleration and with the CoM
   excursion: a passive forced response lags the acceleration by ~180° and is in phase with
   `−x_CoM`; an active balance pump leads the CoM by ~90°.
3. Report the kurtosis/duty-cycle of `neck_pitch`. The Gaussian pays for *dwell* at the
   extremes, not for peak amplitude: a sinusoid of ±15° costs 0.128 per joint, whereas the
   same ±15° excursion with a 10 % duty cycle costs 0.0128 — a **10× discount for spiky
   neck motion**. If the measured trace is spikier than sinusoidal (kurtosis > 1.5), the
   reward is actively shaping the waveform, not just tolerating it.

### B. Leg splay widens the effective support polygon — *hypothesis, magnitude measured*

FK, holding everything else at HOME and varying only `hip_roll`/`hip_yaw` (contact patch =
lowest 2 mm of the sole mesh; "full reach" = the whole sole):

| Configuration | patch centres (separation) | contact polygon width | full sole reach width |
|---|---|---|---|
| HOME (5° knees-in) | ±45.2 mm (90.5) | 119.2 mm | 127.2 mm |
| legs vertical (`hip_roll` dev +5°/+5°) | ±41.6 mm (83.1) | **128.0 mm (+8.8)** | **140.7 (+13.5)** |
| + audited mean `hip_roll` dev | +41.2/−42.8 (84.0) | **125.6 (+6.4)** | **141.3 (+14.1)** |
| + audited mean `hip_yaw` (common-mode 16°) | +47.3/−41.0 (88.3) | 116.2 (−3.0, **asymmetric: [−54.9, +61.3]**) | 133.0 |

So the observed abduction is worth **≈ +6 to +14 mm of lateral support** — comparable to the
±15 mm CoM randomization the curriculum deliberately caps for stability reasons — while the
pose reward charges **0.021/step** for it. Meanwhile the expected cost of *falling* (1.7 % of
episodes end early, mean 983-step episodes, 5.42/step ⇒ ≈ 0.046/step) and of a *tilt*
(`upright` weight 2.0, std² = 0.05) both exceed the entire posture price. In other words:
**a posture change that reduces lateral fall risk by even a few percent is already worth more
than the pose term charges for the whole splay, and the 16° toe-out shrinks one side of the
support polygon by 4.7 mm for which it pays 0.004/step.** The reward structure cannot
discriminate; the balance incentive dominates the posture incentive by ~2–3×.

**Decisive measurements:**
1. **Action trace — the single most valuable measurement.** The raw policy output *is* the
   deviation from HOME (`JointPositionActionCfg.use_default_offset = True`,
   `actions.py:218-219`; `scale = 1.0` at `microduck_velocity_env_cfg.py:429`), so plot the
   raw `hip_roll` action channels against the measured joint positions. If the raw action
   sits at ≈ **+19° (left) / −16° (right)** while the joint sits at +1.6°/+1.5° absolute
   (mean deviations +6.56°/−3.55° from HOME), the observed splay is an *outward command
   partially eaten by the 12.2° inward gravity sag* → the reward is asking for something the
   hardware half-refuses, and the fix is the std (`R3`), not the weight. If instead the raw
   action itself sits at ≈ +1.6°/+1.5°, the pose is a deliberate choice → raise the price
   (`R2`). The FK prediction for the sag (0.0320 N·m ÷ 0.150 N·m/rad = 12.2° *inward*)
   already implies today's +6.6° *outward* must come from an outward command, but ±3 % mass
   and the unknown walking load make the trace the arbiter.
2. Ablate the splay: re-sim with `hip_roll`/`hip_yaw` clamped to HOME. If `pose` rises by
   ~0.03/step while `air_time` + `upright` + `track_*` fall by more than ~0.05/step (or the
   fall rate rises), the defect is profitable; the reward engineering then has to attack
   *why* the wide stance is cheaper (lateral CoM control) rather than the symptom.
3. Log the yaw moment / lateral GRF asymmetry vs `hip_yaw`. A common-mode 16° foot yaw
   predicts a systematic lateral push-off force of `sin16° ≈ 0.28` of the leg force; if the
   measured lateral impulse correlates with the 0.133 rad/s drift, defect B1 causes defect
   B2 and the fix belongs in the yaw/heading term, not in `pose`.

---

## Recommended reward changes

Ranked by expected effect per unit of risk. All line numbers are
`src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py` unless stated otherwise; new
functions belong in `src/mjlab_microduck/tasks/mdp.py` per AGENTS.md.

**R1 — Price heading *position*, not just rate (fixes the circle-walking).** Add a
`heading_error_l1`-style term (or set `command.rel_heading_envs` > 0 at `:996` and add the
missing heading branch to the tracking reward, which the stock function advertises but does
not implement — `rewards.py:47-64`) using `wrap_to_pi(yaw)` against a heading target latched
per command window, e.g. weight 0.5 at the top of the cfg with a 0.3 rad deadband, ramped
from 0 at iter 2000. *Expected effect:* the only term that actually sees a 1.6 rad drift;
0.133 rad/s costs 0.070/step today (the drift retains 97.4 % of the full rate-tracking
credit), whereas an L1 at weight 0.5 with a 0.3 rad deadband charges **0.65/step** for the
same 1.6 rad, i.e. ~9× more.
*Risk:* low — it is a new positive reward with a rate-limited/slewed target, not a tightened
std; must be introduced after walking is consolidated (start ramping ≥ iter 1500) or it will
compete with gait discovery; do **not** gate anything positive on being upright.

**R2 — Replace the collective pose Gaussian with a per-joint mean (or add a targeted L1).**
Either write a microduck-local `pose_mean_of_gaussians` (copy `variable_posture.__call__`
semantics but return `mean_i exp(−err_i²/std_i²)`, `rewards.py:462-466`) or keep the stock
term and add `joint_deviation_l1` (`mdp.py:1971`) on the two `hip_yaw` + two `hip_roll`
joints at a small positive weight. *Expected effect, computed both ways at the measured
state:* the **full** splay/toe-out correction goes from **0.0289/step (0.53 %)** to
**0.296/step (5.5 %)** — a **10.2×** larger incentive — and for a *small* perturbation the
response is ~8.6× larger (a single 5° `hip_roll` error moves the collective form by 0.011
but the per-joint form by 0.095). Note the honest nuance: exactly *at* today's 6.56°
operating point the per-joint Gaussian is itself 2.3σ deep, so its local slope there
(−0.0008/step per degree) is no better than the collective form's (−0.0034); the win is in
the integrated gradient over the deviations the gait actually visits. *Risk:* medium — a
per-joint mean makes the term ~10× more sensitive to *all* 10 joints, including the ankle
excursion that walking requires (exp-units 9.6 + 11.4 today); if it is switched wholesale the
gait may stiffen exactly as the `fine_std=0.1` experiment did (`:1116-1139`). Mitigation:
keep the stock term and add the L1 only on the 4 yaw/roll joints, weight ≤ 0.3, ramped in
after iter 1500. *Note the codebase's own rule: price only the escapable part* — hip_yaw is
fully escapable (vertical axis, 0 gravity moment); hip_roll is not (12.2° identified sag), so
the L1 on hip_roll should use a deadband of ≥ 0.15 rad (8.6°) rather than charging from 0.

**R3 — Stop asking a 2.86° Gaussian to police a load-dominated joint: relax `hip_roll` std
and/or deadband it.** `:315` and `:326` (`.*hip_roll.*: 0.05` in both dicts) → raise to
**0.12–0.15 rad** (= the identified static sag + 1σ of the measured wander) so the target is
*reachable*, and let R2's deadbanded L1 do the policing outside that band. *Expected
effect:* converts a saturating target into an informative one; removes the incentive to
park `hip_roll` near a limit to fight sag. *Risk:* low for the term itself, but it *loosens*
posture — pair it with R2 so the net posture pressure rises rather than falls. Do not touch
`hip_yaw`'s std (0.3) — that joint is fully escapable and should be tightened if anything
(0.3 → 0.15 rad).

**R4 — Give the head wobble a *waveform-aware*, escapable-only price instead of a bigger
Gaussian.** Keep `head_pose_tracking.std = 0.5` untouched (the documented failure mode), and
make the existing escapable DC term slightly better posed: add a small **band-pass/L1 term on
the 1–4 Hz component of `neck_pitch` measured against the trunk's own pitch acceleration**
(e.g. `−w·ReLU(|neck_accel| − α·|trunk_accel| − β)`, w ≈ 0.2), so the price is levied on the
part of the wobble *not* explained by base excitation, with β ≈ the identified passive
deflection (7.6° per m/s²). *Expected effect:* prices the escapable residue without the
0.77/step unescapable tax that killed run 5yay13u4. *Risk:* medium-high (new custom reward,
must be introduced late, needs a measure of base acceleration that the actor can observe —
`body_ang_vel` is in the obs but linear acceleration is not; consider using
`Δ(projected_gravity)`/`Δ(base_ang_vel)` instead). **The honest alternative, if the
measurement in "Is the defect profitable? A" shows the wobble is passive: no reward change —
fix it with servo stiffness (`kp_fw` 32 → higher, or a firmware `kd` that is actually
identified — `characterization.json` notes the effective D gain is unmeasured), because the
reward cannot reach a resonance.**

**R5 — Sanity rails, cheap and independent.** (a) Delete or re-purpose the dead
`angular_momentum` term (`:531`, doubled from the template at `:634`): at −0.02 with
L² ≈ 0.0086² it is numerically inert; if the intent was "don't use the head as a reaction
wheel", replace it with a new `angular_momentum_l1` in `mdp.py` (none exists — grep shows
only L1 helpers for poses/rates) weighted ~0.5 for a constant gradient, and *then* measure
whether head swinging drops. Note the spin env already refuses this term because its 3-D norm
punishes yaw momentum that task needs (`microduck_spin_env_cfg.py:122`) — the same
objection applies here, which is why the yaw component must be excluded or handled by R1
instead. (b) Add
`joint_pos_limit_proximity` (`mdp.py:1989`) on the ankles, which are parking on their soft
limit (the policy pays `−0.0766/step` in `dof_pos_limits`, 2.7× the whole splay price);
AGENTS.md prescribes exactly this fix. (c) If `pose`'s standing/walking switch is to be kept,
make the transition continuous in the command (blend the two stds over 0–0.05 m/s) instead
of the cliff at 0.01 (`:458`), where the term drops from 0.021 to 4e-10.

**Explicitly NOT recommended:** tightening `head_pose_tracking.std` (documented to destroy
walking, `:1116-1139`); any positive-reward gate on an upright/standing condition; raising
`pos`/`action_rate` weights to squeeze the splay indirectly (action_rate is the only term
touching the neck today and it is already at −0.5824/step; raising it further taxes gait
discovery, which is the documented failure mode of early smoothness taxes).

---

### One-line summary of the arithmetic

| Question | Answer |
|---|---|
| % of total reward to swing the neck ±15° | **1.14 %** (neck only), **1.72 %** (4 joints), **2.08 %** with the EMA term |
| % of total reward to hold a 16° toe-out | **0.077 %** (marginal price 0.00024 reward/deg) |
| % of total reward for the whole splay + toe-out | **0.53 %** |
| Is `std_walking` in force while walking? | Yes — `walking_threshold = 0.01`, `total_speed = 0.2` |
| Is `hip_roll` std = 0.05 reachable? | Not reliably — identified 12.2° static sag at 0.150 N·m/rad, ki = 0 |
| Does the double-gate defect affect this env? | No — no gate params are passed (`:1140-1144`) |
| Does anything make the defect profitable? | Yes, structurally: fall avoidance ≈ 0.046/step and the lateral support gain ≈ 6–14 mm both exceed the 0.029/step posture price |
