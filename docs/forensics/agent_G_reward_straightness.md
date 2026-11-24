# Agent G — is straight-line walking priced, observable, and learned?

Scope: **training policy / reward / observations / commands / curriculum / optimisation only**
(actuator data and robot-model geometry belong to other agents; referenced where needed).
Method: static reading of the recipe + **pure-CPU** parsing of recorded logs and traces
(tensorboard tfevents, `docs/forensics/artifacts/**/*.csv|json`, ONNX metadata + CPU
onnxruntime probes, MJCF joint axes via CPU `mj_forward`). **No training, no MuJoCo-Warp sim,
no GPU was used.** No source file was modified.

Evidence base (all reproducible):

| what | where |
|---|---|
| main walking recipe | `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py` (1676 lines) |
| custom MDP | `src/mjlab_microduck/tasks/mdp.py` (7509 lines) |
| stock reward math | `.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/mdp/rewards.py` |
| stock command sampling | `.venv/.../mjlab/tasks/velocity/mdp/velocity_command.py` |
| **newest run** (HD-1910 retrain, 12000 iters, 2026-09-11 20:36 → 09-12 00:18) | `logs/rsl_rl/velocity/Mjlab-Velocity-Flat-MicroDuck/` + `params/env.yaml` |
| frozen recipe of that run | `logs/rsl_rl/velocity/Mjlab-Velocity-Flat-MicroDuck/params/env.yaml:2107-2200` |
| straight-line traces (16 env, vx=0.2, vy=wz=0, 20 s) | `docs/forensics/artifacts/straightness/hd1910_base16.csv` / `.json` |
| old (pre-fix) gait trace | `docs/forensics/artifacts/gf_walk_hd1910_15000_12s.csv` |
| `model_11999` ONNX | `logs/rsl_rl/velocity/Mjlab-Velocity-Flat-MicroDuck/Mjlab-Velocity-Flat-MicroDuck.onnx` |

---

## 0. Verdict

**Yes — the training/reward side plausibly explains the veering. Confidence 高 (high) for
"the recipe cannot price straightness", 中 (medium) for "that is the dominant cause on
hardware".**

Strongest evidence, in order:

1. **Heading is not in the reward at all.** There is no term anywhere in the stack that is a
   function of yaw *angle*. The only yaw-related term (`track_angular_velocity`) prices the
   instantaneous **rate**, through a Gaussian whose combined error is already **1.29 rad/s**
   (raw 0.0357 of 1.0), so a DC yaw bias of 0.1 rad/s — which integrates to **115° of heading
   in 20 s** — costs **0.0014/step = 0.04 % of the return**. Even 0.5 rad/s costs 1.1 %.
   (§1)
2. **Heading and lateral *position* are not observable either.** The actor sees yaw **rate**
   but nothing that tells it it has drifted. Confirmed from the cfg **and** the ONNX metadata.
   The drift state is therefore open-loop and unobservable — the "integrate with no feedback
   path" hypothesis is **verified**. (§2)
3. **The exported `Episode_Reward` breakdown says the same thing: the whole
   angular-velocity objective is 2.1 % of the return** (0.0714 of 3.367/step), against
   `track_linear_velocity` 32.1 %, `head_pose_tracking` 43.6 %, `upright` 41.4 %. The
   task-relevant "go the way you are pointed" signal is a rounding error. (§1)
4. Measured behaviour matches: at the frozen deployment command the policy's **trunk yaw rate
   is 66 deg/s RMS (1.16 rad/s) with only a −4 deg/s DC component** (AC/DC = 16.7×) — the
   drift is a ~6 % **rectification** of a step-locked yaw oscillation produced by the *legs*
   (half-stride yaw asymmetry **+28.9 ± 11.7 deg/s**). A rate-Gaussian priced against a
   1.16 rad/s oscillation cannot see the rectified 0.07 rad/s.
5. **The two historically named suspects are both exonerated by direct measurement**:
   * hip_yaw-at-limit (the 28° duck stance) is **fixed** in `model_11999` — mean
     `hip_yaw` **−1.97° / +2.03°** (was +30.4°/+25.4° *at the mechanical stops*), (§6)
   * the head wobble is a **co-symptom, not the cause** — the current policy's head is nearly
     static in sim (1.8–2.3° std) and the direct head-action ablation *worsens* the drift
     (user's measurement; consistent with my head analysis, §8).
   The defect therefore survives both fixes, which is exactly what a *missing heading term*
   predicts.

Not claimed: that a heading term alone fixes it on hardware. The rectification source is
mechanical (legs), and a heading term only **prices** it — it still has to be learnable through
the available feedback (yaw rate, observed with up to 6° of random-axis IMU misalignment).

---

## 1. Q1 — Is straight-line walking priced? Reward mass, per term

Weights are the live values at iter 11999 from `params/env.yaml` and the curves below.
`Episode_Reward/<term>` is `mean(weight × raw × dt) / max_episode_length_s`
(`mjlab/managers/reward_manager.py:100-131`), i.e. the **time-average of weight × raw** = per-step
weighted value. Read with:

```bash
.venv/bin/python - <<'EOF'
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
p='logs/rsl_rl/velocity/Mjlab-Velocity-Flat-MicroDuck'
ea=EventAccumulator(p,size_guidance={'scalars':0}); ea.Reload()
print({t:dict((e.step,e.value) for e in ea.Scalars(t))[11999] for t in ea.Tags()['scalars'] if t.startswith('Episode_Reward/')})
EOF
```

| term | weight | `Episode_Reward` @11999 | raw | % of net | prices straightness? |
|---|---|---|---|---|---|
| `head_pose_tracking` | +2.0 | **+1.4685** | 0.7342 | **43.6 %** | no (command-following) |
| `upright` | +2.0 | +1.3922 | 0.6961 | 41.4 % | roll/pitch only — **no yaw** |
| `track_linear_velocity` | +2.0 | +1.0811 | 0.5405 | **32.1 %** | **yes, weak** — includes `vy` (see below) |
| `action_rate_l2` | −1.0 | −0.9608 | 0.9608 | −28.5 % | no (all 14 dims) |
| `air_time` | +3.0 | +0.6078 | 0.2026 | 18.1 % | no (command-gated only) |
| `pose` = `per_joint_posture` | +1.0 | +0.4917 | 0.4917 | 14.6 % | **symmetric-to-HOME**; neck/head excluded |
| `neck_motion` | −0.08 | −0.2768 | 3.4598 | −8.2 % | no |
| `head_pose_bias` | +3.0 | −0.2604 | −0.0868 | −7.7 % | no (DC head only) |
| `body_ang_vel` | −0.05 | −0.1271 | 2.5411 | −3.8 % | **no — z excluded by construction** |
| `no_stepping` | −0.4 | −0.1060 | 0.2651 | −3.2 % | only at cmd≈0 |
| `stillness_zero_cmd` | +0.8 | +0.0738 | 0.0922 | 2.2 % | only at cmd≈0 |
| **`track_angular_velocity`** | **+2.0** | **+0.0714** | **0.0357** | **2.1 %** | **yaw RATE only, saturated** |
| `stance_hip_yaw` | −0.8 | −0.0416 | 0.0520 | −1.2 % | symmetric hip_yaw L1 |
| `dof_pos_limits` | −1.0 | −0.0173 | 0.0173 | −0.5 % | hard limits |
| `foot_swing_height` | −0.25 | −0.0172 | 0.0688 | −0.5 % | no |
| `foot_clearance` | −2.0 | −0.0062 | 0.0031 | −0.2 % | no |
| `action_over_limit` | −1.0 | −0.0035 | 0.0035 | −0.1 % | no |
| `self_collisions` | −1.0 | −0.0018 | 0.0018 | −0.1 % | no |
| `foot_slip` | −0.1 | −0.0008 | 0.0080 | −0.02 % | weak (xy slip in contact) |
| `angular_momentum` | −0.02 | **−8.9e-7** | 1.4e-5 | **0.00 %** | **dead** (see §8) |
| `body_pose_tracking` | 0.0 | 0.0000 | — | 0 | disabled by design |

**NET = +3.367/step** (positive stack 5.186, negative stack −1.819, ratio 2.85 : 1).
(The older "≈106/step" figure in `AGENTS.md` is not this run's scale; the `Train/mean_reward`
69.5 per episode ÷ 20.6 s = 3.37/step confirms the numbers above.)

### 1.1 `track_linear_velocity` — does it cover y?

Yes. `mjlab/tasks/velocity/mdp/rewards.py:41-44`:

```python
xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)   # includes vy
z_error  = torch.square(actual[:, 2])
return torch.exp(-(xy_error + z_error) / std**2)                            # std² = 0.1
```

weight **+2.0**, `std = math.sqrt(0.1)` (`microduck_velocity_env_cfg.py:765`, std²=0.1).
So lateral drift *is* priced — but (a) the actor cannot observe `vy` (§2), (b) the policy is
already deep in the Gaussian tail, which halves the marginal price, and (c) the price is
quadratic near zero, i.e. **flat exactly where we care**:

marginal cost of adding a DC lateral bias on top of the current operating point
(`raw = 0.5405 = exp(−e²/0.1)` → e = 0.248 m/s):

| lateral `vy` bias | cost/step | % of net | lateral travel in 20 s |
|---|---|---|---|
| 0.02 m/s | 0.0043 | 0.13 % | 0.40 m |
| 0.05 m/s | 0.0270 | 0.80 % | 1.00 m |
| 0.10 m/s | 0.1081 | 3.2 % | 2.00 m |
| 0.20 m/s | 0.4324 | 12.8 % | 4.00 m |
| 0.30 m/s | 0.9730 | 28.9 % | 6.00 m |

So lateral drift is **partially priced** (a 0.2 m/s crab-walk would cost 13 %), but note the
optimum is *unbiased*, not *straight*: the term is symmetric in sign, so left-drift and
right-drift are equally acceptable, and the actor has no `vy` observation with which to close
the loop on it.

### 1.2 `track_angular_velocity` — does it cover yaw?

Yes, on the **rate** only (`rewards.py:61-64`):

```python
z_error  = torch.square(command[:, 2] - actual[:, 2])   # yaw RATE error
xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)  # roll/pitch rates
return torch.exp(-(z_error + xy_error) / std**2)          # std² = 0.5
```

weight **+2.0**, `std = math.sqrt(0.5)` (`:767`). Raw = 0.0357 → combined error **1.291 rad/s**.
The docstring advertises "reward heading error for heading-controlled envs" — **the heading
branch does not exist in the implementation** (it was never called because
`rel_heading_envs = 0.0`, see §3).

Marginal cost of adding a DC yaw-rate bias (with `exp(−e²/0.5) = 0.0357`):

| yaw-rate bias | cost/step | % of net | heading change over 20 s |
|---|---|---|---|
| 0.02 rad/s | 0.000057 | **0.0017 %** | 22.9° |
| 0.05 rad/s | 0.000357 | **0.011 %** | 57.3° |
| 0.10 rad/s | 0.001428 | **0.042 %** | **114.6°** |
| 0.20 rad/s | 0.005712 | 0.17 % | 229° |
| 0.50 rad/s | 0.035702 | 1.06 % | 573° |

**A yaw-rate bias of 0.1 rad/s — enough to turn the robot 115° in one episode — is free at the
0.04 % level. Heading (integrated) is not priced at all: no such term exists.**
The user's independent measurement agrees in shape: `track_angular_velocity` +0.406/2.0
(20 % of its ceiling).

### 1.3 Terms that *look* like yaw damping but are not

* `body_ang_vel` (w −0.05, mass −0.127): `body_angular_velocity_penalty` reads
  `ang_vel[:, :2]` — **"Don't penalize z-angular velocity"** (`rewards.py:192`) and
  `asset_cfg.body_names = ("trunk_base",)` (`cfg:663`) so the head/neck are out too.
* `angular_momentum` (w −0.02): returns `‖L_root‖²` (`rewards.py:203`) — a **squared** form, so
  its gradient vanishes at the operating point. Measured mass **−8.9e-7/step = 0.00 %**.
  Confirmed dead. It is also un-gated in yaw, so raising it would fight the (desired) turning
  skill — the spin env disabled this term for exactly that reason
  (`microduck_spin_env_cfg.py:122`).
* `foot_slip` (w −0.1, mass −0.0008): the only term touching lateral foot velocity in contact,
  deliberately weak ("-1.0 was too restrictive for this robot's pivot-heavy turning",
  `cfg:665-669`). Effectively off.

---

## 2. Q2 — Is it observable? (61-D actor layout, verified)

From the instantiated cfg (dump the terms in insertion order — that *is* the concatenation
order) and confirmed by the ONNX metadata `observation_names`:

```
idx   term                width   source
0-2   base_ang_vel          3      microduck_mdp.base_ang_vel_imu_misaligned (±6° random axis)
3-5   projected_gravity     3      microduck_mdp.projected_gravity_imu_misaligned
6-19  joint_pos            14      joint_pos_rel (HOME-relative, encoder-biased)
20-33 joint_vel            14      joint_vel_rel (1-ctrl-step delay)
34-47 actions              14      last_action
48-50 command               3      twist  = (vx, vy, wz)
51-54 head_command          4      head_pose deltas
55-60 body_command          6      body_pose deltas
                            = 61  (48 proprio + 13 command block)   ✓ AGENTS.md invariant
```

* **`base_lin_vel` is DELETED from the actor** — `cfg:1045 del cfg.observations["actor"].terms["base_lin_vel"]`,
  re-added to the **critic only** at `cfg:1058` ("real robot can't measure base linear
  velocity"). The comment in the cfg says the actor must not see it.
* `USE_PROJECTED_GRAVITY = True` (`cfg:214`) → the accelerometer channel is *projected gravity*,
  so there is **no linear-acceleration information** either. The actor's only motion sensors
  are the gyro (3) and 14 joint positions/velocities.
* **No heading, no yaw angle, no position** in either group. `projected_gravity` is yaw-invariant
  by construction, so yaw is unobservable *except* through joint/contact consequences.

**Verified hypothesis (high confidence):** the policy observes yaw **rate** but has **no
feedback path for heading**. Combined with §1.2 (rate bias ≈ free), a small per-step yaw bias
integrates into unbounded drift with nothing in the reward or the observation to stop it.
Two aggravating factors in the *observation* of the one signal it does have:

* IMU misalignment DR (`cfg:1141-1148`, `IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0`, random
  axis, per-env constant) rotates `base_ang_vel`, so the observed `ω_z` is contaminated by the
  roll/pitch rates: up to `sin 6° ≈ 0.105 × |ω_xy|`. Measured trunk rate RMS during walking is
  ~1.16 rad/s, so the **cross-talk is of order 0.1 rad/s — the same size as the drift we care
  about**, and it is a *per-env constant* rotation, i.e. it looks like a DC yaw bias to that
  env. (Because it is a rotation of a 3-vector, the actor can in principle invert it — but only
  if it can identify the misalignment, which it cannot distinguish from real motion.)
* `base_ang_vel` noise ±0.03 (`cfg:1127`) plus 0–1 control-step delay (`cfg:1094-1096`).

So even the rate channel is a weak instrument for a 0.07 rad/s DC component.

---

## 3. Q3 — Command distribution: what fraction of experience is the deployment condition?

Exact values from the frozen recipe of the newest run
(`params/env.yaml:2107-2136`) and `cfg:1216-1242`:

| field | value | source |
|---|---|---|
| `heading_command` | **True** (inherited from mjlab) | `env.yaml:2113` |
| `rel_heading_envs` | **0.0** | `cfg:1223` — **kills the built-in heading regulator** |
| `rel_standing_envs` | 0.02 → 0.25 (curriculum) | `cfg:1222`, `cfg:1451-1464` |
| `rel_forward_envs` | **0.2** (inherited, *not* overridden) | `env.yaml:2118` |
| `rel_turn_in_place_envs` | 0.15 | `cfg:1242` |
| `rel_world_envs` | 0.0 | default |
| `init_velocity_prob` | 0.0 | default |
| ranges | `lin_vel_x ±0.4`, `lin_vel_y ±0.3`, `ang_vel_z ±1.0` | `cfg:1231-1233` |
| resample period | 3–8 s (template default) | `velocity_env_cfg.py:180` |
| `head_pose.zero_command_prob` | **0.0** | `env.yaml:2155` |
| `body_pose.zero_command_prob` | **0.0** | `env.yaml:2180` |

**Good news the brief did not expect.** `rel_forward_envs = 0.2` is **inherited from the
upstream velocity template and was never overridden**, and mjlab implements it exactly as the
deployment condition (`velocity_command.py:79-87`):

```python
# Forward-only envs: positive lin_vel_x, zero lateral and angular.
self.vel_command_b[fwd_ids, 1] = 0.0
self.vel_command_b[fwd_ids, 2] = 0.0
self.vel_command_b[fwd_ids, 0] = ...abs().clamp(min=0.3)   # vx ∈ [0.3, 0.4]
```

So an **exact `(vy = 0, wz = 0, vx ∈ [0.3, 0.4])` bucket exists** — the AGENTS.md warning
("uniform sampling makes the all-zero command essentially never occur") applies to
`rel_standing_envs`/`head_pose`, but *not* to the straight-walk bucket:

```
exact straight-walk coverage ≈ 0.20 forward × (1 − 0.15 turn-in-place overwrite)
                              × (1 − 0.25 standing zeroing)
                            ≈ 12.8 % of experience at the end of training (~17 % early)
```

That is a real and adequate slice. **This is not the failure mode.** Two caveats remain:

1. **`rel_heading_envs = 0.0` while `heading_command = True`.** Upstream's default is
   `rel_heading_envs = 0.3` with `heading_control_stiffness = 0.5`
   (`velocity_env_cfg.py:182-185`), which makes 30 % of envs *hold a world heading* by
   computing `wz_cmd = 0.5·wrap(ψ_target − ψ)` every step
   (`velocity_command.py:132-138`). That converts `track_angular_velocity` into an implicit
   **heading** regulator through the back door. Our cfg sets it to 0.0 (`cfg:1223`) and the
   `ranges.heading` slot stays unused. **This is the single largest recipe deletion relative to
   upstream, and it is exactly the mechanism that teaches "hold a heading".**
2. **The exact head-pose deployment command is essentially never sampled.** The four head
   command dims are sampled independently uniform; at the final curriculum caps
   (±1.10, ±1.10, ±1.40, ±0.31 rad, `cfg:1487`) the chance that *all four* land within ±0.05 rad
   of zero is `(0.05/1.10)²·(0.05/1.40)·(0.05/0.31) ≈ 9.5e-5` — about **1 command window in
   10 500**, while the runtime sends exactly 0 for the whole session. AGENTS.md's
   "zero-command behavior must be explicitly trained" applies here and is **not** satisfied
   (`zero_command_prob = 0.0` for both pose commands). Stage 0 of the curriculum (iters 0–500)
   is the only time the head command is near zero.

### 3.1 Curriculum trap in any play/eval (answers the status-check question)

All step-based curricula are functions of `env.common_step_counter`, which **restarts at 0 in
every fresh `play=True` env**. Therefore in the viewer / any eval harness:

| curriculum | value at step 0 | trained value | effect |
|---|---|---|---|
| `stance_tax_weight` (`cfg:1592`) | ramp t=0 | t=1 | `stance_hip_yaw`, `action_over_limit`, `neck_motion`, `stillness_zero_cmd`, `no_stepping` = **weight 0** |
| `head_pose_bias_weight` (`cfg:1574`) | 0.0 | 3.0 | term off |
| `action_rate_weight` (`cfg:1432`) | −0.1 | −1.0 | smoothing 10× weaker |
| `standing_envs`, `com_range`, `head_pose_range` | stage 0 | final | DR/command ranges narrow |

**Consequence:** a `play`-mode reward breakdown is *not* the objective the policy was trained
under, and `head_pose_range` stage 0 (±0.05/±0.05/±0.07/±0.015 rad) means **sim evals never
exercise the head command at all** (a 22 s eval = 1100 steps; stage 1 starts at 12 000 steps =
240 s). This is why `scripts/eval_straightness.py --freeze-pose-cmd` is nearly a no-op on the
observation (it changes a ±3° random command into 0), and why the head is measured nearly
static in sim. Anyone auditing reward mass in `play` mode — including the 7.55/step total in the
status check versus the 3.367/step training log — must say which weight schedule is in force.
It does **not** change the behaviour of a frozen ONNX (a policy is a function of its weights, not
of the reward), so it is a *reporting* trap, not a cause of the defect.

---

## 4. Q4 — Symmetry: is the 61-D table right, and what would enabling it buy?

`ENABLE_SYMMETRY = False` (`cfg:59`), injected as `symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None`
(`cfg:1664`).

### 4.1 Table audit — **the table is correct**

* Layout match: `symmetry.py:9-24` documents `[0:3] ang_vel, [3:6] gravity, [6:20] q,
  [20:34] qd, [34:48] action, [48:51] twist, [51:55] head, [55:61] body` — **identical** to the
  instantiated cfg order (§2) and to the ONNX `observation_names`. The perm/sign blocks
  (`symmetry.py:74-95`) use exactly those offsets.
* `_OBS_PERM` is a valid involution (each block permutes within itself; `_JOINT_PERM`
  `[9,10,11,12,13,5,6,7,8,0,1,2,3,4]` is its own inverse).
* Polar/axial signs: `base_ang_vel [−1,+1,−1]` (axial vector under reflection), gravity
  `[+1,−1,+1]`, twist `[+1,−1,−1]`, head `[+1,+1,−1,−1]`, body `[+1,−1,+1,−1,+1,−1]` — all
  correct for a reflection about the sagittal plane.
* **Joint signs verified against the MJCF joint axes** (this is the part the file's own
  docstring worried about). With `M = diag(1,−1,1)` and the world-frame hinge axes at HOME
  (CPU `mj_forward` on `robot_walk.xml`):

  | pair | axis_L (world) | M·axis_L | axis_R | axis_R·(M·axis_L) | implied sign |
  |---|---|---|---|---|---|
  | hip_yaw | (0,0,−1) | (0,0,−1) | (0,0,−1) | **+1.000** | **−1** |
  | hip_roll | (1,0,0) | (1,0,0) | (1,0,0) | **+1.000** | **−1** |
  | hip_pitch | (0,+0.996,−0.087) | (0,−0.996,−0.087) | (0,−0.996,−0.087) | **+1.000** | **−1** |
  | knee | (0,−0.996,+0.087) | (0,+0.996,+0.087) | (0,+0.996,+0.087) | **+1.000** | **−1** |
  | ankle | (0,+0.996,−0.087) | (0,−0.996,−0.087) | (0,−0.996,−0.087) | **+1.000** | **−1** |
  | neck_pitch / head_pitch | axis ⊥ mirror plane (y) | — | — | a·Ma = −1 | **+1** |
  | head_yaw / head_roll | axis in mirror plane | — | — | a·Ma = +1 | **−1** |

  `_JOINT_SIGN = [−1,−1,−1,−1,−1, +1,+1,−1,−1, −1,−1,−1,−1,−1]` → **matches on all 14 joints**.
  Consistency also requires HOME to be antisymmetric on the swapped pairs, and it is:
  `default_joint_pos` left/right = `(−0.087, −0.458, −0.005, +0.453)` vs `(+0.087, +0.458, +0.005, −0.453)`.
  Repro: `.venv/bin/python -c "import mujoco; ...mj_forward..."` — the block used is in §9.
* One honest caveat: an end-to-end "FK the mirrored configuration and compare body poses"
  test is **not usable** on this robot, because the *body* geometry is not sagittally
  symmetric — e.g. the neck link sits at **y = +14.5 mm** (48.5 g for `neck`+`neck_pitch`)
  while the legs are exactly mirrored. So "the mirror of a pose is a pose the robot can hold"
  is true at the joint-axis level, but the *mass distribution* has no mirror.

### 4.2 What enabling the mirror loss would/wouldn't buy

**Would:** in a periodic gait, a bilaterally mirror-symmetric stride has *zero* net yaw impulse
per stride pair. The measured defect is precisely a half-stride yaw asymmetry of
**+28.9 ± 11.7 deg/s** (§5) that rectifies into the DC bias. A mirror-symmetric *action*
policy is the structural precondition for the impulses to cancel, so `use_mirror_loss=True`
(`symmetry.py:56-61`, coeff 0.5) is a plausible direct attack — cheaper and more general than
the `pose` term because it acts on the 14-D action directly rather than through a joint-space
Gaussian with a 0.3 rad std.

**Wouldn't:** it does not price the *outcome*. A perfectly mirrored policy still drifts if the
two half-strides have different durations or if contact/friction differs; and `use_mirror_loss`
with `use_data_augmentation=False` does **not** touch the critic obs mirroring
(`symmetry.py:148-153` deliberately repeats the unmirrored critic obs). It also cannot express
"hold the commanded heading".

**Risk (medium, and asymmetric):** the mirror function mirrors the **command block** too. With
`head_pose` and `body_pose` commands active, the mirrored target is a *different command*
(head_yaw negated), so a strong mirror loss constrains the policy's head-command response to be
mirror-symmetric — fine in principle, but it interacts with the ±1.40 rad head_yaw command
curriculum. And the robot's mass distribution is *not* symmetric (14.5 mm neck offset, 289 g
head assembly), so a perfectly symmetric gait is not the physical optimum, and AGENTS.md's
standing rule is "symmetry mirror-loss is OFF by default — never for asymmetric tasks". My
assessment: worth an **A/B smoke run**, not a default. The cheaper instrument already in the
repo is `gait_symmetry_penalty` (§7, R3).

---

## 5. Q5 — Is there any term that penalises a LEFT/RIGHT asymmetric gait?

**What exists (grepped over every env cfg):**

| function | file:line | wired into the velocity recipe? |
|---|---|---|
| `gait_symmetry_penalty` (|L−R|/(L+R) swing-time imbalance) | `mdp.py:5164`, used at `microduck_velocity_rollers_env_cfg.py:324` with **weight −1.0** | **NO** |
| `leg_symmetry_reward` (|q_L + q_R| L1) | `mdp.py:5113`, used in `microduck_velocity_swizzle_env_cfg.py:45` | **NO** |
| `bilateral_symmetry_penalty` | `mdp.py:2386` | nowhere |
| `feet_distance_penalty` | `mdp.py:5892` | nowhere |
| `joint_pos_limit_proximity` | `mdp.py:2142` | nowhere |
| `heading_hold_reward` (yaw **angle**) | `mdp.py:5195`, used at `microduck_velocity_rollers_env_cfg.py:345` with **weight +1.0** | **NO** |
| `heading_tracking_reward` | `mdp.py:4819`, used in the swizzle env | **NO** |
| `RelativeHeadingVelocityCommand` (cmd[2] = heading error) | `mdp.py:4761`, used by rollers/env | **NO** |

**What is missing in the walking recipe:**

1. **No heading-angle term of any kind** — not a penalty, not a reward. The only yaw-attitude
   term in the whole stack (`upright`) is tilt-only: `exp(−‖proj_grav_xy‖²/std²)`
   (`rewards.py:108-110`), and projected gravity is yaw-invariant. So the repo prices **roll and
   pitch attitude heavily** (w 2.0, mass 1.392 = 41 % of the return) and **yaw attitude not at
   all**. That asymmetry in the recipe is the core finding.
2. **No yaw-rate damping active when the command is straight.** `body_ang_vel` excludes z by
   construction; `angular_momentum` is numerically dead.
3. **No gait-symmetry term** — the mechanism that rectifies the oscillation into a drift.
4. **No foot-placement / lateral-step term** (nothing prices *where* the foot lands).

**Measured size of the asymmetry being left unpriced** (`hd1910_base16.csv`, 16 envs, 20 s,
vx = 0.2, vy = wz = 0; phase proxy = sign of `hip_pitch_L − hip_pitch_R`, validated at
`corr(φ, foot_z_L − foot_z_R) = −0.75` on the trace that has foot heights):

```
yaw_rate during half-stride φ>0 : +10.6 deg/s
yaw_rate during half-stride φ<0 : −18.3 deg/s
half-stride asymmetry           : +28.9 ± 11.7 deg/s   (per env)
overall DC bias                 :  −3.97 deg/s
yaw-rate AC std (step-locked)   :  66.4 deg/s   →  AC/DC = 16.7×
net yaw over 20 s               :  −79 ± 274 deg   (16 envs)
```

i.e. **the drift is a ~6 % rectification of a step-locked yaw oscillation, and the two
half-strides are 2.7× different in their mean yaw rate.** That is the thing to price, and
nothing in the recipe sees it.

---

## 6. Q6 — History of the defect: did the 2026-09-11 stance-hygiene block actually fix hip_yaw?

**Yes.** Recovered from the newest run's own log, using the identity
`raw = Episode_Reward/stance_hip_yaw ÷ applied_weight`, where the applied weight is the
curriculum ramp (`mdp.py:3617-3646` sets `weight = target × t`, `t = min(1, step/7200)`, logged
as `Curriculum/stance_tax_weight`):

```
iter  ramp_t   w        Episode_Reward   raw Σ relu(|hip_yaw|−0.10)   implied mean |hip_yaw|
  0   0.0017  −0.0014      −0.00000            0.00172                    0.101 rad (5.8°)
299   0.9984  −0.7987      −0.03744            0.04687                    0.123 rad (7.1°)
750   1.0000  −0.8000      −0.04444            0.05555                    0.128 rad (7.3°)
2000  1.0000  −0.8000      −0.03429            0.04286                    0.121 rad (7.0°)
5000  1.0000  −0.8000      −0.03658            0.04572                    0.123 rad (7.0°)
9000  1.0000  −0.8000      −0.03184            0.03980                    0.120 rad (6.9°)
11999 1.0000  −0.8000      −0.04162            0.05203                    0.126 rad (7.2°)
```

(`raw` is the mean over the two `hip_yaw` joints, so the per-joint excess over the 0.1 rad
deadband is `raw/2 ≈ 0.025 rad`; the implied mean deviation is ≈ **7°**, and it is *flat from
iter 750 to 12000*, i.e. stabilised, not still shrinking.)

Cross-checks, all from the same run and the independently recorded trace:

| quantity | pre-fix (`walk_hd1910_15000`) | `model_11999` | source |
|---|---|---|---|
| `hip_yaw` deviation, walking | **+30.4° / +25.4° at the mechanical stops; 48 % of steps within 1° of the stop** | **−1.97° / +2.03° (σ 2.5°)** | `posture_defect_analysis.md:180-187` vs `hd1910_base16.csv` |
| `hip_yaw` raw action, standing | mean \|a\| **10.14** (max 10.27) | `action_over_limit` raw **0.0035 rad** total | `mdp.py:5230` |
| abduction (½(ΔL−ΔR) hip_roll) | **+5.78°** | ~+0.4° (L +1.34°, R −0.57°) | trace |
| trunk height | 91.1 mm (crouch) | **121.0 mm** (official: 119.8 mm) | trace |
| trunk tilt | 3.99° (max 12.5°) | 2.67° | trace |
| `neck_pitch` swing | **16.5° std**, 2.15 Hz | **2.34° std** | trace |
| `neck_motion` raw | 6.72 (my recompute) | 3.46 (logged) | trace / log |
| `pose` (`per_joint_posture`) raw | 0.015–0.0175 (dead) | 0.492 | log |

So the 2026-09-11 block worked as designed: **the toe-out duck stance is gone, the neck pump is
gone (10× reduction), the gait is tall.** And **the veering is still there** (yaw bias
−4 deg/s, `lat_dev_rms` 0.324 m vs the official's 0.064 m, 5× worse). Conclusion: the toe-out
was a *co-symptom* — the same missing posture/heading pricing produced both, and the posture
half got fixed while the heading half was never wired (§7, R1 was proposed by agent A on
2026-09-11 and deliberately deferred, `posture_defect_analysis.md:360-365`).

---

## 7. Q7 — What does the official recipe have that ours doesn't?

`alpha_walking.onnx` is from `apirrone/microduck_runtime` (symlink of
`BEST_alpha_walking_rough.onnx`), task almost certainly `Mjlab-Velocity-Rough-MicroDuck`
(`docs/forensics/agent_B_official_reference.md:26-30`).

**Contract is identical** (measured, not assumed): input `[1,61]`, output `[1,14]`, normalizer
baked in, MLP 512-256-128 ELU, `action_scale` 1.0, `joint_names` and `default_joint_pos`
byte-identical to this repo's HOME, and the **same** `observation_names`
(`base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command,head_command,body_command`)
and `command_names` (`twist,head_pose,body_pose`). Agent B's comment-stripped
`git diff -w origin/develop HEAD -- src/mjlab_microduck/tasks/` is **one functional line**
(`init_std 1.0 → 0.5`). So the *recipe* is not the difference — **and that cuts against the
"the official has a magic term" theory**: the official was trained with the same
`rel_heading_envs = 0.0` deletion and the same absent heading term.

Recipe-level differences that remain, ranked by how much they could explain the straightness gap:

| # | difference | evidence | plausibility as the cause |
|---|---|---|---|
| 1 | **Training task = Rough**, ours = Flat | `agent_B:26-30`; official is quasi-static on our rough task in our env (`agent_B:432-436`) | **medium** — terrain variation forces robustness to lateral perturbations, but agent B could not reproduce the rough condition meaningfully |
| 2 | **Different basin, not a different recipe.** The official converged to a *near-HOME* gait (`pose` = 0.946 at its stance); ours converged (pre-fix) to a crouch with a dead posture gradient (0.015) | `agent_B:337-361` | **high** for the *pre-fix* run; for `model_11999` the basin is fixed (tall gait, pose 0.492) and yet it still veers, so the basin is now a *secondary* factor |
| 3 | **Exploration**: `init_std 0.5` for the (now superseded) HLS run vs the official's 1.0; the newest cfg is back to 1.0 (`cfg:1626`) | `cfg:1627-1642` | low for `model_11999` |
| 4 | **Longer/different schedule**: official head-command normalizer std is 1/1.27 of ours (0.478 vs 0.607 for neck_pitch) → its head-command curriculum topped out lower, or its run saw fewer late-stage samples | `agent_B:100-104` | low, but confirms our curriculum drives the head command *further* than the official's |
| 5 | **Action filtering on the robot**: the runtime applies `action_scale = 0.9` and a first-order low-pass (`head_lowpass 0.5`, `legs_lowpass 0.7`) that **do not exist in this training repo** | `agent_B:76-86`; `AGENTS.md` ("policies are UNFILTERED… trained-with / deployed-without breaks transfer") | **cross-lane but flagged: this is a live sim2real mismatch and a candidate explanation for a hardware-only head wobble** |

Honest bottom line for Q7: the official's straightness is **not** explained by a term we lack —
the *recipe* is the same. It is explained by (i) the official having converged into a basin where
the gait is near-symmetric and the head is near HOME, so the *unpriced* yaw channel happens to be
small, and (ii) the official being evaluated on its native (stiffer, XL330) hardware. The
unpriced-channel defect is real in both, but only our policies landed in the regime where it
matters. A heading term is therefore the right fix regardless: it makes the good basin the only
basin, instead of relying on luck.

---

## 8. Head-wobble hypothesis (first-class, per the follow-up brief)

**Verdict: real reward-side mechanism, but it is a CO-SYMPTOM, not the cause of the veering.**
Confidence 高 that head motion is under-priced/incentivised; 高 that it is not the root cause of
the current drift (the user's head-action ablation made it worse: net yaw −65 ± 180° → −79 ± 249°,
path efficiency 0.412 → 0.257); 中 on the residual hardware-only coupling.

### 8.1 Every reward term that touches the neck/head joints

| term | weight | mass @11999 | mechanism |
|---|---|---|---|
| **`head_pose_tracking`** | **+2.0** | **+1.4685 (43.6 % of net — the largest positive term)** | `mean₄ exp(−(err/0.5)²)` (`mdp.py:5508-5513`) |
| `head_pose_bias` | +3.0 | −0.2604 | L1 on the **1 s EMA** of the same error — DC only (`mdp.py:5550`) |
| `neck_motion` | −0.08 | −0.2768 | `Σ relu(|q̇| − 2.0)²` over the 4 head joints (`mdp.py:1486`) |
| `action_rate_l2` | −1.0 | −0.9608 | all 14 dims, so it *does* touch the neck; on the old policy's trace the 4 head dims are **36 % of Σ(Δa)²** |
| `dof_pos_limits` | −1.0 | −0.0173 | hard-limit overshoot only |
| `self_collisions` | −1.0 | −0.0018 | indirect |
| `angular_momentum` | −0.02 | **−8.9e-7 ≈ 0** | `‖L_root‖²` — squared form, gradient vanishes |
| `body_ang_vel` | −0.05 | −0.1271 | **trunk_base only and xy only** — head and yaw are excluded by construction (`rewards.py:192`) |
| **`pose` (`per_joint_posture`)** | +1.0 | +0.4917 | **EXCLUDED for neck/head** |

* **Confirmed: the posture term does not constrain the head at all.** `cfg:579-581`:
  `SceneEntityCfg("robot", joint_names=(r"^(?!passive_|.*neck.*|.*head.*).*",))`. The reason is
  documented and defensible (it would fight `head_pose_tracking`, `cfg:569-578`) — but the
  consequence is that, when the head command is zero, the *only* posture-to-HOME pressure on the
  head is `head_pose_tracking` itself, whose std is 0.5 rad = **28.6°**, averaged over 4 joints.
* **The head is actively paid to move.** With the final curriculum caps
  (±1.10, ±1.10, ±1.40, ±0.31 rad), a policy that holds the head at HOME while commands resample
  scores `E[exp(−(cmd/0.5)²)]` = 0.402 / 0.402 / 0.317 / 0.885 → mean 0.5015 → **1.003/step of the
  2.0 available**. Full tracking is worth 2.0. So
  **≈ 0.997/step = 29.6 % of the net return is available purely for swinging a 289 g (37 % of
  mass) head**, including ±1.40 rad (80°) of **head_yaw**. During training this is not a side
  reward: it is the #1 positive term (43.6 %), larger than `track_linear_velocity` (32.1 %).
* `zero_command_prob = 0.0` for `head_pose`/`body_pose` (`env.yaml:2155, 2180`), and the exact
  zero 4-vector has probability ≈ 9.5e-5 at the final caps (§3) — while the runtime sends exactly
  0. The deployment head state is effectively untrained for the last ~11 500 iterations.
* `neck_motion`'s deadband (**2.0 rad/s**) means a small head wobble is entirely free: a ±2° wiggle
  at 2.5 Hz is 0.55 rad/s; only ±10°-and-up is taxed. On the recorded policy, head peak |q̇| was
  neck 6.9, head_pitch 5.9, head_yaw 3.9, head_roll 3.1 rad/s — i.e. the tax was clearly active
  then (raw 6.7, ≈ 0.54/step at w −0.08) and is nearly inactive now (raw 3.46).

### 8.2 Can the policy perceive that its head swing is yawing the trunk?

It can see `base_ang_vel` (yaw rate) and the neck/head `joint_pos`/`joint_vel` — but **not** the
heading, not the linear velocity, and not the trunk's angular momentum. So it can react to a
*yaw rate* the head induces, but it cannot perceive the accumulated heading error (§2). The
relevant observable is also corrupted: up to 6° random-axis IMU misalignment mixes roll/pitch
rates into the observed ω_z at ≈0.1 rad/s, i.e. the same order as the drift.

### 8.3 Is there a rectification mechanism? — **Yes, and it is measurable in the existing traces**

Physics: a periodic head swing at the step frequency has zero *mean* reaction torque only if it
is symmetric between the left-stance and right-stance half-cycles and if the contact conditions
are equal. The trace shows it is **not** symmetric:

`gf_walk_hd1910_15000_12s.csv` (the pre-fix policy, head_cmd = 0, 12 s, phase proxy
`φ = hip_pitch_L − hip_pitch_R`, validated at `corr(φ, foot_z_L − foot_z_R) = −0.75`):

| | φ > 0 (267 steps) | φ < 0 (333 steps) | difference |
|---|---|---|---|
| trunk yaw rate | **−17.2 deg/s** | **−6.8 deg/s** | **10.4 deg/s** |
| `neck_pitch` mean | **+12.16°** | **+24.20°** | **−12.0°** |
| `head_pitch` mean | +24.34° | +21.94° | +2.4° |
| `head_roll` mean | +0.82° | −1.76° | +2.6° |
| `head_yaw` mean | −0.35° | −0.32° | ≈0 |

So on that policy the head posture was **locked to the step phase and 12° asymmetric between the
two half-strides**, while the yaw rate was 2.5× larger in one half than the other — exactly the
phase-locked, half-stride-asymmetric pattern that fails to average to zero. `neck_freq_hz ==
gait_freq_hz` in that run's own JSON (`gf_walk_hd1910_15000_12s.json`: 2.167 Hz for the neck, the
trunk bob, `hip_roll` and the gait).

**And in the current policy the mechanism is no longer head-driven:**

| `hd1910_base16.csv` (model_11999, 16 env) | value |
|---|---|
| head joint std | neck_pitch **2.34°**, head_pitch 1.95°, head_yaw 1.84°, head_roll 0.93° (pre-fix: 16.5 / 7.5 / 3.7 / 3.2) |
| head half-stride asymmetry | **≤ 0.6°** on every head joint (pre-fix: 12.0° on neck_pitch) |
| half-stride yaw-rate asymmetry | **+28.9 ± 11.7 deg/s** — unchanged, i.e. **leg-driven** |
| yaw-rate AC std / DC | 66.4 / 3.97 deg/s |

**Q3 answer:** the rectification is real and large, but in `model_11999` it is produced by the
**legs**, not the head. The head ablation confirms this empirically.

### 8.4 The head *can* steer the legs through the policy itself (measured from the ONNX)

A pure-CPU onnxruntime probe of `model_11999` at a nominal observation
(`.venv/bin/python`, `providers=['CPUExecutionProvider']`) — note the action vector is the
HOME-relative joint delta, `scale = 1.0`, `use_default_offset = True`:

| perturbation | effect on the 14 leg actions (rad per unit) |
|---|---|
| `∂/∂ neck_pitch joint-pos obs` | **ankle L −0.652, ankle R +0.749**, knee L +0.252 / R −0.325, hip_yaw L −0.242 / R +0.194 → **max \|leg\| = 0.75** |
| `∂/∂ head_pitch joint-pos obs` | ankle L +0.240 / R −0.281, hip_yaw L +0.085 / R −0.104 → max 0.28 |
| `∂/∂ head_yaw joint-pos obs` | max \|leg\| 0.034 |
| `∂/∂ neck_pitch head-CMD` | hip_pitch **L +0.118 / R −0.108** at cmd 0.15 (antisymmetric) |
| **`∂/∂ twist wz` (the yaw command!)** | **max \|leg\| 0.052** |
| `∂/∂ twist vy` | knee L +0.951 / R +0.979 (symmetric), hip_roll L −0.372 / R −0.516 |

The policy is **~15–20× more sensitive to its head state than to the commanded yaw rate**, and
the head→leg response is strongly **antisymmetric** (L vs R opposite signs), i.e. the head state
directly modulates the left/right asymmetry that produces the yaw drift. So *if* a real head
wobble exists on hardware (a sim2real gap — in sim the head is nearly static at the deployment
command), it will feed back into the gait through these gains. This makes the user's running
ablation worth completing, with one methodological caution: **`--ablate-head` zeroes the head
*actions*, so the head becomes a passive mass driven by reaction forces — it does not hold the
head rigid at HOME.** And the `--freeze-pose-cmd` intervention is nearly a no-op on the
observation (§3.1: the eval never leaves curriculum stage 0, so the head command is ±3° in both
arms); the 16-env run-to-run spread of the yaw bias is ±13.7 deg/s (SE ≈ 3.4), and two 8-env
baseline runs of the same policy differed by 5.3 deg/s — so a 13.9 deg/s difference between two
8-env arms is ≈2σ and **not yet attributable to the head command**. To test the head-command
hypothesis properly, widen the pose-command ranges to the *end-of-curriculum* values in the eval
harness, or accept that only the training-time condition differs.

---

## 9. Ranked, concrete config changes (analysis only — nothing was edited)

Sign convention per AGENTS.md: a function returning **≥ 0** as a *cost* takes a **negative**
weight; a function returning a bounded **reward** (`exp(…)`) takes a **positive** weight;
microduck's self-negating `*_penalty`/`*_l1` take **positive** weights. Every penalty must log
≤ 0 (`Episode_Reward/<penalty>`), and no tax may be introduced before the skill exists.

### R1 — Add a heading-angle (yaw) term, gated to the straight-command condition. **Do this first.**
* **What:** reuse the *existing, proven* instrument — `mdp.heading_hold_reward`
  (`mdp.py:5195`, `exp(−wrap(yaw − yaw_spawn)²/std²)`, corrective: the policy may yaw *back*),
  which the rollers env already uses at **weight +1.0, std 0.4**
  (`microduck_velocity_rollers_env_cfg.py:345`), with the comment "a yaw-RATE penalty … froze the
  yaw and made drift WORSE (tried and reverted)". Copy that precedent instead of inventing one.
* **Edit:** `src/mjlab_microduck/tasks/mdp.py:5195` — add
  `command_name: str | None = None, command_threshold: float = 0.05` and multiply by
  `(‖cmd_xy‖ + |cmd_wz|) < threshold` (mirrors `feet_air_time`'s gate, `rewards.py:230-237`) so it
  never fights a commanded turn. Then wire in
  `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py` right after `cfg:767`
  (`track_angular_velocity`), with `weight = 0.0` at step 0 and a ramp to **+0.5** over iters
  1000–2000 via the existing `microduck_mdp.reward_weight` curriculum pattern (`cfg:1574-1585`).
* **Effect (arithmetic):** at std 0.4 rad, a 1 rad (57°) heading error costs `w·(1−exp(−6.25)) ≈ 0.99w`
  → **0.5/step = 15 % of the current net return**, versus **0.0014/step (0.04 %)** for the
  equivalent rate error today — a ~350× increase in the price of going crooked. A 65° net error
  (the user's measured mean) is priced at ≈0.35/step.
* **Risk: low.** It is a *reward* with a bounded basin and a corrective (not rate-freezing)
  shape, introduced after gait discovery; the gate keeps turning trainable; the reference is the
  spawn heading, so it is invariant to the random reset yaw (`reset_base` samples yaw over ±π).
  Residual risk: it makes the *straight* bucket pay more than the turning buckets, which could
  bias the policy toward refusing to turn — hence the gate at `command_threshold`.

### R2 — Re-enable upstream's heading bucket (alternative or complement to R1). **Cheap, one line.**
* **Edit:** `microduck_velocity_env_cfg.py:1223` — `command.rel_heading_envs = 0.0` → **0.3**
  (keep `heading_command = True`, which is already inherited, and `ranges.heading = (-π, π)`,
  already present).
* **Effect:** for 30 % of envs, `UniformVelocityCommand._update_command`
  (`velocity_command.py:132-138`) recomputes `wz_cmd = 0.5·wrap(ψ_target − ψ)` every step, so
  `track_angular_velocity` becomes a genuine **heading** regulator (the mode its own docstring
  advertises). No new reward code.
* **Risk: low-medium.** It changes the meaning of the `wz` command for those envs (the policy
  must hold a *world* heading), which is a new sub-skill; it also means the deployment's
  `wz = 0` no longer matches a trained command distribution exactly — although `wz_cmd → 0`
  whenever the heading error is 0, which is the straight case. `scripts/eval_straightness.py:120-126`
  already forces `rel_heading_envs = 0` and `heading_command = False`, so evals stay clean.

### R3 — Add the gait-symmetry penalty (attacks the rectification source directly). **Second priority.**
* **What:** `microduck_mdp.gait_symmetry_penalty` (`mdp.py:5164`) — cumulative swing-time
  imbalance `|L−R|/(L+R)`, i.e. it prices *lopsided leg usage*, which is the measured
  `+28.9 ± 11.7 deg/s` half-stride yaw asymmetry. Rollers precedent:
  `microduck_velocity_rollers_env_cfg.py:324`, **weight −1.0** (the function returns ≥ 0, so the
  weight must be negative).
* **Edit:** `microduck_velocity_env_cfg.py`, after the stance-hygiene block at `cfg:633`; start at
  **weight −0.5**, ramped from 0 with the existing `stance_tax_weight` curriculum
  (`cfg:1592-1606`, add `{"reward_name": "gait_symmetry", "weight": -0.5}`).
* **Effect:** it is the only available term that penalises the *mechanism* (unequal left/right
  load) rather than the symptom, and unlike a mirror loss it is a soft penalty on a cumulative
  statistic, so it cannot forbid the instantaneous single-support asymmetry a real stride needs.
* **Risk: medium.** It is an *attempt tax* on the gait; the ramp is mandatory. A lopsided stride
  may also be a legitimate balance strategy for a robot whose mass distribution is not symmetric
  (neck offset 14.5 mm) — watch `Train/mean_episode_length` and `air_time` for a flattening and
  back off first (rollers' rule: "raise if still splayed, lower if it flattens the stride").

### R4 — Sample the exact deployment command bucket explicitly (both twist and head). **Low risk, do with R1.**
* **Edit A:** `microduck_velocity_env_cfg.py:1242` next to `rel_turn_in_place_envs`, add
  `cfg.commands["twist"].rel_straight_envs = 0.25`, and extend
  `VelocityCommandCommandOnly._resample_command` (`mdp.py:4693-4718`) with the mirror-image of the
  turn-in-place branch: force `vy = 0`, `wz = 0`, `vx = sign·U(0.15, 0.4)`, and clear
  `is_standing_env`.
  *Rationale:* `rel_forward_envs = 0.2` already provides this bucket **by inheritance from
  mjlab and undocumented in our recipe** — one refactor away from silently disappearing. Making it
  explicit and raising it to ~0.25 costs 5 % of experience but removes the dependence on an
  inherited default and raises exact-straight coverage from ≈13 % to ≈19 %.
* **Edit B:** `microduck_velocity_env_cfg.py:1262-1290` — set `zero_command_prob = 0.1` on
  **both** `head_pose` and `body_pose` (`UniformPoseCommandCfg`, `mdp.py:5425-5430`, already
  supports it). 0.1 is enough to keep the deployment condition in-distribution without deleting
  the head skill; the exact-zero 4-vector is otherwise sampled at ≈9.5e-5 (AGENTS.md: "zero-command
  behavior must be explicitly trained").
* **Risk: low.** Edit A costs turning samples; Edit B slightly dilutes the head-command training
  distribution. Neither introduces a tax.

### R5 — Fix the *reporting* trap before trusting any further reward audit. **Zero behavioural risk.**
* Any `play=True` / eval harness silently zeroes five hygiene terms, `head_pose_bias`, and weakens
  `action_rate_l2` 10×, because step-based curricula restart at 0 (§3.1). Either replay the
  curricula to their final stage in evaluation, or state the schedule alongside every reward
  number. This is why the measured 7.55/step (play, positive terms only, frozen command) and the
  training log's 3.367/step (iter 11999, all terms) are not comparable.

### R6 — Symmetry mirror loss. **A/B only, not a default.**
`ENABLE_SYMMETRY = True` at `microduck_velocity_env_cfg.py:59` → `SYMMETRY_CFG`
(`symmetry.py:56-61`, `use_mirror_loss=True`, coeff 0.5). The 61-D table is verified correct
(§4.1). Expected benefit: a mirror-symmetric action policy is the precondition for the two
half-strides' yaw impulses to cancel. Risks: (a) it also mirrors the head/body command slots, so
it constrains the head-command response; (b) the robot's mass distribution is not sagittally
symmetric (neck 14.5 mm off-axis), so a perfectly symmetric gait is not the optimum; (c) it
provides no *outcome* signal, so it cannot replace R1. Run it as a 1500-iteration A/B against R1
and compare the half-stride yaw asymmetry from `scripts/eval_straightness.py`, not the total
reward.

### R7 — Things NOT to do
* **Do not tighten `track_angular_velocity`'s std as the primary fix.** The term is deep in its
  tail *because roll/pitch rates dominate the same Gaussian* (`std² = 0.5`, raw 0.036). A tighter
  std there taxes the roll/pitch oscillation that walking requires — the same failure mode as
  `fine_std = 0.1` on the head (documented to kill walking, `cfg:1345-1352`). If you want a
  sharper yaw-rate price, **split** the two (yaw error with its own std ≈ 0.2 rad/s, xy rates
  left alone) — a new mdp function, i.e. a bigger change than R1 for a weaker signal.
* **Do not raise `angular_momentum`** to damp yaw: it is `‖L‖²` (dead at weight −0.02, mass
  −8.9e-7) and it prices yaw momentum that the *turning* skill needs — the spin env disabled it
  for exactly that reason (`microduck_spin_env_cfg.py:122`).
* **Do not add a yaw-rate penalty** in place of R1: the rollers env tried it and recorded that it
  "froze the yaw and made drift WORSE" (`microduck_velocity_rollers_env_cfg.py:340-344`).
* **Do not add a qpos-side hip_yaw limit penalty** — the 2026-09-11 block already fixed that with
  the action-side term, and the rollers note warns a qpos penalty "stole that range and broke the
  gait" (`cfg:88-90`).
* **Do not spend more iterations.** 12 000 iters × 24 steps is already ≈2.1× the AGENTS.md gait
  budget (4000–6000), the stance taxes have been flat since iter 750 (§6), and the defect is a
  missing term, not an unconverged one. Spend the budget on R1+R3 instead.

### Suggested acceptance test (already in the repo)
`scripts/eval_straightness.py` on ≥16 envs, straight command, compared against the official
reference: targets are yaw-rate bias ≤1 deg/s, `lat_dev_rms` ≤0.10 m, path efficiency ≥0.65,
≥80 % of envs within 30° — the official's measured numbers (bias −0.3 ± 0.7 deg/s, net yaw
−6.3 ± 14.2°, path efficiency 0.685, 88 % within 30°) are the bar.

---

## 10. Reproduction commands for every number in this report

```bash
cd /home/joyandai/microduck_rl

# reward mass table + marginal drift prices + head-tracking incentive
.venv/bin/python - <<'EOF'
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
p='logs/rsl_rl/velocity/Mjlab-Velocity-Flat-MicroDuck'
ea=EventAccumulator(p,size_guidance={'scalars':0}); ea.Reload()
S={t:{e.step:e.value for e in ea.Scalars(t)} for t in ea.Tags()['scalars']}
print({k:S[k][11999] for k in S if k.startswith('Episode_Reward/')})
# hip_yaw recovery: raw = Episode_Reward/stance_hip_yaw / (-0.8 * Curriculum/stance_tax_weight)
for it in (0,299,750,2000,5000,9000,11999):
    w=-0.8*S['Curriculum/stance_tax_weight'][it]
    print(it, w, S['Episode_Reward/stance_hip_yaw'][it], S['Episode_Reward/stance_hip_yaw'][it]/w if w else None)
EOF

# command distribution / curricula / zero_command_prob actually used by the newest run
grep -n "rel_forward_envs\|rel_heading_envs\|rel_standing_envs\|rel_turn_in_place_envs\|zero_command_prob\|heading_command" \
  logs/rsl_rl/velocity/Mjlab-Velocity-Flat-MicroDuck/params/env.yaml

# half-stride yaw asymmetry + head amplitudes (16 env, straight command)
.venv/bin/python - <<'EOF'
import numpy as np, csv
d={c:np.array([float(r[c]) for r in csv.DictReader(open('docs/forensics/artifacts/straightness/hd1910_base16.csv'))]) for c in
   csv.DictReader(open('docs/forensics/artifacts/straightness/hd1910_base16.csv')).fieldnames}
for e in np.unique(d['env']).astype(int):
    m=d['env']==e; yr=d['yaw_rate'][m]
    phi=d['left_hip_pitch'][m]-d['right_hip_pitch'][m]; phi-=phi.mean()
    print(e, round(np.rad2deg(yr.mean()),2), round(np.rad2deg(yr.std()),1),
          round(np.rad2deg(yr[phi>0].mean()),1), round(np.rad2deg(yr[phi<0].mean()),1))
EOF

# ONNX head-state -> leg-action gains (CPU only)
.venv/bin/python -c "
import numpy as np, onnxruntime as ort
p='logs/rsl_rl/velocity/Mjlab-Velocity-Flat-MicroDuck/Mjlab-Velocity-Flat-MicroDuck.onnx'
s=ort.InferenceSession(p,providers=['CPUExecutionProvider']); n=s.get_inputs()[0].name
o=np.zeros((1,61),np.float32); o[0,5]=-1; o[0,48]=0.2
r=lambda x: s.run(None,{n:x.astype(np.float32)})[0][0]
print('d leg / d neck_pitch obs =', (r(np.where(np.arange(61)==11,0.3,o))-r(np.where(np.arange(61)==11,-0.3,o)))/0.6)
print('d leg / d twist wz      =', (r(np.where(np.arange(61)==50,0.2,o))-r(np.where(np.arange(61)==50,-0.2,o)))/0.4)"

# symmetry table audit (joint axes)
.venv/bin/python -c "
import numpy as np, mujoco
m=mujoco.MjModel.from_xml_path('src/mjlab_microduck/robot/microduck/robot_walk.xml'); d=mujoco.MjData(m)
nm=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(m.njnt)]
H={'left_hip_roll':-0.087,'left_hip_pitch':-0.458,'left_knee':-0.005,'left_ankle':0.453,'neck_pitch':0.349,'head_pitch':0.349,'right_hip_roll':0.087,'right_hip_pitch':0.458,'right_knee':0.005,'right_ankle':-0.453}
for i,n in enumerate(nm):
    if n in H: d.qpos[m.jnt_qposadr[i]]=H[n]
mujoco.mj_forward(m,d); M=np.diag([1.,-1.,1.])
for b in ['hip_yaw','hip_roll','hip_pitch','knee','ankle']:
    L=d.xaxis[nm.index('left_'+b)]; R=d.xaxis[nm.index('right_'+b)]
    print(b,'axis_R . M axis_L =',round(float(R@(M@L)),3),'-> sign',-1 if R@(M@L)>0 else 1)"
```

---

## 11. Confidence statement

| claim | confidence |
|---|---|
| No reward term prices yaw **angle**; the only yaw term prices rate at 2.1 % of the return | **高** (code + logged mass) |
| A 0.1 rad/s yaw bias (115° in 20 s) costs 0.0014/step = 0.04 % of the return | **高** (closed-form from the logged raw value) |
| The actor cannot observe heading or lateral position; `base_lin_vel` is critic-only | **高** (cfg + ONNX metadata) |
| The exact straight-walk bucket **is** sampled (~13 % of experience) via inherited `rel_forward_envs = 0.2` | **高** (frozen env.yaml + upstream source) |
| `rel_heading_envs = 0.0` deletes upstream's heading regulator | **高** |
| The 61-D symmetry table is correct | **高** (joint-axis verification, 5/5 pairs exact) |
| hip_yaw-at-limit is fixed in `model_11999` (~7° vs 48 % of steps at the stop) | **高** (two independent sources) |
| Head motion is under-priced/incentivised (43.6 % of the return is head tracking; `angular_momentum` dead; `pose` excludes the head; `zero_command_prob = 0`) | **高** |
| Head wobble is a co-symptom, not the cause of the veering | **高** (head static in sim, ≤0.6° half-stride asymmetry; user's ablation worsens drift) |
| The DC drift is a ~6 % rectification of a leg-driven step-locked yaw oscillation | **中-高** (16-env trace; the half-stride asymmetry is measured, the causal attribution to leg mechanics is inference) |
| A heading term (R1) will materially reduce the drift on hardware | **中** — untested; it prices the defect but the correction must still be learned through a noisy, 6°-misaligned yaw-rate channel |
| The official's straightness comes from its basin + native stiffer hardware, not a missing term | **中** |

Speculation is labelled as such inline. Nothing in `src/` was modified by this analysis.
