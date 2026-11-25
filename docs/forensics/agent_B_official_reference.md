# Agent B — official `alpha_walking.onnx` as a reference for gait quality

**Question asked.** The retrained local policy `walk_hd1910_15000.onnx` has two visible
defects: (a) a duck/toe-out, asymmetric leg stance, (b) a head/neck that wobbles far more
than the official policy's. How much of that gap is **physics/actuator** (the XL330 →
HD-1910-C001 swap, 5 V rail, +42 g) and how much is **training**?

**Answer in one line.** Neither defect is visible in the official policy, and neither
survives a 6× stiffening of the servo — **both are training outcomes of the local run**,
enabled by a posture reward that has collapsed to ~1.5 % (walking) / 0 % (standing) of its
gradient in the basin that run converged into. The official policy, by contrast, **cannot
run on the swapped hardware at all** (3–24 falls), so the retrain was necessary — but the
retrain did not reproduce the official gait.

Everything below is a single-env, 20 s, push-free, command-frozen eval in the faithful BAM
training env. Resets count **falls only** (`episode_length_s` was raised to 1000 s so the
20 s episode timeout cannot be mistaken for a fall).

---

## 1. What the official ONNX actually is (the contract)

### 1.1 Provenance

`alpha_walking.onnx` is copied from `apirrone/microduck_runtime` (`policies/README.md`,
`5f3b314`), where it is the symlink target of **`BEST_alpha_walking_rough.onnx`** — i.e. the
run's role name says *rough terrain*, and the training task was almost certainly
`Mjlab-Velocity-Rough-MicroDuck`. It is the deployment's walk/velstand policy; the runtime
also ships `alpha_stand.onnx`, which is what actually runs whenever
`|twist| ≤ DEFAULT_STANDING_THRESHOLD = 0.05` (`duck-control/src/policy.rs:25`).

### 1.2 Graph contract (measured, not assumed)

Dumped with `onnx`:

| property | `alpha_walking.onnx` | `alpha_stand.onnx` | `walk_hd1910_15000.onnx` (local) |
|---|---|---|---|
| input | `obs`, float32, **[1, 61]** | `obs` [1,61] | `obs` [1,61] |
| output | `actions`, float32, **[1, 14]** | `actions` [1,14] | `actions` [1,14] |
| IR / producer | IR 8, pytorch 2.9.1 | IR 8, pytorch 2.9.1 | IR 8, pytorch 2.9.1 |
| graph (9 nodes) | `Sub → Div → (Gemm → Elu)×3 → Gemm` | same | same |
| **obs normalizer baked in?** | **YES** — `/obs_normalizer/Sub` + `/obs_normalizer/Div`, initializers `obs_normalizer._mean [1,61]` and `onnx::Div_24 [1,61]` (= std) | YES | YES |
| MLP | 61→512→256→128→14, ELU | same | same |
| output clipping in graph | **none** (no `Clip` node) | none | none |
| `action_scale` (metadata) | **1.0** | 1.0 | 1.0 |
| `command_names` | `twist,head_pose,body_pose` | same | same |
| `observation_names` | `base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command,head_command,body_command` | same | same |
| `joint_names` | `left_hip_yaw,left_hip_roll,left_hip_pitch,left_knee,left_ankle,neck_pitch,head_pitch,head_yaw,head_roll,right_hip_yaw,right_hip_roll,right_hip_pitch,right_knee,right_ankle` | same | same |
| `default_joint_pos` | `0,-0.087,-0.458,-0.005,0.453,0.349,0.349,0,0, 0,0.087,0.458,0.005,-0.453` | same | same |
| `joint_stiffness` / `joint_damping` | all 1.0 / all 0.0 — **not informative** (BAM is a custom actuator, so `actuator_gainprm/biasprm` carry no servo gains) | same | same |

Two decisive facts fall out of the metadata:

* `default_joint_pos` is **byte-identical to this repo's `HOME_FRAME`** (incl. the STAND2
  revision: hip_pitch 26.24°, ankle 25.95°, knee 0.28°) which landed 2026-06-30
  (`ac10e64 new home position (centered)`). The official policy was therefore exported from a
  revision **at or after** the STAND2 home, and it shares our HOME exactly. "Revision drift in
  the HOME pose" is **not** a confound.
* The observation term names and their order are the 61-D unified-command family
  (48 proprio + `[twist(3), head_pose(4), body_pose(6)]`), matching `AGENTS.md` and matching
  `duck-control/src/obs.rs`'s layout table exactly. The local policy's ONNX has the same
  names, the same MLP shape, and the same normalizer plumbing.

### 1.3 Observation layout (from the deployment's own single source of truth)

`/home/joyandai/microduck/duck-control/src/obs.rs` — "the highest-risk code in the crate":

```
 0..3   gyro (trunk frame, rad/s)          34..48  previous action (raw policy output)
 3..6   projected gravity (trunk frame)    48..51  vx, vy, vyaw
 6..20  joint_pos − home (14, mouth out)   51..55  neck_pitch, head_pitch, head_yaw, head_roll
20..34  joint_vel (14)                      55..57  body x, y  (ALWAYS ZERO)
                                            57..61  body z, roll, pitch, (yaw — always zero)
```

with `targets = DEFAULT_POSITION + action_scale × action` and, on the robot,
`action_scale = 0.9` for walking plus a first-order low-pass on head/leg targets
(`robotd/src/control.rs`, `deploy/robotd.toml`: `head_lowpass = 0.5`, `legs_lowpass = 0.7`,
"the values the alpha policies are TRAINED with").

> **Caveat that matters for defect (b):** no action low-pass exists anywhere in this training
> repo (`grep -rn "LOW_PASS\|low_pass" src/` → nothing; upstream `origin/develop` → nothing;
> mjlab → nothing), yet the daemon applies 0.5/0.7. So *on the robot* the neck's 2.3 Hz pump
> is attenuated relative to what this report measures in sim — for **both** policies
> equally. The comparison below is unaffected; the absolute wobble numbers are an upper
> bound on what the robot shows.

### 1.4 Which upstream config was it trained for?

* `git log origin/develop -5` → tip `5946fd9` (2026-09-01). A comment-stripped, whitespace-free
  diff of `microduck_velocity_env_cfg.py` between `origin/develop` and local `HEAD` is
  **exactly one line**: `init_std: 1.0 → 0.5` (a PPO exploration knob, training-only). The
  reward recipe, command ranges, curricula, DR and obs wiring are **the official recipe**.
* The official policy's ONNX metadata matches that recipe's outputs (61-D layout, `joint_pos`
  relative to HOME, `action_scale` 1.0, MLP 512-256-128).
* Independent numeric corroboration: the baked normalizer's statistics over the **command
  block** are near-identical between the two policies — body-pose std `0.0129,0.0129,0.0129,
  0.0389,0.0389,0.0389` in **both** (identical to 4 decimals), twist std 0.214/0.139/0.523
  (official) vs 0.208/0.135/0.512 (local). Same sampling config.
* One residual difference: the **head_pose** command std is `0.478/0.478/0.605/0.142`
  (official) vs `0.607/0.608/0.770/0.178` (local) — a uniform ×1.27. The two cfgs' head_pose
  curriculum is identical, so the official's training distribution saw a ~27 % *narrower*
  head command range. Unexplained; it does not affect this comparison (head_pose is frozen at
  zero below) and is listed as an open question.

**Verdict on §1:** the official ONNX was trained on the same 61-D unified-command recipe this
repo runs, for the **pre-swap XL330 robot**, on the **rough-terrain** task variant of the
velocity env, and it is a drop-in for the same obs contract. Confounds listed in the brief are
all resolved: obs width matches (61), normalizer is baked in, joint order matches, joint_pos
is HOME-relative, `action_scale` 1.0, no output clipping anywhere, and the "posture flag in
the vx slot" convention belongs to the *sitstand* skill, not to the walking policy.

---

## 2. Experiments

### 2.1 The worktree route was tried first and is blocked — here is why

```
git worktree add /tmp/upstream_wt origin/develop      # succeeds (detached HEAD 5946fd9)
PYTHONPATH=/tmp/upstream_wt/src uv run python -c "import mjlab_microduck.robot.microduck_constants"
   -> TypeError: FrictionDRBamActuatorCfg.__init__() got an unexpected keyword argument
                 'vin_drop_gain_range'
```

`origin/develop`'s `microduck_constants.py` passes `vin_drop_gain_range` (gain × Σ|τ|), while
this fork's vendored `better-actuator-models` **1.0.2** (`vendor/bam`, pinned via
`[tool.uv.sources]`) only accepts `vin_drop_resistance_range` (R × I_bat). Upstream pinned that
BAM from the `mjlab_frictionloss` git branch, so the API renamed between them. The worktree
also has **no `vendor/`** (git-ignored), so it cannot `uv sync` on its own. Upstream
`origin/develop` is therefore **not executable** in this environment.

**Route taken instead (equivalent for evaluation):** run the *current* env cfg and override
(i) the robot MJCF → the upstream pre-swap model (737.24 g, verified below) and (ii) the BAM
actuator args → `xl330/m6.json` **plus** the era's `kp_fw=200, vin_range=(6.5,8.2),
vin_drop=(0,0.2), delay=(3,6)` from `origin/develop:microduck_constants.py`.

Overriding `kp_fw` mattered: `vendor/bam/bam/mjlab.py:217-220` applies `cfg.kp_fw` **on top of**
the JSON, so a `--params-json`-only override (what `eval_onnx_bam.py` does) leaves the HD-1910's
`kp_fw = 32` in place while loading XL330 friction/kt/R — a 6× too-soft servo. See §3.

Harness: a **frozen snapshot** of `scripts/gait_forensics.py`
(md5 `9d02500dffc380724502dfcda887f9bc`, copied to `/tmp/gf_snapshot.py`) plus a launcher that
patches `mjlab.tasks.registry.load_env_cfg`. Sources are saved under
`docs/forensics/artifacts/harness/`.

### 2.2 Nominal-stance reference (CPU MuJoCo, `robot_walk.xml`, HOME angles, `mj_forward`)

| quantity | pre-swap (737.24 g) | post-swap (779.24 g) |
|---|---|---|
| foot separation (`left_foot`↔`right_foot` sites) | **83.642 mm** | **83.642 mm** |
| foot separation in trunk frame (y) | 83.642 mm | 83.642 mm |
| `left_foot` / `right_foot` z in trunk frame | −117.13 mm | −117.13 mm |
| trunk base z | 120.0 mm | 120.0 mm |
| head body (`jaw_soft`) z | **232.60 mm** | 232.60 mm |
| head body HOME quaternion | `(w,x,y,z) = (−0.707107, 0, 0.707107, 0)` | same |
| naive RPY of that quaternion | **`[roll, pitch, yaw] = [0.0, −90.0, 0.0]`** | same |
| rotvec of `R_head·R_head_homeᵀ` at HOME | `[0,0,0]`, angle 2.4e−6 ° | same |

Two conclusions, both used below:

1. The two robot revisions are **kinematically identical** (only the `inertial` mass/inertia of
   the 14 servo bodies changed, +3 g each = +42.0 g total); nominal foot separation is 83.642 mm
   in both, and the 83.6 mm figure in the brief is reproduced exactly.
   *Mass accounting note:* `sum(model.body_mass)` is **737.24 g** for every pre-swap model file
   (`robot_walk.xml` and `robot_allcollisions.xml`, upstream) and **779.24 g** post-swap — so the
   "+42 g" in the brief is exact, but the pre-swap absolute figure is 737.24 g, not the 744.4 g
   quoted in the brief. I could not reproduce 744.4 g from any XML in either revision; the
   comparison is unaffected because the delta is what matters and it is 42.00 g.
   The head assembly (neck + neck_pitch + yaw_roll_motion + jaw_soft) is 289 g, of which the
   `jaw_soft` body that carries the head shell and `head_roll` is 188.8 g — the body whose
   orientation is measured below.
2. **The head body's nominal frame is a pure −90° pitch — an Euler gimbal-lock singularity.**
   Naive `atan2/asin` RPY on the head quaternion is meaningless there (roll and yaw are `atan2(0,0)`).
   Every head number in this report is therefore the **rotation vector of `R_head · R_head_homeᵀ`**
   (plus a ZYX decomposition of the same deviation, which factors the trunk's yaw drift out
   first), computed by the updated `gait_forensics.py`. Raw per-step head quaternions are in the
   CSVs (`quat_head_*`) so the metric can be recomputed.

### 2.3 Run list (exact commands)

Common tail for every row:
`--warmup 2 --seconds 20 --csv docs/forensics/artifacts/<tag>.csv --json-out docs/forensics/artifacts/<tag>.json`

```
# R0  official on the CURRENT (swapped) hardware — the disputed 2026-09-10 scenario
uv run python scripts/gait_forensics.py --onnx /home/joyandai/microduck/policies/alpha_walking.onnx \
    --task Mjlab-Velocity-Flat-MicroDuck --lin-vel-x 0.2 --tag official_HD1910_vx0.2

# R1..R10 via the override launcher (BAM/xml pins from origin/develop:microduck_constants.py)
uv run python /tmp/gf_run.py --bam xl330  --onnx .../alpha_walking.onnx --task Mjlab-Velocity-Flat-MicroDuck --lin-vel-x 0.2 --tag official_PRESWAP_vx0.2
uv run python /tmp/gf_run.py --bam hd1910 --onnx walk_hd1910_15000.onnx            ... --lin-vel-x 0.2 --tag local_HD1910_vx0.2
uv run python /tmp/gf_run.py --bam xl330  --onnx .../alpha_walking.onnx            ... --lin-vel-x 0.0 --tag official_PRESWAP_vx0.0
uv run python /tmp/gf_run.py --bam hd1910 --onnx walk_hd1910_15000.onnx            ... --lin-vel-x 0.0 --tag local_HD1910_vx0.0
uv run python /tmp/gf_run.py --bam hd1910 --onnx .../alpha_walking.onnx            ... --lin-vel-x 0.0 --tag official_HD1910_vx0.0
uv run python /tmp/gf_run.py --bam xl330 --xml src/mjlab_microduck/robot/microduck/robot_walk.xml \
                             --onnx .../alpha_walking.onnx                          ... --lin-vel-x 0.2 --tag official_XL330_779g_vx0.2
uv run python /tmp/gf_run.py --bam xl330  --onnx walk_hd1910_15000.onnx            ... --lin-vel-x 0.2 --tag local_PRESWAP_vx0.2
uv run python /tmp/gf_run.py --bam xl330  --onnx .../alpha_walking.onnx --task Mjlab-Velocity-Rough-MicroDuck --lin-vel-x 0.2 --tag official_PRESWAP_rough_vx0.2

# R11..R14 one-variable actuator decomposition
uv run python /tmp/gf_run.py --bam xl330  --kp-fw 32 --onnx .../alpha_walking.onnx  ... --lin-vel-x 0.2 --tag official_XL330_kp32_vx0.2
uv run python /tmp/gf_run.py --bam xl330  --vin 4.75 5.25 --vin-min 4.0 --onnx .../alpha_walking.onnx ... --lin-vel-x 0.2 --tag official_XL330_5V_vx0.2   # FAILED (see below)
uv run python /tmp/gf_run.py --bam hd1910 --kp-fw 200 --onnx walk_hd1910_15000.onnx ... --lin-vel-x 0.2 --tag local_HD1910_kp200_vx0.2
uv run python /tmp/gf_run.py --bam hd1910 --kp-fw 200 --onnx walk_hd1910_15000.onnx ... --lin-vel-x 0.0 --tag local_HD1910_kp200_vx0.0
```

Every run emitted its resolved override set, e.g.
`[OVR] {"xml": ".../upstream_wt/.../robot_walk.xml", "params_json": ".../xl330/m6.json", "kp_fw": 200.0, "vin_range": [6.5,8.2], "vin_min": 6.0, "vin_drop": [0.0,0.2], "delay": [3,6]}`
(the "delay" unit is sim substeps = 5 ms, so 15–30 ms of actuator lag).

### 2.4 Does the official policy run in the closest faithful configuration? **YES.**

| run | cmd | falls / 20 s | trunk tilt mean (max) | foot sep vs nominal | gait freq | verdict |
|---|---|---|---|---|---|---|
| **official, pre-swap (XL330/m6 + kp_fw 200 + 6.5–8.2 V + 737 g model)** | vx 0.2 | **0** | 3.35° (5.3°) | **−0.68 mm** (83.0 mm) | 1.95 Hz | **walks cleanly** |
| official, pre-swap | vx 0.0 | **0** | 1.56° (1.7°) | +1.09 mm (84.7 mm) | static | **stands rock-steady** |
| official, XL330 params on the **779 g** model | vx 0.2 | **0** | 4.73° (6.2°) | −0.12 mm | 2.05 Hz | walks |
| official, XL330 friction but **kp_fw = 32** | vx 0.2 | **24** | 15.66° (72.5°) | +1.03 mm | 1.15 Hz | constant falling |
| official on the **current HD-1910** hardware | vx 0.2 | 3 | 7.51° (72.7°) | +3.91 mm | — | falls |
| official on the **current HD-1910** hardware | vx 0.0 | 3 | 6.62° (69.4°) | +2.91 mm | — | falls |
| local, its own HD-1910 hardware | vx 0.2 | 0 | 3.99° (12.5°) | **−13.31 mm** | 2.30 Hz | walks |
| local, its own HD-1910 hardware | vx 0.0 | 0 | 0.80° (1.1°) | **−52.46 mm** | static | stands (badly posed) |
| local, pushed onto XL330/737 g | vx 0.2 | 2 | 7.12° (72.7°) | −8.58 mm | — | falls |

**The 2026-09-10 conclusion is falsified.** `docs/policy_failure_analysis.md` §P0-5 claims
(original: *"它在**任何**配置下都失败,**包括它自己那套**(换装前 XL330 机器人 + xl330/m6
参数):指令 0.0(原地站)也摔 14 次 ⇒ **它连站都站不住**"*) that the official policy fails in
every configuration including its own pre-swap robot + params. It does not. In the faithful
pre-swap configuration it stands for 20 s at **tilt 1.56°, yaw drift −0.4°, 0.19° neck-joint
std, foot separation 84.7 mm** — i.e. *better* than nominal — and walks 20 s with **0 falls at
tilt 3.35°**. The earlier attempt did not, in fact, test "its own pre-swap robot+params":
`scripts/eval_onnx_bam.py --params-json` swaps only the BAM JSON (`eval_onnx_bam.py:77-83`), so
it ran the XL330 friction/kt/R model **under the HD-1910's `kp_fw = 32`** and **on the +42 g
HD-1910 model**. Both substitutions are removed here, and the policy works.

That reading is not just plausible, it is *reproduced*: the old note records
`trunk tilt mean 15.6°, max 70.2°, 8 s: 11 falls, achieved vx 0.080/0.20`. My
`official_XL330_kp32_vx0.2` row — XL330 JSON **with `kp_fw` left at 32**, on the pre-swap
737 g model — gives **tilt mean 15.66°, max 72.5°** and 24 falls in 20 s. So the "cannot stand"
result is a *correct measurement of the `kp_fw = 32` configuration*, which is not the
configuration the policy was trained for. The single experiment that flips it is raising
`kp_fw` 32 → 200 (row R1 above); nothing else about the actuator needs to change.

(The one non-bug caveat: on the rough-terrain task variant the policy stayed
quasi-static — trunk z 317.8 mm = raised platform, 0.05 Hz — so the "rough-terrain policy"
confound is not positively resolved, only rendered moot: it walks fine on flat.)

**Why the official policy cannot run on the current hardware.** The decomposition above isolates
it to the actuator, not the mass:

* +42 g on its own (XL330 params, HD model): **0 falls**, foot separation −0.12 mm, head pitch
  std 3.88° → the mass change is *not* what breaks it. (It does change straight-line behaviour
  dramatically: yaw drift +9.6° on 779 g vs +279.9° on 737 g — see open questions.)
* `kp_fw` 200 → 32 alone, everything else XL330 (same model, same 7.4 V rail, same friction
  model): **24 falls in 20 s**. This single firmware gain is the dominant cause: the policy's
  commanded offsets were learned against a servo 6× stiffer, and `kp_fw` is applied *over* the
  JSON (`bam/mjlab.py:217`), so the mismatch is easy to create by accident.
* Voltage alone could not be tested: `official_XL330_5V_vx0.2` aborted at `KeyError:
  'Mjlab-Velocity-Flat-MicroDuck'` — at that moment a **parallel agent's in-flight edit** of
  `src/mjlab_microduck/tasks/{mdp,microduck_velocity_env_cfg}.py` had broken the task-package
  import, so the registry was empty. Same cause killed two other runs (`RuntimeError: a Tensor
  with 4 elements cannot be converted to Scalar` in `curriculum_manager.reset`). See §5.

---

## 3. Side-by-side: official vs local on **their own** hardware, cmd vx = 0.2, 20 s

| metric | **official** (XL330/737 g, its own hw) | **local** (HD-1910/779 g, its own hw) | local with `kp_fw=200` (same hw, 6× stiffer servo) |
|---|---|---|---|
| falls / 20 s | 0 | 0 | 0 |
| trunk z | 119.8 mm | **91.1 mm** (−29 mm) | 98.3 mm |
| trunk tilt mean / max | **3.35° / 5.3°** | 3.99° / 12.5° | 5.11° / 13.4° |
| **foot separation** (trunk-y) | **82.96 mm** | **70.33 mm** | 67.9 mm |
| — vs nominal 83.642 mm | **−0.68 mm** | **−13.31 mm** | −15.7 mm |
| — min…max | 70.7…98.3 mm | 32.0…118.6 mm | 27.8…105.3 mm |
| `hip_roll` dev L / R | **+0.21° / −0.54°** | +7.55° / −4.02° | +4.21° / −4.53° |
| abduction ½(ΔL−ΔR) | **+0.37°** | **+5.78°** | +4.37° |
| `hip_yaw` dev L / R (**toe-out**) | **−4.56° / +5.49°** | **+15.74° / +16.35°** | **+16.53° / +16.76°** |
| `hip_pitch` dev L / R | +5.29° / −5.63° | **−34.16° / −53.01°** | −28.75° / −48.70° |
| `knee` dev L / R | +2.46° / −3.45° | +27.99° / +5.13° | +29.17° / +9.01° |
| `ankle` dev L / R | −4.36° / +4.69° | **+45.47° / +49.61°** | +43.97° / +45.85° |
| `neck_pitch` std / p2p | **1.37° / 5.42°** | **18.09° / 60.43°** | 19.10° / 65.08° |
| `neck_pitch` osc. amplitude @ gait freq | **0.75°** | **16.44°** | 14.77° |
| `head_pitch` joint std / p2p | 1.02° / 3.28° | 9.17° / 34.81° | 9.11° / 35.72° |
| head world pitch std / p2p | **3.51° / 10.15°** | **17.38° / 60.56°** | 8.43° / 31.97° |
| head **rel. to trunk** pitch std / p2p | **1.69° / 6.88°** | **22.48° / 78.29°** | 24.63° / 84.69° |
| head rel. to trunk total-rotation std | **1.55°** | 11.81° | — |
| dominant gait frequency | 1.95 Hz | 2.30 Hz (user reports 2.25 Hz) | 2.20 Hz |
| mean \|action\| | **0.160** | **0.395** (2.5×) | 0.378 |
| action-rate (raw units/step) | 0.062 | 0.0825 | 0.085 |
| **official posture reward at the measured stance** | **0.946** | **0.015** | ~0 |
| yaw drift over 20 s (@wz = 0) | +279.9° | +327.8° | −144.4° |
| approx. forward speed (proxy, see §5) | 0.10 m/s | 0.14 m/s | 0.19 m/s |

Same table, **cmd vx = 0.0** (the deployment idle state; note the real runtime would hand this
to `alpha_stand`, not to the walking policy):

| metric | official (pre-swap) | local (HD-1910) | local, `kp_fw=200` |
|---|---|---|---|
| falls / 20 s | 0 | 0 | 0 |
| **foot separation** | **84.74 mm** (+1.09) | **31.18 mm** (**−52.46**) | 32.2 mm (−51.4) |
| `hip_yaw` dev L / R (toe-out) | +0.78° / +0.07° | **+30.46° / +25.40°** | +30.38° / +25.40° |
| `hip_pitch` dev L / R | +1.50° / −1.01° | −44.71° / −47.33° | — |
| `ankle` dev L / R | +0.17° / −0.29° | +53.96° / +54.0° | — |
| trunk z | 115.5 mm | 91.5 mm | 93.7 mm |
| trunk tilt | 1.56° | 0.80° | 1.07° |
| `neck_pitch` std | 0.19° | 0.97° | 2.60° |
| head pitch (world) std | 0.42° | 0.81° | 1.50° |
| mean \|action\| | 0.064 | **1.195** (19×) | 1.201 |
| **posture reward at the stance** | **0.968** | **0.0000** | ~0 |

### 3.1 Cross-hardware rows (isolating physics from policy)

| run | falls | foot sep | abduction | toe-out L/R | neck joint std | head rel-trunk pitch std |
|---|---|---|---|---|---|---|
| official @ XL330/737 g (own) | 0 | 82.96 | +0.37 | −4.56/+5.49 | 1.37 | 1.69 |
| official @ XL330/**779 g** | 0 | 83.52 | −0.15 | −1.78/+0.18 | 1.48 | 2.38 |
| official @ HD-1910/779 g | 3 | 87.55 | +2.57 | −2.31/+2.77 | 16.47 | 24.83 |
| official @ XL330, **kp_fw 32** | 24 | 84.60 | +0.68 | −1.14/−2.34 | 15.16 | 19.66 |
| local @ HD-1910/779 g (own) | 0 | 70.33 | +5.78 | +15.74/+16.35 | 18.09 | 22.48 |
| local @ HD-1910/779 g, **kp_fw 200** | 0 | 67.90 | +4.37 | +16.53/+16.76 | 19.10 | 24.63 |
| local @ XL330/737 g | 2 | 75.06 | +14.28 | +9.72/+19.17 | 34.18 | 49.62 |

The two rows that matter most:

* **The official policy keeps its nominal stance and quiet head under every physics that still
  lets it stand** (abduction +0.37° / −0.15°, head rel-trunk pitch std 1.69° / 2.38°). Its
  *failure mode* under HD-1910 physics is falling, not splaying — it does not develop the duck
  stance even while falling (abduction +2.57°).
* **The local policy keeps its duck stance and its head pump when the servo is stiffened 6×**
  (toe-out +16.5°/+16.8° vs +15.7°/+16.4°; abduction 4.37° vs 5.78°; head rel-trunk pitch std
  24.6° vs 22.5°; zero-command stance foot separation 32.2 mm vs 31.2 mm, action 1.201 vs
  1.195). If actuator compliance were the cause, this single-variable change would have moved
  them.

### 3.2 Why the stance reward stopped constraining the local policy

`variable_posture` (mjlab `velocity/mdp/rewards.py:385`) is
`exp(−mean_j (Δq_j / σ_j)²)` over the 10 leg joints — a **mean**, so one bad joint is diluted by
nine good ones, and the exponential collapses once the mean exceeds ~2. Evaluating it at each
policy's own measured joint deviations (script
`docs/forensics/artifacts/harness/pose_cost.py`, stds from the cfg:
walking `hip_roll σ = 0.05 rad = 2.9°`; standing `hip_yaw σ = 0.1 rad`, `ankle σ = 0.1 rad`):

| stance evaluated | pose-reward value (max 1.0) | mean (Δq/σ)² | worst joints |
|---|---|---|---|
| official @ vx 0.2 | **0.946** | 0.06 | nothing beyond 0.4σ |
| official @ vx 0.0 | **0.968** | 0.03 | nothing beyond 0.4σ |
| local @ vx 0.2 | **0.015** | 4.18 | both ankles +3.2…3.5σ, left hip_roll +2.6σ |
| local @ vx 0.0 | **0.0000** | 28.4 | both ankles +9.4σ, right hip_pitch −5.5σ |
| local under XL330 physics | **0.0000** | 11.1 | right hip_roll −7.0σ, both ankles +4.2σ |

This reproduces the training log exactly (`Episode_Reward/pose = 0.0175` at iteration 14 999,
weight 1.0) and means the term supposed to define the stance contributes **1.5 % of its nominal
gradient** while walking and **0 %** while standing, in a reward stack worth ~107/step. The
official recipe therefore *asks* for the nominal stance (worth 0.95/step, which the official
policy collects) but **cannot pull back a policy that has already drifted several σ out**: the
gradient is gone exactly when it is needed. That is a recipe-level fragility; it does not by
itself create the duck stance, because the official policy under the same recipe never left the
basin.

---

## 4. Verdict, defect by defect

### Defect (a) — duck/toe-out, asymmetric leg stance

| hypothesis | verdict | evidence |
|---|---|---|
| (i) inherent to the official reward recipe | **contributes as an enabler, not a cause** | the official policy, trained with the identical recipe, holds foot sep −0.68 mm from nominal, `hip_yaw` dev < 5.6°, abduction +0.37° at vx 0.2 and +1.09 mm / +0.78° at vx 0.0. The recipe's `pose` term *penalises* exactly this (σ = 2.9° on hip_roll) — but its exponential form makes it self-defeating (local pose-reward value 0.015 vs official 0.946), so the recipe fails to *correct* the drift rather than causing it. |
| (ii) HD-1910 physics | **rejected** | stiffening `kp_fw` 32 → 200 with the HD-1910 friction model and 5 V rail leaves every posture metric essentially unchanged (toe-out +16.53/+16.76°, abduction +4.37°, zero-cmd foot sep 32.2 mm, action 1.201). Under XL330/737 g physics the local policy's splay gets *worse* (abduction +14.3°) while it falls. The official policy is unchanged by the +42 g model (foot sep 83.52 mm, abduction −0.15°). |
| (iii) undertraining of the local run | **primary cause, with a "stuck in a dead basin" caveat** | the deficit is a stable, reproducible fixed point of that policy: at vx 0.0 the local stance is `hip_yaw +30°/+25°`, `hip_pitch −44.7°/−47.3°`, `ankle +54°/+54°`, feet 31 mm apart, 28 mm below nominal trunk height, with a *constant* action vector of mean magnitude 1.195 (rate 0.007 — it is parked, not balancing). The policy's own baked normalizer reproduces it (`joint_pos` block means: `right_hip_pitch −0.918 rad = −52.6°` vs measured −53.0°; `hip_yaw +0.318/+0.214 rad = +18.2°/+12.3°`), i.e. it is the *converged* posture of that run, not an eval artifact. The run was also still improving when it stopped (mean reward 91→108 and `error_vel_yaw` 4.14→2.57 rad/s between iters 1500→15000, `track_angular_velocity` monotonically up), so 15 000 iterations was not a plateau. Because the posture gradient is dead at that stance, more iterations at the *same* weights may not fix it — a posture-side fix (or a reward the gradient can see) is the reliable lever. |

### Defect (b) — exaggerated head/neck wobble

| hypothesis | verdict | evidence |
|---|---|---|
| (i) inherent to the official reward recipe | **rejected as a cause** | the official policy on the same recipe wobbles **0.75°** of `neck_pitch` at the gait frequency, head-relative-to-trunk pitch std **1.69°**, total head-attitude deviation std 1.55°. Nothing in the recipe forces a neck pump; by design it prices head *DC* bias (`head_pose_bias`, 1 s EMA) and head pose *tracking*, and deliberately does not price the oscillation a 289 g head produces. |
| (ii) HD-1910 physics | **rejected** | the wobble is a commanded motion of the joint, not passive compliance: `kp_fw` 200 vs 32 leaves the head-relative-to-trunk pitch std at 24.6° vs 22.5° and the neck joint std at 19.1° vs 18.1°. (Independent support, same session, another agent: freezing the neck action removes the wobble but triples trunk tilt — the head is used as a balance reaction mass.) For contrast, the official policy *run under HD-1910 physics* does show a wobble (neck joint std 16.5° vs 1.4°) — but that run is falling, so it measures the failure mode, not the actuator's passive effect. |
| (iii) undertraining | **cause** | the local policy is the only configuration that pumps the neck while walking stably, at 5× (world) to 13× (rel. trunk) the official's amplitude, at a comparable gait frequency (2.30 vs 1.95 Hz), with 2.5× the mean action magnitude and a stance 29 mm lower. Its `head_pose_tracking` reward is healthy (1.677/2.0 in training), so the wobble is not a tracking failure the reward could see — it is an unpriced degree of freedom the optimizer chose to use. |

**Both defects are training-related; neither is the actuator swap.** The actuator swap is a
*separate*, much larger effect: it is what makes the official policy unusable here (24 falls
from `kp_fw` alone) and therefore what forced the retrain. The retrain bought stability
(0 falls, 4° tilt) but did not buy the official gait.

---

## 5. Confidence and open questions

**High confidence**

* ONNX contract: 61-D in / 14-D out, normalizer baked in (`Sub`/`Div`), no output clipping,
  512-256-128 ELU, `action_scale` 1.0, joint order and `default_joint_pos` identical to this
  repo's `HOME_FRAME`. Read off the graph and metadata, and corroborated by the policy actually
  standing and walking in our env.
* The 2026-09-10 "it cannot even stand on its own pre-swap robot+params" claim is wrong: in the
  faithful pre-swap config it stands (tilt 1.56°, 0 falls) and walks (tilt 3.35°, 0 falls) for
  20 s. The earlier run omitted `kp_fw` (left at 32) and used the +42 g model.
* The official policy's failure on the current hardware is actuator-driven: `kp_fw` 32 vs 200 is
  sufficient on its own (24 falls); +42 g alone is not (0 falls).
* Both local defects survive a 6× servo stiffening and persist under XL330 physics → not
  actuator compliance.
* Nominal reference: 83.642 mm foot separation, 232.60 mm head z, head body HOME quaternion is a
  pure −90° pitch (gimbal lock confirmed — naive head RPY is unusable).
* The posture reward collapses to ~0 gradient at the local policy's stance (0.015 walking,
  0.0000 standing, matching the training log's 0.0175).

**Uncertain / not established**

* **Yaw drift is unattributed and looks fragile.** At `wz = 0` both policies veer
  (~0.24–0.29 rad/s), yet the official policy veers only +9.6° over 20 s on the 779 g model and
  +279.9° on the 737 g model — same policy, same actuator. This is either a genuine
  near-bifurcation of straight-line walking or a single-DR-draw artifact (a single env with seed
  42 still samples CoM ±3 mm, IMU misalignment ≤6°, encoder bias ±0.86°, armature ±10 %,
  friction ±10 %). It is **not** one of the two defects, and my defect metrics are stable across
  the three official physics configurations, so the conclusions do not depend on it — but the
  absolute drift numbers should not be quoted without a multi-env/multi-seed repeat.
* **"Will more training remove the defects?"** Not tested (no training runs allowed). The
  evidence is mixed: the run had clearly not converged (reward and ang-vel error still
  improving at iteration 15 000), *but* the posture gradient is already dead at the converged
  stance, which is the classic signature of a basin that more PPO at the same weights will not
  leave. The head pump might yield to the action-rate curriculum (already at its final weight
  −1.0), the stance probably will not.
* **Achieved forward velocity is not measured.** The harness logs no base velocity; the numbers
  in the last row of §3 are a proxy (head-camera site displacement projected on the trunk
  heading) and they read low for *both* policies (0.10 / 0.14 m/s at cmd 0.2), which conflicts
  with the user's 0.202 m/s for the local policy from `eval_onnx_bam.py`. Treat that row as
  indicative only; the commands are demonstrably respected (vx 0.0 vs 0.2 changes the behaviour
  qualitatively in every configuration).
* **Rough terrain not resolved.** The official policy is named `..._rough`; on
  `Mjlab-Velocity-Rough-MicroDuck` it stayed quasi-static (trunk z 317.8 mm — it spawns on the
  raised platform; 0.05 Hz; mean |action| 0.092). I did not chase that, because it walks on flat
  in the pre-swap config, so the "it's a rough-terrain policy" confound cannot explain any
  failure seen here.
* **Voltage alone not isolated.** `official_XL330_5V_vx0.2` crashed for an unrelated reason (a
  concurrent editor broke the task-package import).
* **The head_pose command-range difference** (official normalizer std ×1/1.27 of ours) is
  unexplained and suggests the official run's curriculum topped out lower than the current cfg's
  ±1.10 rad. Irrelevant to this comparison (head_pose was frozen at zero, and the official policy
  works in that condition) but it is one more hint that the official run predates some
  non-eval-relevant tuning.
* **Deployment filters.** The daemon applies `action_scale = 0.9` and a 0.5/0.7 head/leg
  low-pass that do not exist in this training repo. Both policies are affected equally, so the
  *comparison* holds, but the official gait as seen on the robot is smoother than measured here
  — the real head wobble of a deployed policy is likely below every number in §3.

**Concurrency caveat (affects provenance, not numbers).** A parallel agent edited
`scripts/gait_forensics.py` at 12:18 (I froze a snapshot at md5 `9d02500dffc380724502dfcda887f9bc`
and ran everything through it), then edited the **tracked** files
`src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py` (12:25:22) and `.../mdp.py` (12:26:17),
adding zero-weight, ramp-from-zero reward/curriculum terms for exactly these two defects
(`stance_hip_yaw`, `stance_hip_roll`, `action_over_limit`, `neck_motion`, `stance_tax_weight`).
Runs R1–R10 executed before those edits, against pristine recipes; runs R11–R14 executed after
them, and my launcher **pruned** the new terms so the term set matched. The diff is reward-only —
no observation, command, actuator, event or model change — and a fixed ONNX policy's dynamics do
not depend on reward terms, so all rows are comparable. Three runs aborted on the editors'
in-flight states (`RuntimeError: a Tensor with 4 elements cannot be converted to Scalar`,
`KeyError: 'Mjlab-Velocity-Flat-MicroDuck'`); those runs are the control/extra ones, not the
table rows.

---

## Artifacts

`docs/forensics/artifacts/` — `<tag>.csv` (per-step 20 Hz trace incl. head/trunk quaternions, all
14 joint positions/actions, foot-site positions, body-frame foot separation) and `<tag>.json`
(summary metrics) for every run above; `logs/<tag>.log` (full stdout);
`harness/` (`gf_snapshot.py`, `gf_run.py`, `ovr.py`, `batch.sh`, `batch2.sh`, `home_ref.py`,
`table.py`, `pose_cost.py`).

Primary sources: `/tmp/upstream_wt` (throwaway `git worktree` of `origin/develop`, created for
this task — it is not executable, see §2.1); `/home/joyandai/microduck/policies/README.md`;
`/home/joyandai/microduck/duck-control/src/{obs,policy}.rs`;
`/home/joyandai/microduck/robotd/src/control.rs`;
`vendor/bam/bam/mjlab.py:159,217`; `docs/policy_failure_analysis.md` §P0-5.
