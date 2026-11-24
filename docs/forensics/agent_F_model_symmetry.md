# Agent F — robot MODEL (MJCF) left/right symmetry & mass-distribution audit

**Scope.** The robot model only: is there a systematic left/right asymmetry or an off-centre
mass distribution in the MJCF that would push Microduck to veer? Actuator identification and
reward/training are other agents' scope. All analysis static / CPU-only (`mujoco` 3.10 without a
GPU, `mj_forward` only — the simulator is never stepped).

---

## 0. Verdict up front

**低 (LOW) confidence that the robot MODEL is a plausible cause of the systematic veering.**

| | |
|---|---|
| **Single strongest measured number** | **Whole-robot lateral CoM offset in the HOME pose = −0.031 mm (−31 µm)**, against a foot support whose inner edges are at ±24.85 mm — the CoM sits **800× inside** its own support polygon. |
| Second strongest | The **largest** left/right mass error anywhere in the model is **0.5 mg** (the ankle link: `0.0300246` vs `0.0300251` kg), i.e. 0.06 % of that link, worth a roll moment of **0.17 µN·m** = 1.5e-7 of the HD-1910's 1.18 N·m stall torque. |
| Third | The model's intrinsic CoM error is **100–500× smaller** than the CoM randomization the policy already trains under (`±3 mm` initial → `±15 mm`, `microduck_velocity_env_cfg.py:223,881,1532-1535`). A policy that survives that DR cannot be steered by a 31 µm offset. |

**Mathematically non-zero but physically negligible** (this is the correct label for every
deviation found):

* ankle L/R: 0.5 mg mass, 2.0 µm CoM, 4.4e-10 kg·m² inertia (rel. 4.0e-5 of that link);
* sole collision meshes: mirror-equivalent to 3.3 µm mean / 1.84 mm worst-vertex, contact patch
  0.123 mm different at the inner edge (0.36 % of foot width);
* hip link: 30 nm CoM, 3.3e-12 kg·m² inertia (rel. 1.9e-6).

**Not big enough to steer the robot.** Nothing in the model creates a systematic bias
comparable to the observed drift. This is consistent with the independent measurement reported by
the delegating agent — holding the actuator at the real HD-1910 parameters, swapping the MODEL
from HD-1910 geometry to official XL330 geometry moved path efficiency 0.24 → 0.41 (i.e. it
changed, and in the *better* direction) while median |net yaw| was unchanged at 89° vs 89°. My
audit **does not contradict** that; it removes the model as a first-order candidate.

**One real finding that is *not* an asymmetry but is worth a loud flag** (§3): the two hip-yaw
joints share the *same* world axis direction `(0,0,−1)`, so **`+θ` on the left and `+θ` on the
right rotate both feet the same way in the world**. "Both hip_yaw joints at their + hard stop"
therefore produces a *steering* stance (feet as a pair rotated ~27.5° clockwise, 5° apart), not a
mirrored stance. The joint *ranges* are correctly mirrored (both feet get the identical
25°-out / 30°-in world-yaw travel) — but a policy that saturates both joints at the same *signed*
value gets an asymmetric foot pair. This is present in the **official** runtime model too, so it
predates the servo swap; it is a policy/reward-visible convention trap, not a servo-swap defect.

---

## 1. Method, provenance and reproducibility

Reproducible script (new):

```
.venv/bin/python docs/forensics/artifacts/harness/model_symmetry_audit.py     # ~90 s, CPU only
```

`docs/forensics/artifacts/harness/model_symmetry_audit.py`. Two mujoco-3.10 traps this script
handles explicitly (`data.ximat` is now the *inertia principal* frame, not the body frame, and
`data.cinert` is not a plain 3×3 → both are unusable for a naive mirror comparison):

* body inertia is rebuilt as `I_body = R(iquat) · diag(body_inertia) · R(iquat)ᵀ` — the script
  self-tests this against the XML `fullinertia` of `trunk_base`
  (`robot_walk.xml:87`; agreement 3.9e-13 kg·m², limited by the 10 digits printed in the XML);
* world inertia is `R(xmat) · I_body · R(xmat)ᵀ`. Mirror (`M = diag(1,−1,1)`) comparisons are
  `M·I_L·M` vs `I_R` and `M·xipos_L` vs `xipos_R`, i.e. **frame-independent**;
* whole-body CoM is `data.subtree_com[0]` after `mj_forward` at the requested joint
  configuration (with the free joint placed at `(0,0,0.12)`, identity quaternion → the trunk
  frame *is* the world frame), **never** a naive sum of body origins.

Models compared (all committed / on this machine):

| tag | path | total mass |
|---|---|---|
| POST-HD1910 | `src/mjlab_microduck/robot/microduck/robot_walk.xml` (used by the walking task) | 779.2432 g |
| PRE-XL330 | `.../robot_walk_preswap.xml` == `robot_walk.xml.bak_servoswap`, git blob-identical to `1ee2e3c` (`git log --all` blob match `fe7cac82…`) | 737.2432 g |
| WALK-BACKLASH / ALLCOLLISIONS / ROLLERS | `robot_walk_backlash.xml`, `robot_allcollisions.xml`, `robot_allcollisions_rollers.xml` | 779.2432 / 779.2432 / 778.8296 g |
| OFFICIAL-runtime | `/home/joyandai/microduck/kinematics/assets/alpha/robot_walk.xml` (runtime repo, commit `cc972c5`) | 744.3611 g |

The official file is **kinematics-only**: no `<geom>`, no `<actuator>`, no `<default>` classes
(`nu=0`, no `*_foot_collision`), so it can arbitrate joint axes/ranges but not masses, contacts or
friction. Note it is the source of the "744.4 g" figure quoted in the brief — the repo's own
pre-swap re-export is **737.24 g**, a 7.12 g difference (§5.3).

---

## 2. Q1 — Mirror-symmetry audit (masses, inertias, ipos)

Reported at the HOME pose (`microduck_constants.HOME_FRAME`, `microduck_constants.py:73-97`),
left body reflected through `y=0` and compared to its right counterpart.

### 2.1 Measured world-frame residuals — the authoritative table

| body pair (left ↔ right) | Δmass | ΔCoM (M·L − R) | max \|ΔI_world\| | relative to \|I_R\| |
|---|---|---|---|---|
| `yaw2roll` ↔ `bearing_roll` (`robot_walk.xml:114` / `:267`) | **0.0000 mg** | (0, 0, 0) mm — exactly 0 | **0** kg·m² | 0 |
| `hip_l` ↔ `hip_l_2` (`:127` / `:280`) | **0.0000 mg** | (0, 0, **+0.00003**) mm = 30 nm | 3.29e-12 | 1.9e-6 |
| `upper_leg_left` ↔ `upper_leg_right` (`:136` / `:289`) | **0.0000 mg** | (0, 0, 0) mm | 3.4e-21 | 1.6e-16 (float64 noise) |
| `leg` ↔ `leg_2` (`:151` / `:304`) | **0.0000 mg** | (0, 0, 0) mm | 2.0e-20 | 3.8e-15 (float64 noise) |
| **`ankle_left` ↔ `ankle_right`** (`:161` / `:314`) | **−0.5000 mg** | (**+0.00199, +0.00200**, +0.00007) mm ≈ **2.0 µm** | **4.44e-10** | 4.0e-5 |
| leg chain total (`yaw2roll…ankle`, 5 bodies) | left 141.0456 g vs right 141.0461 g → **Δ = 0.5 mg** | | | |

**Largest deviation in the whole model: 0.5 mg mass / 2.0 µm CoM / 4.4e-10 kg·m² on the ankle.**
Identical on the pre-swap model (same numbers to all printed digits, `PRE-XL330` section of the
script output) ⇒ it predates the servo swap and is an Onshape float-rounding artifact
(`mass="0.0300246"` vs `mass="0.0300251"`, `robot_walk.xml:161` vs `:314`).

### 2.2 Raw XML inertial data, as written (for the record)

| body | file:line | `pos` (ipos) [m] | mass [kg] | fullinertia (Ixx Iyy Izz Ixy Ixz Iyz) |
|---|---|---|---|---|
| `yaw2roll` | `robot_walk.xml:114` | `-8.8e-11 0.00099414 0.0168917` | 0.0260406 | `4.74279202e-06 3.862498e-06 2.6294218e-06 -0 -0 -5.75059577e-08` |
| `bearing_roll` | `:267` | *identical* | *identical* | *identical* |
| `hip_l` | `:127` | `0.014815 1.6179e-08 -0.00829992` | 0.00918934 | `1.27777516e-06 1.71858982e-06 9.67415536e-07 1.48470435e-12 5.788595e-07 -1.48470435e-12` |
| `hip_l_2` | `:280` | `0.014815` **`1.6212e-08`** `-0.00829992` | 0.00918934 | *identical* |
| `upper_leg_left` | `:136` | `0.006766 0.0228838 0.0134531` | 0.0512067 | `1.79254839e-05 1.101439e-05 2.10607679e-05 -4.6934934e-06 2.42169779e-07 4.13524798e-07` |
| `upper_leg_right` | `:289` | `-0.006766 0.0228838 0.0134531` | 0.0512067 | same with 4th/5th components sign-flipped (`+4.6934934e-06 -2.42169779e-07`), 6th identical |
| `leg` | `:151` | `-1.52e-10 0.0318937 -0.00953505` | 0.0245844 | `5.9803315e-06 2.45460161e-06 4.94235919e-06 -1.13898927e-12 0 8.71332487e-07` |
| `leg_2` | `:304` | `-0.0318937 -1.52e-10 -0.00953505` | 0.0245844 | Ixx/Iyy swapped + sign-flipped off-diagonals — exactly the mirror in the mirrored frame (world residual 0) |
| `ankle_left` | `:161` | `-0.00615738 -0.0137483 -0.0156586` | **0.0300246** | `5.91525e-06 1.11534e-05 7.74015e-06 -2.0535e-07 8.2152e-08 5.6714e-08` |
| `ankle_right` | `:314` | `0.00615539 -0.0137482` **`-0.0156566`** | **0.0300251** | `5.91534e-06 1.11534e-05 7.74016e-06 2.04877e-07 -8.247e-08 5.7017e-08` |

Raw `ipos`/`fullinertia` numbers must **not** be compared field-by-field across L/R: the two
sides' body frames differ (e.g. `hip_l` quat `0.707107 -0.707107 -0 -0` at `:124` vs `hip_l_2`
quat `0 -0 0.707107 0.707107` at `:277`), so a legitimate mirror makes several components change
sign or swap. §2.1 is the frame-independent comparison. (The `hip_l` y-component difference
`1.6179e-08` vs `1.6212e-08` = 3.3 pm is present verbatim in the official model too
— `1.5269e-08` vs `1.5309e-08` — confirming it is a CAD export artifact, not a servo-swap effect.)

### 2.3 Is 0.5 mg / 2 µm / 31 µm enough to steer?

No, by four orders of magnitude:

* ankle asymmetry roll moment: `Δm·g·r = 5e-7 kg × 9.81 × 0.035 m = 1.7e-7 N·m`
  = **0.17 µN·m**, vs the HD-1910's ~1.18 N·m stall (`12 kg·cm`) → **1.5e-7**.
* its contribution to the measured whole-body CoM offset: `0.5 mg × 35 mm = 0.0175 g·mm` out of
  the total first moment `779.24 g × 0.0312 mm = 24.3 g·mm` → **0.07 %**. The remaining 99.9 %
  comes from the midline bodies' own y-offsets (head shell, speaker, RPi HAT PCB), which are
  single bodies with no left/right counterpart.
* it is 100–500× below the trained DR: `COM_RANDOMIZATION_RANGE = 0.003` m ramping to 0.015 m
  (`microduck_velocity_env_cfg.py:223,888,1532-1535`), i.e. the policy is explicitly trained to
  survive a ±3…15 mm lateral CoM error.

---

## 3. Q2 — Joint axes & limits: the "asymmetric hip_yaw limit" claim is **REFUTED**

### 3.1 Measured world axes (neutral pose) and ranges

| joint | world axis | range [deg] | source |
|---|---|---|---|
| `left_hip_yaw` | **(0, 0, −1)** | **[−25, +30]** | `robot_walk.xml:113` |
| `right_hip_yaw` | **(0, 0, −1)** | **[−30, +25]** | `robot_walk.xml:266` |
| `left_hip_roll` | (+1, 0, 0) | [−22, +22] | `:126` |
| `right_hip_roll` | (+1, 0, 0) | [−22, +22] | `:279` |
| `left_hip_pitch` | (0, +1, 0) | [−90, +90] | `:135` |
| `right_hip_pitch` | (0, −1, 0) | [−90, +90] | `:288` |
| `left_knee` | (0, −1, 0) | [−90, +90] | `:150` |
| `right_knee` | (0, +1, 0) | [−90, +90] | `:303` |
| `left_ankle` | (0, +1, 0) | [−90, +90] | `:160` |
| `right_ankle` | (0, −1, 0) | [−90, +90] | `:313` |
| `neck_pitch` / `head_pitch` / `head_yaw` / `head_roll` | (0,−1,0) / (0,+1,0) / (0,0,+1) / (−1,0,0) | [−90,+60] / [−90,+90] / [−170,+170] / [−25,+25] | `:181,:194,:203,:216` |

`autolimits="true"` (`:5`) ⇒ every listed `range` is an active limit.

### 3.2 The refutation, with the mechanism

Both hip-yaw joints have the **same** world axis `(0,0,−1)` (their parent bodies `yaw2roll`
`:111` and `bearing_roll` `:264` carry quaternions `0 -0.707107 -0.707107 -0` and
`0 0.707107 0.707107 0`, which are **negatives of each other, i.e. the same rotation**). Writing
ψ for the foot's world yaw, ψ = −θ for *both* legs. Therefore:

* left foot world-yaw travel: θ_L ∈ [−25°,+30°] → **ψ_L ∈ [−30°, +25°]**
* right foot world-yaw travel: θ_R ∈ [−30°,+25°] → **ψ_R ∈ [−25°, +30°]**
* the left's ψ-set is the exact negative of the right's ⇒ **ψ_R = −ψ_L is achievable throughout ⇒
  the mechanism is mirror-symmetric.** Both feet can toe-out 25° and toe-in 30°.

So the env-cfg comment
(`microduck_velocity_env_cfg.py:98,144`: "hip_yaw 的**指令**是 +70°(左)/+35°(右),远超 +30°/+25°
的硬限位") reads `+30 (L)` and `+25 (R)` as if they were the same physical direction. They are
not. `+30°` on the left is **left toe-in 30°**; `+25°` on the right is **right toe-out 25°**.
"Both hip_yaw joints at their + stop" — the converged stance of the local policy — is therefore
**not** a symmetric duck stance: it is both feet rotated the same way in the world
(−30° and −25°), i.e. the *feet as a pair* yawed ~27.5° clockwise with a 5° residual differential.
That is exactly the "双脚整体扭转约 28°" measurement in the same comment — and it is a *steering*
posture, not a splayed one.

**But this is not a left/right asymmetry and not a servo-swap artifact:** the official runtime
model has the identical axes and the identical ranges
(`/home/joyandai/microduck/kinematics/assets/alpha/robot_walk.xml`: `left_hip_yaw` [−25,+30],
`right_hip_yaw` [−30,+25], both local axis `0 0 1` in identically-oriented parent frames), and
git shows the ranges have been this exact asymmetric pair since `55aebea` (`[…-25,+30]`), with the
earlier exports symmetric `±30` (`git log --format=%h -- …/robot_walk.xml` + `git show <c>:…`).
The official *policy* simply never drives both joints to the same stop.

**No other L/R range asymmetry exists anywhere in the model** (table 3.1: every remaining pair is
either identical, or an exact negation with an exact-negated world axis).

**Quantified effect of the limits.** Driving both joints to their + hard stop gives the stale
foot pair described above. Its *static* consequences (script `SAT-YAW` row,
left_hip_yaw = +30°, right_hip_yaw = +25°):

| stance | CoM y | distance to nearest support edge (24.85 mm) |
|---|---|---|
| HOME | −0.031 mm | 24.88 mm |
| both hip_yaw at + stop | **+1.157 mm** | 23.69 mm |

So even the extreme asymmetric-yaw stance only moves the CoM 1.16 mm laterally — 4.7 % of the way
to the support edge. **The model's limits cannot unbalance the robot through mass distribution.**
Any veer they cause must be *kinematic steering* (where the feet point), which is a
policy/action-space question, not a model-asymmetry question. Flagging it as a hypothesis for the
actuator/reward agents: *if* the policy saturates both hip-yaw joints (forensics measured 48 % of
walking steps at the stop), the feet are steered ~27.5° in the same world direction with a 5°
differential — that is a plausible veer *mechanism*, enabled by the joint-coordinate convention
(shared world axis). **Cannot distinguish, from the model alone, whether the trained policy
actually relies on this**; that requires the rollout data the other agents have.

---

## 4. Q3 — Whole-body centre of mass & support polygon

`mj_forward` + `data.subtree_com[0]`, free joint at `(0,0,0.12)`, identity quaternion
(script §`whole-body CoM`; the script asserts `subtree_com[0]` == the mass-weighted `Σmᵢ·xiposᵢ`
to 1e-9, so this is not a naive-origin sum).

| model | total mass | CoM @ NEUTRAL [mm] | CoM @ **HOME** [mm] | **lateral offset** |
|---|---|---|---|---|
| POST-HD1910 | 779.2432 g | (+1.4534, −0.0312, +144.8243) | (+0.4292, **−0.0312**, +140.3750) | **−0.031 mm = −31 µm** |
| PRE-XL330 | 737.2432 g | (+1.7356, −0.0136, +146.2030) | (+0.5608, **−0.0136**, +141.7486) | **−0.014 mm = −14 µm** |
| OFFICIAL-runtime | 744.3611 g | (+0.1596, +0.1555, +140.5832) | (−0.4429, **+0.1555**, +136.6574) | **+0.156 mm = +156 µm** |

Foot support (`left_foot_collision` / `right_foot_collision` mesh contact patches at HOME, points
within 1 mm of the sole's lowest vertex; script §`foot geoms`):

* left patch: y ∈ [+24.853, +58.872] mm, x ∈ [−15.775, +29.793] mm, z = 2.8176 mm
* right patch: y ∈ [−58.872, **−24.976**] mm, x ∈ [−15.775, +29.793] mm, z = 2.8176 mm
* ⇒ **inner edges ±24.85 mm (gap 49.7 mm), outer width 117.74 mm, foot length 45.57 mm**;
  left/right foot separation 83.64 mm between the `left_foot`/`right_foot` sites (reproduces
  agent B's 83.642 mm).

**Significance.** The 31 µm lateral CoM offset is:

* **800× inside** the support polygon's inner edge (±24.85 mm);
* a static roll moment `m·g·Δy = 0.7792 × 9.81 × 3.12e-5 = 2.4e-4 N·m` → **0.24 mN·m**, i.e.
  **0.02 % of the HD-1910's 1.18 N·m stall torque**; during double support the ground reaction
  simply redistributes and the net moment is zero — the robot does not "lean" at all;
* **100× below** the smallest CoM randomization the policy trains with (±3 mm) and 500× below the
  final curriculum stage (±15 mm) — see §2.3.
* the forward offset is also harmless: +0.43 mm (HOME) vs a patch spanning x ∈ [−15.8, +29.8] mm.

Verdict: **below the noise floor** by ~2 orders of magnitude relative to training DR and ~3
relative to the support polygon. It cannot account for a systematic veer.

---

## 5. Q4 — Does the pre-swap (XL330) model have the same asymmetry?

### 5.1 Provenance trap — `a91465c^` is **not** the XL330 model

```
$ for c in a91465c a91465c^ 79a993a 79a993a^ 1ee2e3c; do
    git show $c:src/mjlab_microduck/robot/microduck/robot_walk.xml |
      grep -o 'mass="[0-9.e-]*"' | grep -o '[0-9.e-]*' | awk '{s+=$1} END{printf "%s %.4f g\n","'$c'",s*1000}'; done
a91465c   779.2432 g      # HD-1910 (current)
a91465c^  800.2432 g      # <-- the HLS2909 era, NOT the XL330 model
79a993a   800.2432 g      # HLS2909 swap commit
79a993a^  737.2432 g      # XL330 (the real pre-swap baseline)
1ee2e3c   737.2432 g
```

`git show a91465c -- src/mjlab_microduck/robot/microduck/robot_walk.xml` therefore diffs HD-1910
against **HLS2909** (−21.0 g: −4.5 g trunk, −1.5 g on each of 11 servo-bearing bodies), *not*
against the XL330 robot. The commit message's "737.2 g → 779.2 g" is a comparison against the
XL330 baseline. The file that *is* the XL330 model is
`src/mjlab_microduck/robot/microduck/robot_walk_preswap.xml` (= `robot_walk.xml.bak_servoswap`,
blob-identical to `1ee2e3c`, `8479079`, `ddb729b` — verified by `git log --all` blob md5 match),
and it is what the script audits as `PRE-XL330`.

### 5.2 XL330 (737.2432 g) → HD-1910 (779.2432 g): exactly +42.000 g = 14 × 3.000 g

| body | XL330 [g] | HD-1910 [g] | Δ |
|---|---|---|---|
| `trunk_base` | 199.2240 | 208.2240 | **+9.0000** (3 servos) |
| `yaw2roll` / `bearing_roll` | 23.0406 | 26.0406 | +3.0000 each |
| `hip_l` / `hip_l_2` | 6.1893 | 9.1893 | +3.0000 each |
| `upper_leg_left` / `_right` | 48.2067 | 51.2067 | +3.0000 each |
| `leg` / `leg_2` | 21.5844 | 24.5844 | +3.0000 each |
| `ankle_left` / `ankle_right` | 30.0246 / .0300251 | *unchanged* | **0** |
| `neck` | 36.8414 | 39.8414 | +3.0000 |
| `neck_pitch` | 5.7200 | 8.7200 | +3.0000 |
| `yaw_roll_motion` | 48.6000 | 51.6000 | +3.0000 |
| `jaw_soft` (head) | 188.7660 | 188.7660 | **0** |

**In every changed body the inertia scales with the mass and `ipos` is bit-identical** —
`git show a91465c -- …/robot_walk.xml` shows *only* `<inertial …>` lines changed, and within each
line only `mass` and the `fullinertia` numbers; no body `pos`/`quat`, no joint, no geom, no
actuator line. Confirmed by the script: both files report the same total, the same joint axes and
ranges, the same actuator gear/ctrlrange/forcerange, and byte-identical foot contact patches.

### 5.3 Everything that differs beyond "+3 g/servo" — and the answer to priority 3

| quantity | official runtime (744.3611 g) | repo pre-swap XL330 (737.2432 g) | comment |
|---|---|---|---|
| total mass | 744.3611 g | 737.2432 g | **+7.118 g unexplained by servos** |
| `trunk_base` | 264.3850 g | 199.2240 g | **+65.161 g** |
| `yaw_roll_motion` | 24.9200 | 48.6000 | **−23.680 g** |
| `jaw_soft` / `bottom_head_shell` | 172.6490 | 188.7660 | **−16.117 g** |
| `upper_leg` / `left_upper_leg` | 43.1500 | 48.2067 | −5.057 g |
| `ankle_left` | 26.4653 | 30.0246 | −3.559 g |
| `hip_l` | 5.2100 | 6.1893 | −0.979 g |
| `leg`, `neck`, `neck_pitch` | 21.5844, 36.8414, 5.7200 | identical | 0 |
| `neck_pitch` ipos | `0 -0.025 0.0141665` | identical | 0 |
| `yaw_roll_motion` ipos | `-0.00140611 -0.00599577 0.0155412` | `-0.000166657 -0.00500964 0.0178065` | **≈2.5 mm CoM move** |
| `jaw_soft` ipos | `0.00368889 0.00145319 -0.0329861` | `0.00460523 0.00120047 -0.0324761` | **≈1.1 mm** |
| `hip_l` ipos | `0.0152731 … -0.00887078` | `0.014815 … -0.00829992` | **≈0.7 mm** |
| `trunk_base` ipos | `-0.0184191 6.58793e-06 0.00237545` | `-0.0226332 -2.70828e-05 0.00280064` | **≈4.2 mm in x** |
| joint axes & ranges | identical to the repo model (all 14) | identical | hip_yaw [−25,+30]/[−30,+25] in **both** |
| structure | no geoms, no actuators, no `imu`… but an extra site `imu_bno`; link names differ (`left_upper_leg` vs `upper_leg_left`); `ankle_right` body quat `0 -0.707107 0.707107 0` vs the repo's `0 -0.707107 -0.707107 0` (a rotation about the joint axis — not physical: the world axis still comes out (0,−1,0)) | | |

So, **beyond the +3 g/servo, the re-export (`1ee2e3c` era) moved ~65 g out of the trunk and
redistributed it into the head-train and legs relative to the official runtime model, while
keeping the total 7.1 g lighter.** That is a substantial revision difference between "the official
robot" and "our model" — but it is *shared* (it predates the HD-1910 swap, it is the model the
local policy's predecessor presumably used) and it is not a left/right asymmetry. The script shows
all three revisions have a lateral CoM offset ≤ 0.16 mm, so this redistribution does not create a
lateral bias either (official: +0.156 mm, XL330: −0.014 mm, HD-1910: −0.031 mm).

### 5.4 Same asymmetry pre/post? — YES, all of it predates the swap

* ankle 0.5 mg / 2.0 µm / 4.4e-10 kg·m²: **identical** in PRE-XL330 and POST-HD1910.
* foot mesh mirror distance (max 1.842423 mm, mean 0.003319 mm) and contact patches
  (inner edges 24.8532 vs 24.9763 mm): **identical to all printed digits** in PRE and POST.
* hip_yaw ranges, all joint axes, actuator gear/ctrlrange/forcerange: **identical**.
* the lateral CoM offset **changed** from −0.0136 mm to −0.0312 mm (Δ = −17.6 µm) — i.e. the
  re-export moved the CoM laterally by 18 µm, still 3 orders of magnitude below significance.

**Conclusion: the servo swap introduced no asymmetry. Every deviation found exists in the
pre-swap model as well, and the hip_yaw range asymmetry exists in the official runtime model.**

---

## 6. Q5 — Support polygon / foot geometry / friction

| property | left | right | symmetric? |
|---|---|---|---|
| collision geom | `left_foot_collision` (`robot_walk.xml:164`, mesh `sole_left`) | `right_foot_collision` (`:319`, mesh `sole_right`) | yes |
| contact patch y | [+24.8532, +58.8722] mm | [−58.8721, −24.9763] mm | outer edges mirror to **0.1 µm**; inner edges differ by **0.123 mm** |
| patch x / z | [−15.7752, +29.7930] mm, z = 2.8176 mm | identical | yes |
| patch area proxy | 756 verts within 1 mm of z_min | 758 verts | mesh tessellation differs (7896 vs 7946 verts) |
| mesh mirror distance | — | — | mean **3.3 µm**, worst vertex **1.84 mm** |
| `condim` / `priority` | 3 / 0 (raw XML) | 3 / 0 | yes |
| `friction` | `[1.0, 0.005, 0.0001]` | identical | yes |
| `solref` / `solimp` | `[0.02, 1.0]` / `[0.9, 0.95, 0.001, 0.5, 2.0]` | identical | yes |
| `contype` / `conaffinity` | 1 / 1 | 1 / 1 | yes |

mjlab's collision config (`microduck_constants.py:99-104`) applies **both** sides with one regex:
`condim={r"^(left|right)_foot_collision$": 3, ".*_collision": 1}`,
`priority={r"^(left|right)_foot_collision$": 1}`,
`friction={r"^(left|right)_foot_collision$": (1.0,)}` → **`priority=1` and `friction=(1.0,)` on
both feet; still symmetric.**

The only asymmetry is the right foot's inner edge sitting 0.123 mm further from the midline
(foot width 34.019 vs 33.896 mm, −0.36 %). Same effect size as the ankle mass: a 0.36 % smaller
contact patch changes that foot's available lateral friction by 0.36 % of its ~3.8 N normal load
≈ **14 mN** — 0.002 g-equivalent on a 780 g robot. **Physically negligible.** (Present identically
in PRE-XL330, so not swap-related.)

---

## 7. Q6 — Actuators, `ctrlrange`, gearing, `armature`

All 14 actuators instantiate the **same** default class (`robot_walk.xml:37-42`, applied at
`:411-425` via `class="chosen_actuator"`): `armature=0.0018`, `damping=0.053`,
`frictionloss=0.0048`, `position kp=0.55 kv=0 forcerange="-0.96 0.96" ctrlrange="-10 10"`.
Compiled-model check (script §`joints/actuators`, identical for the pre-swap model):

```
nu=14   distinct gear={(1.0,0,0,0,0,0)}   ctrlrange={(-10.0,10.0)}   forcerange={(-0.96,0.96)}
armature = 0.001800 for all 14 DOFs
```

No per-joint overrides, no L/R override — **zero actuator asymmetry in the XML.**

Under the real training stack this is overridden uniformly anyway: `FrictionDRBamActuatorCfg`
with a single `motor_name="hd1910"` and `target_names_expr=(r"^(?!passive_).*",)`
(`microduck_constants.py:127-167`) converts **every** matched actuator identically
(`vendor/bam/bam/mjlab.py:292-306`): `set_to_motor()`, `gear=[1,0,0,0,0,0]`,
`forcerange=±vin·kt/R`, `joint.armature = actuator.get_extra_inertia()` (one scalar for all 14),
`damping=0`, `frictionloss=0`. One motor model ⇒ one set of parameters ⇒ no L/R split is possible
without a code change.

---

## 8. What could not be established / explicit limits of this audit

* **Labeled speculation (not measurement):** §3.2's "the policy saturates both hip-yaw joints and
  therefore steers both feet the same way" is inferred from the model + the *other* agent's
  reported 48 %-at-the-stop observation. I did not run rollouts (no GPU; and rollouts are the
  other agents' scope). The model-level facts (shared world axis, mirrored ranges, 27.5°/-5°
  outcome of the both-at-+stop pose) are measured; the causal link to the observed veer is not.
* **Cannot distinguish** whether the ~18 µm CoM change between XL330 and HD-1910 or the 7.1 g
  official-vs-repo mass redistribution has *any* dynamical consequence — statically they are
  irrelevant, and the reported model-swap A/B (path efficiency 0.41 vs 0.24, |net yaw| 89° vs 89°)
  says the geometry/mass change is not the dominant cause. I did not measure dynamics.
* **Not checked:** the run-time collision/contact settings *as compiled by mjlab* for the actual
  walking task (I read the XML-compiled model + the cfg regexes, which agree, but I did not build
  the mjlab scene — that needs warp).
* **Not checked:** whether the sole STL assets themselves are exact mirror copies at the CAD
  source; the *exported* meshes differ by 1.84 mm at the worst vertex with a different vertex
  count (7896 vs 7946), which is most consistent with a different tessellation, but I cannot prove
  that from the STL alone. The contact patch (the part that touches) is mirror-equal except for
  the 0.123 mm inner edge.
* `robot_allcollisions_rollers.xml` uses different leg body names, so the script skips its per-body
  pair table (it still reports total mass 778.8296 g and CoM y = −0.039 mm at HOME). The walking
  task does not use that model.

---

## 9. Exact commands

```bash
# authoritative run (all numbers in this report)
.venv/bin/python docs/forensics/artifacts/harness/model_symmetry_audit.py

# provenance of the "pre-swap" file
git log --all --format=%h -- src/mjlab_microduck/robot/microduck/robot_walk.xml   # blob md5 match -> 1ee2e3c
for c in a91465c a91465c^ 79a993a 79a993a^ 1ee2e3c; do git show $c:src/mjlab_microduck/robot/microduck/robot_walk.xml \
  | grep -o 'mass="[0-9.e-]*"' | grep -o '[0-9.e-]*' | awk '{s+=$1} END{printf "'$c' %.4f g\n",s*1000}'; done

# servo-swap diff (masses/inertias only)
git show a91465c -- src/mjlab_microduck/robot/microduck/robot_walk.xml
# the real XL330 baseline (NOT a91465c^, which is the 800.24 g HLS2909 model)
git show 79a993a^:src/mjlab_microduck/robot/microduck/robot_walk.xml | diff - src/mjlab_microduck/robot/microduck/robot_walk_preswap.xml

# hip_yaw range history
for c in $(git log --format=%h -- src/mjlab_microduck/robot/microduck/robot_walk.xml); do \
  git show $c:src/mjlab_microduck/robot/microduck/robot_walk.xml | grep -o 'name="left_hip_yaw"[^/]*'; done

# official runtime reference
cat /home/joyandai/microduck/kinematics/assets/alpha/robot_walk.xml
```

Artifacts: `docs/forensics/artifacts/harness/model_symmetry_audit.py` (script, self-testing),
raw stdout kept at `/tmp/audit_out.txt` and `/tmp/audit_out2.txt` during this session.
