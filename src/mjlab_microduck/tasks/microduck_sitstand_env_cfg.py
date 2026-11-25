"""Microduck *sitstand* task (v1.5, mjlab 1.3.0) — commanded sit ↔ stand, GENTLY.

One policy, both directions, driven by a posture command:
    cmd (twist slot) = [sit_flag, 0, 0]   sit_flag ∈ {0 = STAND, 1 = SIT}
"Stand" is the all-zero command — the same deployment idle as every other
policy. The command flips mid-episode with a dwell time of a few seconds, so
each episode trains descents, seated rest, rises and standing rest, plus
"hold what you're already doing" (reset state × command are independent).

2026-08 rebuild from scratch (the old phase-cycle env predates the 1.3.0
migration and every sit/standup lesson). Design synthesis:
  - Posture-conditioned single-target rewards (mdp posture_*): the sit env's
    minimum-viable "organic discovery" stack, but the target (SIT keyframe +
    SIT_Z vs HOME + STAND_Z) is selected per env from the live command. No
    trajectory, no waypoints, no phase timing — the policy discovers its own
    transition path, in as many steps as it likes (knee-down first, head
    assist, etc. are all allowed: full-collision model, no head-ground
    penalty, no fall termination).
  - Gentleness both ways: descent-speed cap (sit env's proven recipe, -10
    from step 0) AND a mirrored rise-speed cap (introduced by curriculum
    AFTER the rise is discovered — the standup attempt-tax lesson), plus the
    |a_z| shock penalty throughout.
  - Rest quality: posture_stillness (velocity-Gaussian at the commanded
    height, tilt-gated) + posture_composite (multiplicative height·upright·
    pose vs the commanded target — partial-sum exploits like plank/flop/lean
    collapse to ~0).
  - Head commandable in BOTH postures (head_pose command + tracking, exactly
    like velocity/standup), body_command slot zero-padded → 61D obs parity.
  - Sim2real: velocity-parity DR / obs noise / delays / regularisers (the
    transferring recipe), sit env's contact-solver hardening (nconmax=200,
    iters 30/50 — seated contact NaN fix), delayed push ramp (pushes early
    made the sit env unlearn sitting).

Keyframes (stability-verified, keep in sync with sit/standup envs):
  SIT  = knee ±1.35, hip_pitch ∓0.4079, ankle/hip_roll 0, trunk z 0.060
         (swept 2026-07-27 — the old keyframe tipped over; verify TILT in sim
         before changing this pose).
  STAND = HOME joints, trunk z 0.115 (measured standing equilibrium).

Joint layout (14 actuated joints):
    0-4 : left  leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)
    5-8 : neck/head (neck_pitch, head_pitch, head_yaw, head_roll)
    9-13: right leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)
"""

import math
import os as _os
from copy import deepcopy


def _env_bool(name: str, default: bool) -> bool:
    value = _os.environ.get(name)
    return default if value is None or value == "" else value.strip().lower() in (
        "1", "true", "yes", "on"
    )


def _env_float(name: str, default: float) -> float:
    value = _os.environ.get(name)
    return default if value is None or value == "" else float(value)

# Symmetry
ENABLE_SYMMETRY = False

# ── Domain randomisation (matched to the velocity env for sim2real parity) ────
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True   # match velocity: randomize head-assembly CoM
ENABLE_KP_RANDOMIZATION              = False  # match velocity (OFF)
ENABLE_KD_RANDOMIZATION              = False  # match velocity (OFF)
ENABLE_MASS_INERTIA_RANDOMIZATION    = True   # match velocity: dr.pseudo_inertia (mass+inertia)
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True   # match velocity: FrictionDRBamActuator.friction_scale
ENABLE_ARMATURE_RANDOMIZATION        = True   # match velocity: reflected rotor inertia
ENABLE_VELOCITY_PUSHES               = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True   # match velocity: obs-level per-env misalignment
ENABLE_ENCODER_BIAS                  = True   # match velocity: per-env joint encoder offset (actor obs)

# ── Ranges (matched to the velocity env) ──────────────────────────────────────
COM_RANDOMIZATION_RANGE             = 0.003           # ramped to 0.015 via com_range curriculum
HEAD_COM_RANDOMIZATION_RANGE        = 0.003           # ramped to 0.01 via head_com_range curriculum
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
KP_RANDOMIZATION_RANGE              = (0.85, 1.15)    # unused (kp DR off)
KD_RANDOMIZATION_RANGE              = (0.9, 1.1)      # unused (kd DR off)
VELOCITY_PUSH_INTERVAL_S            = (3.0, 6.0)
# Final magnitude matches velocity's ±0.3 but the ramp is DELAYED (see the
# push_magnitude curriculum): the sit env's lesson — pushes mid-descent before
# the transition motions have consolidated make the policy unlearn them and
# converge to "just stand doing nothing".
VELOCITY_PUSH_RANGE                 = (-0.3, 0.3)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0  # match velocity (obs-level, zero-centered random axis)

# Episode length: room for 2-3 posture segments (dwell 3.5-6.5 s each), i.e.
# at least one full sit → rest → rise → rest cycle per episode.
EPISODE_LENGTH_S = 12.0
# Dwell time in each commanded posture before a resample may flip it. The
# lower bound must comfortably exceed a gentle transition (~1.5 s) plus some
# rest, so "arrive, then hold still" is always trained.
POSTURE_DWELL_S  = (3.5, 6.5)
# Probability a resample commands SIT (vs STAND). 0.5 → all four combinations
# of (reset state × command) get equal coverage, including both holds.
# These are intentionally environment overrides rather than a new default
# curriculum.  R4 first learns the stand endpoint with a neutral head, then
# restores the original 50/50 reset/command mix while resuming the same policy.
# The defaults remain the established task recipe.
SIT_PROB         = _env_float("MICRODUCK_SIT_PROB", 0.5)
SITTING_INIT_PROB = _env_float("MICRODUCK_SITTING_INIT_PROB", 0.5)
if not 0.0 <= SIT_PROB <= 1.0:
    raise ValueError(f"MICRODUCK_SIT_PROB must be in [0, 1], got {SIT_PROB}")
if not 0.0 <= SITTING_INIT_PROB <= 1.0:
    raise ValueError(
        "MICRODUCK_SITTING_INIT_PROB must be in [0, 1], "
        f"got {SITTING_INIT_PROB}"
    )

# ── SIT keyframe (joint_pos index → angle in rad). Single fixed target. ─────
# STABILITY-VERIFIED 2026-07-27 (sit env, scratchpad sweep_sit_pose2.py):
# knee ±1.35, hip_pitch = HOME ∓ 0.05 lean, ankle 0, hip_roll 0 settles at
# 3-5° tilt for 95-100% of noisy resets. The old keyframe (knee ±1.0472,
# hip_pitch HOME) is NOT statically stable — it tips to ~88° in 1 s and
# silently drove the sit env's whole hop/back-flop/plank exploit chain.
# If the robot or keyframe changes, RE-RUN THE SWEEP — verify tilt, not z.
# Keep in sync with microduck_sit_env_cfg.SITTING_TARGET_OVERRIDES and
# microduck_standup_env_cfg.SITTING_JOINT_OVERRIDES.
SITTING_TARGET_OVERRIDES = {
    1:   0.0,      # left  hip_roll   (HOME -0.0873)
    2:  -0.4079,   # left  hip_pitch  (HOME -0.4579; +0.05 = slight fwd lean)
    3:   1.35,     # left  knee       (HOME 0)
    4:   0.0,      # left  ankle      (HOME +0.4530)
    # neck/head intentionally omitted → steered by the head_pose command.
    10:  0.0,      # right hip_roll   (HOME +0.0873)
    11:  0.4079,   # right hip_pitch  (HOME +0.4579)
    12: -1.35,     # right knee       (HOME 0)
    13:  0.0,      # right ankle      (HOME -0.4530)
}

_LEG_JOINTS  = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_NECK_JOINTS = [5, 6, 7, 8]
_NECK_JOINT_NAMES = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")

# ── 头部姿态 L1（2026-09-16，实测后新增）─────────────────────────────────────
# 为什么需要它（实测见 docs/q1_q2_root_cause_2026-09-16.md §5.1c 与
# scripts/probe_sitstand_head_terms.py）：三脚架修好后，策略**仍然**把 neck_pitch
# 停在硬限位（差 110°）、头高从 162 掉到 147 mm，而**两个已有的头部项都拉不动它**：
#   * head_pose_tracking = mean_j exp(−(err_j/std)²) —— 对 4 个关节取**均值**（docstring
#     写明是为了"单个关节坏了别把整项打死"）。副作用是相反方向的失效：3 个关节跟得好时，
#     脖子差 110° 只让该项 1.0 → 0.667，只收 0.25/步 ⇒ **对"单一关节停死"近乎失明**。
#   * posture_composite 的头因子 = exp(−mean(err²)/head_std²) **看得见**脖子（0.0021），
#     但它是**乘性**的，把自己的梯度也乘没了：∂/∂neck ≈ −2·err/σ²·head_factor·其余因子·w
#     ≈ **−0.09 /rad**。脖子挪 0.2 rad 只值 0.018/步，**低于 PPO 优势估计的噪声底**。
#     那 +1.8/步的余量**存在但不可达**（正是仓库已记录的"高斯基死在尾部"）。
# ⇒ 修法不是调大权重、也不是调 std，而是给一个**在大误差处梯度不消失**的形式：
#   对头关节**求和**的 L1（不是取均值），只在 STAND 静息时生效（过渡期保留用头辅助的自由）。
ENABLE_HEAD_L1 = _env_bool("MICRODUCK_HEAD_L1", False)
# 标定：静息时 L1 和 = |−1.93| − 0.10 ≈ 1.83 rad ⇒ 权重 1.0 时罚 −1.83/步，
# 而展开脖子可拿回约 +1.8(composite) + 0.25(tracking) = +2.05/步
# ⇒ 摆动 ≈ 3.9/步（约总奖励 10.3/步的 38%），梯度 −1.0/rad 是 composite 现状的 11 倍。
HEAD_L1_WEIGHT = 1.0
HEAD_L1_DEADBAND = 0.10

# ── Head-tripod rest fix (2026-09-16) ────────────────────────────────────────
# The posture ramp is POSTURE_RAMP_S = 2 s and `alpha` slews the setpoint; the
# command is a binary flag, so "the ramp has settled" is |flag - alpha| < eps.
RAMP_DONE_EPS = 0.02
# Load below this is a light brush, not support: the feet carry 4.7 N in the
# tripod pose and the head carries 3.16 N, so 0.5 N is unambiguously "resting on
# the head" while ignoring incidental contact during a wobbly stand.
HEAD_GROUND_FORCE_THRESHOLD_N = 0.5
# Weight calibration: the tripod's measured advantage is +3.69/step against a
# ~3.16 N head load, so ~1.4 /N flips it. Ramp 0 -> target over 300 iters of env
# steps (the same "no attempt-tax while a skill is still being discovered" rule
# the rest of this file follows, and by then the rise is long since learned).
HEAD_GROUND_WEIGHT = 1.4
NECK_LIMIT_PROXIMITY_MARGIN = 0.20   # rad of "keep out" band before the hard stop
NECK_LIMIT_PROXIMITY_WEIGHT = -1.0
HEAD_FIX_RAMP_ITERS = 300

# Trunk height targets (m) — both MEASURED in sim, never carried across robot
# or keyframe changes (sit run-1 / standup lessons).
STAND_Z = 0.115
SIT_Z   = 0.060

# Upright gating window for ``upright_while_tall``: full upright incentive
# above STAND_UPRIGHT_Z, fades to 0 at SIT_UPRIGHT_Z (committed to the sit).
# Blocks the "tip backward while still high" descent exploit; the always-on
# upright_linear floor covers the seated regime.
STAND_UPRIGHT_Z = 0.10
SIT_UPRIGHT_Z   = 0.075

# Target-ramp duration (s): the command term slews an internal target blend
# STAND↔SIT over this time, and the posture rewards track the MOVING target.
# THE anti-crash mechanism (run-1 failure: near-instant transitions). With a
# binary target, arriving early pays the full goal jackpot (~7/step) for
# every step saved, while the linear speed caps integrate to a bounded
# excess-distance cost (~50 total for an instant drop) — crashing won ~7×.
# With the ramp, being AHEAD of the setpoint zeroes the height/composite
# stack for the ramp remainder, so tracking the slow setpoint is the argmax.
# 55 mm over 2 s ≈ 0.028 m/s, comfortably under both caps below.
POSTURE_RAMP_S = 2.0

# Vertical-speed caps (m/s) — now BACKSTOPS for overshoot/bounce around the
# slewed target (see POSTURE_RAMP_S), not the primary gentleness mechanism.
# The rise cap is looser (rising against gravity needs some momentum to get
# over the heels) and is introduced by curriculum only after the rise motion
# has been discovered — see the rise_speed_weight curriculum.
MAX_DESCENT_SPEED = 0.05
MAX_RISE_SPEED    = 0.08

# ── Seated fall-back hardening (2026-10) ──────────────────────────────────────
# Real robot: an abrupt head move while seated tips it BACKWARD; it stays
# leaned ~30° in the sit pose and the neck servo overloads. Sim battery
# (claude_experiments/sitstand_head_bench.py, r78hyhck@1250): fast head
# wiggles / pitch-down+yaw jumps tip 30-45% of seated robots, none recover,
# neck |τ| 0.13-0.19 Nm vs 0.02 calm. A backward kick of |ω| ≈ 3 rad/s leaves
# 69% stuck leaned (claude_experiments/sitstand_tip_probe.py), ≥ 4 rad/s
# flops to ~65° on the back. Training had neither fast head commands (the
# head command only resamples every few s) nor any leaned-back seated state.
#  - seated_head_jerk: abrupt head-command jumps / sign flips while seated.
#  - seated_tip: backward pitch kicks while seated → recovery frontier.
#  - neck_strain_leaned: neck torque cost while leaned BACK in the sit.
# Both events are introduced by curriculum only once the sit exists (the
# push-ramp lesson: perturbing an unconsolidated transition unlearns it).
ENABLE_SEATED_FALLBACK_HARDENING = True
SEATED_HEAD_JERK_INTERVAL_S = (0.3, 0.8)
SEATED_TIP_INTERVAL_S       = (3.0, 6.0)

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MICRODUCK_ROUGH_TERRAINS_CFG,
    HEAD_BODY_NAMES,
    HEAD_POSE_CMD_RESAMPLE_S,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_sitstand_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Microduck sitstand environment configuration."""

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    # NOTE: the SIT env penalises head-ground contact; this env deliberately did
    # NOT, so that the head could be used as a third support point during a
    # transition. That freedom was right, but it also left the head-tripod REST
    # pose unpriced — and the policy parked there (see `head_ground` below and
    # docs/policy_failure_analysis.md §8). The fix is not to re-add a blanket
    # contact penalty; it is to price the LOAD only once the posture ramp has
    # settled (`mdp.posture_head_ground_penalty`), so the manoeuvre stays free.
    head_ground_cfg = ContactSensorCfg(
        name="head_ground_contact",
        # jaw_soft carries the head collision geoms (top_head_shell = the flat
        # top, jaw, bottom_head_shell) in robot_allcollisions.xml — same primary
        # as the roulade env's head pivot sensor.
        primary=ContactMatch(mode="body", pattern="jaw_soft", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    # Standup robot variant: full collision meshes — the body must physically
    # rest on the ground while seated, and knees/head may touch mid-transition.
    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors  = (feet_ground_cfg, self_collision_cfg, head_ground_cfg)
    cfg.viewer.body_name = "trunk_base"

    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Actions ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Rewards: drop walking-specific terms ──────────────────────────────────
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── Rewards: posture-conditioned single-target stack ──────────────────────
    # Every task term below reads the commanded posture and selects its target
    # (SIT keyframe + SIT_Z vs HOME + STAND_Z) per env. Weights mirror the sit
    # env's proven stack (positive task mass ≈ velocity scale, so the shared
    # sim2real regularisers act at the same RELATIVE strength — the standup
    # transfer lesson).

    # Pose target — legs only (head is command-steered). Generous std keeps
    # gradient alive from either end (~1.35 rad knee delta).
    cfg.rewards["posture_pose_legs"] = RewardTermCfg(
        func=microduck_mdp.posture_pose_match,
        weight=4.0,
        params={
            "command_name":  "twist",
            "std":           0.5,
            "joint_indices": _LEG_JOINTS,
            "sit_overrides": SITTING_TARGET_OVERRIDES,
        },
    )

    # Head pose tracking (commandable head control, like velocity/standup) —
    # active in BOTH postures. Weight kept light so a transient head-assist
    # during a transition only pays a small tracking cost.
    cfg.rewards["head_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.head_pose_tracking,
        weight=0.75,
        params={"command_name": "head_pose", "std": 0.5},
    )

    # Head DC-droop penalty (velocity/standup's fix, adapted for sitstand).
    # L1 on a 1s EMA of the head tracking error — prices only the SUSTAINED
    # gravity sag the policy can cancel by biasing the neck command up; transient
    # motion averages out. head_pose_tracking (Gaussian, std=0.5) alone lets the
    # 37%-mass head sag ~10° for ~0.03 reward, so without this the policy just
    # lets the head droop, which reads as "can stand but won't lift the head".
    #
    # Weights are scaled by the CURRENT head mass ratio from the HD-1910 servo
    # swap (XL330 18g -> HD-1910 21g = +3g,
    # 14 servos = +42g, robot 737g -> 779g). Head assembly went 279.9g -> 288.9g
    # (+3.2%), so the gravity droop load grows x1.032; standup's final
    # 1.5 * 1.032 = 1.55. Use the MASS ratio, not the body-share (share rose to
    # 37.1% only because the whole robot got heavier; droop scales with absolute
    # head mass).
    #
    # Upright gate (same values as standup): the gate zeroes the error feeding
    # the EMA below gate_height_low, so the seated/rising phase (SIT_Z≈0.060,
    # head may touch the ground) accumulates NOTHING — no tax on the transition
    # head-pivot; only the standing rest (z>0.11) is priced. Starts at 0 and is
    # introduced by curriculum below (discovery-vs-refinement timing).
    cfg.rewards["head_pose_bias"] = RewardTermCfg(
        func=microduck_mdp.head_pose_bias_penalty,
        weight=0.0,  # ramped by head_pose_bias_weight curriculum below
        params={
            "command_name":       "head_pose",
            "tau_s":              1.0,
            "gate_height_low":    0.09,
            "gate_height_high":   0.11,
            "gate_tilt_full_deg": 20.0,
            "gate_tilt_zero_deg": 45.0,
        },
    )

    # L1 bootstrap — constant gradient toward the commanded pose.
    cfg.rewards["posture_pose_l1"] = RewardTermCfg(
        func=microduck_mdp.posture_pose_l1,
        weight=1.0,
        params={
            "command_name":  "twist",
            "joint_indices": _LEG_JOINTS,
            "sit_overrides": SITTING_TARGET_OVERRIDES,
        },
    )

    # Trunk height — two-layer Gaussian (standup recipe: wide layer for the
    # bootstrap pull across the 55 mm travel, sharp layer so the final cm has
    # real gradient instead of a saturated plateau) + L1 transition driver.
    cfg.rewards["posture_height"] = RewardTermCfg(
        func=microduck_mdp.posture_height_gaussian,
        weight=1.0,
        params={
            "command_name": "twist",
            "sit_z":        SIT_Z,
            "stand_z":      STAND_Z,
            "std":          0.04,
        },
    )
    cfg.rewards["posture_height_sharp"] = RewardTermCfg(
        func=microduck_mdp.posture_height_gaussian,
        weight=1.0,
        params={
            "command_name": "twist",
            "sit_z":        SIT_Z,
            "stand_z":      STAND_Z,
            "std":          0.015,
        },
    )
    # L1 weight 6.0: between sit's 5.0 and standup's 7.5 — resting in the
    # WRONG posture must be clearly net-negative in both directions (staying
    # seated under a stand command was the standup env's stall mode at low L1).
    cfg.rewards["posture_height_l1"] = RewardTermCfg(
        func=microduck_mdp.posture_height_l1,
        weight=6.0,
        params={
            "command_name": "twist",
            "sit_z":        SIT_Z,
            "stand_z":      STAND_Z,
        },
    )

    # Rise bootstrap — pays for upward motion itself when STAND is commanded
    # and the trunk is below 0.125 (just ABOVE the target so the final cm
    # still pays). Destination-only rewards have zero gradient at zero motion;
    # without this the standup env parked seated. Zero under a SIT command.
    cfg.rewards["rise_bootstrap"] = RewardTermCfg(
        func=microduck_mdp.posture_rise_bootstrap,
        weight=0.75,
        params={
            "command_name": "twist",
            "max_height":   0.125,
            "max_vz":       MAX_RISE_SPEED,  # explosive launch can't out-earn a gentle rise
        },
    )

    # ── Gentleness (the point of this env) — three complementary signals ─────
    #  - ``descent_speed``: per-step penalty on downward vz beyond 0.05 m/s.
    #    THE anti-brutality term for the sit: a fast drop pays on every step
    #    of the fall so it can't be amortised. -10 from step 0 (sit lesson:
    #    at -5 a crash-sit was net-positive), tightened to -20 by curriculum.
    #  - ``rise_speed``: the mirror cap for the stand-up (0.08 m/s). Starts at
    #    weight 0 and is introduced at iter 750 by curriculum — the standup
    #    lesson: a motion-tax active while the skill is being DISCOVERED makes
    #    exploratory attempts net-negative and the skill is never found. The
    #    sit-keyframe start is easy (no prone flips), so 750 is late enough.
    #  - ``gentle_motion``: |a_z| shock penalty, both directions, always on.
    #
    # ⚠️ POSITIVE weights, deliberately: these three functions ALREADY return
    # negative values (-clamp(...), -|a_z|), same convention as the *_l1_penalty
    # helpers (used with +1/+6 here). Run 7ev90yd9 (2026-08-12) had them at
    # negative weights — the double negative made them REWARDS for violence
    # (wandb: Episode_Reward/descent_speed +4.6, rise_speed +2.1, gentle_motion
    # +0.57, the three biggest positive terms) and trained a butt-hopping,
    # crash-sitting policy. Same bug class roller_standup found in gentle_rise.
    # After any reward change, check wandb Episode_Reward/<penalty> stays ≤ 0.
    cfg.rewards["descent_speed"] = RewardTermCfg(
        func=microduck_mdp.trunk_downward_velocity_penalty,
        weight=10.0,
        params={
            "max_down_vel": MAX_DESCENT_SPEED,
            "asset_cfg":    SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["rise_speed"] = RewardTermCfg(
        func=microduck_mdp.trunk_upward_velocity_penalty,
        weight=0.0,
        params={
            "max_up_vel": MAX_RISE_SPEED,
            "asset_cfg":  SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["gentle_motion"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # Two-layer upright pressure (sit env values — the anti-flop calibration):
    #  - always-on linear floor: holds the trunk vertical at BOTH rests; at 2.5
    #    "lie on your back" trails upright rest by ~4.5/step (sit run-2 fix).
    #  - height-gated booster: blocks the "tip backward while tall" descent
    #    exploit; during the rise it doubles as an arrival-uprightness pull.
    cfg.rewards["upright_linear"] = RewardTermCfg(
        func=microduck_mdp.body_upright_linear,
        weight=2.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    cfg.rewards["upright_while_tall"] = RewardTermCfg(
        func=microduck_mdp.upright_while_tall,
        weight=1.5,
        params={
            "height_low":  SIT_UPRIGHT_Z,
            "height_high": STAND_UPRIGHT_Z,
            "asset_cfg":   SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Stillness at the commanded posture — "arrive, then rest QUIETLY, UPRIGHT"
    # as an explicit positive peak. The z gate is a band around the commanded
    # height (inactive during transitions); the tilt gate pays nothing for a
    # tilted rest (back/face/side flops earn zero — the sit run-2 exploit).
    cfg.rewards["posture_stillness"] = RewardTermCfg(
        func=microduck_mdp.posture_stillness,
        weight=2.0,
        params={
            "command_name":  "twist",
            "sit_z":         SIT_Z,
            "stand_z":       STAND_Z,
            "band_full":     0.012,
            "band_zero":     0.03,
            "vel_std":       0.05,
            "tilt_full_deg": 25.0,
            "tilt_zero_deg": 60.0,
        },
    )

    # Multiplicative goal score vs the COMMANDED target — kills partial-sum
    # farming in both postures (plank, flop, lean, park-1cm-short). Broad stds
    # keep gradient visible far from the goal (standup's proven calibration).
    # head_std adds the neck/head-at-command factor: the first sign-fixed run
    # rested with the head DANGLING to the floor (trunk/legs/z all on target →
    # full composite, only the 0.75 tracking term lost, and the hanging head
    # adds passive stability). With the head factor, the goal state itself
    # requires the head up at its commanded pose; transient head assist
    # mid-transition stays free (composite ≈0 there anyway).
    cfg.rewards["posture_composite"] = RewardTermCfg(
        func=microduck_mdp.posture_composite,
        weight=3.0,
        params={
            "command_name":  "twist",
            "sit_overrides": SITTING_TARGET_OVERRIDES,
            "joint_indices": _LEG_JOINTS,
            "sit_z":         SIT_Z,
            "stand_z":       STAND_Z,
            "height_std":    0.03,
            "upright_std":   0.40,   # ≈ 23° effective — plank (~70°+) scores ~0
            "pose_std":      0.40,
            "head_std":      0.40,   # head fully dropped (~1.2 rad) → factor ~0.01
        },
    )

    # ── Head-tripod terminal rest: price the LOAD, once the rise has settled ──
    # The measured exploit (docs/policy_failure_analysis.md §8): the standing rest
    # is head-planted, head-ground force 3.16 N = 37 % of body weight, and that
    # pose beats an upright stand by +3.69/step because the two trunk-HEIGHT terms
    # saturate in it while the only objecting term (the multiplicative
    # posture_composite) is too light to outbid them. head_pose_bias is worth
    # -0.00084/step there, i.e. 0.1 % of the gap — turning its weight up ~1000x
    # would wreck every other term instead.
    #
    # So remove the SUPPORT, not the symptom: charge the vertical load the head
    # carries, and only while STAND is commanded AND the 2 s posture ramp has
    # settled (|flag - alpha| < ramp_eps). Mid-transition head assist — which this
    # env deliberately allows and which the rise genuinely uses — is untouched.
    #
    # Returns <= 0 -> POSITIVE weight. Calibration: the tripod's advantage is
    # +3.69/step and it loads the head with ~3.16 N, so a weight near
    # (3.69 / (3.16 - 0.5)) ~= 1.4 /N makes the tripod net-negative. Start at 0
    # and ramp (discovery-vs-refinement timing, same as every other tax here).
    cfg.rewards["head_ground"] = RewardTermCfg(
        func=microduck_mdp.posture_head_ground_penalty,
        weight=0.0,  # ramped by head_ground_weight curriculum below
        params={
            "sensor_name":       head_ground_cfg.name,
            "command_name":      "twist",
            "ramp_eps":          RAMP_DONE_EPS,
            "force_threshold_n": HEAD_GROUND_FORCE_THRESHOLD_N,
            "use_normal_force":  True,
        },
    )

    # Neck/head joints parked on their HARD stops. Measured on the delivered
    # policy: neck_pitch sits at -1.583 rad against a hard lower limit of
    # -1.5708 rad, in every phase. The stock dof_pos_limits term only fires past
    # the soft limit (last ~7.5 % of range) and only by the overshoot magnitude,
    # so it reads ~0 for a joint parked ON the stop. Returns >= 0 -> NEGATIVE
    # weight. Same gate: only the standing rest is priced.
    cfg.rewards["neck_limit_proximity"] = RewardTermCfg(
        func=microduck_mdp.joint_limit_proximity_standing,
        weight=0.0,  # ramped by head_ground_weight curriculum below
        params={
            "command_name": "twist",
            "ramp_eps":     RAMP_DONE_EPS,
            "margin":       NECK_LIMIT_PROXIMITY_MARGIN,
            "asset_cfg":    SceneEntityCfg("robot", joint_names=_NECK_JOINT_NAMES),
        },
    )

    # 头部姿态 L1（见常量区注释）。返回 ≤0 ⇒ 正权重。默认关，供受控 A/B。
    if ENABLE_HEAD_L1:
        cfg.rewards["head_pose_l1"] = RewardTermCfg(
            func=microduck_mdp.head_pose_error_l1_standing,
            weight=0.0,  # ramped by head_l1_weight curriculum
            params={
                "command_name": "head_pose",
                "ramp_eps":     RAMP_DONE_EPS,
                "deadband":     HEAD_L1_DEADBAND,
            },
        )

    # ── Sim2real regularisers — MATCHED to velocity ─────────────────────────
    # velocity's exact set and absolute weights:
    #   • action_rate_l2: -0.1 at stage 0, ramped -0.1 → -1.0 by iter 1500
    #   • body_ang_vel -0.05, angular_momentum -0.02
    #   • soft_landing dropped; joint_torques_l2 / neck_action_rate_l2 not added
    # Plus joint_torque_rate_l2 (anti-jitter), phased in once the transition
    # motions exist. Both caps + |a_z| already push toward slow-careful motion;
    # per the regularizer-type lesson these smoothness terms damp jitter
    # WITHOUT blocking a slow big motion, so heavier-than-velocity would also
    # be defensible — start at parity, tighten only if the real robot shakes.
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2, weight=0.0
    )

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05      # velocity value
    cfg.rewards["angular_momentum"].weight = -0.02  # velocity value
    cfg.rewards.pop("soft_landing", None)           # velocity removes it

    # Neck strain while leaned BACK in the sit (the overload symptom). Cost
    # function (≥ 0) → negative weight. ≈ -1.1/step at overload torque
    # (0.15 Nm → (0.15/0.1)² = 2.25), zero when upright / standing / rising /
    # leaning forward, so it never taxes sit or rise discovery → on from 0.
    if ENABLE_SEATED_FALLBACK_HARDENING:
        cfg.rewards["neck_strain_leaned"] = RewardTermCfg(
            func=microduck_mdp.neck_torque_when_tilted,
            weight=-0.5,
            params={
                "command_name":   "twist",
                "tilt_start_deg": 12.0,
                "tilt_full_deg":  25.0,
                "torque_scale":   0.1,
            },
        )

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # Drop the base "upright" Gaussian — replaced by the two-layer upright above.
    if "upright" in cfg.rewards:
        del cfg.rewards["upright"]

    # ── Observations (identical layout to walking / sit / standup policies) ───
    del cfg.observations["actor"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    # mjlab 1.3.0 base template adds sensor-based foot_height + height_scan obs.
    # Sitstand has no terrain-height sensor (and drops the walking foot rewards),
    # so remove these terms. foot_air_time/foot_contact(_forces) use the
    # feet_ground_contact sensor, which sitstand does define, so they stay.
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )

    # IMU obs delay: max_lag 1 — velocity's 2026-07 audit value (real dxl IMU
    # path is fast, ±20 ms envelope).
    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    # Obs noise matched to the velocity env.
    cfg.observations["actor"].terms["base_ang_vel"].noise    = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise       = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise       = Unoise(n_min=-0.25, n_max=0.25)

    # IMU mounting-misalignment DR (match velocity): per-env constant rotation of
    # the IMU-derived actor obs; critic keeps the true values.
    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    # 1-ctrl-step lag on joint_vel (Dynamixel present_velocity is ~1 period old).
    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    # Deepcopy joint_pos/joint_vel per group (they share base-template objects) so
    # the encoder-bias `biased` flag below applies to the actor only.
    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    # Encoder-bias DR (match velocity): actor sees joint_pos + per-env bias;
    # critic keeps the true joint pos.
    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # ── Head pose command (commandable head control, like velocity/standup) ──
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=HEAD_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.05, 0.05),    # neck_pitch
            (-0.05, 0.05),    # head_pitch
            (-0.07, 0.07),    # head_yaw
            (-0.015, 0.015),  # head_roll
        ),
    )

    # Command obs slots. head_command is the real head_pose command;
    # body_command stays zero-padded (body control not used here).
    # Layout parity with velocity/standup: [twist(3), head_pose(4), body_pose(6)].
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": "head_pose"},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # ── Command: sit/stand posture flag in the twist slot ────────────────────
    # cmd = [sit_flag, 0, 0]; dwell-time resampling flips the posture mid-
    # episode. "Stand" is the all-zero command (deployment idle parity). The
    # runtime drives this by writing 0/1 into the vx slot of the command
    # buffer. Internally the term slews a target blend over POSTURE_RAMP_S
    # that the posture rewards track (see the constant's comment); the OBS
    # stays the raw binary flag.
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs  = 0.0
    command.heading_command   = False
    command.ranges.heading    = None
    command.resampling_time_range = POSTURE_DWELL_S
    command.debug_vis = False
    cfg.commands["twist"] = microduck_mdp.SitStandCommandCfg(
        **{
            **vars(command),
            "sit_prob": SIT_PROB,
            "ramp_s":   POSTURE_RAMP_S,
            "sit_z":    SIT_Z,
            "stand_z":  STAND_Z,
        }
    )

    # ── Terminations ──────────────────────────────────────────────────────────
    # No fall termination: wobbles/tips during transitions must play out so the
    # policy experiences the impact/upright costs instead of a truncated episode.
    if "fell_over" in cfg.terminations:
        del cfg.terminations["fell_over"]
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )

    # ── Events ────────────────────────────────────────────────────────────────
    # BAM (mjlab_frictionloss branch) writes per-env dof_frictionloss/dof_damping
    # every step; this no-op event registers those fields for per-world expansion.
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )

    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)  # match velocity

    # Base reset: standing, just above the measured equilibrium (STAND_Z=0.115).
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.11, 0.12)

    # Reset-state mix: 50% standing / 50% already seated (SIT keyframe with
    # joint/tilt noise). Combined with the independent 50/50 posture command
    # this trains all four cases — sit-from-stand, rise-from-sit, hold-stand,
    # hold-sit — and hands the policy both goal states' values directly (the
    # sit env's discovery-bootstrap lesson, extended to both ends).
    cfg.events["set_ground_state"] = EventTermCfg(
        func=microduck_mdp.set_random_ground_state,
        mode="reset",
        params={
            "face_down_prob":          0.0,
            "face_up_prob":            0.0,
            "sitting_prob":            SITTING_INIT_PROB,
            "standing_prob":           1.0 - SITTING_INIT_PROB,
            "sitting_joint_overrides": SITTING_TARGET_OVERRIDES,
            "sitting_joint_noise_std": 0.10,           # ≈ 6° per joint
            "sitting_tilt_max":        math.radians(8),
            "sitting_z_min":           0.06,            # settles to the 0.060 rest
            "sitting_z_max":           0.075,
            "standing_z_min":          0.11,
            "standing_z_max":          0.12,
        },
    )

    # MuJoCo physics robustness (sit env's contact NaN fix). The standup XML
    # has full collisions on every body; the seated pose puts trunk + folded
    # legs + head all in close ground/self contact. Default nconmax=35 and
    # solver iters=10 overflow the contact solver on sit attempts → NaN →
    # nan_state terminations that punish the descent itself ("learn then
    # unlearn by iter 500" pattern).
    cfg.sim.nconmax = 200
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.ls_iterations = 50

    if ENABLE_VELOCITY_PUSHES:
        interval = (0.5, 1.0) if play else VELOCITY_PUSH_INTERVAL_S
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=interval,
            params={
                "velocity_range": {
                    "x": VELOCITY_PUSH_RANGE,
                    "y": VELOCITY_PUSH_RANGE,
                },
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    if ENABLE_SEATED_FALLBACK_HARDENING:
        # Both start disabled (prob 0 / zero kick) and are ramped by the
        # seated_head_jerk_prob / seated_tip_range curricula below.
        cfg.events["seated_head_jerk"] = EventTermCfg(
            func=microduck_mdp.seated_head_command_jerk,
            mode="interval",
            interval_range_s=SEATED_HEAD_JERK_INTERVAL_S,
            params={"prob": 0.0, "flip_frac": 0.5},
        )
        cfg.events["seated_tip"] = EventTermCfg(
            func=microduck_mdp.seated_backward_tip,
            mode="interval",
            interval_range_s=SEATED_TIP_INTERVAL_S,
            params={
                "ang_vel_range": (0.0, 0.0),
                "prob":          0.5,
                "lateral_frac":  0.3,
            },
        )

    if ENABLE_COM_RANDOMIZATION:
        # mjlab 1.3.0: stock dr.body_ipos (operation="add") reads the compile-time
        # default each reset → non-accumulating natively.
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
        kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)
        cfg.events["randomize_motor_gains"] = EventTermCfg(
            func=microduck_mdp.randomize_delayed_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "kp_range": kp_range,
                "kd_range": kd_range,
            },
        )

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        # match velocity: physics-consistent mass+inertia via pseudo_inertia
        # (alpha scales both by e^(2α), CoM untouched). Startup mode.
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )

    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        # match velocity: scale BAM's friction budget per-env via the
        # FrictionDRBamActuator hook (dof_frictionloss is zeroed under BAM).
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    # NOTE: IMU mounting-misalignment is applied at the OBSERVATION level above
    # (matching velocity) — the old event-based randomize_imu_orientation wrote
    # site_quat, which under mjlab 1.3.0 is neither per-env nor read by the obs.

    # ── Terrain ───────────────────────────────────────────────────────────────
    if not rough:
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
    else:
        cfg.scene.terrain.terrain_type = "generator"
        cfg.scene.terrain.terrain_generator = MICRODUCK_ROUGH_TERRAINS_CFG
        if play:
            cfg.scene.terrain.terrain_generator.curriculum = False
            cfg.scene.terrain.terrain_generator.num_cols = 5
            cfg.scene.terrain.terrain_generator.num_rows = 5

    # ── Curriculum ────────────────────────────────────────────────────────────
    if not rough:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # Head pose command range curriculum — same per-joint widening as the
    # velocity/standup envs (5% → 100% of each joint's reachable delta).
    cfg.curriculum["head_pose_range"] = CurriculumTermCfg(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "head_pose",
            "range_stages": [
                {"step": 0,         "ranges": ((-0.05, 0.05),  (-0.05, 0.05),  (-0.07, 0.07),  (-0.015, 0.015))},
                {"step": 500 * 24,  "ranges": ((-0.17, 0.17),  (-0.17, 0.17),  (-0.21, 0.21),  (-0.047, 0.047))},
                {"step": 1000 * 24, "ranges": ((-0.39, 0.39),  (-0.39, 0.39),  (-0.49, 0.49),  (-0.11, 0.11))},
                {"step": 1500 * 24, "ranges": ((-0.72, 0.72),  (-0.72, 0.72),  (-0.91, 0.91),  (-0.20, 0.20))},
                {"step": 2000 * 24, "ranges": ((-1.10, 1.10),  (-1.10, 1.10),  (-1.40, 1.40),  (-0.31, 0.31))},
            ],
        },
    )

    # CoM-randomization range curricula — match velocity (trunk capped at ±15 mm,
    # head at ±10 mm, per the 2026-07 audit).
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                    {"step": 1500 * 24, "range": 0.015},
                ],
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                ],
            },
        )

    # Push curriculum — delayed significantly (sit env lesson): a push
    # mid-transition tips the robot into configurations it can't recover from
    # before the motions have consolidated; early pushes made the sit policy
    # unlearn sitting and converge to "just stand doing nothing".
    if ENABLE_VELOCITY_PUSHES:
        cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
            func=microduck_mdp.push_curriculum,
            params={
                "event_name": "push_robot",
                "push_stages": [
                    {"step": 0,         "velocity_range": {"x": (0.0, 0.0),    "y": (0.0, 0.0)}},
                    {"step": 1000 * 24, "velocity_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)}},
                    {"step": 1500 * 24, "velocity_range": {"x": (-0.10, 0.10), "y": (-0.10, 0.10)}},
                    {"step": 2000 * 24, "velocity_range": {"x": (-0.20, 0.20), "y": (-0.20, 0.20)}},
                    {"step": 2500 * 24, "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE}},
                ],
            },
        )

    # action_rate curriculum — velocity's exact ramp (-0.1 → -1.0 by iter 1500).
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "action_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": -0.1},
                {"step": 500 * 24,   "weight": -0.2},
                {"step": 750 * 24,   "weight": -0.4},
                {"step": 1000 * 24,  "weight": -0.6},
                {"step": 1250 * 24,  "weight": -0.8},
                {"step": 1500 * 24,  "weight": -1.0},
            ],
        },
    )

    # Descent-speed cap tightening: discover the sit under magnitude 10
    # (crash-sit already net-negative), then tighten to 20. POSITIVE weights —
    # the function is self-negating (see the sign-convention warning at the
    # reward definitions).
    cfg.curriculum["descent_speed_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "descent_speed",
            "weight_stages": [
                {"step": 0,          "weight": 10.0},
                {"step": 500 * 24,   "weight": 20.0},
            ],
        },
    )

    # Rise-speed cap — introduced only AFTER the rise motion exists (the
    # standup attempt-tax lesson: any motion-tax during discovery makes
    # exploratory attempts net-negative and the skill is never found).
    # Pushed 750/1250 → 1500/2500: the rise needs a brief dynamic burst to
    # rock over the heels (vz > 0.08 for a few steps), and the first
    # sign-fixed run stalled in a head-down forward fold — a half-finished
    # rise — consistent with the cap taxing the final weight shift while it
    # was still being consolidated. Sit-direction gentleness doesn't depend
    # on this cap (descent_speed covers it), so late is cheap. If the rise
    # degrades when this kicks in, soften the final stage — never earlier.
    cfg.curriculum["rise_speed_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "rise_speed",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 1500 * 24,  "weight": 5.0},
                {"step": 2500 * 24,  "weight": 10.0},
            ],
        },
    )

    # Seated fall-back hardening ramps — after the sit/rise exist (the
    # sitstand recipe has both by ~1250) and alongside the head range
    # reaching its full width (2000). Tip kicks ramp from "mostly catchable"
    # (≤ 3 rad/s) to "flops onto the back" (4 rad/s) so the policy learns
    # the catch first, then the recovery. If a seated metric steps DOWN at
    # these boundaries, stretch them — never earlier.
    if ENABLE_SEATED_FALLBACK_HARDENING:
        cfg.curriculum["seated_head_jerk_prob"] = CurriculumTermCfg(
            func=microduck_mdp.event_param_curriculum,
            params={
                "event_name": "seated_head_jerk",
                "param_stages": [
                    {"step": 0,         "params": {"prob": 0.0}},
                    {"step": 1500 * 24, "params": {"prob": 0.2}},
                    {"step": 2000 * 24, "params": {"prob": 0.4}},
                ],
            },
        )
        cfg.curriculum["seated_tip_range"] = CurriculumTermCfg(
            func=microduck_mdp.event_param_curriculum,
            params={
                "event_name": "seated_tip",
                "param_stages": [
                    {"step": 0,         "params": {"ang_vel_range": (0.0, 0.0)}},
                    {"step": 1500 * 24, "params": {"ang_vel_range": (1.0, 2.5)}},
                    {"step": 2000 * 24, "params": {"ang_vel_range": (1.5, 3.25)}},
                    {"step": 2500 * 24, "params": {"ang_vel_range": (2.0, 4.0)}},
                ],
            },
        )

    # Torque-rate anti-jitter — phased in once both transition motions exist.
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 750 * 24,   "weight": -5e-4},
                {"step": 1250 * 24,  "weight": -1e-3},
            ],
        },
    )

    # head_pose_bias weight — same discovery-vs-refinement timing as standup
    # (3k-4k), and scaled by the HD-1910 head-mass ratio (see the reward above):
    # standup 0.5/1.5 * 1.032 = 0.52/1.55. If the standing head is STILL down
    # after a run, raise the last stage — do NOT move the introduction earlier.
    cfg.curriculum["head_pose_bias_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "head_pose_bias",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 3000 * 24,  "weight": 0.52},
                {"step": 4000 * 24,  "weight": 1.55},
            ],
        },
    )

    # Head-tripod fix (2026-09-16) — both terms ramp 0 -> target once the rise is
    # long since discovered (it appears at 2000-4000 iters; see the queue doc §8).
    cfg.curriculum["head_ground_weight"] = CurriculumTermCfg(
        func=microduck_mdp.staged_reward_weight_ramp,
        params={
            "ramp_steps": HEAD_FIX_RAMP_ITERS * 24,
            "stages": [
                {"reward_name": "head_ground", "weight": HEAD_GROUND_WEIGHT},
                {"reward_name": "neck_limit_proximity",
                 "weight": NECK_LIMIT_PROXIMITY_WEIGHT},
            ],
        },
    )

    # 头部 L1 权重课程：0 → 目标，300 迭代（与其它尝试税同样的"摸索期不上税"）
    if ENABLE_HEAD_L1:
        cfg.curriculum["head_l1_weight"] = CurriculumTermCfg(
            func=microduck_mdp.staged_reward_weight_ramp,
            params={
                "ramp_steps": HEAD_FIX_RAMP_ITERS * 24,
                "stages": [{"reward_name": "head_pose_l1", "weight": HEAD_L1_WEIGHT}],
            },
        )

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckSitStandRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # matches velocity; normalizer MUST be baked into ONNX by export.py
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="microduck_sitstand",
    run_name="microduck_sitstand",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=15_000,
)


def configure_openmd_standing_support(cfg: ManagerBasedRlEnvCfg) -> None:
    """Account for standing on the OpenMD head shell and loaded neck joints."""
    head_ground = ContactSensorCfg(
        name="head_ground_contact",
        primary=ContactMatch(mode="body", pattern="bottom_head_shell", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )
    cfg.scene.sensors = tuple(
        sensor for sensor in cfg.scene.sensors if sensor.name != head_ground.name
    ) + (head_ground,)
    cfg.rewards["head_ground"] = RewardTermCfg(
        func=microduck_mdp.posture_head_ground_penalty,
        weight=0.0,
        params={
            "sensor_name": head_ground.name, "command_name": "twist", "ramp_eps": 0.02,
            "force_threshold_n": 0.5, "use_normal_force": True,
        },
    )
    cfg.rewards["neck_limit_proximity"] = RewardTermCfg(
        func=microduck_mdp.joint_limit_proximity_standing,
        weight=0.0,
        params={
            "command_name": "twist", "ramp_eps": 0.02, "margin": 0.2,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=("neck_pitch", "head_pitch", "head_yaw", "head_roll")
            ),
        },
    )
    cfg.curriculum["head_ground_weight"] = CurriculumTermCfg(
        func=microduck_mdp.staged_reward_weight_ramp,
        params={
            "ramp_steps": 300 * 24,
            "stages": [
                {"reward_name": "head_ground", "weight": 1.4},
                {"reward_name": "neck_limit_proximity", "weight": -1.0},
            ],
        },
    )
