"""Microduck velocity (walking) environment.

The main locomotion task: velocity-command tracking + head-pose commands.
The reward/regularization recipe is locomotion-focused (lean tracking +
gait/feet terms, curriculum-ramped action-rate smoothing), with:

  - foot_slip kept at -0.1 (deliberately weak — stronger was too restrictive
    for this robot's pivot-heavy turning)
  - fixed, modest command ranges (ang ±1.0 makes turning learnable) instead of
    a widening curriculum that outpaced the robot's capability
  - turn-in-place: 15% of envs get lin=0 + |ang| ∈ [0.4, 1.0] (2026-07 audit:
    independent uniform sampling makes spin-on-the-spot ~2% of data → untrained)
  - head_pose_tracking as a primary objective, plus an EMA-based head_pose_bias
    penalty that prices only the escapable DC head droop (see below)
  - body_pose tracking infra kept intact but DISABLED (weight 0) so the obs

================================================================================
OFFICIAL ALIGNMENT (2026-09-17) — read this before changing a default
================================================================================
The DEFAULT recipe below is now **functionally identical to the official
upstream recipe** (``reference_official_velocity_env_cfg.py``, pollen-robotics
origin/develop @ 3412ae5) with exactly two exceptions:

  1. **BAM actuator + servo parameters** — deliberately ours (HD-1910-C001,
     kp_fw=32, 7.5 V rail, m5 params, lag 4..7). They live in
     ``robot/microduck_constants.py`` and are NOT part of this file.
  2. **The head-stability block** (``ENABLE_HEAD_STABILITY``, default True) —
     two terms the user asked for on 2026-09-17: a head-wobble tax and a
     one-sided head-droop tax. Set ``MICRODUCK_HEAD_STABILITY=0`` to get the
     byte-exact official recipe. ``tests/test_official_alignment.py`` proves
     the equality with that switch off.

WHY THE FLIP: **six switches covering twelve deviations were ON by default** —
stance hygiene (7 reward terms), the pose reducer (`per_joint_posture`), the
air-time progress gate, `heading_hold`, the widened `hip_roll` pose std (0.12),
and the pose zero-command bucket (0.1). Each was added to patch a symptom of a
previous policy, and every one of them was validated by comparing
INDEPENDENTLY TRAINED runs whose within-arm seed spread exceeded the effect
(AGENTS.md: "4 arms x 3 seeds could not separate any arm"). Several of the
newer ones made things worse, not better — the ``hip_roll`` L1 arms drove
zero-command survival 100 % -> 0 % (``logs/axis/THREE_AXIS_SUMMARY.txt``) and
the newest such arm logged ``Episode_Termination/fell_over`` 25 -> 69 in its
first iterations (``logs/w5_train.log``).

Every deviation is still one environment variable away, so the A/B is not
lost — only the DEFAULT is. The switches and their official values:

  MICRODUCK_STANCE_HYGIENE=1    adds stance_hip_yaw / action_over_limit /
                                neck_motion / standing_leg_vel /
                                stillness_zero_cmd / no_stepping /
                                yaw_hold_zero_cmd            (official: absent)
  MICRODUCK_PER_JOINT_POSTURE=1 pose = mean_j(exp(-err^2/std^2)) instead of
                                mjlab's exp(-mean_j(err^2/std^2))
  MICRODUCK_AIR_TIME_GATE=1     air_time gated by actual progress (anti-march)
  MICRODUCK_HEADING_HOLD=1      yaw-angle hold reward
  MICRODUCK_HIP_ROLL_STD=0.12   the widened hip_roll pose tolerance
  MICRODUCK_POSE_ZERO_CMD_PROB  explicit all-zero head/body command bucket
  MICRODUCK_ABDUCTION=1         leg-abduction task-level tax
  MICRODUCK_YAW_DC=1            net-yaw-rate tax
  MICRODUCK_TURN_FIRST=1        turn-only command curriculum
  MICRODUCK_SYMMETRY=1          mirror loss
    slot stays alive for envs that use it
"""

# ============================================================
# 【中文导读】本文件是 Microduck 的"行走主任务"(velocity)环境配置,
# 也是所有其他任务环境的共享底座 —— 机器人模型、域随机化(DR)、
# 观测噪声、动作延迟、命令槽都从这里继承,其他环境要么依赖它、
# 要么镜像它。
#
# 本文件内部结构(建议按此顺序阅读):
#   1. 顶部常量区 (L22–87)   —— ENABLE_* 开关 + 所有 DR/命令的取值,
#                                 相当于"配方表"。先看这里再看代码。
#   2. MICRODUCK_ROUGH_TERRAINS_CFG (L125+) —— 粗糙地形生成配置
#     (Microduck 脚只能抬 1–2 cm,地形被压得很温和)。
#   3. make_microduck_velocity_env_cfg() 工厂函数 (L193+) —— 真正的
#     组装逻辑,按 场景→动作→奖励→事件→观测→命令→课程 的顺序搭建。
#   4. MicroduckRlCfg (L912+) —— PPO 训练参数(网络结构、学习率等)。
#
# 贯穿全文件的核心约定(详见 AGENTS.md):
#   - 观测合同是 61 维,全策略家族共享,保证运行时可以热切换策略;
#   - 被动关节统一命名 passive_*,所有选择器用 ^(?!passive_).* 排除;
#   - 致动器是 BAM(电压控制 + 摩擦),不是理想 PD;
#   - 域随机化只能按环境在重置时重采样,绝不允许跨重置累积。
# ============================================================

import math
import os as _os


def _env_bool(name: str, default: bool) -> bool:
    """EXPERIMENT HOOK (2026-09-14 night queue).

    Defaults reproduce the committed behaviour exactly; the env vars exist so one
    checkout can run several A/B arms without editing source between runs. Every
    run records its RESOLVED cfg in params/env.yaml, so results stay reproducible.
    """
    _v = _os.environ.get(name)
    return default if _v is None or _v == "" else _v.strip().lower() in (
        "1", "true", "yes", "on")

from copy import deepcopy

# 每个环境的步数 = 每迭代的 env steps 数(PPO roll-out 长度)。
# 注意:本仓库所有课程(Curriculum)的 step 步长都是"环境步",不是迭代数,
# 所以换算公式是 iteration × NUM_STEPS_PER_ENV(即 iteration × 24)。
NUM_STEPS_PER_ENV = 24

# Fraction of envs commanded to spin on the spot (lin=0, |ang| ∈ [0.4·max, max]).
# 原地转身(原地旋转)的环境比例:15% 的环境获得 lin=0(直线速度命令为零)
# + |ang| ∈ [0.4, 1.0](角速度命令较大)的命令。
# 【为什么需要单独一档】2026-07 审计发现:如果只是独立均匀采样,
# "原地转圈"只占约 2% 的经验数据 → 这个技能永远训练不出来。
# 显式开一档才能保证原地转有足够的训练样本。
TURN_IN_PLACE_FRACTION = 0.15

# 对称性(左右镜像,参考 symmetry.py 的 61D 表)。
# 默认关闭 —— 镜像损失只适用于对称任务(如匀速行走),对非对称任务禁止开。
#
# 【2026-09-16 为什么现在要开】64 env × 5 seed 的直行矩阵(scripts/q1_straightness_matrix.sh)
# 把"走歪"的机制定位到了**步态锁相的偏航不对称**:
#   我们的策略  yaw DC = -12.1 deg/s,且与左/右脚单支撑相位锁定的分量 AC = -26.3 deg/s
#              (即左半步与右半步的偏航冲量差 ≈ 26 deg/s),转弯半径 3.2 m ⇒ 绕圈;
#   官方策略    yaw DC = -0.23 deg/s,转弯半径 ~12 m ⇒ 直走。
# 注意:两份策略的 hip_yaw 都**不在**限位上(0.0% 时间)、双脚对扭转角 <1°
# ⇒ 5V 时代那个"hip_yaw 顶死 + 整体扭转站姿"的旧缺陷在当前策略里已经不存在,
# 剩下的就是纯粹的左右步态不对称。而奖励栈里**没有任何一项**为左右对称付钱:
#   * heading_hold 是角度项(治"朝哪走"),不是对称项;
#   * gait_symmetry_penalty 只罚**累计摆动时间**的失衡(见 mdp.py),对"两个半步
#     偏航冲量不等"几乎不敏感;
#   * ENABLE_SYMMETRY(镜像损失)从未开启过。
# 所以这一项是唯一直接作用于"左右不对称"的机制。开启前必须先通过
# tests/test_symmetry_table.py(61 维置换/符号表的逐项证明,含自逆、HOME 自镜像、
# 与实际模型关节顺序一致)——表错一项就是整轮训练的静默错误梯度。
# 【开关】MICRODUCK_SYMMETRY=0/1 显式覆盖,便于同一 checkout 跑 A/B;
# 每个 run 目录的 git diff / params/env.yaml 仍会记录当次状态(本仓库溯源惯例)。
ENABLE_SYMMETRY = _env_bool("MICRODUCK_SYMMETRY", False)

# -------------------- 姿态卫生 / 执行器卫生(2026-09-11) --------------------
# 【为什么需要这一组】2026-09-11 的步态取证(run 2026-09-10_18-43-53)发现
# 走路策略的"奇怪姿势"不是训练不足,而是**奖励里根本没有给这些自由度定价**:
#
#   1. hip_yaw 被顶死在机械限位上:走路时 48% 的步子在限位处
#      (左 +30.4° / 限位 +30°,右 +25.4° / 限位 +25°),站在原地时 100%;
#      原始网络输出 |a| 高达 10.14(正常应 ≈1)。后果是双脚整体扭转约 28°
#      (外八字),站姿变成"下蹲 + 前后劈叉 + 双脚内收 32 mm"的怪姿势。
#   2. neck_pitch 被策略**主动**驱动成 ±15.7° 的 2.15 Hz 正弦(等于步频),
#      头相对躯干点头 ±22.6°(峰峰值 76°)。注意:是**指令目标**在摆,
#      关节只是以 66°(86 ms)滞后跟随 —— 所以这是策略的选择,不是舵机
#      拉不动,也不是重力下垂。消融实验(把脖子动作冻结在 HOME)能彻底消除
#      头摆,但躯干倾角从 2.6° 涨到 5.9° —— 说明它在拿 289 g 的头当
#      **配重**(占整机 37%)。
#
# 【为什么奖励没管住】mjlab 的 pose 项是 exp(-mean_j(err_j²/std_j²)),
# 走路时脚踝一项就 ~10.9 → 整项饱和在 0.0154/1.0(wandb 实测),
# 占总奖励(≈106/步)的 **0.014%**。hip_yaw 偏 16° 摊到 10 个关节里
# 只值 0.0087,再进 exp 基本等于 0。头摆的代价同样只有 ~0.17/步(0.16%)。
# 两个缺陷都是"几乎免费"的,所以策略理性地选了它们。
#
# 【为什么用这几个函数】同仓库的 velocity_rollers 环境早就写过同一类修法
# (见 microduck_velocity_rollers_env_cfg.py 的 action_over_limit /
# hip_roll_neutral),而且 mdp.py 里的 joint_deviation_l1 /
# joint_pos_limit_proximity / neck_joint_vel_l2 文档里**点名**了 hip_yaw /
# hip_roll —— 只是从没接进走路环境。这里把它们接上,不新造轮子。
#
# ⚠️ 注意 rollers 环境的记录:"a qpos penalty stole that range and broke the gait"
# —— qpos 侧限位惩罚会偷走关节可用行程。所以限位问题用**动作侧**惩罚
# (action_over_limit_penalty),它只罚"命令远超限位",不罚"关节到达限位"。
ENABLE_STANCE_HYGIENE = _env_bool("MICRODUCK_STANCE_HYGIENE", False)
# ↑ DEFAULT IS NOW False = OFFICIAL. See the OFFICIAL ALIGNMENT block in the
#   module docstring. Turning it back on restores the 2026-09-11 stance /
#   actuator-hygiene taxes, which is what every run through 2026-09-17 used.
#
# ── 只带"零命令三件套"的窄开关（2026-09-21 的统一模型用）────────────────────
# MICRODUCK_ZERO_CMD_TERMS=1 只注册 stillness_zero_cmd + no_stepping +
# yaw_hold_zero_cmd（外加给它们爬坡的 zero_cmd_tax_weight 课程），**不带** stance
# hygiene 里的姿态税（stance_hip_yaw / action_over_limit / neck_motion /
# standing_leg_vel）。
#
# 为什么需要它（实测）：官方配方 + 头块 + flight 那一族会走也会站（60/64、站立
# 0 resets、倾角 5.6°），但**站着时脚以 2.15 Hz 反复抬起 14.7 mm** —— 踏步。同速度
# 对照（都走 ~0.115 m/s）：12 偏差 + flight 的 ours_flight_s42_r2 站立离地只有
# 0.34 mm，差别就在这三项。原因很直接：零命令下 tracking 奖励只要求"不移动"，
# 而**原地抬脚并不违反它**，所以踏步在官方配方里完全没有价格。
# 这三项的 docstring 原文就是 "Discourages marching in place when the robot should
# stand still"，而且都按 `cmd_speed < 0.01` 门控 —— 走路时恒为 0，
# 对步态零影响（这是构造上的保证，不是经验判断）。
ENABLE_ZERO_CMD_TERMS = _env_bool("MICRODUCK_ZERO_CMD_TERMS", False)

# hip_yaw 回中(L1,梯度恒定)。L1 而不是高斯:高斯在 16° 处已经很平,
# 而 L1 在任何偏差处都给同样的力。rollers 环境的调参经验:
# "raise if still splayed, lower if it flattens the stride"。
#
# 【deadband = 0.10 rad(5.7°)的理由】2026-09-11 的原始动作取证:
# hip_yaw 的**指令**是 +70°(左)/+35°(右),远超 +30°/+25° 的硬限位;
# 而 hip_roll 的**指令**只有 +3.75°/−3.31°(外张),关节误差 <1°。
# 所以 hip_yaw 是"故意摆出来的姿势"(绕垂直轴的重力矩恰好 0.000 N·m,
# 不可能是重力下垂),可以放心收;hip_roll 那 3.5° 是策略有意为之但
# 幅度小,而且该关节在负载下的可辨识静态下垂就有 ~12° —— 惩罚前 12°
# 等于惩罚"逃不掉的部分",所以 hip_roll **不加 L1**,只把它的 pose std
# 从 0.05 放宽到 0.12(别再要求做不到的事)。

# ── 走路外八字（hip_roll 外张）的两个可切换修法（2026-09-16） ──────────────
# 实测（scripts/compare_walk_angles.py，vx=0.2，seed 0，12 s，均值）：
#   官方 alpha_walking     L-R hip_roll 偏 HOME = 0.054 rad（约 3°）
#   我们 ck7000 / ck12000  L-R = 0.456 / 0.477 rad（约 26–27°），L+R ≈ 0
#   ⇒ 是**镜像对称的"掰开"**，不是整体倾斜。
# 根因（多 agent 审计，带 file:line）：唯一给 hip_roll 定价的项是 pose 高斯，
# 而它的 std 在 2026-09-16 从 0.05 放宽到 0.12；**官方配方用的是 0.05**，原注释
# 就是 "hold the 5-degree-inward stance (sole sits flat), stop leg splay"
# (reference_official_velocity_env_cfg.py:225,234)。其余正则项对**静态偏置**全部
# 失明：action_rate/body_ang_vel 是速率项（常数偏置代价恰为 0）、
# dof_pos_limits 的软限位在 ±0.346 rad 之外（观测 |q| = 0.154）、
# action_over_limit 要 |action| > 0.77（观测 0.24）。而 stance_hip_yaw 的关节
# 选择器只匹配 hip_yaw，**看不见 hip_roll**。
#
# 两个开关彼此独立，便于归因：
#   MICRODUCK_HIP_ROLL_STD=0.05   把 pose 的 hip_roll std 调回官方值
#   MICRODUCK_STANCE_HIP_ROLL=1   额外挂一条 hip_roll 的 L1 回中（rollers 环境
#                                 已用同一形状的项治过同一个病，权重 -2.0）
HIP_ROLL_POSE_STD = float(_os.environ.get("MICRODUCK_HIP_ROLL_STD", "0.05"))
# ↑ DEFAULT IS NOW 0.05 = OFFICIAL (was 0.12 from 2026-09-11). The 0.12 value
#   was justified by a "12 deg static sag under load" that was later shown to be
#   (a) calibrated on the RETIRED 5 V rail and (b) pointing INWARD while the
#   observed splay points OUTWARD (docs/forensics/agent_A_reward_audit.md:88-90)
#   — i.e. it was a widened target for a defect it did not explain, and
#   tightening it back was never re-tested in a controlled way.
ENABLE_STANCE_HIP_ROLL = _env_bool("MICRODUCK_STANCE_HIP_ROLL", False)
STANCE_HIP_YAW_L1_WEIGHT = -0.8
# The original 0.10 rad deadband was chosen for hip_yaw, whose historical
# command overshoot was large.  It is too generous for the hip-roll residual:
# the R3 L1 arm left one leg at -0.169 rad while the other sat inside this
# deadband.  Keep the legacy default byte-for-byte equivalent, but allow the
# focused R4 arm to test a smaller *hip-roll-only* tolerance without changing
# the hip-yaw hygiene recipe.
STANCE_HIP_ROLL_L1_DEADBAND = float(
    _os.environ.get("MICRODUCK_STANCE_HIP_ROLL_DEADBAND", "0.10")
)
# 权重单独可覆盖。默认与 stance_hip_yaw 相同（-0.8）；想"加大惩罚"就调这个，
# 而不要再去动 pose 的 std —— 后者已被实测证伪（外八 0.454 -> 0.583，更糟）。
STANCE_HIP_ROLL_L1_WEIGHT = float(
    _os.environ.get("MICRODUCK_STANCE_HIP_ROLL_WEIGHT", str(STANCE_HIP_YAW_L1_WEIGHT))
)

# ── 腿外展 abduction（任务层面，但**对称量**）—— 修"外八"的正确形状 ──────────
# 【为什么是这个量，而不是 hip_roll、也不是脚间距】
# 2026-09-17 用仓库自己的验收工具（`scripts/check_posture_acceptance.py`）
# 对 ck7000 逐项判定，结果 15/17 通过，**只挂两条**：
#     falls in 20 s            FAIL
#     abduction [deg]  12.2    FAIL（官方 0.4，门槛 3.0）
# 而 `foot sep (walk)` / `foot sep (stand)` / `toe_out` / `hip_yaw` 全部 PASS。
#   ⇒ **外八的客观指纹是 abduction，不是脚间距。**
# `gait_forensics.py:565-568` 的定义：
#     abduction_deg = 0.5 * (left_hip_roll_dev - right_hip_roll_dev)
#   = 两条腿**对称地一起外展**的量（正值=向外）。它对"整体倾斜"免疫
#     （整体倾斜时两项同号、相减抵消），只保留"两条腿一起张开"。
# 【为什么 L1 在这个量上安全，而 L1 在单条 hip_roll 上会毁掉站立】
# 实测证伪过两次：单关节 hip_roll 的 L1（deadband 0.10 / 0.03）把 abduction
# 压下去了，但**策略从膝（±0.43）和踝（±0.58）逃逸、塌成深蹲**，零命令存活
# 100% → 0%、tilt 2° → 47°。
# abduction 是**两条腿的对称组合**：要把它压到 0，只能让两条 hip_roll 同时回到
# 标称 —— 单靠一条腿摆姿势无法抵消（因为减号在中间）。同时它**不比 hip_roll 更
# 好逃逸**：膝/踝内收同样会改变 foot 位置，但 abduction 只看 hip_roll，所以
# "深蹲"这条路对 abduction 是**无效的**（它不会降低 abduction），因此梯度会
# 指向真正的腿构型，而不是把整机压扁。
# 【门控】只在非零指令（走路）时收费。零命令站立时 ck7000 本来就略内收
#   （abduction 接近 0），而站立是唯一还完好的行为，不该被收税。
ENABLE_ABDUCTION = _env_bool("MICRODUCK_ABDUCTION", False)

# ── 站距上限（2026-09-18，agent D 的处方②）───────────────────────────────────
# 【要修的是什么】实测站距（两个独立工具方向一致）：
#     官方 65 mm（`logs/queue_align/*_accept/walk.json`）/ 84 mm（agent D 探针）
#     我们 ck7000 130 / 122 mm，align_fwd 131 / 133 mm
#   ⇒ **我们的脚间距是官方的 1.6–2 倍**，这正是用户反复说的"腿像两侧迈开"。
# 【为什么不用仓库里已有的 `feet_distance_penalty`】它罚的是 `clamp(min_dist − d)`
#   ＝**脚太近**，方向是反的；而且它自己的 docstring 就写着在我们工作点上恒为 0
#   （双脚 133 mm ≫ 78 mm 下限）⇒ 接上去**一点梯度都没有**。所以另写一个 max 版本。
# 【为什么不是关节惩罚】单关节 hip_roll 的两次尝试都已证伪（收 pose std 更糟；
#   L1 把外八压下去但站立存活 100%→0%）。**脚的位置是任务层面的量**：任何关节
#   重排都改变不了"两脚相距 130 mm"这个事实 ⇒ 绕不过去。
# 【默认关】要开必须显式打开；打开后按非零指令门控（站立不收费）。
ENABLE_FEET_MAX_WIDTH = _env_bool("MICRODUCK_FEET_MAX_WIDTH", False)
# ── 腾空相惩罚（2026-09-18，路线 A：对齐官方的"走"）─────────────────────────
# 【实测依据（同一个检查点、只改指令速度，64 环境）】
#     指令 0.05/0.10 → 离地 1–2 mm、每步偏航冲量 ±0.5、净偏航 2–3°（但**根本不动**）
#     指令 0.20      → 离地 27.4 mm、冲量 **−31.5**、腾空 14%、净偏航 **114°**
#   官方：离地 12.5 mm、触地占空比 0.585、**腾空 0.000**、前进 0.038 m/s（是"走"）
# ⇒ **我们的策略没有"走"这个模式**：要么原地蹭、要么跑。跑的时候每步产生 14 倍大的
#   偏航扰动，靠对消压到 2.8%（大而随机的残差 ⇒ 偏置符号乱翻）；官方靠**不产生扰动**。
# 【为什么不是去调 air_time】已核实 `mjlab/.../rewards.py:209-238`：它奖励的是
#   **摆动时长**落在 [0.125,0.30] s，不直接奖励腾空 —— 不腾空的走（官方摆动 ~0.24 s）
#   拿到的分一样多。所以直接罚"两只脚同时离地"这个**任务层面的状态**才是对症的。
# 【为什么绕不开】腾空是接触状态，任何关节重排都藏不住它（与站距项同理；与已被证伪的
#   单关节 hip_roll 惩罚相反）。退化解"干脆不抬脚"由 foot_clearance(−2.0) 与
#   foot_swing_height(−0.25, target 0.02) 定价 —— 这正是官方所在的那个平衡点。
ENABLE_FLIGHT_PENALTY = _env_bool("MICRODUCK_FLIGHT_PENALTY", False)
# ── 偏航率单独定价（去稀释）—— 2026-09-18，agent A 的 4.2× 发现 ──────────────
# 【问题】官方那条 `track_angular_velocity` 是
#     exp(−[(cmd_z−ω_z)² + ω_x² + ω_y²]/std²)
# 把**未指令的 roll/pitch 角速度**和偏航误差放在同一个指数里。走路时 |ω_xy|≈0.87
# rad/s、std²=0.5 ⇒ 整项（连带偏航梯度）被乘上 exp(−1.52)≈**0.22**，
# 于是偏航通道实际只有权重的 ~1/4.4 的权威，一次完美转弯只值 +0.28–0.37/步。
# 【修法】把**同一个误差**单独、不加稀释地定价一次。roll/pitch 的阻尼本来就是
# `body_ang_vel` 的职责（它构造上排除 z），所以拆开不丢任何东西 ——
# 这是给已有目标重新配权，不是新增税。
ENABLE_YAW_Z_TRACK = _env_bool("MICRODUCK_YAW_Z_TRACK", False)
YAW_Z_TRACK_WEIGHT = float(_os.environ.get("MICRODUCK_YAW_Z_TRACK_WEIGHT", "2.0"))
YAW_Z_TRACK_STD = float(_os.environ.get("MICRODUCK_YAW_Z_TRACK_STD", str(math.sqrt(0.5))))

# 【符号：**正**权重】`flight_penalty` 返回 ≤0（自我取负），配负权重会双重取负
# ⇒ 变成"越爱飞越奖励"（AGENTS.md 记录过同类事故）。判据：日志里必须 ≤0。
# 强度标定：ck7000 在 0.2 指令下腾空 14% ⇒ 权重 +2.0 时约 −0.28/步（总奖励 ~3.4）。
FLIGHT_PENALTY_WEIGHT = float(_os.environ.get("MICRODUCK_FLIGHT_WEIGHT", "2.0"))
FLIGHT_GRACE_STEPS = int(_os.environ.get("MICRODUCK_FLIGHT_GRACE_STEPS", "0"))

# 0.100 m 起罚：官方 65–84 mm 在死区内，我们 122–133 mm 立即计费
# （(0.130−0.100)=0.030 m/步 × 权重；权重 3.0 ⇒ −0.09/步，量级与其它税项相当）。
FEET_MAX_WIDTH = float(_os.environ.get("MICRODUCK_FEET_MAX_WIDTH_M", "0.100"))
# 【符号：必须是**正**权重】`feet_distance_max_penalty` 返回 ≤0（自我取负），
# 配负权重会双重取负 ⇒ **变成"脚张得越开奖励越多"**（AGENTS.md 记录过同类事故：
# 权重 −2.0 时 `Episode_Reward/abduction` 读到 +0.0042）。判据：日志里必须 ≤0。
FEET_MAX_WIDTH_WEIGHT = float(_os.environ.get("MICRODUCK_FEET_MAX_WIDTH_WEIGHT", "3.0"))
# 3° 起罚：与验收表 `abduction le 3.0` 同口径；官方 0.4°、ck7000 12.2°
# ⇒ 现交付件立即计费（约 (12.2-3)/57.3 = 0.16 rad/步），梯度清晰可见。
ABDUCTION_DEADBAND = float(_os.environ.get("MICRODUCK_ABDUCTION_DEADBAND", "0.052"))
# **正权重**：`leg_abduction_penalty` 返回 <= 0（自我取负风格，与 stance_hip_yaw 同类），
# 所以必须配正权重。给负权重会双重取负 ⇒ **变成"外展越多奖励越多"**
# （实测：权重 -2.0 时 `Episode_Reward/abduction` 读到 **+0.0042**，正是 AGENTS.md
#  那条"负权重配自我取负的惩罚 = 奖励违规"的翻版。判据：惩罚项必须 <= 0）。
ABDUCTION_WEIGHT = float(_os.environ.get("MICRODUCK_ABDUCTION_WEIGHT", "2.0"))
STANCE_HIP_YAW_L1_DEADBAND = 0.10

# 命令远超硬限位的惩罚(动作侧,能随 ONNX 一起部署到真机)。
# overshoot 0.3 rad 是给低 kp 舵机留的"够到限位"的余量,只罚野过头。
# 【权重 -1.0 的来历】实测 raw 值跨 rollout 变化很大(0.06 ~ 1.01 rad):
# 中位数为 0、只有 ~30% 的步非零,但尾部可以单步超限 9 rad。
# -1.0 让"偶发野过头"值 0.06~1.0/秒(≈1~13% 总奖励),既不常驻收费,
# 又能在策略真的把 hip_yaw 顶到限位外时形成实质压力。
ACTION_OVER_LIMIT_WEIGHT = -1.0
ACTION_OVER_LIMIT_OVERSHOOT = 0.3

# 脖子"抽动"惩罚:只罚超过 deadband 的关节速度,慢速摆头(执行头部姿态命令
# 需要的那部分)免费。deadband 2.0 rad/s ≈ 2.15 Hz 下 ±8.5° 的等效幅度
# —— 正常行走时躯干只俯仰 ±2°,脖子不需要这么大的摆动。
# 【权重 -0.08 的来历】deadband=2.0 时该 term 的实测 raw 值 = 7.49,
# 即当前策略抽动脖子要付 7.49×0.08 ≈ 0.60/秒 ≈ 8% 的总奖励。
# ⚠️ 参考量级:AGENTS.md 记载 run 5yay13u4 在**不可逃脱**的 0.77/步
# 头税下直接放弃了迈步。本项是**可逃脱**的(不抽动就付 0),所以 0.60
# 不会逼出"站着不动最优",但也不再是现在那 0.16% 的噪音级信号。
NECK_MOTION_WEIGHT = -0.08
NECK_MOTION_DEADBAND_RAD_S = 2.0

# 这几个税项在 STANCE_TAX_RAMP_ITERS 迭代内线性爬坡(AGENTS.md:技能还在
# 摸索时上税会让"什么都不做"变成最优解)。300 迭代时步态已经出现。
STANCE_TAX_RAMP_ITERS = 300

# 【最重要的一个开关】把 pose 项从"集体高斯"换成"逐关节高斯均值"。
#
#   原式(mjlab variable_posture): exp(-mean_j(err_j²/std_j²))   ← 会整体塌掉
#   新式(mdp.per_joint_posture)  : mean_j(exp(-err_j²/std_j²))  ← 不会塌掉
#
# 【为什么要换 —— 2026-09-11 Agent B 的对照实验,同一套官方配方】
#   official alpha_walking  : 站姿贴着 HOME          → pose 项 = 0.946
#   本地 walk_hd1910_15000  : 深蹲(躯干 91 mm vs 官方 120 mm)→ pose 项 = 0.015
#                                                         (站着时 0.0000)
# 集体式把每个关节的误差先除以关节数再取 exp:走路时脚踝必然大幅摆动(实测
# 偏离 HOME +44°/+48°),它一个人就把整项按到 0 —— 于是 hip_yaw / hip_roll
# 的梯度也跟着一起死了。0.015 意味着"姿态"这一项在奖励里等于不存在,
# 而 hip_yaw 就是在这里失守的(指令 +70° vs 限位 +30°,48% 的步子在限位上)。
# 官方策略没掉进这个陷阱,所以对它来说集体式够用 —— 问题是一个**正反馈陷阱**,
# 不是公式本身错。
#
# 逐关节均值让每个关节自己那一份梯度永远活着:本地站姿下光 hip_yaw 一项就还有
# exp(-(0.28/0.3)²)=0.42,而不是被 1/10 稀释成 0。最优解不变(同样的 std,
# 同样是"处处贴 HOME"),不引入新盆地,只是让信号不再死掉。
#
# 设 False 可一键回到官方原式做 A/B。
ENABLE_PER_JOINT_POSTURE = _env_bool("MICRODUCK_PER_JOINT_POSTURE", False)
# ↑ DEFAULT IS NOW False = OFFICIAL (mjlab's collective Gaussian). The swap to
#   the per-joint mean was made to stop the collective form collapsing to
#   ~0.015 when one joint (the ankle) travels far — a real effect, but it was
#   never A/B-validated, and its own known cost is that a single far-travelling
#   joint now only costs 1/N of the term (see docs/next_steps_2026-09-17.md §2.1).

# 【原地踏步的修法 —— 开关】air_time(权重 3.0,全配方最大的正奖励)只按
# **命令**门控,不按**实际速度**门控:
#     cfg.rewards["air_time"].params["command_threshold"] = 0.01
# 于是"命令非零"就够拿 3.0/步的抬脚奖励 —— **原地踏步是一条合法的刷分路径**。
# 以前没人管是因为姿态没定价(策略没理由少迈腿);2026-09-11 把 pose 修活之后
# (raw 0.011 → 0.49),"迈步但不前进"变成了一个有吸引力的中间盆地:
# 实测 iter 3500 时 air_time 0.98 而 track_linear_velocity 只有 0.90/2.0。
#
# 同仓库的 rollers 环境早就踩到并修过这个洞(原文:
# "otherwise a fast in-place flutter farmed this"),用 vel_gate_ref 把抬脚奖励
# 按实际前进速度门控。这里把那套搬到走路任务上。
#
# 0.0 = 关闭(与环境当前行为**逐字节相同**,连函数都不会被换掉)。
# >0  = 启用:按"沿命令方向的实际体速"门控;原地转命令改用实际偏航角速度门控。
# 另有 gate_floor=0.25 保底 —— 见 mdp.feet_air_time_progress_gated 的说明:
# 硬零会让"还不会走路"的策略连抬脚的动机一起丢掉(AGENTS.md 的死循环风险)。
AIR_TIME_PROGRESS_GATE_V_REF = 0.15   # 2026-09-11 @4273 判定"原地踏步"后启用
AIR_TIME_PROGRESS_GATE_TURN_REF = 0.3
# DEFAULT IS NOW OFF = OFFICIAL stock feet_air_time. The env var still means
# "1 = gate ON", so every historical launch command keeps its meaning; only the
# default moved. (See the OFFICIAL ALIGNMENT block in the module docstring.)
if not _env_bool("MICRODUCK_AIR_TIME_GATE", False):
    # stock `feet_air_time` (official behaviour)
    AIR_TIME_PROGRESS_GATE_V_REF = 0.0
    AIR_TIME_PROGRESS_GATE_TURN_REF = 0.0
AIR_TIME_PROGRESS_GATE_FLOOR = 0.25

# 【零命令站定】权重(见下面 rewards 区 stillness_zero_cmd / no_stepping 的注释)。
# stillness 是单调正奖励(站定 ≈ +0.8),no_stepping 罚脚离地(踏步 −0.4~−0.8)。
# 两者都按命令门控,**对行走步态零影响**。
STILLNESS_AT_ZERO_CMD_WEIGHT = 0.8
NO_STEPPING_WEIGHT = -0.4

# 【零命令定点航向】(2026-09-15) 站立时把航向钉在"停下那一刻"的朝向。
# 为什么必须补：heading_hold 的乘性速度门(vel_gate_ref=0.15)在站立时把整项归零（刻意防刷分），
# 于是**静止时没有任何项约束偏航** —— 实测 11999 迭代策略零命令 12 s 漂 −144°(≈12°/s)，
# 而同策略 trunk 其它指标正常(tilt 2.9°、步频 0.083 Hz)。航向是主杠杆：night_queue 12 run
# 统计 |yaw_bias| ↔ 20 s 净前进 r = −0.900 (r²=0.81)、↔ 路径效率 r = −0.781。
# 角项(exp(−err²/std²))可回纠 + 角速度项给稠密梯度；门控用整条 twist 指令 ⇒ 命令一给即返回 0，
# 对行走步态零影响。权重与零命令三件套一起由 stance_tax_weight 课程爬坡。
YAW_HOLD_AT_ZERO_CMD_WEIGHT = 0.8       # 与 stillness_zero_cmd 同量级（两者都只在零命令下付钱）
YAW_HOLD_AT_ZERO_CMD_STD = 0.6          # rad ≈ 34°
YAW_HOLD_AT_ZERO_CMD_RATE_WEIGHT = 1.0  # 每 (rad/s) 扣多少（再乘外层权重）
# 实测(scripts/probe_yaw_hold_term.py)：零命令站姿下该项非零、0.2 m/s 下严格为 0。
# 训练层面的有效性（能否把 2°/s 的慢漂压掉）**尚未经训练验证** ——
# 2026-09-15 那次 walk run 不含此项（它是"修前"基线），下一次 walk 训练才是它的验证。
#
# 【开关】MICRODUCK_YAW_HOLD_ZERO_CMD=0 关掉本项（默认开），用于做"有/无该项"的受控对照，
# 不必改代码；run 目录里的 git diff 仍会记录当次状态（本仓库的溯源惯例）。
if not _env_bool("MICRODUCK_YAW_HOLD_ZERO_CMD", True):
    YAW_HOLD_AT_ZERO_CMD_WEIGHT = 0.0

# 【零命令"踏步"的直接解药 —— 腿关节速度 L2】
# 实测(2026-09-14/15):零命令下"腿关节速度平方和"把行为分得干干净净,而且**稠密**:
#     踏步策略(3000~3500 迭代) 74.7~76.0   → 权重 -0.02 时罚 -1.49~-1.52/步
#     walk_8250(基本站定)        24.8        → -0.50/步
#     model_11999(完美站定)       0.008      → -0.0002/步(几乎免费)
# 已接线的两项都不够稠密:stillness 只看**瞬时**躯干速度(每步只在过零点附近有值),
# no_stepping 是二元脚触地事件。这一项给出与"腿动得多快"成正比的连续梯度。
# 仅由命令门控(零命令才生效)⇒ 对走路步态零影响。
STANDING_LEG_VEL_L2_WEIGHT = -0.02
# 被推时的豁免阈值(m/s):用**平滑后**的躯干速度判断。推一下(0.3 m/s)能解锁
# 恢复步,而步频级的躯干晃动(平滑后 ~0.03)解锁不了。
STANDING_LEG_VEL_PUSH_EXEMPT = 0.2

# -------------------- 直行保持 / 步态对称(2026-09-14) --------------------
# 【为什么需要这一组 —— 走歪归因的结论】
# 2026-09-14 的多 agent 取证(docs/forensics/走歪归因报告.md)定位到:
# 整个奖励栈里**没有任何一项是"偏航角"的函数**。唯一与偏航有关的是
# track_angular_velocity,它只对**角速度误差**计费,而且只占收益的 2.1%
# (iter 11999 实测 raw +0.0357/1.0 → 合成角速度误差已达 1.291 rad/s,深在高斯尾部)。
# 而 61 维观测里**没有航向、也没有机体线速度**,策略只能看到偏航"角速度"。
# 后果:一个未被压住的偏航角速度会**开环积分**成越来越大的航向误差 ——
# 实测直行命令下 20 秒净偏航 −103°±193°、路径效率 0.249(官方同条件 0.84)。
#
# 实测的缺陷形态:两个半步的偏航率是 +10.6 / −18.3 deg/s
# (半步不对称 +28.9 deg/s),而整段直流偏置只有 −4 deg/s —— **AC/DC = 16.7×**,
# 即漂移是约 6% 的**整流残差**。这种"步频锁相振荡 + 微小整流"正是
# 速率高斯看不见的工况,所以必须改用**角度**项。
#
# 【为什么用角度项而不是偏航率惩罚】rollers 环境已经试过并回退了,原话在
# microduck_velocity_rollers_env_cfg.py 里:"a yaw-RATE penalty froze the yaw
# and made drift WORSE (tried and reverted)"。角度项是**可纠正**的:
# 漂了就掉分,策略可以把 yaw 转回来补回分数。
#
# ⚠️ 本组修改**只动奖励**,不碰 61 维观测合同 ⇒ **运行时不需要改**,
# 策略仍可与官方策略热插拔。
ENABLE_HEADING_HOLD = _env_bool("MICRODUCK_HEADING_HOLD", False)
# ↑ DEFAULT IS NOW False = OFFICIAL (the official recipe has no yaw-angle term
#   at all; its straightness comes from the policy, not from a reward).
#   Still one env var away — see the OFFICIAL ALIGNMENT block in the docstring.

# 角度项权重。量级标定(参照 AGENTS.md"比较奖励质量而不是权重"):
# 训练日志里单步净收益约 +3.4,本项满分 +2.5。
#   37°(0.65 rad,当前策略的实际工作点)→ 损失 1.87/步
#   17°(0.30 rad)                      → 损失 0.88/步
# 对比现状:同样 1 rad 误差在 track_angular_velocity 下只值 0.04% —— 差数百倍。
#
# 【2026-09-14 调参依据 —— 权重和 std 是配套改的,别只改一个】
# 上一轮 run(weight 1.5 / std 0.4)在 5500 次迭代里这一项的原始值**一直卡在
# 0.07**(等效航向误差 37°),完全没改善,而同时 track_linear 从 0.51 涨到 0.93。
# 说明策略在学,只是**这一项推不动**。原因算出来是**梯度形状**问题:
#   高斯梯度 ∝ err·exp(−err²/std²),峰值在 err = std/√2 处
#   std=0.4 rad(23°)⇒ 峰值在 **16°**,而策略实际在 **37°**
#   ⇒ 在 37° 处 hold 只有 0.07,梯度只剩峰值的 **28%**
# 所以把 std 放宽到 0.55 rad(31.5°)、权重提到 2.5:
#   37° 处梯度 **0.16 → 0.48 /rad(3.0 倍)** —— 正好作用在策略所在的位置
#   8° 处梯度 0.418 → 0.390(几乎不损失,不会牺牲将来的精度)
#   该项占净收益的比例 ~4% → ~13%
HEADING_HOLD_WEIGHT = 2.5
HEADING_HOLD_STD = 0.55
# 门控阈值(rad/s):只在"命令接近直行"时计费,避免和任务的转弯指令打架。
# ang_vel_z 均匀采样 ∈ (−1,1) ⇒ 该阈值覆盖约 20% 的经验
# (再加上 rel_forward_envs 给的那 20% 直行桶 —— 见 §"命令分布")。
HEADING_HOLD_COMMAND_THRESHOLD = 0.2
# 【速度门 —— 必须要有,否则会挖出"干脆站住"的坑】
# 不加速度门时,"站着不动"完美保持航向 ⇒ 白拿满额奖励。按实测数字推演:
#   站定 ≈ track_linear 1.34 + heading_hold 1.5 = 2.84
#   歪着走 ≈ track_linear 1.63 + heading_hold ~0.2 = 1.83
# 即加了这项之后"站住"反而**比"歪着走"分高** —— 对还不会走直线的策略来说,
# 这是一个"停下就赢"的局部最优(AGENTS.md 明令禁止:正奖励不许落在可刷分的状态上)。
# 修法沿用仓库自己的先例:feet_air_time_progress_gated 就是用 vel_gate_ref
# 防"原地踏步刷分"(原文:"otherwise a fast in-place flutter farmed this")。
# 乘性门:静止 → 0,达到参考速度 → 满额。当前策略实测 vx≈0.207 > 0.15 ⇒ 门全开,
# 不会削弱它现在拿到的梯度。
HEADING_HOLD_VEL_GATE_REF = 0.15
# 【速度门的平滑时间常数与死区 —— 必须要有】
# 门必须用"平滑后的带符号速度",不能用瞬时 |v|:实测(2026-09-14)用瞬时值时,
# **原地踏步的策略把门顶开了 59.7%,而站定不动只有 40.3%** ——
# 等于付钱让策略踏步(正反馈,策略会一直踏下去)。
# 对幅值做平滑也没用(E|v| 对任何振荡都 >0),只有"先平滑带符号分量再取模"
# 才能把振荡抵消掉。加死区后实测:静止 0~2%,真走路 99.9%。
HEADING_HOLD_VEL_GATE_TAU = 1.0      # s，一阶低通时间常数
HEADING_HOLD_VEL_GATE_DEADBAND = 0.05  # m/s，静止时躯干漂移的余量
# 课程:先让步态出现,再引入本项(AGENTS.md:技能摸索期不能上新目标)。
# 用 reward_weight 的分段阶跃(它不是插值,必须离散成台阶)。
HEADING_HOLD_RAMP_ITER_START = 300
HEADING_HOLD_RAMP_ITER_END = 1300

# 左右步态对称性惩罚(**默认关闭**:本次"最优单改"先只上 heading_hold)。
# 代码与接线都留好了,作为下一步独立的 A/B:
#   打开只需把 ENABLE_GAIT_SYMMETRY 改 True,其余自动生效。
#
# 【为什么这次不开】
#   1. 它是**代理指标**:罚的是"累计摆动时间不平衡",而真正的病因是净偏航力矩。
#      heading_hold 直接给"结果"(航向被保持住)定价,策略可以自由选择
#      "把步态改对称"或"反向补偿"来拿分 —— 不必规定它怎么做。
#   2. 它**会和转弯命令打架**:转弯本来就需要不对称迈步,而本项无门控时会把
#      合法的转弯当成缺陷来罚(rollers 环境没这个问题,因为它的 heading 命令
#      是关掉的;走路环境的 |wz| 最大到 1.0,还有 15% 原地旋转桶)。
#      要开的话必须先像 heading_hold 那样加一道偏航门控。
#   3. 一次只改一件事,失败时才分得清是谁的责任。
ENABLE_GAIT_SYMMETRY = False
GAIT_SYMMETRY_WEIGHT = -0.5
GAIT_SYMMETRY_RAMP_ITER_START = 500
GAIT_SYMMETRY_RAMP_ITER_END = 1500

# -------------------- 走路净偏航率(2026-09-16) --------------------
# 【实测】64 env × 5 seed 的直行矩阵(scripts/q1_straightness_matrix.sh):
#     官方 alpha_walking   yaw DC -0.23 deg/s,步态内摆动 AC -0.66 deg/s ⇒ 转弯半径 ~12 m
#     我们 walk_hd1910_12000 yaw DC -12.08 deg/s,AC -26.28 deg/s   ⇒ 转弯半径 3.2 m
# 一个周期内航向变化 = DC × T ⇒ 恒定 -12 deg/s 就是 30 秒转一整圈,这就是"最后会转圈"。
# 5V 时代那个"hip_yaw 顶死限位 / 双脚整体扭转 28°"的旧缺陷在当前策略里**已不存在**
# (贴限位 0.0% 时间、双脚对扭转 <1°),剩下的是一个**近似恒定**的偏航率偏置。
#
# 【估计量】y = EMA_tau(wz),tau≈0.8 s ≈ 两个步态周期 ⇒ 步频摆动被平均掉(衰减约 10 倍),
# 恒定偏置几乎原样通过。实测信噪:偏置 12 deg/s vs 摆动泄漏 ~2 deg/s ⇒ 测的是偏置不是迈步。
#
# 【为什么不是 |wz|】rollers 环境试过裸的偏航率惩罚并回退("froze the yaw and made drift
# WORSE")—— |wz| 会把每一次转向修正也一起罚掉。本项的关键差别是**对零均值摆动免疫**:
# 步态需要的摆动一分钱不花,只有残余 DC 收费。且门控在**直线命令**上,转弯命令永不收费。
#
# 【与 heading_hold 的关系】heading_hold 是**角度**项(定价"最终朝哪"),允许把漂移纠回来,
# 但它作用在一个**策略观测不到**的量上(61 维观测里没有航向),在工作点附近梯度很弱,
# 而且它是"结果"不是"原因"。本项提供每步稠密、且因果直接的信号。两者互补,不是替代。
# 【开关命名】本项是"净偏航率(时间平均)"惩罚,不是镜像损失 —— 镜像损失的开关是
# ENABLE_SYMMETRY / MICRODUCK_SYMMETRY。历史上本项复用了 MICRODUCK_YAW_SYMMETRY
# 这个名字,结果 2026-09-16 排组合臂时把两者搞混(冒烟里镜像损失没出现才发现)。
# 现在首选名是 MICRODUCK_YAW_DC,旧名保留为兼容别名。
ENABLE_YAW_SYMMETRY = _env_bool(
    "MICRODUCK_YAW_DC", _env_bool("MICRODUCK_YAW_SYMMETRY", False))
YAW_SYMMETRY_WEIGHT = 3.0            # 实测 |EMA(wz)| ≈ 0.21 rad/s ⇒ 约 -0.63/步(总奖励 ~3.4)
YAW_SYMMETRY_TAU_S = 0.8             # ≈ 两个步态周期(步频 2.3 Hz)
YAW_SYMMETRY_WZ_CMD_THRESHOLD = 0.15
YAW_SYMMETRY_RAMP_ITERS = 300        # 技能摸索期不上税(AGENTS.md)

# -------------------- 转向优先课程(2026-09-16) --------------------
# 【要修的是什么】当前走路策略**不会转弯**:偏航指令 ±34 deg/s 只换来 <2% 的实际偏航率
# 变化(原地时连符号都不对),而 track_angular_velocity(权重 2.0,参数与官方逐字相同)
# 只拿到 0.119/1.0 —— 策略**白扔 23% 总奖励**也不转;官方策略同样指令能跟到 +18.5 deg/s。
# ⇒ 不是"定价不够",是**这个步态盆地做不到**:它收敛成了一个固定碎步极限环,
#   落足位置与偏航指令解耦。
#
# 【机制】按 AGENTS.md 自己的课程规则("每个阶段要和策略实际学会的东西对齐";
# "罕见但重要的命令区域要显式分桶"):给转向一个"只要求转向"的阶段,让这个技能
# 在**前进步态还没形成、无处可躲**的时候先被独立发现,再放开通前进指令。
# 前进范围留一个**小而非零**的值(±0.05 而不是 0)——AGENTS.md:从不非零的命令输入
# 会让输入神经元永久死掉。
#
# 阶段(env steps = iteration × 24):
#   0        : lin ±0.05(仅供神经元存活), ang ±1.0  ⇒ 只练转
#   1200     : lin_x ±0.20, lin_y ±0.15,     ang ±1.0  ⇒ 混入前进
#   2500     : 完整范围(见下面的命令区常量)              ⇒ 恢复正常任务
# 【开关】MICRODUCK_TURN_FIRST=1 启用(默认关,便于受控 A/B;run 目录 git diff 自证)。
ENABLE_TURN_FIRST_CURRICULUM = _env_bool("MICRODUCK_TURN_FIRST", False)
TURN_FIRST_STAGE1_ITERS = 1200
TURN_FIRST_STAGE2_ITERS = 2500
TURN_FIRST_TINY_LIN = 0.05           # 非零,保住输入神经元

# 【部署对齐】head_pose / body_pose 的"精确零命令"概率。
# 现状是 0.0 —— 连续均匀采样下 4 维同时精确为 0 的概率约 1e-4,
# 而**部署时恰恰发的就是全零**("没有姿态请求")。AGENTS.md 明确记录过这个陷阱
# ("Zero-command behavior must be explicitly trained: uniform sampling
#  essentially never produces the all-zero command, which is exactly the
#  deployment idle state")。给 10% 的显式零命令桶即可覆盖。
# 只改采样分布,不改观测形状/范围 ⇒ 不破坏 61 维合同。
POSE_ZERO_COMMAND_PROB = float(
    _os.environ.get("MICRODUCK_POSE_ZERO_CMD_PROB", "0.0")
)
# ↑ DEFAULT IS NOW 0.0 = OFFICIAL. The 0.1 bucket was added because deployment
#   sends the ALL-ZERO head/body command, which continuous uniform sampling
#   essentially never produces (AGENTS.md). It is a real argument — but it is
#   also a deviation, and it was never A/B-validated, so it is now opt-in.

# -------------------- 头部稳定性 / 头下垂(2026-09-17,用户要求) --------------------
# 【用户要求原文】"头要稳定，因为头的重量增加了，增加头乱动的惩罚，不能低头，
#   低头的惩罚"。这是**官方配方之外唯一保留的一处增项**(见模块顶部的
#   OFFICIAL ALIGNMENT 说明),用 MICRODUCK_HEAD_STABILITY=0 可以回到逐字节官方。
#
# 【头的重量:实测,不是估计】从 MJCF 直接读:
#     jaw_soft 188.77 g + yaw_roll_motion 51.60 + neck 39.84 + neck_pitch 8.72
#     = 288.93 g / 整机 779.24 g = **37.1%**;力臂 75.6 mm。
#   注意一个必须记录的事实:**"头变重了"在当前模型里只成立 +9 g(+3.2%)** ——
#   preswap(官方 XL330 时代)是 279.93 g,现在的 288.93 g 全部来自颈部链上
#   那 3 个 HD-1910 舵机各 +3 g(整机 +42 g = 14 舵机 × 3 g)。
#   头占整机的**比例其实从 38.0% 降到 37.1%**。
#   ⇒ 头"重"是**绝对量**(37% 的整机质量挂在 75 mm 力臂上)这个理由,
#     而不是"这次换舵机把头加重了"。惩罚仍然成立,但理由要写对。
ENABLE_HEAD_STABILITY = _env_bool("MICRODUCK_HEAD_STABILITY", True)

# 【头顶乱动】只罚超过 deadband 的颈部关节速度。deadband 的标定(实测):
#   * 官方 alpha_walking:颈部步频幅度 0.75° ⇒ 峰值角速度 ≈0.18 rad/s ⇒ **免费**;
#   * 我们的坏策略:16.4° @2.15 Hz ⇒ 峰值 ≈3.87 rad/s ⇒ 计费。
#   deadband=2.0 rad/s 落在两者之间,所以它是"把坏步态拉回官方水平"的税,
#   而不是"禁止头动"的税。执行 head_pose 命令所需的慢速摆头(2–5 s 重采样、
#   峰值 1.7–3.5 rad/s 的短促段)只有一部分会越界,且这是**可逃脱**的。
# 【权重 -0.08 的来历】deadband=2.0 时坏策略该项 raw ≈ 7.49 ⇒ 约 −0.60/s
#   ≈ 8% 总奖励;官方质量的步态付 0。AGENTS.md 记录的不可逃脱头税是 0.77/步
#   (run 5yay13u4 直接放弃迈步)—— 本项远低于它,且可逃脱。
# 【为什么必须爬坡】AGENTS.md:技能还在摸索时上税会让"什么都不做"成为最优解。
#   与官方 head_pose_bias 同一条时间线:600 迭代前为 0,1500 迭代到满权。
HEAD_MOTION_WEIGHT = float(_os.environ.get("MICRODUCK_HEAD_MOTION_WEIGHT", "-0.08"))
HEAD_MOTION_DEADBAND_RAD_S = float(
    _os.environ.get("MICRODUCK_HEAD_MOTION_DEADBAND", "2.0")
)

# 【不能低头】单侧、相对命令的 pitch L1(见 mdp.head_droop_penalty 的长说明)。
#   deadband=0.15 rad(8.6°):37% 质量的头在迈步中无法避免的摆动免费,
#   持续的低头(实测坏策略 −1.934 rad)照常计费。
# 【权重 2.0 的标定】坏策略脖子 −1.93 rad ⇒ 扣 (−1.93+0.15)×2.0 ≈ −3.6/步,
#   远大于它从"折脖子"里能拿到的任何好处;官方姿态(−0.13 rad,在 deadband 内)
#   付 0。注意它与官方 head_pose_bias(3.0,对称、含 yaw/roll、只罚 DC)不冲突:
#   那项罚"平均偏差",本项罚"方向",两者一起把低头变成单向代价。
HEAD_DROOP_WEIGHT = float(_os.environ.get("MICRODUCK_HEAD_DROOP_WEIGHT", "2.0"))
HEAD_DROOP_DEADBAND = float(_os.environ.get("MICRODUCK_HEAD_DROOP_DEADBAND", "0.15"))
# 【头要一直向前】yaw/roll 的 L1(双侧、相对 HOME、按命令门控)。
# 【实测基线(2026-09-17,compare_walk_angles.py,1 env × seed 0 × 12 s,均值 ± std)】
#                   走路 vx=0.2                站立 vx=0.0
#   joint        官方          我们        官方          我们
#   head_yaw   -0.014±0.045  -0.023±0.029  +0.018±0.023  -0.006±0.019
#   head_roll  -0.024±0.032  +0.060±0.015  -0.035±0.004  **+0.119±0.004**
# ⇒ 两条结论,都直接决定死区:
#   ① 走路时**两台策略的头都朝着正前方**(1–3.5°,峰值 ≈ mean+2σ:yaw 0.104、
#      roll 0.088)⇒ 死区必须 ≥0.10 才不会去税官方步态里不可避免的摆动;
#   ② **站立时我们有一个真实缺陷:head_roll 偏 +0.119 rad = 6.8°**(官方 2°,
#      std 只有 0.004 ⇒ 是稳定的静态偏斜,不是抖动)⇒ roll 死区取 0.10 就能
#      把官方(峰值 0.088)放过去、同时给我们这个 6.8° 的歪头计费。
#   AGENTS.md 的规矩:deadband 用实测的不可逃逸值,而不是拍脑袋 —— 上面这张表就是。
# 【门控】命令(±1.40 rad 的 head_yaw 指令)非零时免费 —— 否则本项会和
#   head_pose_tracking 抢同一个关节(仓库明令:同一关节不能被两个目标拉扯)。
#   部署发的是全零命令 ⇒ 门恒开 ⇒ 真机上就是"一直向前"。
#   要"连命令也不许转"(更严格的读法)就把 gate 设成 0.0。
HEAD_FORWARD_WEIGHT = float(_os.environ.get("MICRODUCK_HEAD_FORWARD_WEIGHT", "2.0"))
HEAD_FORWARD_DEADBAND_YAW = float(
    _os.environ.get("MICRODUCK_HEAD_FORWARD_DEADBAND_YAW", "0.12")
)
HEAD_FORWARD_DEADBAND_ROLL = float(
    _os.environ.get("MICRODUCK_HEAD_FORWARD_DEADBAND_ROLL", "0.10")
)
HEAD_FORWARD_COMMAND_GATE = float(
    _os.environ.get("MICRODUCK_HEAD_FORWARD_CMD_GATE", "0.05")
)

# 与官方 head_pose_bias 的时间线一致(0 直到 600 迭代,1500 到满权)。
HEAD_STABILITY_RAMP_ITER_START = 600
HEAD_STABILITY_RAMP_ITER_MID = 1000
HEAD_STABILITY_RAMP_ITER_END = 1500

# -------------------- 域随机化(DR)总开关 --------------------
# Domain randomization toggles
# 每个 ENABLE_* 控制下面工厂函数里对应的一段 cfg.events[...] 是否注册。
# 这里是"改实验配置的第一站":想在某个新实验里去掉/加上某类随机化,
# 直接改这些布尔值,不用动组装代码。
# 域随机化 = 让仿真环境和真机之间的差异(质量分布、摩擦、噪声、延迟…)
# 按环境随机变化,迫使策略学出鲁棒行为,而不是过拟合到一个精确的仿真。
ENABLE_COM_RANDOMIZATION = True
ENABLE_HEAD_COM_RANDOMIZATION = True  # Randomize CoM of the head assembly bodies
ENABLE_KP_RANDOMIZATION = False # Was True
ENABLE_KD_RANDOMIZATION = False # Was True
ENABLE_MASS_INERTIA_RANDOMIZATION = True  # Can enable once walking is stable
ENABLE_JOINT_FRICTION_RANDOMIZATION = True  # Scales BAM's friction budget per-env via FrictionDRBamActuator.friction_scale
ENABLE_JOINT_DAMPING_RANDOMIZATION = False
ENABLE_ARMATURE_RANDOMIZATION = True  # Reflected rotor inertia (microban-style). DOES affect BAM (armature is set, not zeroed).
ENABLE_VELOCITY_PUSHES = True  # Velocity-based pushes for robustness training
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True  # Simulates mounting errors
ENABLE_ENCODER_BIAS = True  # Per-env joint encoder calibration offset (actor obs sees joint_pos + bias)
ENABLE_BASE_ORIENTATION_RANDOMIZATION = False  # Randomize initial tilt to force reactive behavior

# Head/body pose command tracking (replaces the old neck-offset disturbance scheme).
# Head pose: 4D deltas-from-HOME on neck/head joints; vel env tracks these as a
# primary objective. Body pose: 6D delta in [x, y, z, roll, pitch, yaw]; vel env
# samples small ranges + tiny reward weight so input neurons stay alive but
# tracking isn't the priority (standup env raises the weight).
#
# 【命令重采样周期】命令不是常驻的,而是每隔 HEAD_POSE_CMD_RESAMPLE_S /
# BODY_POSE_CMD_RESAMPLE_S 秒重新采样一次(均匀随机),形成"维持一段时间
# 的目标",策略在两次重采样之间跟踪它。范围见下面 cfg.commands 部分。
HEAD_POSE_CMD_RESAMPLE_S = (2.0, 5.0)
BODY_POSE_CMD_RESAMPLE_S = (2.0, 5.0)

# Observation configuration
USE_PROJECTED_GRAVITY = True  # If True, use projected gravity instead of raw accelerometer
# 【解释】projected_gravity:把重力向量投影到机体坐标系(3 维),这是
# 大多数行走策略用的姿态观测(比原始加速度计更干净)。
# 选 False 则用原始加速度计(raw_accelerometer)。

# -------------------- 域随机化取值范围(配方) --------------------
# 下面是各 DR 项的数值范围。注意:这些范围是"踩过坑后收敛出来的",
# 改之前要看后面的注释和 AGENTS.md 里的教训。
# Conservative ranges proven to be stable - can increase gradually if needed
COM_RANDOMIZATION_RANGE = 0.003  # ±3mm initial, ramped to ±8mm via curriculum
# 【为什么不能无限加大】躯干质心(CoM)偏移若超过脚支撑多边形
# (脚跟距脚踝仅 20 mm),随机化的质心会落在支撑面外,强迫策略学会
# 又宽又快的过度反应步态,而且后向平衡会无法训练 —— 见下面
# com_range 课程中记录的 2026-07 审计。
# Head CoM randomization: applied per-episode to every body of the head assembly
# (neck → neck_pitch → yaw_roll_motion → head-roll body). Same non-accumulating
# mechanism as the trunk CoM randomization above. The head-roll body is named
# bottom_head_shell in the walk model and jaw_soft in the 2026-07 roller model,
# hence the alternation. NOTE: bearing_roll is NOT a head body — in both models
# it is the right-hip-yaw link (child of trunk_base); it has always been listed
# here by mistake and is kept only to preserve existing DR behavior.
# 【中文解释】头部组件(脖子→头)逐体加随机 CoM 偏移。head 体重约
# 280 g,占整机 38%,是巨大的"力臂",所以单独随机化——现实中 3D
# 打印件/电子件的质量分布不可能和 CAD 完全一致。
HEAD_COM_RANDOMIZATION_RANGE = 0.003  # ±3mm initial, ramped via curriculum
HEAD_BODY_NAMES = (
    "neck",
    "neck_pitch",
    "yaw_roll_motion",
    "(bottom_head_shell|jaw_soft)",
    "bearing_roll",
)
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)  # ±5% applied to BOTH mass and inertia together.
KP_RANDOMIZATION_RANGE = (0.85, 1.15)  # ±15%
KD_RANDOMIZATION_RANGE = (0.9, 1.1)  # ±10% (can increase to 0.8-1.2)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
JOINT_DAMPING_RANDOMIZATION_RANGE = (0.9, 1.1)
ARMATURE_RANDOMIZATION_RANGE = (0.9, 1.1)  # ±10% reflected rotor inertia (microban: dr.joint_armature, same range)
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)  # Apply pushes every 3-6 seconds
VELOCITY_PUSH_RANGE = (-0.3, 0.3)  # Velocity change range in m/s. Was ±0.5 — an
# ADDITIVE kick larger than max walk speed (0.4) every 3-6 s trains a permanently
# nervous fall-recovery gait (2026-07 audit). ±0.3 keeps push robustness while
# letting a calmer gait be optimal.
# 【中文解释】速度扰动:每 3–6 秒给机体一次随机速度变化(模拟被绊/被推)。
# 原来 ±0.5 m/s 比最大步行速度(0.4)还大,等于每几秒被撞一次,
# 训练出的步态永远处于"准备摔倒的紧张状态"。缩到 ±0.3 保留鲁棒性
# 同时允许更平静的步态成为最优解。
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0  # up-to-6° random-axis IMU mounting error. NOTE: zero-centered (random axis) — trains tolerance to misalignment *magnitude*, NOT a pitch bias. The real board's systematic ~5° pitch offset is corrected at the source in the runtime (imu-pitch-offset), not here.
# 【中文解释】IMU 安装误差随机化:随机轴、最大 6°。注意是零均值(随机轴),
# 只能训练对"误差大小"的容忍度,无法补偿系统性的俯仰偏置 ——
# 真机板子那个 ~5° 的系统偏差在运行时源码里修正(imu-pitch-offset),不是靠 DR。
ENCODER_BIAS_RANGE = (-0.015, 0.015)  # ±0.86° per-joint encoder offset (constant per env)
# 【中文解释】编码器偏置:每个环境给每个关节一个恒定的编码器校准误差
# (±0.015 rad ≈ ±0.86°)。真实舵机零位不可能标定得完全一致,策略必须
# 学会在观测有微小偏差的情况下仍然工作。只作用于 actor(策略看到的),
# critic 看到真值(特权信息)。
BASE_ORIENTATION_MAX_PITCH_DEG = 10.0  # ±10° forward/backward tilt at episode start
BASE_ORIENTATION_MAX_ROLL_DEG = 5.0  # ±5° side-to-side tilt at episode start

import mujoco as _mujoco
import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    MetricsTermCfg,
    CurriculumTermCfg,
    EventTermCfg,
    MetricsTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


# ============================================================
# 【粗糙地形配置】Microduck 专用的粗糙地形,比 mjlab 默认的
# ROUGH_TERRAINS_CFG 温和得多:这个机器人脚只能抬 1–2 cm,
# 所以台阶高度被压到 ≤1.5 cm(默认是 10 cm)。
# ============================================================
# Microduck-specific rough terrain: much gentler than the default ROUGH_TERRAINS_CFG.
# The robot can only lift its feet ~1-2 cm, so steps are capped at 1.5 cm.
MICRODUCK_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),          # 每块"补丁"地形 8m × 8m
    border_width=20.0,        # 边界宽度(留白)
    num_rows=10,              # 地形行数 = 难度等级数(课程用 terrain_levels 控制)
    num_cols=20,              # 列数(每列一种地形类型?不,是每行一个难度)
    sub_terrains={
        # 四种地形按比例混合。注意每个比重(proportion)加起来 ≈ 1.0。
        # 平坦地面占 25%。
        "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.25),
        # 金字塔台阶(层层递减的方箱):
        #   step_height_range=(0.0, 0.015) —— 台阶最大 1.5 cm(默认是 10 cm!)
        #   step_width=0.15 —— 台阶宽度(必须 ≥ 机器人步宽,否则像走楼梯)
        #   platform_width=2.0 —— 每行顶端的平台宽度
        "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.0, 0.015),  # max 1.5 cm (vs 10 cm default)
            step_width=0.15,
            platform_width=2.0,
            border_width=1.0,
        ),
        # NOTE: BoxInvertedPyramidStairsTerrainCfg removed — it sets env_origin_z to the pit
        # bottom (negative), causing resets at root_z = 0.12 + env_origin_z ≈ −0.10 m which
        # places the robot below the pit floor and makes it fall through the ground.
        # Uneven cobblestone-like ground: random per-cell height offsets.
        # grid_width=0.12 on an 8m patch = 66×66 = 4 356 boxes/patch → ~261 K total → OOM.
        # 0.45 m gives 17×17 = 289 boxes/patch → ~17 K total (border = 0.35 m ✓).
        # Must not divide evenly into terrain size (8.0 m): 0.45 × 17 = 7.65 ✓
        # 【中文解释】随机方块地面(类似鹅卵石):
        #   极危险的一个坑:grid_width 设 0.12 m 会在 8m 见方里生成
        #   66×66=4356 个盒子 → 内存爆(OOM)。0.45 m → 17×17≈289 个盒,
        #   而且 0.45 不能整除 8.0(8.0/0.45=17.78),这样边界才不出鬼。
        #   另外千万不要用倒金字塔台阶:它会把 env_origin_z 设为坑底
        #   (负值),重置时 root_z ≈ −0.10 m,机器人直接掉到地底下。
        "random_grid": terrain_gen.BoxRandomGridTerrainCfg(
            proportion=0.30,
            grid_width=0.45,
            grid_height_range=(0.0, 0.010),  # max 1 cm
            platform_width=1.5,
        ),
        # Gentle slopes (heightfield pyramid, platform on TOP — robot spawns on
        # the flat platform and walks down/up/across the slope as commands
        # resample). slope_range is rise/run: 0.03→0.10 ≈ 1.7°→5.7° by
        # difficulty — small robot, small slopes. NOT inverted (see the
        # inverted-pyramid env_origin note above — same pit-spawn risk class).
        # vertical_scale=0.001 keeps quantization steps at 1 mm so a gentle
        # slope is smooth instead of a staircase of 5 mm ledges.
        # 【中文解释】缓和坡道(高度场金字塔,平台在顶部 —— 机器人出生在
        # 平台上,随命令重采样走向下坡/上坡/横穿)。
        #   slope_range = 高/水平距离:0.03→0.10 约等于 1.7°→5.7°,
        #   按难度递增 —— 小机器人配小坡。
        #   vertical_scale=0.001 把高度量化到 1 mm,否则缓坡会变成
        #   5 mm 级别的"台阶楼梯"。
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.20,
            slope_range=(0.03, 0.10),
            platform_width=2.0,
            vertical_scale=0.001,
        ),
    },
    add_lights=False,
)


# ============================================================
# 【地形接触软化】spec_fn:在 MJCF 编译后、仿真开始前对场景做的最后加工
# (类似 MuJoCo 的 callback spec 阶段)。只用于粗糙地形。
# ============================================================
def _soften_terrain_contacts(spec: _mujoco.MjSpec) -> None:
    """Soften terrain box geom contacts to reduce edge-contact NaN instability.

    Box terrains place adjacent geoms at different heights. The hard edges where
    heights change cause contact normal instability when feet land on them, which
    can produce impulsive NaN forces in the MuJoCo solver.

    Doubling the solref time constant (0.02 → 0.04 s) makes contact springs
    2× softer — enough to damp the instability without noticeably changing the
    macro-level walking physics. Applied to all geoms in the "terrain" body,
    which contains every box generated by TerrainGenerator.
    """
    # 【中文解释】方块地形相邻块高度不同,硬边会让接触法线在脚落地时
    # 失稳 → 求解器产生冲激式 NaN 力。把接触弹簧的 solref 时间常数
    # 从 0.02 加倍到 0.04 s,弹簧变软 2 倍,足以压制失稳,又基本不
    # 改变宏观行走物理。只作用于 TerrainGenerator 生成的 terrain body。
    body = spec.body("terrain")
    count = 0
    for geom in body.geoms:
        geom.solref = [0.04, 1.0]   # 2× softer time constant (default: 0.02)
        geom.solimp = [0.85, 0.95, 0.001, 0.5, 2.0]  # slightly softer impedance
        count += 1
    print(f"[rough terrain] spec_fn: softened {count} terrain geoms (solref=0.04)")


# ============================================================
# 【主工厂函数】make_microduck_velocity_env_cfg
# 返回一个 ManagerBasedRlEnvCfg —— mjlab 的"环境配置对象"。
# 它本身不含任何训练逻辑,是一张"配方单":用哪些 robot 模型、
# 哪些传感器、哪些奖励项、哪些事件、哪些观测、哪些命令、哪些课程。
# 工厂参数:
#   play  = True → 用于 viewer/回放(地形关闭课程、缩短推送间隔等);
#   rough = True → 启用粗糙地形(MICRODUCK_ROUGH_TERRAINS_CFG)。
# 任务注册在 tasks/__init__.py 里调用本函数(平地和粗糙两个变体)。
# ============================================================
def make_microduck_velocity_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Microduck velocity tracking environment configuration."""

    # ----------------------------------------------------------
    # 【姿态奖励的两种标准差(高斯)】
    # pose 奖励 = 每个关节对"默认站姿"的高斯跟踪奖励:
    #   reward = exp(-0.5 * ((q - q_home)/std)^2)
    # std 越小 = 奖励越尖锐 = 策略越被迫贴住 HOME 姿态。
    # std_standing:命令为零(站定)时用 —— 收紧,逼策略站姿标准;
    # std_walking:行走命令下发后用 —— 放松,允许迈步所需的偏差。
    # ----------------------------------------------------------
    std_standing = {
        # Lower body — tighter to keep the robot in home pose when standing
        r".*hip_yaw.*": 0.1,
        r".*hip_roll.*": HIP_ROLL_POSE_STD,  # 0.05 → 0.12 (2026-09-11): the identified static
                                # sag of this joint under load is ~12 deg, so 0.05
                                # (2.9 deg) asked for something the actuator cannot
                                # hold and the Gaussian simply went flat.
        r".*hip_pitch.*": 0.15,
        r".*knee.*": 0.15,
        r".*ankle.*": 0.1,
    }
    # 【中文解释】hip_roll 从 0.05 放宽到 0.12:0.05 rad(2.9°)比舵机自身
    # 在负载下的静态下垂(实测 ~12°)还小 —— 目标根本不可达,高斯就变平了,
    # 等于没有梯度。正确的做法是给一个"够得着"的目标,而不是要求做不到的事。

    std_walking = {
        # Lower body
        r".*hip_yaw.*": 0.3,
        r".*hip_roll.*": HIP_ROLL_POSE_STD,  # 0.05 → 0.12, same reason as std_standing above
        r".*hip_pitch.*": 0.4,
        r".*knee.*": 0.4,
        r".*ankle.*": 0.25, # was 0.15
    }
    # 【中文解释】行走时对腿部放松 2–3 倍(std 0.3–0.4 vs 站立 0.1–0.15),
    # 脚踝从 0.15 提到 0.25 —— 太紧的脚踝会在枢轴式转弯时卡住策略。
    # 注意这些 dict 只匹配"腿"关节 —— 全程不含 head/neck!
    # (头/颈由 head_pose_tracking 命令奖励驱动,见下面 pose 奖励的
    # asset_cfg 排除正则。)

    # 两只脚对应的 site 名,供多个传感器/奖励共用(顺序:左,右)。
    site_names = ["left_foot", "right_foot"]

    # Contact sensor for feet - LEFT, RIGHT order
    # 【中文解释】脚部接触传感器:检测"脚碰撞体 ↔ 地面"是否接触。
    #   primary  : 机器人上的脚碰撞体(正则 ^(left_foot_collision|right_foot_collision)$)
    #   secondary: 地面(body 名为 terrain)
    #   fields=("found","force"):输出"是否接触"与"接触力"
    #   reduce="netforce" : 多接触点合并为净力
    #   num_slots=1
    #   track_air_time=True: 顺便记录"离地时间"(用于 air_time 奖励)
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",  # LEFT foot first, RIGHT foot second
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    # 【中文解释】自碰撞传感器:检测躯干子树之间是否互相碰撞
    # (比如腿打到自己身上的电池盒)。primary/secondary 都是
    # trunk_base 子树,reduce="none" —— 只输出"有没有碰"(found)。
    # 下面 self_collisions 奖励用它惩罚策略拼命往身体里挤腿。
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    # mjlab 1.3.0: foot_height obs + foot_clearance/foot_swing_height rewards are
    # now driven by a per-foot terrain-height ray sensor (was site_pos based).
    # Mirrors microban's foot_height_scan.
    # 【中文解释】脚底地形高度扫描传感器(向下打射线,测"脚到地的高度"):
    #   frame=ObjRef(site, 左右脚) —— 从每只脚的位置发射
    #   RingPatternCfg.single_ring(radius=0.04, num_samples=2) —— 每脚 2 根射线
    #   ray_alignment="yaw" —— 射线朝向对齐机体偏航
    #   max_distance=1.0 m;exclude_parent_body 排除父体(防止射到自己的腿)
    # mjlab 1.3.0 起,脚高度观测与 foot_clearance/foot_swing_height 奖励
    # 都由这个射线传感器驱动(以前是 site 位置),与 microban 一致。
    foot_height_scan_cfg = TerrainHeightSensorCfg(
        name="foot_height_scan",
        frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in site_names),
        pattern=RingPatternCfg.single_ring(radius=0.04, num_samples=2),
        ray_alignment="yaw",
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=False,
    )

    # 脚摩擦随机化作用的碰撞体名字(见下方 foot_friction 事件)。
    foot_frictions_geom_names = (
        "left_foot_collision",
        "right_foot_collision",
    )

    # ----------------------------------------------------------
    # 【第一步:从 mjlab 的 velocity 模板拿一个"通用走路环境"基础配置】
    # make_velocity_env_cfg() 是 mjlab 库自带的模板(定义了标准动作
    # joint_pos、标准观测、命令 twist、一批常规奖励/事件)。
    # 之后所有 cfg.xxx = ... 都是在这个模板上"打补丁"。
    # 注意:返回的对象带共享可变引用 —— 所以下面多处要用 deepcopy
    # 防止其他环境文件(standup/ground_pick)在 import 时互相污染。
    # ----------------------------------------------------------
    cfg = make_velocity_env_cfg()

    # Robot setup
    # 【机器人】用 MICRODUCK_WALK_ROBOT_CFG —— "walk" 模型:
    # 去掉了躯干/头部的碰撞体(摔倒代价便宜,适合走路任务)。
    # 注意:其他任务(standup、roulade 等)用的是 robot_allcollisions.xml,
    # 因为它们的任务是"躺在地上一动不动",必须保留全身碰撞。
    cfg.scene.entities = {"robot": MICRODUCK_WALK_ROBOT_CFG}
    # 【场景传感器】注册上面定义的三个传感器:脚接触 / 自碰撞 / 脚高度扫描
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg, foot_height_scan_cfg)
    cfg.viewer.body_name = "trunk_base"

    # Action configuration
    # 【动作】velocity 模板自带 joint_pos 动作(每个舵机一个位置命令)。
    # scale=1.0:动作输出 × 1.0 直接作为关节角度目标(rad)。
    # critical:动作维度必须 = 舵机数 14(dof 总数 16 里有 2 个被动关节,
    # 被动关节不驱动)。观察维度也因此在后面被过滤为 14。
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ============================================================
    # === REWARDS(奖励项)===
    # mjlab 的奖励是"项"列表,每项 = {func, weight, params}。
    # 总奖励 = Σ weight × func(...)。本文件对模板奖励做定制:
    # 改权重、改参数、删项、加项。
    # 【符号约定提醒(AGENTS.md)】mjlab 模板的代价函数返回 ≥0,
    # 用负权重;microduck 自写的 *_penalty 自取负(返回 ≤0),用正权重。
    # 判断方法:wandb 里每个 Episode_Reward/<penalty> 必须 ≤ 0!
    # ============================================================
    # Pose reward configuration
    # 【姿态奖励 pose】每个关节对 HOME 姿态的高斯奖励。
    cfg.rewards["pose"].params["std_standing"] = std_standing  # tight when command=0
    cfg.rewards["pose"].params["std_walking"] = std_walking
    cfg.rewards["pose"].params["std_running"] = std_walking
    # Pose reward operates on LEG joints only. Head/neck are command-driven
    # (head_pose_tracking) — if they were in this reward too, it would pull
    # them to HOME while head_pose_tracking pulls them to the command, and the
    # policy converges to "ignore the command" because pose reward dominates
    # once head_pose_tracking's gradient dies at large commands.
    # 【中文解释】pose 奖励只作用在腿部关节,用正则排除 passive_(被动
    # 关节)和 *neck*/*head*(头颈)。为什么:头颈由 head_pose_tracking
    # 命令奖励驱动。如果 pose 奖励也拉它们回 HOME,两个奖励就会打架:
    # 命令偏差一大,pose 的高斯梯度就没了、而 pose 奖励权重是 1.0,
    # 策略学会"无视命令"。这个坑的教训:同一关节不能被两个目标拉扯。
    cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(r"^(?!passive_|.*neck.*|.*head.*).*",)
    )
    cfg.rewards["pose"].params["walking_threshold"] = 0.01
    cfg.rewards["pose"].weight = 1.0

    # Structural fix: swap the collective Gaussian for a per-joint mean so one
    # far-travelling joint (the ankle, in a crouched gait) can no longer mute the
    # posture gradient for every other joint. Same stds, same weight — only the
    # reducer changes. See ENABLE_PER_JOINT_POSTURE above for the 0.946-vs-0.015
    # evidence and mdp.per_joint_posture for the derivation.
    if ENABLE_PER_JOINT_POSTURE:
        cfg.rewards["pose"].func = microduck_mdp.per_joint_posture

    # ----------------------------------------------------------
    # === 姿态卫生 / 执行器卫生(2026-09-11,见文件顶部常量区)===
    # 这一组是"畸形姿势"的直接解药。每一项都对应一个**实测**的缺陷,
    # 权重由"缺陷代价占奖励的比例"反推(见顶部注释)。全部走课程爬坡。
    # ----------------------------------------------------------
    if ENABLE_STANCE_HYGIENE:
        # hip_yaw / hip_roll 回中。注意只作用在这两个关节上 —— 不是所有
        # 腿关节:hip_pitch/knee/ankle 在走路时**必须**大幅偏离 HOME,
        # 把它们也 L1 回中会直接把步态压平(这也是 stock pose 项饱和的原因)。
        cfg.rewards["stance_hip_yaw"] = RewardTermCfg(
            func=microduck_mdp.joint_deviation_l1,
            weight=0.0,  # ramped by the stance_tax_weight curriculum
            params={
                "deadband": STANCE_HIP_YAW_L1_DEADBAND,
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=(r"^(?!passive_).*hip_yaw.*",)
                ),
            },
        )

        # hip_roll 的可选 L1 回中。默认关：2026-09-11 的判断是该关节负载下有
        # ~12° 静态下垂，L1 会去税一个舵机逃不掉的部分。但 2026-09-16 的多 agent
        # 审计指出：观测到的偏置是**向外**的，而静态下垂是**向内**的
        # (docs/forensics/agent_A_reward_audit.md:88-90)，所以下垂解释不了外八；
        # 且 12° 那个数是在**已退役的 5 V 轨**上标定的。rollers 环境用同一形状的
        # 项治好了同一个病。开关独立，便于 A/B。
        if ENABLE_STANCE_HIP_ROLL:
            cfg.rewards["stance_hip_roll"] = RewardTermCfg(
                func=microduck_mdp.joint_deviation_l1,
                weight=0.0,  # ramped by the stance_tax_weight curriculum
                params={
                    "deadband": STANCE_HIP_ROLL_L1_DEADBAND,
                    "asset_cfg": SceneEntityCfg(
                        "robot", joint_names=(r"^(?!passive_).*hip_roll.*",)
                    ),
                },
            )
        # 动作侧限位越界惩罚。**为什么不是 qpos 侧**:rollers 环境的记录
        # 明确写过 "a qpos penalty stole that range and broke the gait" ——
        # qpos 惩罚会偷走低 kp 舵机"冲过目标再被限位挡住"所必需的行程。
        # 动作侧只罚"命令远超限位 + overshoot",关节仍能到达整个可用范围,
        # 而且因为约束的是策略输出,行为会被烘进 ONNX、能带到真机。
        # 【顺带修掉的部署隐患】当前策略在 hip_yaw 上输出 |a|=10.14,
        # 真机运行时会把目标算成 HOME+10 rad —— 这个惩罚让网络不再这么干。
        cfg.rewards["action_over_limit"] = RewardTermCfg(
            func=microduck_mdp.action_over_limit_penalty,
            weight=0.0,  # ramped
            params={
                "action_name": "joint_pos",
                "overshoot": ACTION_OVER_LIMIT_OVERSHOOT,
            },
        )
        # 脖子"抽动"惩罚(带 deadband)。详见 mdp.neck_joint_vel_deadband_l2:
        # 慢速摆头免费(执行 head_pose 命令需要),步频级的泵动收费。
        cfg.rewards["neck_motion"] = RewardTermCfg(
            func=microduck_mdp.neck_joint_vel_deadband_l2,
            weight=0.0,  # ramped
            params={"deadband": NECK_MOTION_DEADBAND_RAD_S},
        )
        # Direct cure for marching in place at zero command (see the constant's
        # comment for the measured calibration).
        cfg.rewards["standing_leg_vel"] = RewardTermCfg(
            func=microduck_mdp.joint_vel_l2_when_standing,
            weight=0.0,  # ramped
            params={
                "command_name": "twist",
                "command_threshold": 0.01,
                "body_vel_threshold": STANDING_LEG_VEL_PUSH_EXEMPT,
            },
        )

    # ----------------------------------------------------------
    # === 直行保持 / 步态对称(2026-09-14,见文件顶部常量区)===
    # 见顶部注释:奖励栈里原本**没有任何一项是偏航角的函数**,而观测里又没有
    # 航向 ⇒ 走歪既未被定价也无法闭环纠正。这一组补上"角度"这一项。
    # ----------------------------------------------------------
    if ENABLE_HEADING_HOLD:
        # 唯一"偏航角"函数。用已有的 heading_hold_reward(rollers 已验证),
        # 新加 command_name/command_threshold 门控:命令在转弯时该项返回 0
        # (转弯本身既不加分也不扣分),只在命令接近直行时才把"保持航向"计费,
        # 且参考航向在转弯期间持续刷新 —— 转弯结束的瞬间参考自动落在
        # "直线段起点"的航向上,不需要边沿检测。
        # 门控关闭返回 0 而不是 1:两者对梯度的作用完全相同,但常数 1.0 会给
        # 正奖励栈注入约 +1.2/步的**无梯度**奖励,稀释其他所有项的相对影响
        # (AGENTS.md:"比较奖励质量而不是权重")。仓库既有惯例也是返回 0。
        cfg.rewards["heading_hold"] = RewardTermCfg(
            func=microduck_mdp.heading_hold_reward,
            weight=0.0,  # ramped by the heading_hold_weight curriculum
            params={
                "std": HEADING_HOLD_STD,
                "asset_cfg": SceneEntityCfg("robot"),
                "command_name": "twist",
                "command_threshold": HEADING_HOLD_COMMAND_THRESHOLD,
                # Multiplicative speed gate — see the constant's comment: without
                # it a STATIONARY policy collects the full reward (it holds the
                # heading perfectly), which would score higher than veering.
                "vel_gate_ref": HEADING_HOLD_VEL_GATE_REF,
                "vel_gate_tau": HEADING_HOLD_VEL_GATE_TAU,
                "vel_gate_deadband": HEADING_HOLD_VEL_GATE_DEADBAND,
            },
        )

    if ENABLE_GAIT_SYMMETRY:
        # 累计摆动时间不平衡 |L−R|/(L+R)。真实步态里"此刻只有一只脚在摆"
        # 是正常的,所以只罚**累计**不平衡。当前策略左右膝差 6.8°、踝差 6.6°
        # (官方 <1°),这条腿的不对称就是净偏航力矩的来源。
        cfg.rewards["gait_symmetry"] = RewardTermCfg(
            func=microduck_mdp.gait_symmetry_penalty,
            weight=0.0,  # ramped by the gait_symmetry_weight curriculum
            params={"sensor_name": "feet_ground_contact"},
        )

    if ENABLE_YAW_SYMMETRY:
        # 走路时的净偏航率(net yaw rate)惩罚 —— 返回 ≤0 ⇒ 权重取正。
        # 门控:直线命令(|cmd_wz| ≤ 阈值)且确实在走(|cmd_xy| > 0.05)。
        # 上线后必须校验 wandb Episode_Reward/yaw_dc ≤ 0(AGENTS.md 的符号规则)。
        cfg.rewards["yaw_dc"] = RewardTermCfg(
            func=microduck_mdp.yaw_dc_rate_penalty,
            weight=0.0,  # ramped by the yaw_dc_weight curriculum
            params={
                "command_name":         "twist",
                "wz_command_threshold": YAW_SYMMETRY_WZ_CMD_THRESHOLD,
                "command_threshold":    0.05,
                "tau_s":                YAW_SYMMETRY_TAU_S,
            },
        )

    # Body-specific reward configurations
    # 【直立体 upright】躯干(trunk_base)要保持直立的高斯奖励。
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    # upright: deliberately strong (2.0 / std²=0.05, was 1.0 / std²=0.1).
    # 2026-07 pitch-vs-speed eval: the policy walks with a +2-4° steady forward
    # lean (p90 ~6-8°) and ~2/3 of push-induced falls at speed are FORWARD. At
    # weight 1.0 / std²=0.1 a 4° lean cost ~0.05/step — effectively free. At
    # 2.0 / std²=0.05 it costs ~0.19/step: enough gradient to hold the trunk
    # level in steady gait while transient lean (push recovery, accel) stays
    # affordable.
    # 【中文解释】故意加得很强(2.0 / std²=0.05,原来是 1.0/0.1)。
    # 2026-07 的速度-俯仰评估发现:策略走路时带着 +2–4° 的稳定前倾
    # (p90 约 6–8°),而且受推后摔倒约 2/3 是向前摔。旧权重下 4° 前倾
    # 每步只花 0.05 —— 等于免费。新权重下花 0.19/步:足以在平稳步态中
    # 把躯干拉直,同时瞬态前倾(受推恢复、加速)仍然"付得起"。
    cfg.rewards["upright"].weight = 2.0
    cfg.rewards["upright"].params["std"] = math.sqrt(0.05)

    # Foot-specific configurations. In mjlab 1.3.0 foot_swing_height is fully
    # sensor-driven (no asset_cfg); only foot_clearance/foot_slip still carry an
    # asset_cfg whose site_names select the feet.
    # 【中文解释】把脚部相关奖励的作用对象限定到两只脚(site)。
    for reward_name in ["foot_clearance", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names
    # 【脚间距下限（任务层面）】见文件顶部 ENABLE_FEET_DISTANCE 的长注释。
    # 默认关闭 ⇒ 不打开开关时配置与改动前**逐字节一致**（可用 tests/ 与 smoke 验证）。
    # 打开后：走路时（非零指令）两脚横向间距不得小于 FEET_DISTANCE_MIN。
    # 这是**唯一一个改在任务层面、单关节绕不过去**的外八处方
    # （关节空间的两种形状都已被实测证伪，见顶部注释）。
    if ENABLE_YAW_Z_TRACK:
        cfg.rewards["track_angular_velocity_z"] = RewardTermCfg(
            func=microduck_mdp.track_angular_velocity_z_only,
            weight=YAW_Z_TRACK_WEIGHT,
            params={
                "std": YAW_Z_TRACK_STD,
                "command_name": "twist",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    if ENABLE_FLIGHT_PENALTY:
        # 返回 ≤0 ⇒ **正权重**（见常量区的符号说明）。
        cfg.rewards["flight"] = RewardTermCfg(
            func=microduck_mdp.flight_penalty,
            weight=FLIGHT_PENALTY_WEIGHT,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "command_threshold": 0.01,
                "grace_steps": FLIGHT_GRACE_STEPS,
            },
        )

    if ENABLE_FEET_MAX_WIDTH:
        # 返回 ≤0 ⇒ **正权重**（见常量区的符号说明）。site_names 与
        # foot_clearance/foot_slip 共用左右脚 site。
        cfg.rewards["feet_max_width"] = RewardTermCfg(
            func=microduck_mdp.feet_distance_max_penalty,
            weight=FEET_MAX_WIDTH_WEIGHT,
            params={
                "max_dist": FEET_MAX_WIDTH,
                "asset_cfg": SceneEntityCfg("robot", site_names=tuple(site_names)),
                "command_name": "twist",
                "command_threshold": 0.01,
            },
        )

    if ENABLE_ABDUCTION:
        cfg.rewards["abduction"] = RewardTermCfg(
            func=microduck_mdp.leg_abduction_penalty,
            weight=ABDUCTION_WEIGHT,
            params={
                "deadband": ABDUCTION_DEADBAND,
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*hip_roll.*",)),
                "command_name": "twist",
                "command_threshold": 0.01,
            },
        )


    # Body-specific configurations
    # 【机体角速度惩罚 body_ang_vel】只惩罚躯干,不惩罚头(头部摆动是
    # 必然的,见后面 head_pose_bias 的讨论)。
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)

    # foot_slip deliberately weak (-0.1, not -1.0): -1.0 was too restrictive
    # for this robot's pivot-heavy turning.
    # 【中文解释】脚滑惩罚故意弱(-0.1):这个机器人转身是"以脚掌为
    # 轴心"的方式 -1.0 会直接把转身练死。
    cfg.rewards["foot_slip"].weight = -0.1
    cfg.rewards["foot_slip"].params["command_threshold"] = 0.01

    # 移掉模板的 soft_landing(软着陆)奖励:对这个小而轻的机器人,
    # 落地冲击惩罚由别的项负责,这里不再重复。
    cfg.rewards.pop("soft_landing", None)

    # Self-collision penalty: discourages legs from crashing into the trunk
    # battery holder (the self_collision_only-classed geoms on leg, leg_2,
    # battery_holder). With proper joint-range limits the policy can't actually
    # reach the body, but a positive signal here keeps it well clear.
    # 【中文解释】自碰撞惩罚(新加项,func 是 mjlab 的 self_collision_cost):
    # 防止腿打到躯干下的电池盒。虽然关节限位下很难真碰到,但给个
    # 惩罚信号能让策略保持安全距离。
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # air_time window [0.125, 0.300] s. NOTE: standing still at zero command is
    # taught by the standing_envs curriculum (→25% standing envs by ~iter 2000),
    # not by an explicit stillness/no-stepping term.
    # 【中文解释】离地时间奖励 air_time:鼓励双脚交替离地在
    # [0.125, 0.300] s 区间 —— 低于下限=碎步拖沓,高于上限=大跳。
    # 注意:命令为零时"站定不动"不是靠专门的不动惩罚教的,而是靠
    # standing_envs 课程(把 25% 环境赋予零命令,到约 2000 迭代)。
    cfg.rewards["air_time"].weight = 3.0
    cfg.rewards["air_time"].params["command_threshold"] = 0.01
    cfg.rewards["air_time"].params["threshold_min"] = 0.125
    cfg.rewards["air_time"].params["threshold_max"] = 0.300

    # Optional progress gate on air_time (see AIR_TIME_PROGRESS_GATE_V_REF at the
    # top). While it is 0.0 the term keeps mjlab's function and the recipe is
    # byte-identical to before; enabling it swaps in the gated version so that
    # lifting the feet without moving the body stops paying full price.
    if AIR_TIME_PROGRESS_GATE_V_REF > 0.0 or AIR_TIME_PROGRESS_GATE_TURN_REF > 0.0:
        cfg.rewards["air_time"].func = microduck_mdp.feet_air_time_progress_gated
        cfg.rewards["air_time"].params["vel_gate_ref"] = AIR_TIME_PROGRESS_GATE_V_REF
        cfg.rewards["air_time"].params["turn_gate_ref"] = AIR_TIME_PROGRESS_GATE_TURN_REF
        cfg.rewards["air_time"].params["gate_floor"] = AIR_TIME_PROGRESS_GATE_FLOOR

    # ----------------------------------------------------------
    # 【零命令 = 站定,不许原地踏步】(2026-09-11,用户明确要求)
    # 需求原话:"我想要一个不踏步,按键盘走再走的模型" —— 不按键就站着不动,
    # 按了键才走。
    #
    # 【为什么官方配方没这个】AGENTS.md 的原设计是:零命令站定靠
    # standing_envs 课程(把 25% 的环境命令清零)去"自然学会",
    # **刻意不加任何 stillness 项**。practice 上这确实常常够用 ——
    # 昨天那个训完的模型在 vx=0 时确实是完全静止的(倾角 0.4°、零漂移)。
    # 但它没有上界:一旦策略学到"踏步也能拿分",就没有任何东西把它拉回来。
    #
    # 【为什么这两个工具是现成的】mdp.py 里 `stillness_at_zero_command` 和
    # `no_stepping_penalty` 就是为这件事写的(后者 docstring 原文:
    # "Discourages marching in place when the robot should stand still"),
    # 但 grep 全仓库:**从没接进任何 velocity 环境**。这次接上。
    #
    # ⚠️ 两个都按命令门控(cmd_speed < 0.01),所以**对行走步态零影响** ——
    # 不存在"为了治站姿把走路练坏"的风险。
    # 【为什么用"正奖励 + 惩罚"两条而不是只加惩罚】
    #   * stillness 是单调递减的正奖励:动得越快越少分,**没有阈值可以跨过去
    #     逃掉**(不像 gate 型惩罚能被"刚好卡在阈值下"利用),AGENTS.md 推荐
    #     的形态;
    #   * no_stepping 直接罚"脚离地"这个具体动作,是踏步的直接代价。
    # 两者一起:站定 ≈ +0.8,踏步 ≈ −0.4 ~ −0.8,落差 ~1.2/步,足够翻盘。
    #
    # ⚠️ 【2026-09-17 对齐官方】这三项**只在 ENABLE_STANCE_HYGIENE 打开时注册**。
    # 理由不是审美:它们的权重**由 stance_tax_weight 课程**爬坡,而那个课程本身
    # 就在 ENABLE_STANCE_HYGIENE 块里 —— 关掉 hygiene 时权重永远停在 0.0,
    # 注册它们等于在奖励管理器里挂三个恒为 0 的死项(还破坏与官方的逐项等价)。
    if ENABLE_STANCE_HYGIENE or ENABLE_ZERO_CMD_TERMS:
        cfg.rewards["stillness_zero_cmd"] = RewardTermCfg(
            func=microduck_mdp.stillness_at_zero_command,
            weight=0.0,  # ramped by the stance_tax_weight curriculum
            params={
                "command_name": "twist",
                "command_threshold": 0.01,
                "vel_std": 0.1,
            },
        )
        cfg.rewards["no_stepping"] = RewardTermCfg(
            func=microduck_mdp.no_stepping_penalty,
            weight=0.0,  # ramped
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "command_threshold": 0.01,
            },
        )
        # 【零命令定点航向】站定时按住"停下那一刻"的朝向（角项可回纠 + 角速度项稠密）。
        # 门控是**整条 twist 指令**，所以走路时恒为 0 —— 与 heading_hold 不重叠、不打架。
        cfg.rewards["yaw_hold_zero_cmd"] = RewardTermCfg(
            func=microduck_mdp.yaw_hold_at_zero_command,
            weight=0.0,  # ramped by the stance_tax_weight curriculum
            params={
                "command_name": "twist",
                "command_threshold": 0.01,
                "std": YAW_HOLD_AT_ZERO_CMD_STD,
                "rate_weight": YAW_HOLD_AT_ZERO_CMD_RATE_WEIGHT,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    # 【平滑类正则(注意都是负权重,模板 cost 返回 ≥0)】
    # body_ang_vel:惩罚躯干角速度(转头/翻滚)。对动态任务要放轻!
    cfg.rewards["body_ang_vel"].weight = -0.05
    # angular_momentum:惩罚整体角动量(防止转体/蹦跳混进行走里)。
    cfg.rewards["angular_momentum"].weight = -0.02

    # Velocity tracking rewards
    # 【速度跟踪】这是行走任务的主奖励:线性速度命令 + 角速度命令的高斯跟踪。
    # std 的含义:你"还关心"的误差大小。std=sqrt(0.1)≈0.316 m/s:
    # 速度误差超过这个量级奖励才开始明显下降。太松在小误差处没有梯度。
    cfg.rewards["track_linear_velocity"].weight = 2.0
    cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.1)
    cfg.rewards["track_angular_velocity"].weight = 2.0
    cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.5)

    # Action smoothness: stage-0 value; the action_rate_weight curriculum below
    # ramps it -0.1 → -1.0 by iter 1500.
    # 【中文解释】动作平滑惩罚(相邻步动作变化率)。初始只有 -0.1,
    # 靠下面的 action_rate_weight 课程逐步加到 -1.0(1500 迭代前)。
    # 【为什么不能一开始就重】步态还在摸索时,任何"动作税"都会让
    # "什么都不做"变成最优解 —— 平滑类正则必须在技能学会后再上强度。
    cfg.rewards["action_rate_l2"].weight = -0.1

    # 【脚部抬升类】command_threshold=0.01:只有命令绝对值 > 0.01 时
    # 才计入(站定时不惩罚)。target_height=0.02:离地高度目标 2 cm
    # (原来 1 cm —— 太矮导致拖脚)。
    cfg.rewards["foot_clearance"].params["command_threshold"] = 0.01
    cfg.rewards["foot_clearance"].params["target_height"] = 0.02  # Increased from 0.01 to penalize dragging

    cfg.rewards["foot_swing_height"].params["command_threshold"] = 0.01
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.02  # Increased from 0.01 to force foot lifting

    # NOTE: no neck-only action-rate term — the shared action_rate_l2 sums over
    # ALL action dims (neck included), and head_pose_tracking below gives the
    # 4 neck/head DOFs a position objective, so the neck is fully shaped.
    # 【中文解释】不单独加脖子动作率惩罚:shared 的 action_rate_l2 已经把
    # 所有动作维度(含脖子)都算了,而 head_pose_tracking 给 4 个头颈
    # 自由度提供位置目标 —— 头颈已经"被完全塑形",无需额外项。

    # ----------------------------------------------------------
    # Events(事件):按 mode 分类 —— "startup"(仿真启动时执行一次)、
    # "reset"(每次环境重置时执行)、"interval"(周期性)。
    # 事件是"对物理量/状态做操作"的机制,与奖励("给分")相对。
    # ----------------------------------------------------------
    # BAM (mjlab_frictionloss branch) writes per-env dof_frictionloss/dof_damping
    # every step; this no-op event registers those fields for per-world expansion.
    # 【中文解释】BAM 致动器每步会写每个环境的 dof_frictionloss/dof_damping;
    # 这个"无操作"事件(mode=startup)的作用是让 mjlab 把这些字段注册为
    # 按环境展开(per-world expansion)—— 没有它,BAM 的摩擦参数会退化
    # 成所有环境共享一份。【任何独立编写的环境都必须注册这个事件!】
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )

    # 【重置事件】每次重置时清零"动作历史"(action 的上一帧/上两帧)。
    # 这是 action_rate 类惩罚所需的状态 —— 不重置的话,动作率惩罚在
    # 环境切换后仍带着上一集的动作,给出错误信号。
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )

    # 【脚与地摩擦随机化】模板自带 foot_friction 事件,这里把作用对象
    # 限定为两只脚底的碰撞体,并收紧范围 (0.7, 1.3):
    # 从 (0.3, 1.2) 收窄 —— 原来 0.3 的摩擦系数太滑,这个机器人走不了
    # (脚小、抓地能力有限),0.7 下限是"能走且仍带随机性"的折中。
    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)  # Grippier footpad — narrowed from (0.3, 1.2)
    # Terminate environments that have gone numerically unstable (NaN physics).
    # MuJoCo can produce NaN joint positions on extreme contact impulses.
    # Terminating immediately resets to a valid state before NaN propagates
    # into the observation buffer and corrupts network weights.
    # 【中文解释】NaN 终止(仿真守门员):极端接触冲量可能让 MuJoCo
    # 算出 NaN 关节位置。立即终止并重置,避免 NaN 进入观测缓冲区
    # 污染网络权重(一次 NaN 会毁掉整个训练!r sl_rl 的 check_nan 会
    # 直接崩掉 run)。sensor_names 传入脚接触传感器:一是为了用
    # 传感器数值辅助判定,二是把这个传感器注册为读取路径。
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": (feet_ground_cfg.name,)},
    )

    # 【初始位姿 z 范围】重置时机体根高 0.12–0.13 m(站姿小腿长)。
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.12, 0.13)

    # Velocity-based pushes for robustness training
    # 【鲁棒性推送】按概率/间隔给机体随机速度扰动(模拟被绊/被推)。
    if ENABLE_VELOCITY_PUSHES:
        # In play mode, use shorter interval for better visibility
        # play 模式下改为 0.5–1.0 s:方便人在 viewer 里看清"被推后如何恢复"。
        interval = (0.5, 1.0) if play else VELOCITY_PUSH_INTERVAL_S

        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=interval,
            params={
                # 只在 x(前后)/y(左右)方向加扰动 —— z 方向(垂直)不加。
                "velocity_range": {
                    "x": VELOCITY_PUSH_RANGE,
                    "y": VELOCITY_PUSH_RANGE,
                },
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    # ----------------------------------------------------------
    # 【域随机化(DR)区块】上面每个 ENABLE_* 开关对应下面一段。
    # 关键约定(AGENTS.md):DR 绝不能跨重置累积。
    #   mjlab 1.3.0 的 dr.* 算子(operation="add"/"scale")原生不累积
    #   —— 它们每次重置都重读编译期默认值(uses_defaults=True)。
    #   所以直接用 dr.* 就是安全的;自写的 DR 函数必须"先恢复再施加",
    #   累计式 CoM 随机化曾经毁掉过几个月的长训练。
    # 另一个约定:改物理参数要动 mu,奖励要动 reward —— 事件改的是
    # 物理世界,不是策略的世界。
    # ----------------------------------------------------------
    # Domain randomization — re-sampled per episode at reset. In mjlab 1.3.0 the
    # stock dr.* ops with operation="add"/"scale" read from the compile-time
    # default field each reset (Operation.uses_defaults=True), so they are
    # NON-accumulating natively — this upstream behavior replaces microduck's old
    # custom restore-then-add functions that worked around the accumulation footgun.
    if ENABLE_COM_RANDOMIZATION:
        # 【躯干质心随机化】给 trunk_base 质心加 [-R, +R] 的随机偏移。
        # R 从 ±3 mm 起步,由 com_range 课程升到 ±15 mm(封顶)。
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
        # Randomize the CoM of the head assembly bodies (per-body fresh offset each reset).
        # 【头部质心随机化】对 HEAD_BODY_NAMES 里每个头部件逐一加偏移。
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        # Randomize motor PD gains
        # Uses custom function that handles DelayedActuator
        # 【舵机 PD 增益随机化】用自写函数 randomize_delayed_actuator_gains,
        # 因为本仓库的致动器是带延迟的 BAM(DelayedActuator 派生),模板的
        # dr.kp/dr.kd 对它无效。当前两者都关(False),所以等效不注册:
        # kp_range/kd_range 为 (1.0, 1.0) 也完全随机化。
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
        # Physics-consistent mass + inertia randomization via mjlab's pseudo_inertia:
        # alpha scales BOTH mass and inertia by e^(2*alpha) with the CoM unchanged
        # (so it does NOT conflict with randomize_com). alpha_range is derived from
        # the ±5% mass scale range: e^(2*alpha) ∈ [0.95, 1.05].
        # Replaces the old custom randomize_mass_and_inertia, which was a silent
        # no-op under mjlab 1.3.0 (direct per-env body_mass/body_inertia writes are
        # not expanded and collapse to a single shared value). Startup mode = fixed
        # per env for the whole run (standard for mass DR; no accumulation).
        # 【中文解释】质量+惯量联合随机化(mode="startup" = 每个环境在整个
        #   训练期固定一个值,不像 reset 模式每集重采 —— 这是质量 DR 的
        #   标准做法,而且不存在累积问题)。
        #   alpha 同时缩放质量和惯量 e^(2α),且不变质心 —— 所以和
        #   randomize_com 不冲突。α = ln(range)/2,把 ±5% 质量范围
        #   换算成 α。
        #   为什么不用以前的自写函数:mjlab 1.3.0 下直接写 per-env 的
        #   body_mass/body_inertia 不会被展开,退化成所有环境共享一个值
        #   (静默失效)。pseudo_inertia 是物理一致的官方路径。
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
        # Joint-friction DR under BAM: scales BAM's velocity-independent friction
        # budget (Coulomb + Stribeck + load) per-env via the FrictionDRBamActuator
        # friction_scale hook. MuJoCo's dof_frictionloss is zeroed under BAM, so the
        # stock dr.dof_frictionloss is a no-op — this is the BAM-native path.
        # 【中文解释】关节摩擦随机化(这是 BAM 原生路径!):
        # BAM 致动器下 MuJoCo 的 dof_frictionloss 被清零,所以模板的
        # dr.dof_frictionloss 是静默无效的。正确做法:通过
        # FrictionDRBamActuator 的 friction_scale 钩子缩放 BAM 自身的
        # 摩擦预算(库仑+Stribeck+负载)。随机化范围 ±10%(0.9,1.1)。
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_JOINT_DAMPING_RANDOMIZATION:
        # Randomize joint damping (lubrication, temperature effects).
        # Custom non-accumulating scaler. NOTE: no-op under BAM (dof_damping
        # zeroed in edit_spec); only affects the XML position actuator.
        # 【中文解释】关节阻尼随机化。注意:BAM 下 dof_damping 也被清零
        # (edit_spec 里),所以这个事件目前是 no-op —— 它只对 XML 里
        # 的"位置致动器"起作用。默认关闭(False),留下的目的是备忘:
        # 想随机化"润滑/温度"效果时,应该去动 BAM 的摩擦,而不是这个。
        cfg.events["randomize_joint_damping"] = EventTermCfg(
            func=microduck_mdp.randomize_dof_field_scaled,
            mode="reset",
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "field": "dof_damping",  # required by domain_randomization=True
                "scale_range": JOINT_DAMPING_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_ARMATURE_RANDOMIZATION:
        # Randomize reflected rotor inertia (armature), microban-exact
        # (dr.joint_armature, scale, ±10%). Non-accumulating (uses_defaults). DOES
        # affect the BAM actuator — BAM sets dof_armature (~0.0018), it isn't zeroed.
        # 【中文解释】反射转子惯量(armature)随机化 ±10%,与 microban 一致。
        # 注意:BAM 会设置 dof_armature(~0.0018),不会清零 —— 所以这个
        # 随机化对 BAM 有效(模拟不同批次电机转子惯量的差别)。
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    # IMU orientation randomization (mounting error) is applied at the OBSERVATION
    # level below (per-env constant rotation of projected_gravity + base_ang_vel).
    # The old event-based randomize_imu_orientation wrote site_quat, which under
    # mjlab 1.3.0 is neither per-env expanded nor read by these obs — a no-op.
    # 【中文解释】IMU 姿态随机化不在事件区做,而在观测区做(见下方):
    # 旧做法 write site_quat 在 mjlab 1.3.0 下既不按环境展开、观测也不读它,
    # 是静默无效的。正确做法:在观测函数里对 projected_gravity 和
    # base_ang_vel 施加一个 per-env 的常值旋转。

    # Base orientation randomization (forces reactive behavior)
    # 【初始姿态随机化】给初始俯仰/翻滚一定扰动,逼策略学会"反应式"
    # 调整。默认关闭(False)—— 对速度任务来说,初始倾斜会导致
    # 策略把"从倾斜恢复"和"行走"混在一起,通常弊大于利。
    if ENABLE_BASE_ORIENTATION_RANDOMIZATION:
        cfg.events["randomize_base_orientation"] = EventTermCfg(
            func=microduck_mdp.randomize_base_orientation,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "max_pitch_deg": BASE_ORIENTATION_MAX_PITCH_DEG,
                "max_roll_deg": BASE_ORIENTATION_MAX_ROLL_DEG,
            },
        )

    # ----------------------------------------------------------
    # Observations(观测)
    # 分 actor(策略看到的,61 维)和 critic(评论家看到的,含特权信息)
    # 两组。核心约定:actor 的观测布局是全家族共享的、
    # 61 维:48 维基础本体感觉 + 13 维命令 [twist(3), head_pose(4), body_pose(6)]。
    # 不用某个命令槽的环境一律"零填充"(保留观测项、采样极小范围),
    # 绝不删槽 —— 否则 ONNX 形状改变、运行时无法热切换策略。
    # ----------------------------------------------------------
    # Observations
    # 【删除 base_lin_vel(线速度)—— 特权信息!】
    # 真机上拿不到准确的机体线速度(IMU 只能测角速度+加速度)。
    # 所以策略(actor)不许看线速度;只有 critic 可以(集中式训练,
    # 评论家评估"好状态",给策略更多信息能学得更快)。
    del cfg.observations["actor"].terms["base_lin_vel"]
    # mjlab 1.3.0 adds a height_scan term (terrain ray scan) to both groups by
    # default. The microduck has no such body-mounted terrain sensor for the
    # policy, so drop it from both (mirrors microban).
    # 【删除 height_scan】模板默认给两组都加了地面射线扫描项,但
    # Microduck 没有这个传感器(策略在真机上读不到),必须删掉,
    # 否则训练和部署的观测布局不一致。
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    # Add base_lin_vel to critic only (privileged information)
    # 【critic 保留真实线速度】特权观测:评论家看到真相,但策略决策时看不到,
    # 这是"集中式/不对称评论家"训练的常规做法。
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel,
        scale=1.0,
    )

    # Determine gravity/accelerometer term name based on flag
    # 根据 USE_PROJECTED_GRAVITY 决定用哪个重力观测项名。
    gravity_term_name = "projected_gravity" if USE_PROJECTED_GRAVITY else "raw_accelerometer"

    # Replace projected_gravity with raw_accelerometer if flag is False
    # 【二选一】USE_PROJECTED_GRAVITY=False 时:删掉 projected_gravity,
    # 换成自写的 raw_accelerometer(原始加速度计读数,更接近真实 IMU
    # 但噪声大)。当前为 True,不触发。
    if not USE_PROJECTED_GRAVITY:
        # Remove projected_gravity and add raw_accelerometer
        del cfg.observations["actor"].terms["projected_gravity"]
        cfg.observations["actor"].terms["raw_accelerometer"] = ObservationTermCfg(
            func=microduck_mdp.raw_accelerometer,
            scale=1.0,
        )

    # 【deepcopy 理由】actor 和 critic 从模板拿到的是"同一个 term 对象"!
    # 后面我们各自改它们的延迟、噪声、函数 —— 必须先深拷贝拆开,
    # 否则改 actor 会污染 critic(下面 joint_pos/joint_vel 也同理)。
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )

    # 【传感器延迟建模】真实 IMU 读数有延迟。delay_max_lag=1(原来是 3,
    # 即最坏 60 ms):2026-07 审计发现真实 DXL IMU 路径很快,±20 ms 就够,
    # 3 的延迟会训练出"预判型"策略,部署时反而被打乱。
    # delay_update_period=64:延迟值每 64 步更新一次(抖动感)。
    #   注意:mujoco 的延迟单位是"控制步",1 步 = 1/50 Hz = 20 ms。
    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1  # was 3 (=60 ms worst case); real dxl IMU path is fast — ±20 ms envelope (2026-07 audit)
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64

    # 重力(姿态)观测同样加 IMU 延迟(同真实硬件路径)。
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1  # was 3 (=60 ms worst case); real dxl IMU path is fast — ±20 ms envelope (2026-07 audit)
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    # The critic's sensor-derived terms are the one obs path `nan_state` cannot
    # protect (it checks joint + root state; these read raycast/contact sensor
    # data, which MuJoCo can return non-finite for while the state is still
    # clean). A single NaN here kills the whole run via rsl_rl's check_nan —
    # that is the 2026-08-21 Velocity2-Rough-Backlash crash. Critic-only, so
    # sanitizing costs the policy nothing.
    # 【中文解释】critic 的传感器派生项是 nan_state 保护不到的路径:
    # nan_state 只查关节+机体状态,而这些项读射线/接触传感器 ——
    # MuJoCo 可能在状态仍然干净时就返回非有限值(一次 NaN 就会经
    # rsl_rl 的 check_nan 毁掉整个 run —— 2026-08-21 的真实崩溃)。
    # 把 critic 的这三个项换成 *_safe 版本(夹紧/清洗 NaN)。
    # 因为只改 critic,策略的观测完全不受影响(零代价保险)。
    for _term, _safe in (
        ("foot_contact_forces", microduck_mdp.foot_contact_forces_safe),
        ("foot_height", microduck_mdp.foot_height_safe),
        ("foot_air_time", microduck_mdp.foot_air_time_safe),
    ):
        if _term in cfg.observations["critic"].terms:
            cfg.observations["critic"].terms[_term].func = _safe

    # Observation noise configuration (edit these values as needed)
    # 【观测噪声】给策略的传感器读数加均匀噪声,模拟真实传感器误差。
    # 数值大幅调小(如 base_ang_vel 从 ±0.2 到 ±0.03):噪声太大会把
    # 有用的信号淹没,这版数值是贴近真实 DXL/IMU 误差的。
    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03) # was 0.2
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)  # was 0.15
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)  # was 0.05
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)  # was 2.0

    # IMU mounting-misalignment DR (per-env constant rotation of the IMU-derived
    # observations). Applied to the ACTOR only (the policy sees a slightly rotated
    # IMU frame, like a real mounting error); the critic keeps the true values.
    # 【中文解释】IMU 安装误差 DR(在观测层做!):把 IMU 派生的两个观测
    # 换成 misaligned 版本 —— per-env 常值旋转(最大 6°,随机轴)。
    # 只作用于 actor:策略看到"装歪的 IMU",评论家看到真值。
    # 记住(IMU_ORIENTATION_RANDOMIZATION_ANGLE 的注释):这是零均值
    # 随机轴,只能训练对"偏差大小"的鲁棒性;系统性的固定偏置由
    # 运行时校准解决,DR 帮不上。
    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        if USE_PROJECTED_GRAVITY:
            g = cfg.observations["actor"].terms[gravity_term_name]
            g.func = microduck_mdp.projected_gravity_imu_misaligned
            g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    # 1-ctrl-step lag on joint_vel: the Dynamixel firmware computes
    # present_velocity via a moving-average over the previous position-sample
    # window, so the value the policy actually reads is ~1 control period old.
    # Matches reality and stops the policy relying on instantaneous qdot feedback.
    # 【中文解释】joint_vel(关节速度)固定延迟 1 个控制步:真实 Dynamixel
    # 固件用"前一个位置采样窗口"的滑动平均计算 present_velocity,
    # 所以策略读到的速度大约早了一个控制周期。不建模这个延迟,
    # 策略会依赖"瞬间速度反馈",部署时被打乱。
    # delay_update_period=0:固定延迟(不抖动,确定性)。
    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    # Exclude passive_* joints (jaw linkage) from joint_pos/vel obs so the
    # observation dim matches the action dim (14) instead of the raw articulation (16).
    # Deepcopy each joint_pos/joint_vel term first — actor and critic share the
    # same term objects/params dicts from the base template, so mutating one would
    # leak into the other (e.g. the encoder-bias `biased` flag below).
    # 【中文解释】关键不变量:观测维度必须 = 动作维度 = 14(舵机数)。
    # 机器人总关节数 16(dof),多出来的是 2 个被动关节(名字以 passive_
    # 开头,如下颌连杆)。用正则 ^(?!passive_).* 只保留舵机关节。
    # 先 deepcopy 再改:模板里 actor 和 critic 共享同一 term 对象,
    # 直接改会互相泄漏(见下面 biased 标志的例子)。
    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    # Encoder-bias DR: the base template samples a per-env constant joint-encoder
    # offset (startup event "encoder_bias"), but joint_pos_rel ignores it unless
    # biased=True. Feed the biased joint pos to the ACTOR only (what the real
    # encoders report); the critic keeps the true joint pos (privileged).
    # 【中文解释】编码器偏置 DR:模板会给每个环境采样一个常值关节编码器
    # 偏移(startup 事件 encoder_bias),但观测函数只有在 biased=True 时
    # 才会把这个偏移加进去。这里:actor(策略)= 加偏的版本(和真机编码器
    # 一样"不准"),critic = 真值(特权)。如果 ENABLE_ENCODER_BIAS 关闭,
    # 就把整个事件删掉(不留死代码)。
    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # ----------------------------------------------------------
    # Commands(命令)— 策略的"遥控器输入"
    # 命令槽在观测里被拼成固定顺序:[twist(3), head_pose(4), body_pose(6)]
    # = 13 维命令块,这就是 61 维 = 48 + 13 的由来。
    # 关键教训(AGENTS.md):
    #   * 命令输入若永远为零 → 对应输入神经元的权重永远是死的。
    #     所以每个槽位从第 0 步就保持微小的非零采样范围;
    #   * 全零命令(部署时的待机态)必须显式训练(zero_command 类采样),
    #     均匀采样几乎永远不会产生全零;
    #   * 罕见但重要的命令区域需要显式分桶(见 rel_turn_in_place_envs)。
    # ----------------------------------------------------------
    # Commands — deepcopy to avoid shared-state corruption from other env cfgs
    # (make_velocity_env_cfg() returns objects with shared mutable references;
    # standup/ground_pick envs mutate commands["twist"] in place, zeroing ranges)
    # 【中文解释】必须先 deepcopy!make_velocity_env_cfg() 返回的对象带着
    # 共享可变引用;其他环境文件(standup、ground_pick)会就地修改
    # commands["twist"](比如把范围清零),import 顺序不同就会互相污染 ——
    # 这里拆开,保证本文件的命令配置独立。
    command: UniformVelocityCommandCfg = deepcopy(cfg.commands["twist"])
    cfg.commands["twist"] = command
    # 【站立环境比例】2% 的环境命令全零(站定)。从第 0 步就非零、
    # 再由 standing_envs 课程往上加 —— 保证"站定"行为从训练早期就在学,
    # 而且"全零命令"在经验数据里一定有出现(部署待机态!)。rel_heading_envs=0:
    # 不指定"朝向命令"(朝向由 twist 的 yaw 隐含)。
    command.rel_standing_envs = 0.02  # small but non-zero from the start, ramped up by curriculum
    command.rel_heading_envs = 0.0
    # Modest, FIXED command ranges (no widening curriculum): a ramp to
    # lin ±0.4 / ang ±2.0 outpaced the robot's capability and tracked a
    # post-iter-1000 reward/episode-length decline. ang ±1.0 is the big
    # change — it makes turning learnable.
    # 【中文解释】命令范围固定且克制(不开加宽课程):以前尝试过把
    # 角速度上限拉到 ±2.0,结果超过机器人能力,1000 迭代后奖励和
    # 回合长度一起下跌。改为固定 ±1.0 —— 这个改动让"转弯"变得可学。
    command.ranges.lin_vel_x = (-0.4, 0.4)
    command.ranges.lin_vel_y = (-0.3, 0.3)
    command.ranges.ang_vel_z = (-1.0, 1.0)
    # ── 命令速度上限（可选开关，2026-09-20）────────────────────────────────
    # MICRODUCK_CMD_LIN_MAX=<m/s>：把 lin_vel_x 两侧钳到该值，lin_vel_y 按同一
    # 比例缩放（保持方向分布形状），ang_vel_z 不动 —— 单一变量：只降线速度。
    # 为什么需要它：冻结 ck7000 的速度扫描（logs/probe_speed/speed_vs_flight.tsv）
    # 测到 0.10→0.20 m/s 之间有**相变**：每步偏航冲量 ×93、离地间隙 ×12、
    # 20 s 净转角 2°→114°；而 <0.10 时策略根本不走。所以"把命令压在相变以下"
    # 是一个从未在**正确配方族**（官方配方+头块）上试过的杠杆，也正是用户
    # 2026-09-18 提的"是不是速度太快导致的"。默认 1.0 = 不缩放 ⇒ 官方对齐不变
    # （tests/test_official_alignment.py 仍然通过）。
    _CMD_LIN_SCALE = float(_os.environ.get("MICRODUCK_CMD_LIN_MAX", "0") or 0.0)
    if _CMD_LIN_SCALE > 0.0:
        _lo, _hi = command.ranges.lin_vel_x
        _ratio = _CMD_LIN_SCALE / max(abs(_lo), abs(_hi))
        command.ranges.lin_vel_x = (-_CMD_LIN_SCALE, _CMD_LIN_SCALE)
        command.ranges.lin_vel_y = tuple(v * _ratio for v in command.ranges.lin_vel_y)
    command.viz.z_offset = 0.5
    # 【换成"只保留命令、没有扰动"的自定义命令类】VelocityCommandCommandOnlyCfg:
    # 不带 mjlab 模板的 speed_limits 噪声/扰动抽取,命令更"干净",
    # 与运行时对接更直接(vars(command) 把现有字段拷贝过去)。
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))
    # Explicit turn-in-place bucket (see TURN_IN_PLACE_FRACTION above).
    # 【原地转身显式分桶】15% 的环境强制采样"原地转"(lin=0 且 |ang| 大)。
    # 见顶部 TURN_IN_PLACE_FRACTION 的解释。
    cfg.commands["twist"].rel_turn_in_place_envs = TURN_IN_PLACE_FRACTION

    # Head pose command (4D deltas from HOME, in joint order:
    #   neck_pitch, head_pitch, head_yaw, head_roll). Tracked as a primary
    # reward — see "head_pose_tracking" added below. Initial ranges are small
    # non-zero so input neurons stay alive from step 0; curriculum widens them.
    # Per-joint final caps reflect each joint's mechanically reachable delta
    # from HOME (XML limits minus HOME offset, with ~10% safety margin):
    #   neck_pitch / head_pitch: ±1.10 rad (limit ±π/2 with HOME=±20°)
    #   head_yaw                : ±1.40 rad (limit ±π/2 with HOME=0)
    #   head_roll               : ±0.31 rad (limit ±20°)
    # Initial ranges are small non-zero so input neurons stay alive from step 0.
    # 【中文解释】头部姿态命令:4 维"离 HOME 的偏差"(关节顺序:
    # neck_pitch, head_pitch, head_yaw, head_roll),是走路任务的主目标之一
    # (真机上这个按钮让机器人转头/抬头看)。
    #   * 初始范围小但非零:保证输入神经元从第 0 步就有信号;
    #   * 最终上限 = 各关节从 HOME 出发的机械可达范围(XML 限位减去
    #     HOME 偏移,再留 ~10% 安全裕度):
    #       neck/head_pitch ±1.10 rad;head_yaw ±1.40;head_roll ±0.31;
    #   * 范围由 head_pose_range 课程 5 级放大(5%→100%)。
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=HEAD_POSE_CMD_RESAMPLE_S,
        zero_command_prob=POSE_ZERO_COMMAND_PROB,  # deployment sends all-zero
        ranges=(
            (-0.05, 0.05),    # neck_pitch
            (-0.05, 0.05),    # head_pitch
            (-0.07, 0.07),    # head_yaw
            (-0.015, 0.015),  # head_roll (tighter — much smaller mechanical range)
        ),
    )
    # Body pose command (6D delta from nominal standing: [x, y, z, roll, pitch, yaw]).
    # Vel env carries this slot for runtime obs-shape parity; tracked at a tiny
    # weight to keep the input neurons alive but not steer the policy. The
    # standup env raises the weight + widens the ranges.
    # 【中文解释】机体姿态命令:6 维"离名义站姿的偏差"[x, y, z, roll,
    # pitch, yaw]。走路任务只是**占住这个槽位**,保证观测形状与运行时
    # 一致(61 维合同);奖励权重是 0.1?不,是 0(见下面 body_pose_tracking),
    # 但保持微小的采样范围让神经元活着。真正大力使用这个槽位的是
    # standup 环境(它提高权重、放大范围)。
    cfg.commands["body_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=BODY_POSE_CMD_RESAMPLE_S,
        zero_command_prob=POSE_ZERO_COMMAND_PROB,  # deployment sends all-zero
        ranges=(
            (-0.005, 0.005),  # x (m)
            (-0.005, 0.005),  # y (m)
            (-0.005, 0.005),  # z (m)
            (-0.05, 0.05),    # roll (rad)
            (-0.05, 0.05),    # pitch (rad)
            (-0.05, 0.05),    # yaw (rad)
        ),
    )

    # Append head + body command obs terms to both policy and critic groups.
    # Order matters for the runtime obs layout: [twist(3), head_pose(4), body_pose(6)].
    # 【中文解释】把两个命令项追加进 actor 和 critic 的观测组。
    # 【顺序敏感!】运行时观测布局是固定的 [twist(3), head_pose(4),
    # body_pose(6)] —— 追加顺序就是拼接顺序,不能乱。actor 和 critic
    # 都要加(评论家也要知道命令才能评估"这个状态好不好")。
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "head_pose"},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "body_pose"},
        )
    # ----------------------------------------------------------
    # === 诊断指标（**不是奖励**，不改变策略/观测/动力学）===
    # 【为什么加】agent E（2026-09-18）：训练日志里**根本没有偏航 DC 这个量**，
    # 只能靠隔一段时间花 GPU 跑一次 P64 抽检；而现成的代理量更糟 ——
    # `Metrics/twist/error_vel_yaw` 被**步内偏航振荡**主导（稳态 3.3–4.0 rad/s
    # vs 指令上界 1.0），且在 night_queue 四臂里把排序**弄反**（该值最低的臂
    # 走直最差，rho = −0.32）。
    # 【关键】看 `yaw_dc_abs`（每 env 取绝对值再平均）：**带符号的平均会互相抵消** ——
    # 每个 env 的偏置符号是自发对称破缺（同一次训练 12 个走路检查点六正六负），
    # 所以均值读到 ~0，而每条腿 20 s 仍漂 78–118°。绝对值不抵消，才是与官方的差距。
    # 门控会把均值按"当前走直线的 env 比例"稀释（该比例由命令课程固定），
    # 所以**跨臂比较必须同迭代比**。
    cfg.metrics["yaw_dc_abs"] = MetricsTermCfg(
        func=microduck_mdp.yaw_wander_metric,
        params={
            "command_name": "twist",
            "wz_command_threshold": YAW_SYMMETRY_WZ_CMD_THRESHOLD,
            "command_threshold": 0.05,
            "tau_s": YAW_SYMMETRY_TAU_S,
            "signed": False,
        },
    )
    cfg.metrics["yaw_dc_signed"] = MetricsTermCfg(
        func=microduck_mdp.yaw_wander_metric,
        params={
            "command_name": "twist",
            "wz_command_threshold": YAW_SYMMETRY_WZ_CMD_THRESHOLD,
            "command_threshold": 0.05,
            "tau_s": YAW_SYMMETRY_TAU_S,
            "signed": True,
        },
    )


    # === Pose tracking rewards ===
    # head_pose: primary objective in vel env — the whole point of the rewrite.
    # std=0.5 with per-joint Gaussian (see head_pose_tracking in mdp.py): at the
    # full ±1.0 rad command, a non-tracking policy still sees per-joint reward
    # exp(-(1/0.5)²)=exp(-4)≈0.018 — a small but non-zero gradient — so the
    # curriculum widening doesn't kill the signal. Final reward is the mean
    # over 4 joints, so partial tracking is partial reward (no all-or-nothing).
    # 【中文解释】头部姿态跟踪奖励(主目标之一,权重 2.0):
    #   按关节独立高斯,std=0.5。注意这个设计的精妙之处:命令拉满
    #   (±1.0 rad)时,一个完全不跟的策略每关节仍得到 exp(-4)≈0.018
    #   —— 微小但非零的梯度,所以课程放大范围的过程不会杀掉信号;
    #   最终奖励取 4 个关节的均值 → 跟一半就是一半奖励(不是全有全无)。
    cfg.rewards["head_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.head_pose_tracking,
        weight=2.0,
        params={"command_name": "head_pose", "std": 0.5},
    )
    # body_pose: infra kept intact but DISABLED (weight 0) — the obs slot and
    # command stay alive for envs that raise the weight (standup).
    # 【中文解释】机体姿态跟踪奖励:基础设施完整保留但权重=0(禁用)——
    # 因为走路任务不追求躯干精确位姿(要的是跟住速度命令),而 standup
    # 环境会把这个权重提上来。nominal_height=0.095:名义站姿躯干高度,
    # 用于 z 维高斯。
    cfg.rewards["body_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.body_pose_tracking_6d,
        weight=0.0,
        params={
            "command_name": "body_pose",
            "nominal_height": 0.095,
            "xy_std": 0.05,
            "z_std": 0.02,
            "angle_std": math.radians(15),
        },
    )

    # Head droop fix (2026-08-20). The head walks pitched ~15° down (measured:
    # run ww1g2198 head_pose_tracking 1.544/2.0 → 14.6° mean joint error).
    # DO NOT fix this by tightening head_pose_tracking's std: run 5yay13u4 tried
    # fine_std=0.1 and the policy stopped walking entirely by iter 300 (air_time
    # 1.01 → 0.02, peak foot height 15 mm → 2 mm, entropy collapsed 10.9 → 1.9).
    # An instantaneous tight tolerance taxes walking 0.77/step — 76% of the whole
    # air_time reward — and is UNESCAPABLE, since a 280 g head (38% of robot
    # mass) must oscillate while stepping. Standing still scored higher, so it
    # stood still.
    # The DC bias, unlike the oscillation, IS escapable (bias the neck command up
    # to cancel gravity sag), so price only that: L1 on a 1 s EMA of the error.
    # At the optimum this costs a walking policy nothing.
    # 【中文解释】"头下垂"修正(2026-08-20 的教训,非常经典):
    #   实测策略走路时头垂 ~15°(run ww1g2198 平均关节误差 14.6°)。
    #   ⚠️ 绝对不要用"收紧 head_pose_tracking 的 std"修:试过 fine_std=0.1,
    #   run 5yay13u4 在 300 迭代内整个步态崩了(air_time 1.01→0.02,
    #   脚峰值高度 15mm→2mm,熵 10.9→1.9)。为什么:瞬时紧容差相当于
    #   每步收走路税 0.77 = air_time 奖励的 76%,而且**不可逃脱** ——
    #   280 g 的头(占机体 38%)在迈步中**必须**来回摆。站着不动得分
    #   更高 → 策略选择了站。
    #   但是"直流偏置"(头一直垂着)和"摆动"不同:它可以逃 —— 把
    #   脖子命令向上偏,抵消重力下垂。所以只惩罚 DC:对 1 秒 EMA 的
    #   误差做 L1。最优情况下,走路策略一分钱都不用付。
    #   【这就是 AGENTS.md 里"只惩罚可逃脱部分"的例子。】
    cfg.rewards["head_pose_bias"] = RewardTermCfg(
        func=microduck_mdp.head_pose_bias_penalty,
        weight=0.0,  # ramped by the head_pose_bias_weight curriculum below
        params={"command_name": "head_pose", "tau_s": 1.0},
    )
    # ----------------------------------------------------------
    # 【头部稳定性层】(2026-09-17,用户要求;官方配方之外的唯一增项)
    # 见文件顶部 ENABLE_HEAD_STABILITY 的长注释与 mdp.head_droop_penalty。
    # 两项**都从 0 权重起步**,由 head_stability_weight 课程按官方
    # head_pose_bias 的同一条时间线爬上来(600 迭代前免费,1500 到满权)。
    # ----------------------------------------------------------
    if ENABLE_HEAD_STABILITY:
        # 头"乱动":颈部关节速度超过 deadband 的部分,平方和。官方步态免费,
        # 步频锁相的抽动计费。关节选择在 mdp 内部(脖子/头是 5..8 号关节)。
        cfg.rewards["head_motion"] = RewardTermCfg(
            func=microduck_mdp.neck_joint_vel_deadband_l2,
            weight=0.0,  # ramped by the head_stability_weight curriculum
            params={"deadband": HEAD_MOTION_DEADBAND_RAD_S},
        )
        # 头下垂:单侧、相对命令的 pitch L1(返回 ≤0 ⇒ **正权重**)。
        cfg.rewards["head_droop"] = RewardTermCfg(
            func=microduck_mdp.head_droop_penalty,
            weight=0.0,  # ramped by the head_stability_weight curriculum
            params={
                "command_name": "head_pose",
                "deadband": HEAD_DROOP_DEADBAND,
            },
        )
        # 头要一直向前:yaw/roll 双侧 L1,按命令门控(命令要转头时免费)。
        cfg.rewards["head_forward"] = RewardTermCfg(
            func=microduck_mdp.head_forward_penalty,
            weight=0.0,  # ramped by the head_forward_weight curriculum
            params={
                "command_name": "head_pose",
                "deadband_yaw": HEAD_FORWARD_DEADBAND_YAW,
                "deadband_roll": HEAD_FORWARD_DEADBAND_ROLL,
                "command_gate": HEAD_FORWARD_COMMAND_GATE,
            },
        )


    # ----------------------------------------------------------
    # Terrain(地形):工厂参数 rough 决定平地 or 粗糙地形。
    # ----------------------------------------------------------
    # Terrain
    if not rough:
        # 平地:"plane" 类型,不生成地形。
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
    else:
        # 粗糙地形:用上面 MICRODUCK_ROUGH_TERRAINS_CFG 生成。
        cfg.scene.terrain.terrain_type = "generator"
        cfg.scene.terrain.terrain_generator = MICRODUCK_ROUGH_TERRAINS_CFG

        # Soften terrain box contacts: adjacent boxes at different heights create
        # hard edges that destabilise the contact solver and produce NaN forces.
        # 【粗糙地形专属设置 1:软化接触】见 _soften_terrain_contacts。
        cfg.scene.spec_fn = _soften_terrain_contacts

        # The velocity env default nconmax=35 is tight for rough terrain: when the
        # robot falls and multiple body links hit multiple boxes simultaneously,
        # contacts overflow → some are silently dropped → sudden decompression → NaN.
        # 【粗糙地形专属设置 2:接触上限】模板 nconmax=35 太紧:粗糙地形
        # 摔倒时多个连杆同时撞多个盒子 → 接触溢出 → 部分接触被静默丢弃
        # → 突然解压 → NaN。放到 200。
        cfg.sim.nconmax = 200   # was 35

        # The velocity env uses only 10 solver iterations (vs the default 100),
        # which is too few to resolve edge contacts on rough box terrain.
        # Tripling iterations significantly reduces contact resolution failures
        # with a modest compute cost on GPU (MJWarp parallelises across envs).
        # 【粗糙地形专属设置 3:求解迭代】模板只有 10 次(默认 100),
        # 不足以解出方块地形边缘接触。×3 到 30 显著减少接触求解失败,
        # GPU 上 MJWarp 按环境并行,代价温和。
        cfg.sim.mujoco.iterations = 30    # was 10
        cfg.sim.mujoco.ls_iterations = 50  # was 20

        # play 模式:关地形课程、缩小地形规模(人眼只看几秒,不用看满难度谱)。
        if play:
            cfg.scene.terrain.terrain_generator.curriculum = False
            cfg.scene.terrain.terrain_generator.num_cols = 5
            cfg.scene.terrain.terrain_generator.num_rows = 5

    # ----------------------------------------------------------
    # Curriculum(课程)= 训练过程中的"进度表"。
    # 每个课程项在指定 step(环境步,不是迭代数!)改写某个参数。
    # 换算:step = iteration × NUM_STEPS_PER_ENV(24)。
    # 核心原则(AGENTS.md):
    #   * 阶段必须和策略实际学到的能力对齐 —— 技能还没出现就上难度,
    #     会让"什么都不做"成为最优;
    #   * wandb 指标在课程阶段边界处**下降**,说明节奏错了(阶段太早/太长),
    #     只往回调,绝不提前;
    #   * 改参数要走 managers(env.event_manager.get_term_cfg),
    #     别直接写 env.cfg —— 但配置侧(本文件)用 CurriculumTermCfg 声明,
    #     运行时由 manager 按 step 应用。
    # ----------------------------------------------------------
    # action_rate weight ramp: gentle smoothing while the gait bootstraps, then
    # tighten to -1.0 by iter 1500.
    # 【课程 1:动作平滑权重爬坡】步态启动期保持温和(-0.1),
    # 到 1500 迭代收到 -1.0(每 250 迭代一档)。
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.1},
                {"step": 500 * NUM_STEPS_PER_ENV, "weight": -0.2},
                {"step": 750 * NUM_STEPS_PER_ENV, "weight": -0.4},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -0.6},
                {"step": 1250 * NUM_STEPS_PER_ENV, "weight": -0.8},
                {"step": 1500 * NUM_STEPS_PER_ENV, "weight": -1.0},
            ],
        },
    )

    # Gradually increase standing env fraction after walking is established
    # 【课程 2:站立环境比例】步态建立后再逐步提高"站定"环境的占比
    # (2% → 25%,到 2000 迭代)。这是"全零命令"行为(部署待机态)的
    # 显式训练 —— 见命令部分的注释。
    cfg.curriculum["standing_envs"] = CurriculumTermCfg(
        func=microduck_mdp.standing_envs_curriculum,
        params={
            "command_name": "twist",
            "standing_stages": [
                {"step": 0,           "rel_standing_envs": 0.02},
                {"step": 500 * 24,    "rel_standing_envs": 0.05},
                {"step": 750 * 24,    "rel_standing_envs": 0.1},
                {"step": 1000 * 24,   "rel_standing_envs": 0.15},
                {"step": 1500 * 24,   "rel_standing_envs": 0.2},
                {"step": 2000 * 24,   "rel_standing_envs": 0.25},
            ],
        },
    )

    # NOTE: no velocity-command-range curriculum — ranges are fixed (see the
    # command section above).
    # 【注意】没有速度命令范围课程 —— 范围是固定的(见命令区注释)。

    # Head pose command range curriculum — per-joint, scaled to each joint's
    # reachable delta from HOME (with ~10% margin from XML limits). Same 5-stage
    # shape as before (5% → 15% → 35% → 65% → 100% of each joint's final cap).
    # neck/head pitch final ±1.10 rad, head_yaw ±1.40, head_roll ±0.31.
    # 【课程 3:头部命令范围】5 级放大(最终上限的 5%→15%→35%→65%→100%)。
    # 为什么不是一开始就全范围:命令范围太大而策略还在学走路,头部目标
    # 会抢走路任务的学习资源;5 阶段曲线与策略能力对齐。
    cfg.curriculum["head_pose_range"] = CurriculumTermCfg(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "head_pose",
            "range_stages": [
                # step,                ranges = ((neck_pitch), (head_pitch), (head_yaw),  (head_roll))
                {"step": 0,         "ranges": ((-0.05, 0.05),  (-0.05, 0.05),  (-0.07, 0.07),  (-0.015, 0.015))},
                {"step": 500 * 24,  "ranges": ((-0.17, 0.17),  (-0.17, 0.17),  (-0.21, 0.21),  (-0.047, 0.047))},
                {"step": 1000 * 24, "ranges": ((-0.39, 0.39),  (-0.39, 0.39),  (-0.49, 0.49),  (-0.11, 0.11))},
                {"step": 1500 * 24, "ranges": ((-0.72, 0.72),  (-0.72, 0.72),  (-0.91, 0.91),  (-0.20, 0.20))},
                {"step": 2000 * 24, "ranges": ((-1.10, 1.10),  (-1.10, 1.10),  (-1.40, 1.40),  (-0.31, 0.31))},
            ],
        },
    )

    # Body pose command range curriculum: stay small in vel env. Standup env
    # overrides this curriculum with wide ranges + heavy reward weight.
    # 【课程 4:机体命令范围】走路任务里保持小范围(只有第 0 阶段)。
    # standup 环境会整体覆盖这个课程(大范围 + 高权重)。
    cfg.curriculum["body_pose_range"] = CurriculumTermCfg(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "body_pose",
            "range_stages": [
                {"step": 0, "ranges": (
                    (-0.005, 0.005),  # x (m)
                    (-0.005, 0.005),  # y (m)
                    (-0.005, 0.005),  # z (m)
                    (-0.05, 0.05),    # roll
                    (-0.05, 0.05),    # pitch
                    (-0.05, 0.05),    # yaw
                )},
            ],
        },
    )

    # CoM randomization range curriculum - start small, ramp up
    # 【课程 5:质心随机化范围爬坡(仅当 ENABLE_COM_RANDOMIZATION)】
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    # Capped at ±15 mm (2026-07 audit): the previous ramp to ±30 mm
                    # exceeded the foot support polygon (heel is only 20 mm behind
                    # the ankle) — the randomized CoM could sit entirely outside
                    # support, forcing a wide/fast hyper-reactive gait and making
                    # BACKWARD balance untrainable. Regression timeline matched the
                    # ramp increases: 0.015 → 0.02 → 0.03 as policies got worse.
                    # 【为什么封顶 ±15 mm】曾加到 ±30 mm:超过了脚的支撑
                    # 多边形(脚跟只在脚踝后 20 mm)—— 随机化质心可能整个
                    # 落在支撑面外,逼出又宽又快的超反应步态,而且"后向
                    # 平衡"直接不可训练。回归时间线与爬坡完全吻合:
                    # 0.015→0.02→0.03,策略越训越差。
                    {"step": 0,          "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24,  "range": 0.01},
                    {"step": 1500 * 24,  "range": 0.015},
                ],
            },
        )

    # Head CoM randomization range curriculum - start small, ramp up
    # 【课程 6:头部质心随机化爬坡(仅当 ENABLE_HEAD_COM_RANDOMIZATION)】
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    # Capped at ±10 mm (2026-07 audit — same over-conservatism
                    # concern as trunk CoM; head is a large lever arm).
                    # 【封顶 ±10 mm】理由同躯干:头部是巨大的杠杆臂。
                    {"step": 0,          "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24,  "range": 0.01},
                ],
            },
        )

    # Disable default curriculum
    # 【删除模板自带课程】terrain_levels(地形难度课程)只在粗糙地形下保留:
    # 平地没有地形可分级;command_vel(速度命令加宽课程)统一删掉 ——
    # 本任务命令范围是固定的(见命令区)。
    if not rough:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # head_pose_bias ramp: OFF until iter 600, then 1.0 → 3.0 by iter 1500.
    # Held at 0 early because a posture-precision term is a distraction before
    # a gait exists. At weight 3.0 a 15° residual bias costs 0.79/step and a
    # 2° bias costs 0.10/step.
    # 【课程 7:头下垂惩罚权重】第 600 迭代前为 0(步态还没出现时,
    # 姿态精度项只是干扰),然后 1.0→2.0→3.0 爬升。
    # 权重 3.0 时:残余 15° 下垂每步罚 0.79,2° 只罚 0.10 ——
    # 量级设计成"大偏差贵、小偏差便宜",策略有梯度可走。
    cfg.curriculum["head_pose_bias_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "head_pose_bias",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 600 * NUM_STEPS_PER_ENV, "weight": 1.0},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": 2.0},
                {"step": 1500 * NUM_STEPS_PER_ENV, "weight": 3.0},
            ],
        },
    )
    # 【课程 7b:头部稳定性权重】(2026-09-17)与 head_pose_bias **同一条时间线** ——
    # 600 迭代前为 0,1500 迭代到满权。理由见文件顶部 ENABLE_HEAD_STABILITY。
    # 用 reward_weight(阶跃函数)而不是 staged_reward_weight_ramp:两项的
    # deadband 已经把"不可避免的部分"排除掉了,不需要再线性淡入,
    # 而且与官方 head_pose_bias 的节奏保持一致更好归因。
    if ENABLE_HEAD_STABILITY:
        cfg.curriculum["head_stability_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "head_motion",
                "weight_stages": [
                    {"step": 0, "weight": 0.0},
                    {"step": HEAD_STABILITY_RAMP_ITER_START * NUM_STEPS_PER_ENV,
                     "weight": 0.25 * HEAD_MOTION_WEIGHT},
                    {"step": HEAD_STABILITY_RAMP_ITER_MID * NUM_STEPS_PER_ENV,
                     "weight": 0.50 * HEAD_MOTION_WEIGHT},
                    {"step": HEAD_STABILITY_RAMP_ITER_END * NUM_STEPS_PER_ENV,
                     "weight": HEAD_MOTION_WEIGHT},
                ],
            },
        )
        # head_droop 是**惩罚**(返回 ≤0 ⇒ 正权重),它没有理由等到 600 迭代 ——
        # 但按 AGENTS.md,任何"在上税"都要等技能出现,所以同样分三段爬坡。
        cfg.curriculum["head_droop_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "head_droop",
                "weight_stages": [
                    {"step": 0, "weight": 0.0},
                    {"step": HEAD_STABILITY_RAMP_ITER_START * NUM_STEPS_PER_ENV,
                     "weight": 0.25 * HEAD_DROOP_WEIGHT},
                    {"step": HEAD_STABILITY_RAMP_ITER_MID * NUM_STEPS_PER_ENV,
                     "weight": 0.50 * HEAD_DROOP_WEIGHT},
                    {"step": HEAD_STABILITY_RAMP_ITER_END * NUM_STEPS_PER_ENV,
                     "weight": HEAD_DROOP_WEIGHT},
                ],
            },
        )
        # 头要一直向前:同一时间线、同一形状。
        cfg.curriculum["head_forward_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "head_forward",
                "weight_stages": [
                    {"step": 0, "weight": 0.0},
                    {"step": HEAD_STABILITY_RAMP_ITER_START * NUM_STEPS_PER_ENV,
                     "weight": 0.25 * HEAD_FORWARD_WEIGHT},
                    {"step": HEAD_STABILITY_RAMP_ITER_MID * NUM_STEPS_PER_ENV,
                     "weight": 0.50 * HEAD_FORWARD_WEIGHT},
                    {"step": HEAD_STABILITY_RAMP_ITER_END * NUM_STEPS_PER_ENV,
                     "weight": HEAD_FORWARD_WEIGHT},
                ],
            },
        )


    # Posture / actuator-hygiene taxes (2026-09-11) fade in over the first
    # STANCE_TAX_RAMP_ITERS iterations. Held at 0 at step 0 so the gait is not
    # taxed while it is still being discovered (AGENTS.md), then ramped
    # linearly so there is no weight cliff for the policy to trip over.
    # 【课程 8:姿态卫生税】0 → 目标权重,在 300 迭代内线性爬坡。
    if ENABLE_STANCE_HYGIENE:
        cfg.curriculum["stance_tax_weight"] = CurriculumTermCfg(
            func=microduck_mdp.staged_reward_weight_ramp,
            params={
                "ramp_steps": STANCE_TAX_RAMP_ITERS * NUM_STEPS_PER_ENV,
                "stages": [
                    # 只在开关打开时才存在；staged_reward_weight_ramp 对不存在的
                    # 项会报错，所以这里必须条件化。
                    *([{"reward_name": "stance_hip_roll",
                        "weight": STANCE_HIP_ROLL_L1_WEIGHT}]
                      if ENABLE_STANCE_HIP_ROLL else []),
                    {"reward_name": "stance_hip_yaw", "weight": STANCE_HIP_YAW_L1_WEIGHT},
                    {"reward_name": "action_over_limit", "weight": ACTION_OVER_LIMIT_WEIGHT},
                    {"reward_name": "neck_motion", "weight": NECK_MOTION_WEIGHT},
                    {"reward_name": "standing_leg_vel",
                     "weight": STANDING_LEG_VEL_L2_WEIGHT},
                    {"reward_name": "stillness_zero_cmd",
                     "weight": STILLNESS_AT_ZERO_CMD_WEIGHT},
                    {"reward_name": "no_stepping", "weight": NO_STEPPING_WEIGHT},
                    {"reward_name": "yaw_hold_zero_cmd",
                     "weight": YAW_HOLD_AT_ZERO_CMD_WEIGHT},
                ],
            },
        )

    # 【课程:零命令三件套的爬坡】(2026-09-21 统一模型)只在 ZERO_CMD_TERMS 单开
    # （= 不带 stance hygiene）时注册，避免和上面的 stance_tax_weight 撞名/撞项。
    # 同样的理由：技能还在摸索时上税会让"什么都不做"成为最优解，所以 0 → 目标权重
    # 在 STANCE_TAX_RAMP_ITERS 内线性爬坡。
    if ENABLE_ZERO_CMD_TERMS and not ENABLE_STANCE_HYGIENE:
        cfg.curriculum["zero_cmd_tax_weight"] = CurriculumTermCfg(
            func=microduck_mdp.staged_reward_weight_ramp,
            params={
                "ramp_steps": STANCE_TAX_RAMP_ITERS * NUM_STEPS_PER_ENV,
                "stages": [
                    {"reward_name": "stillness_zero_cmd",
                     "weight": STILLNESS_AT_ZERO_CMD_WEIGHT},
                    {"reward_name": "no_stepping", "weight": NO_STEPPING_WEIGHT},
                    {"reward_name": "yaw_hold_zero_cmd",
                     "weight": YAW_HOLD_AT_ZERO_CMD_WEIGHT},
                ],
            },
        )

    # 【课程 9:直行保持】(2026-09-14)0 → HEADING_HOLD_WEIGHT,分两级台阶。
    # 为什么不在 step 0 就给:本项是正奖励,而"站着不动"天然完美满足它。
    # 虽然它和 track_linear_velocity 对冲后不会把策略推向静止,但在步态
    # 还没出现的早期,满权重的航向项会削弱"迈步"的动机(AGENTS.md 的
    # "技能还在摸索时上目标会让什么都不做变成最优解")。
    # 用 reward_weight 的分段阶跃 —— 它是阶跃函数不是插值(AGENTS.md)。
    if ENABLE_HEADING_HOLD:
        cfg.curriculum["heading_hold_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "heading_hold",
                # 4 级台阶而不是 2 级:reward_weight 是阶跃函数(不是插值),
                # 2 级会在 iter 300 处产生 0 → 0.75 的悬崖。AGENTS.md 记录过
                # "wandb 指标正好在课程边界处下台阶 = 课程节奏错了"。
                "weight_stages": [
                    {"step": 0, "weight": 0.0},
                    {"step": HEADING_HOLD_RAMP_ITER_START * NUM_STEPS_PER_ENV,
                     "weight": 0.23 * HEADING_HOLD_WEIGHT},
                    {"step": int(0.47 * HEADING_HOLD_RAMP_ITER_END) * NUM_STEPS_PER_ENV,
                     "weight": 0.50 * HEADING_HOLD_WEIGHT},
                    {"step": int(0.73 * HEADING_HOLD_RAMP_ITER_END) * NUM_STEPS_PER_ENV,
                     "weight": 0.77 * HEADING_HOLD_WEIGHT},
                    {"step": HEADING_HOLD_RAMP_ITER_END * NUM_STEPS_PER_ENV,
                     "weight": HEADING_HOLD_WEIGHT},
                ],
            },
        )

    # 【课程 10:步态对称】同上,晚一点引入(它属于"尝试税",早期上会让
    # 策略连不对称的步态都不敢试 —— 而早期恰恰需要先迈出腿来)。
    if ENABLE_GAIT_SYMMETRY:
        cfg.curriculum["gait_symmetry_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "gait_symmetry",
                "weight_stages": [
                    {"step": 0, "weight": 0.0},
                    {"step": GAIT_SYMMETRY_RAMP_ITER_START * NUM_STEPS_PER_ENV,
                     "weight": 0.5 * GAIT_SYMMETRY_WEIGHT},
                    {"step": GAIT_SYMMETRY_RAMP_ITER_END * NUM_STEPS_PER_ENV,
                     "weight": GAIT_SYMMETRY_WEIGHT},
                ],
            },
        )

    # 【课程 11:净偏航率】(2026-09-16)0 → 目标权重,300 迭代线性爬坡。
    # 与 stance_tax_weight 同样的理由:技能还在摸索时上税会让"什么都不做"成为最优解。
    if ENABLE_YAW_SYMMETRY:
        cfg.curriculum["yaw_dc_weight"] = CurriculumTermCfg(
            func=microduck_mdp.staged_reward_weight_ramp,
            params={
                "ramp_steps": YAW_SYMMETRY_RAMP_ITERS * NUM_STEPS_PER_ENV,
                "stages": [
                    {"reward_name": "yaw_dc", "weight": YAW_SYMMETRY_WEIGHT},
                ],
            },
        )

    # 【课程 12:转向优先】(2026-09-16)见常量区注释。前 1200 迭代只发转弯指令,
    # 让"用落足位置产生偏航"这个技能在没有前进步态可依赖时先长出来。
    if ENABLE_TURN_FIRST_CURRICULUM:
        cfg.curriculum["turn_first"] = CurriculumTermCfg(
            func=microduck_mdp.velocity_command_range_curriculum,
            params={
                "command_name": "twist",
                "stages": [
                    {"step": 0,
                     "lin_vel_x": (-TURN_FIRST_TINY_LIN, TURN_FIRST_TINY_LIN),
                     "lin_vel_y": (-TURN_FIRST_TINY_LIN, TURN_FIRST_TINY_LIN),
                     "ang_vel_z": (-1.0, 1.0)},
                    {"step": TURN_FIRST_STAGE1_ITERS * NUM_STEPS_PER_ENV,
                     "lin_vel_x": (-0.20, 0.20), "lin_vel_y": (-0.15, 0.15),
                     "ang_vel_z": (-1.0, 1.0)},
                    {"step": TURN_FIRST_STAGE2_ITERS * NUM_STEPS_PER_ENV,
                     "lin_vel_x": (-0.40, 0.40), "lin_vel_y": (-0.30, 0.30),
                     "ang_vel_z": (-1.0, 1.0)},
                ],
            },
        )

    return cfg


# ============================================================
# 【PPO 训练参数】MicroduckRlCfg —— 只含算法/网络/日志配置,
# 与上面的环境配置(env cfg)分离。任务注册时把两者绑在一起
# (见 tasks/__init__.py 的 register_mjlab_task)。
# ============================================================
MicroduckRlCfg = RslRlOnPolicyRunnerCfg(
    # ---- Actor(策略网络)----
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),   # 3 层 MLP,宽度递减(输入 61D → 输出 14D 动作)
        activation="elu",
        obs_normalization=True,        # ★ 观测归一化开着 → 导出 ONNX 时必须
                                       # 用 scripts/export.py 把归一化器烘焙进图!
                                       # (否则部署时策略看到未归一化输入,直接失灵)
        distribution_cfg={
            # 高斯分布动作(连续控制的标准选择)。
            "class_name": "GaussianDistribution",
            "init_std": 1.0,  # ★ 官方值。2026-09-11 改回 1.0(见下方说明)
            # 【中文解释】初始探索标准差。
            # 2026-09-04 曾为 HLS 舵机降到 0.5(理由:HLS 的双限幅把大动作
            # 卡死 → 初始 1.0 的标准差大部分打到饱和 → "乱蹬" → 学不动)。
            # 2026-09-11 改回官方的 1.0,理由是**对齐官方**:
            #   * `git diff -w origin/develop HEAD -- src/mjlab_microduck/tasks/`
            #     显示这是我们相对官方**唯一**的功能性偏离(其余全是注释);
            #   * 官方策略(init_std=1.0)在同一套配方下走出了完全正确的步态
            #     (躯干 119.8 mm、外张 +0.37°、颈部摆动 0.75°),
            #     而我们的 0.5 那一轮走进了深蹲/外八字/甩头的坏盆地;
            #   * 0.5 的原始理由(HLS 双限幅)**已经随换舵机失效** ——
            #     现在是 HD-1910,而且 `action_over_limit` 会惩罚野过头的目标。
            # 探索不足 → 早期就收敛到一个"能拿分但难看"的局部最优,正是
            # Agent C 观察到的"从 iter 2500 起 pose/头部跟踪就平了"。
            "std_type": "scalar",       # 所有动作共享一个标量标准差(简单、稳)
        },
    ),
    # ---- Critic(价值网络)----
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),   # 同样结构(但输入含特权观测,见观测区)
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        # ---- PPO 超参数 ----
        value_loss_coef=1.0,            # 价值损失系数
        use_clipped_value_loss=True,    # 价值损失也做裁剪(PPO 的标准变体)
        clip_param=0.2,                 # PPO 裁剪范围(标准值 0.2)
        entropy_coef=0.01,              # 熵正则:鼓励探索,防止过早收敛到确定性
        num_learning_epochs=5,          # 每次采集后更新轮数
        num_mini_batches=4,             # mini-batch 切分
        learning_rate=1.0e-3,
        schedule="adaptive",            # 自适应学习率(按 KL 调整,配 desired_kl)
        gamma=0.99,                     # 折扣因子(50Hz → 0.99 ≈ 0.2s 视野)
        lam=0.95,                       # GAE λ(偏差-方差权衡)
        desired_kl=0.01,                # 目标 KL:每次更新不能偏离旧策略太多
        max_grad_norm=1.0,              # 梯度裁剪(防爆炸)
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
        # 【对称性】ENABLE_SYMMETRY=True 时注入左右镜像损失(61D 表在
        # symmetry.py);当前 False = 不用(速度任务的非对称性不强,
        # 镜像反而可能限制技能)。
    ),
    # ---- 日志与运行参数 ----
    wandb_project="mjlab_microduck",    # wandb 项目名
    experiment_name="velocity",  # Directory name  # 本地日志目录 logs/velocity/
    run_name="velocity",  # Appended to datetime in wandb: <datetime>_velocity
    save_interval=250,                  # 每 250 迭代存一次 checkpoint(model_XXXX.pt)
    num_steps_per_env=24,               # 每迭代每环境采集 24 步(与上方 NUM_STEPS_PER_ENV 一致)
    max_iterations=50_000,              # 训练上限(预算参考:走路类任务 4000–6000 迭代)
)


def configure_openmd_locomotion(
    cfg: ManagerBasedRlEnvCfg, *, final_weights: bool = False
) -> None:
    """Add OpenMD head stabilization and straight-walking diagnostics."""
    for metric_name, signed in (("yaw_dc_abs", False), ("yaw_dc_signed", True)):
        cfg.metrics[metric_name] = MetricsTermCfg(
            func=microduck_mdp.yaw_wander_metric,
            params={
                "command_name": "twist",
                "wz_command_threshold": 0.15,
                "command_threshold": 0.05,
                "tau_s": 0.8,
                "signed": signed,
            },
        )

    # A warm-started VelStand inherits the final walking stage at step zero.
    stages = ((0, 1.0),) if final_weights else (
        (0, 0.0), (600, 0.25), (1000, 0.5), (1500, 1.0)
    )
    for name, schedule, function, params, weight in (
        ("head_motion", "head_stability_weight", microduck_mdp.neck_joint_vel_deadband_l2,
         {"deadband": 2.0}, -0.08),
        ("head_droop", "head_droop_weight", microduck_mdp.head_droop_penalty,
         {"command_name": "head_pose", "deadband": 0.15}, 2.0),
        ("head_forward", "head_forward_weight", microduck_mdp.head_forward_penalty,
         {"command_name": "head_pose", "deadband_yaw": 0.12,
          "deadband_roll": 0.10, "command_gate": 0.05}, 2.0),
    ):
        cfg.rewards[name] = RewardTermCfg(
            func=function, weight=weight if final_weights else 0.0, params=params
        )
        cfg.curriculum[schedule] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": name,
                "weight_stages": [
                    {"step": iteration * NUM_STEPS_PER_ENV, "weight": factor * weight}
                    for iteration, factor in stages
                ],
            },
        )
