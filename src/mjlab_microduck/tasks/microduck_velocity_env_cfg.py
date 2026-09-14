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
ENABLE_SYMMETRY = False

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
        r".*hip_roll.*": 0.05,  # 0.1→0.06→0.05 — hold the 5°-inward stance (sole sits flat), stop leg splay
        r".*hip_pitch.*": 0.15,
        r".*knee.*": 0.15,
        r".*ankle.*": 0.1,
    }
    # 【中文解释】hip_roll 定为 0.05 是历史调参结果(0.1→0.06→0.05):
    # 保持 5° 内八站姿(脚掌能完全贴平地面),阻止腿向外劈开。

    std_walking = {
        # Lower body
        r".*hip_yaw.*": 0.3,
        r".*hip_roll.*": 0.05,  # 0.1→0.06→0.05 — hold the 5°-inward stance, stop the leg splay to vertical
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
            "init_std": 0.5,  # was 1.0 (xl330); HLS 双限幅下大动作→饱和乱蹬→学不动 (2026-09-04 诊断)
            # 【中文解释】初始探索标准差。从 1.0 降到 0.5(2026-09-04 诊断):
            # 动作范围由 HLS 双限幅卡住,初始 1.0 的标准差产生的动作大部分
            # 会打到饱和限位 → 输出全是"乱蹬" → 学不动。0.5 起步让早期
            # 探索更接近有效动作区间。
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
