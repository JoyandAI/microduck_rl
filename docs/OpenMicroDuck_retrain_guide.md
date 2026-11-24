# OpenMD 任务、物理模型与公开策略

使用 HD‑1910 舵机时，先按动作选择完整任务 ID，再核对物理模型。首次安装与训练可按 [AutoDL 起立训练指南](autodl_standup_retrain_guide.md)操作。资产校验、奖励调参、曲线判读、跨仿真验证和上机步骤见 [OpenMicroDuck PPO 训练教程](OpenMicroDuck_PPO训练教程.md)。

## 训练哪一个任务

下表列出 14 个基础任务。每个任务都在注册表中关联训练环境、回放环境和 PPO 配置。`Flat` 和 `Rough` 选择地形，`Backlash` 选择带被动齿隙关节的机器人。使用表中的完整 ID，同一动作的不同变体可能共用实验目录。[任务注册](../src/mjlab_microduck/tasks/__init__.py) · [OpenMD 环境工厂](../src/mjlab_microduck/tasks/openmd_env_cfg.py)

| 行为 | 任务 ID | 机器人 XML |
|---|---|---|
| 平地速度跟踪 | `Mjlab-Velocity-Flat-OpenMD` | `robot_walk.xml` |
| 平地速度跟踪，计入齿隙 | `Mjlab-Velocity-Flat-Backlash-OpenMD` | `robot_walk_backlash.xml` |
| 起立并站稳 | `Mjlab-StandUp-Flat-OpenMD` | `robot_stand.xml` |
| 行走与跌倒恢复，平地 | `Mjlab-VelStand-Flat-OpenMD` | `robot_allcollisions.xml` |
| 行走与跌倒恢复，崎岖地形 | `Mjlab-VelStand-Rough-OpenMD` | `robot_allcollisions.xml` |
| 行走与跌倒恢复，平地、带齿隙 | `Mjlab-VelStand-Flat-Backlash-OpenMD` | `robot_allcollisions_backlash.xml` |
| 行走与跌倒恢复，崎岖地形、带齿隙 | `Mjlab-VelStand-Rough-Backlash-OpenMD` | `robot_allcollisions_backlash.xml` |
| 按指令坐下或站起，平地 | `Mjlab-SitStand-Flat-OpenMD` | `robot_stand.xml` |
| 按指令坐下或站起，崎岖地形 | `Mjlab-SitStand-Rough-OpenMD` | `robot_stand.xml` |
| 低头用喙触地并起身，平地 | `Mjlab-GroundPick-Flat-OpenMD` | `robot_stand.xml`，工厂添加 `mouth_tip` 定位点 |
| 低头用喙触地并起身，崎岖地形 | `Mjlab-GroundPick-Rough-OpenMD` | `robot_stand.xml`，工厂添加 `mouth_tip` 定位点 |
| 右脚踢球 | `Mjlab-BallKick-Flat-OpenMD-Right` | `robot_stand.xml`，环境另加球 |
| 左脚踢球 | `Mjlab-BallKick-Flat-OpenMD-Left` | `robot_stand.xml`，环境另加球 |
| 前滚翻并落地 | `Mjlab-Roulade-Flat-OpenMD` | `robot_stand.xml` |

走路模型用脚底接触地面。行走与跌倒恢复的联合任务需要躯干及其他部件也能接触地面。训练环境按任务构造机器人，加载同目录的网格、HOME 和执行器配置。安装后运行 `list-envs | grep -F OpenMD` 核对任务，运行 `python scripts/check_openmd_training.py` 核对执行器和物理资产。[机器人定义与 HOME](../src/mjlab_microduck/robot/openmd_constants.py) · [XML 与网格](../src/mjlab_microduck/robot/openmd/) · [HD‑1910 m6 标定](../vendor/bam/bam/params/hd1910/m6.json)

## 实机部署参考

OpenMicroDuck 公开策略包含以下六份 ONNX。用任务表选择训练目标，用部署索引中的 `models.<槽位>.onnx_sha256` 核对下载的文件。例如，在包含权重的目录执行 `sha256sum OpenMD_walk.onnx` 即可得到完整哈希。[公开策略文件](https://github.com/JoyandAI/OpenMicroDuck/tree/main/policies) · [部署参数和权重索引](https://github.com/JoyandAI/OpenMicroDuck/blob/main/policies/OpenMD_policy_settings_20261028.json)

| 槽位 | 公开权重 | 用途与训练目标 |
|---|---|---|
| `walk` | `OpenMD_walk.onnx` | 非零速度指令时行走。参考权重为 m9999，训练目标对应平地、带齿隙的 Velocity。 |
| `stand` | `OpenMD_stand.onnx` | 参考权重为编号 17500 的 VelStand 检查点所导出的策略。公开配置在零速度指令时切换到此槽。 |
| `sitstand` | `OpenMD_sitstand.onnx` | 按指令坐下或站起，对应 SitStand。 |
| `ground_pick` | `OpenMD_ground_pick.onnx` | 俯身触地并回起，对应 GroundPick。公开权重的踝部倍率已包含在计算图中。 |
| `kick_right` | `OpenMD_kick_right.onnx` | 右踢。也可以从注册的左踢任务训练自己的策略，公开配置的 `kick_left` 槽位为空。 |
| `roulade` | `OpenMD_roulade.onnx` | 前滚翻，对应 Roulade。 |

起立入门任务训练从坐姿等初态起身和保持站立。若目标是复核公开的站立槽，请加载表中的 VelStand 权重，并与配套部署参数一起回放。配置的 `home_deg` 按名称包含嘴部关节，策略只输出 14 维腿部和头颈动作。按关节名建立对应关系。训练中的 BAM 舵机增益、运行时的动作倍率和滤波是不同参数。[公开部署说明](https://github.com/JoyandAI/OpenMicroDuck/blob/main/policies/README.md)

基础行走和起立可以从零训练。VelStand 默认启用专家行为克隆，通过 W&B 获取起立教师 `model_9750.pt` 和行走教师 `model_3750.pt`。使用已有教师时，在行为克隆配置中分别设置本地 `checkpoint_path` 和 `anchor_checkpoint_path`。复训参考策略 VelStand 17500 需另行匹配当次的初始化检查点、两个教师、观测顺序与课程起点。[教师加载与默认配置](../src/mjlab_microduck/tasks/distill.py)

继续 PPO 训练需要与任务相容的 `.pt` 检查点。公开部署 ONNX 可作动作对照。保存每次训练的完整任务 ID、随机种子、实验目录中的 `params/env.yaml`、`params/agent.yaml`、选中检查点和代码版本，便于复核候选策略。
