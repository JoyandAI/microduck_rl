# OpenMicroDuck PPO 训练教程

本教程用飞特 HD‑1910 舵机的 OpenMicroDuck 训练平地行走。任务名中的 `OpenMD` 指这套机器人的仿真配置。使用配套的训练 XML 和网格，可从第 2 节开始，在云服务器上完成 PPO 训练、回放和 ONNX 导出。更换打印件、轴承或脚底后，先按第 1 节验收物理资产，再按第 7、8 节评估已有策略。

`m5`、`m6` 指 BAM 执行器模型。走路参考策略 m9999 使用 HD‑1910 m6 和带关节间隙的仿真模型。下文示范用这个任务训练新策略。要复现某个指定检查点，还需当次实际使用的源码、参数、初始化方式和运行记录。[动作与任务对照](OpenMicroDuck_retrain_guide.md)

## 1. 从 CAD 交付到可训练的物理模型

### 机械同事需要交什么

| 文件或数据 | 到手后怎样验收 |
|---|---|
| 同一装配版本导出的 URDF、MJCF，以及它们引用的全部文件 | 打包完整目录，保留 XML 的 include、网格路径和网格缩放。URDF 用来核对连杆树、关节和惯性。本项目的训练工厂消费 MJCF。在另一目录解压后，仍能打开同一台机器人。 |
| 姿态与地面场景 | 原生成器读取机器人源文件 `openmicroduck.xml`、场景 `openmicroduck_flat_scene.xml` 及相邻的 `meshes/stl/`。场景需要 `INIT`、`STAND`、`SIT`、`FOLD`。沿用 14 主动关节时，源关键帧有 21 个 qpos（基座位置与四元数共 7 个，加 14 个关节）。姿态已包含在单份 MJCF 时，可从中整理出生成器需要的场景输入。 |
| 新版质量属性和实物记录 | 每个运动连杆的质量、质心、惯量、坐标系及单位。已正确写入 XML 的数据可以直接读取。附装电池后的实测总质量、轴承位置与数量、打印填充或可称重部件的实测质量。钢轴承和打印件按装配归到实际承载的连杆。 |
| 改过的关节和接触面 | 记录零位、轴方向、活动范围、反向空行程、脚底厚度/轮廓，以及 IMU 和电池安装位置。若舵机、供电或传动也改了，附型号、供电电压和一段带时间戳的目标角/实测角往返记录。 |

**训练已有机型时直接使用配套的 MJCF、网格和执行器参数。** 从新装配导出时，URDF 和 MJCF 可以作为主体，还要交齐外部引用、姿态和实测质量。需重新生成资产时，从训练工程根目录运行生成器。配套配置从 `cad/openmd/ros_description/` 读取 `openmicroduck.xml`、`openmicroduck_flat_scene.xml` 和 `meshes/stl/`。[模型生成器](../src/mjlab_microduck/robot/openmd/build_models.py) · [走路模型输入配置](../src/mjlab_microduck/robot/openmd/config_mjcf_walk.json)

**给导出人员的顺序。** 打开与实机一致的最终装配 → 按哪些零件会相对运动划分连杆，把固定连接的轴承座、螺丝等计入相应连杆 → 定义关节父子关系、轴线、正方向和零位 → 给实际材质或实测质量 → 用同一装配版本导出 URDF、MJCF 和网格。保留各网格的局部原点、方向和缩放。一份整机合并 STL 会丢掉各关节的相对运动。源 URDF 的 `<inertial>`、`<joint>`、`<limit>`、`<collision>` 应能逐项对应到 MJCF 的 body/inertial、joint、限位和碰撞 geom。

在装有 URDF 工具的机械或 ROS 工作站，运行 `check_urdf 路径/机器人.urdf`，检查从根连杆到两脚及头部的父子树。若交的是 xacro，先按交付方的配置展开为 URDF。再在能解析 `package://` 的查看器中载入交付包，逐关节转动，核对同名网格、尺度和方向。树解析成功后仍要打开碰撞显示，并按下节独立编译 MJCF，两边的几何和惯性可能在转换时发生变化。

### 怎样核对动力学与几何

1. **逐连杆对表。** URDF 和 MJCF 的坐标可不同，先变换到同一连杆坐标系、同一质心再比质量与惯量。用 kg、m、kg·m²。注意 CAD 的 kg·mm² 换为 kg·m² 需乘 10⁻⁶。检查主惯量为正且满足三角不等式。给打印件填材料后，仍要用实际装配称重交叉核对。整机质量相同，脚踝处多出的质量依然会改变摆腿惯量。[MuJoCo 刚体惯性定义](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-inertial)
2. **从基座到足底查运动学。** 在 HOME 看足底高度、双脚支撑面、左右姿态。逐个给小正角位移，确认轴向、左右符号、单位和编码器零位。在 SIT、FOLD 和极限附近查连杆穿插。机械硬限位、训练的软限位和板端保护分别登记。配套生成器采用 0.12 m 的初始躯干高度，并将 IMU site 放在躯干惯性中心。替换结构时，请用实物的站姿高度和 IMU 安装位重新核对这两项。
3. **把舵机和轴承放到正确的模型层。** 连杆惯量写入刚体，折算到驱动轴的转子惯量由 BAM 的 armature 注入。驱动关节的摩擦、黏性阻尼和力矩由 BAM 路径处理。记录真实供电和电流限制，将最终执行器的力矩上限、可用力矩随速度的变化、反馈增益和延迟与同电压下的规格或实测对照。XML 的裸 position 执行器限幅与经过 BAM 构建的模型需分别查。再用改动关节的往返实测校验回差与摩擦，避免在 XML 和 BAM 中重复加入同一份转子惯量或摩擦。配套走路模型使用峰峰值 2° 的回差。传动机构或轴承改变后，按实测值设定。[机器人执行器配置](../src/mjlab_microduck/robot/microduck_constants.py) · [BAM 实现](../vendor/bam/bam/mjlab.py) · [回差配置](../src/mjlab_microduck/robot/openmd/config_mjcf_walk_backlash.json)
4. **分开视觉网格和碰撞几何。** 配套走路模型使用左右脚底网格碰撞。为新的平底脚建模时，可先用与实物同宽、同长、同厚的薄 Box，圆弧或突出部位再用少量简单几何近似。保持实际支撑边缘、接地高度与摩擦。若脚底本身是曲面，就对照网格版测滚动支点和滑移。坐站、踢球、翻滚和跌倒恢复还需检查小腿、躯干、头壳等任务相关接触。生成器按网格名识别左右脚，改为 Box/Capsule 时要同步适配选择逻辑，保留 `left_foot_collision`、`right_foot_collision` 和足部 site。[碰撞生成逻辑](../src/mjlab_microduck/robot/openmd/build_models.py) · [任务实际加载的机器人](../src/mjlab_microduck/robot/openmd_constants.py)
5. **验最终运行模型。** 为修改过的资产准备独立的工程副本和空输出网格目录，保留原模型用于对照。生成器向自身目录写输出，并在目标已有 STL 时跳过网格复制。将新资产接入机器人工厂，分别编译源 MJCF 和任务工厂最终使用的模型。做静置、下落、双脚支撑和侧躺试验，观察哪些几何体接触、接触力及穿透深度。最后用同一份 ONNX 和固定指令比较结构变动前后的表现。

装完第 3 节的环境后，可用下面的命令读出**交付场景**实际编译得到的质量属性和关键帧。示例读取配套走路场景。验收新资产时换成对应文件的实际路径。惯量是质心主轴坐标下的三主值，`iquat` 给出该坐标系在连杆系中的方向。板端轴向、摩擦和触地仍需上述实物/运行检查。

```bash
python - src/mjlab_microduck/robot/openmd/scene_walk_backlash.xml <<'PY'
import sys
import mujoco as mj
m = mj.MjModel.from_xml_path(sys.argv[1])
print("whole scene kg", m.body_mass.sum(), "nq/nv/nu", m.nq, m.nv, m.nu)
print("keyframes", [m.key(i).name for i in range(m.nkey)])
for i in range(1, m.nbody):
    print(m.body(i).name, "kg", m.body_mass[i], "CoM", m.body_ipos[i],
          "principal inertia", m.body_inertia[i], "iquat", m.body_iquat[i])
PY
```

称重对照时用只含机器人和静态地面的场景。若场景含球或道具，单独统计机器人子树。

**训练跌倒恢复前，再检查运行时接触。** 从工程根目录运行 `python scripts/audit_openmd_collision_runtime_20261023.py --json outputs/openmd_contacts.json`。VelStand 的环境会按名称筛选碰撞体。有些 CAD 导出器把头壳、下颌和躯干的碰撞副本命名为 `.001`；直接加载这样的 XML，运行时可能只保留脚底和舵机壳。配套加载器会把这些几何体映射到任务选择的名称。查看报告中两个 `allcollisions` 模型的 `current_loader`，本套资产的有效碰撞体应各为 32 个，`lost_on_entity_build` 应为空，`floor_probe` 应记录头壳接触地面且法向力大于零。普通走路和带间隙走路使用左右两块足底碰撞。该静态探针用于验收模型，恢复成功率要在策略回放中另测。[运行时碰撞检查](../scripts/audit_openmd_collision_runtime_20261023.py)

## 2. 用 m6 训练 HD‑1910，以及怎样对照 m5

参考走路策略 m9999 的物理配置是 `motor_name="hd1910"`、`model="m6"`、`kp_fw=5`，训练供电范围 7.1～7.9 V。`hd1910/m6.json` 写有 `actuator="sts3215"`，它指定这份辨识参数配套的电压控制律。保留该字段以及 `error_gain_ratio`、力矩、负载摩擦等整套参数。m5 的这份文件使用 `actuator="hd1910"`，含单独的 `kd`。训练侧的 `kp_fw=5` 属于该仿真拟合，不能作为指令写回板端寄存器。[m6 参数](../vendor/bam/bam/params/hd1910/m6.json) · [m5 参数](../vendor/bam/bam/params/hd1910/m5.json) · [配置入口](../src/mjlab_microduck/robot/microduck_constants.py)

**配套任务已选用 m6，先运行第 3 节的预检即可确认实际生效的模型。** 单独替换 JSON 文件名、照抄 m5 的 kd 或更改板端增益，会同时改变执行器语义。已训练的 m5 ONNX 仍然是原来的策略。用 m6 仿真播放它，可以检验策略面对不同动力学的表现。要得到 m6 训练的策略，需要用 m6 配置训练并重新导出。

| 想回答的问题 | 实验做法 | 记录结果 |
|---|---|---|
| 旧步态对执行器模型有多敏感？ | 冻结同一个 m5 ONNX、同一机器人场景、HOME、动作倍率、指令和种子，分别用匹配各自控制律的 m5、m6 执行器评估。打印实际加载的配置及 JSON 哈希。 | 固定时长存活率、实际前进速度、偏航、滑步、关节跟踪误差。通用 CPU 脚本的 `--model m5` 不能切换其正常 BAM 路径。 |
| 用 m6 重新训练是否更好？ | A 用可追溯的原始 m5 配方，B 用 m6 配方。结构、PPO、奖励、命令分布、训练预算和配对种子保持一致。两组均从零起步，留多个检查点。若原 m5 源码或标定未知，标记为新建 m5 对照。 | 两组都在相同目标条件下评估，跨种子确认，再在同一台机器人、同一组板端参数下比较。验证 m5/m6 后再单独改动机器人结构，并保留原场景作为对照。 |

## 3. 云端环境与第一次检查

从新实例开始，请先完成 [Linux GPU 实例、源码与 Python 环境准备](autodl_standup_retrain_guide.md#1-准备实例和源码)和[依赖安装与预检](autodl_standup_retrain_guide.md#2-安装并确认训练环境)。这两节给出持久卷、NVIDIA 驱动、Python 3.12、HTTPS 包源和实际 wheel 下载的检查方法。起立和行走共用同一个环境。跟随本教程时，把安装命令里的任务换成下列带回差行走任务，其余安装和验证步骤相同。

```bash
bash scripts/autodl_setup.sh --task Mjlab-Velocity-Flat-Backlash-OpenMD
```

脚本从 `uv.lock` 生成版本约束。配套清单固定 torch 2.9.1、warp-lang 1.12.0、mujoco 3.10.0、mujoco-warp 3.8.1 和 mjlab 1.3.0。完成 editable 安装、`uv pip check`、关键包导入和 CUDA 运算检查后，运行 `python scripts/check_openmd_training.py`。看到 `HD-1910 m6 OK`，并确认 `Mjlab-Velocity-Flat-Backlash-OpenMD` 对应 `get_openmd_walk_backlash_spec`，再开始行走训练。预检也会核对当前源码、BAM 导入位置及实际 XML/网格。缺少模型时检查是否取得完整训练工程；任务未出现时检查虚拟环境与 editable 安装位置。之后在同一个已激活环境中使用 `train`、`play` 和导出脚本。[训练工程配置](../pyproject.toml) · [依赖清单](../requirements-autodl.txt) · [预检源码](../scripts/check_openmd_training.py)

## 4. 找代码、选任务、改奖励

| 要做的事 | 读或改哪里 |
|---|---|
| 选择训练入口、任务、地形 | `pyproject.toml` 注册 mjlab 插件。[任务表](../src/mjlab_microduck/tasks/__init__.py)绑定训练工厂、play 工厂、PPO 配置和 runner。 |
| 找到训练实际使用的机器人 | [OpenMD 环境工厂](../src/mjlab_microduck/tasks/openmd_env_cfg.py)选择机器人，[机器人配置](../src/mjlab_microduck/robot/openmd_constants.py)选择 HOME、XML 和碰撞族。执行器来自上一节列出的 BAM 入口。 |
| 了解每步奖励与观测 | [Velocity 配方](../src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py)设置命令范围、奖励权重、课程、随机化和 PPO 参数。[MDP 模块](../src/mjlab_microduck/tasks/mdp.py)实现自定义奖励和事件。[Backlash 包装](../src/mjlab_microduck/tasks/backlash.py)改变间隙模型的编码器观测。 |
| 切换动作 | [SitStand](../src/mjlab_microduck/tasks/microduck_sitstand_env_cfg.py)用第一个 twist 槽的 0/1 表示站/坐。[VelStand](../src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py)联合训练行走与恢复。其余任务与所用 XML 见[对应表](OpenMicroDuck_retrain_guide.md)。 |

`Velocity` 表示速度跟踪任务，`Flat` 表示平地。普通平地走路用 `Mjlab-Velocity-Flat-OpenMD`，计入传动间隙用 `Mjlab-Velocity-Flat-Backlash-OpenMD`。联合恢复用 `Mjlab-VelStand-Flat-OpenMD`，起伏地形用 `Mjlab-VelStand-Rough-OpenMD`。从 `list-envs` 选择完整任务名。基础走路可从零训练。VelStand 默认启用专家行为克隆，启动前应确认能读取起立和行走两个 PPO 教师。默认检查点和本地路径字段见[教师配置](../src/mjlab_microduck/tasks/distill.py)。

若目标是复训参考策略 VelStand 17500，还需匹配当次训练的配方、walk m9999 初始化检查点，以及当次的行走和 StandUp 教师。部署的 SitStand 14999 用于指令坐站。详见[任务与公开策略对照](OpenMicroDuck_retrain_guide.md)。

### 第一次用默认奖励，第二次只验证一个假设

PPO 的 actor 输出动作，critic 估计后续累计回报。环境决定“一步动作值多少分”。先用默认配置留一条可重复的基线。基础走路配方的线速度、角速度跟踪权重各 2.0，动作变化平方惩罚 `action_rate_l2` 从 −0.1 逐级加至 −1.0。m9999 的初始学习率 0.001、adaptive 调度、目标 KL 0.01、clip 0.2、每批 5 个 epoch／4 个 mini-batch。改变动作目标时，先固定这些优化参数。

| 观察到的现象 | 从哪里下手 | 怎样发现策略钻了空子 |
|---|---|---|
| 前进时拖脚 | 先查 `foot_height_scan` 在新足底的零点。现有 `foot_clearance` 用脚底离地高度误差乘足部水平速度，默认目标 0.02 m、指令阈值 0.01。核对传感器后再调高度或权重。 | 同时量实际前进速度、滑移、脚尖净空。若机器人站住后代价下降，需要一起检查速度跟踪。 |
| 步态缺乏左右交替 | 先检查已有接触、腾空时间和滑移项。若确需步态相位约束，用左右脚触地/离地事件测量交替。仅在有移动命令且实际发生移动时给分。 | 测直行、转弯和零指令三种工况。只奖励交替次数容易产生原地踏步。部署仍使用 61 维观测时，可先把相位放在奖励里。若把时钟加入 actor，须连同板端推理接口一起设计。 |
| 零指令站立晃动 | 先查训练是否实际采到全零命令。现有 `joint_vel_l2_when_standing` 可约束零指令下的腿速。对姿态恢复可另量倾角、双脚支撑和保持时间。 | 并测从静止起步、受扰恢复和零指令停稳。过重的稳态惩罚可能连第一步或起身尝试也一起压掉。若运行时在零指令时由 walk 切到 stand 槽，须分别回放两个槽位策略，再测切换瞬间。 |

例如试验“降低目标抬脚高度能否减轻过度抬腿”，先在新实验的环境工厂继承同一走路配置，仅把 `cfg.rewards["foot_clearance"].params["target_height"]` 从 0.02 调为一个有测量依据的值。A/B 用同一批种子、同一地面、指令和预算，比较摆动脚净空、滑移和速度误差。若某项受课程控制，要修改对应阶段表。只改初始权重，阶段切换时会被覆盖。走路课程里 `step = PPO 更新次数 × 24`。

新增自定义奖励时，将计算函数放在 `tasks/mdp.py`，返回每个并行环境一个标量。用 `RewardTermCfg(func=..., weight=..., params=...)` 挂到目标任务工厂。先用指定状态检查“做对”和“做错”的带权分值。原始量是正代价时用负权重。函数直接返回负代价时用正权重。注册独立任务时同时提供 train/play 工厂和独立的 `experiment_name`。用实际模型验证关节选择、权重符号、阈值两侧的值，再跑 64 环境短训。带间隙模型按名字取主动或编码器关节，避免把被动关节当成第 15 个动作。

右踢和翻滚可直接沿用各自的配套奖励。

- 右踢任务让球出现在右脚前方，奖励向前的球速，并鼓励左支撑脚保持接触。分别记录踢球命中率、球前进速度、支撑脚是否离地和踢后是否站稳。Actor 的输入不含球位置，测试时要固定或记录球与脚的起始相对位置。[右踢任务](../src/mjlab_microduck/tasks/microduck_ball_kick_env_cfg.py)
- 翻滚任务分别计算翻转进度和完成后的落地姿态。调节落地奖励前，先看早期检查点能否完成翻转，再统计着地后的倾角和能否恢复站姿。保持同一组初始姿态比较候选。[翻滚任务](../src/mjlab_microduck/tasks/microduck_roulade_env_cfg.py)

## 5. 无头并行训练与 TensorBoard

下面在同一已激活的 Bash 会话中运行配套的默认走路任务。示例分别保存短训和正式训练的日志。训练默认无头，无需启动 viewer 或录视频。这个命令可训练新的 m6 步态，同一任务 ID 下的不同训练配置和种子会得到不同的策略。

```bash
TASK=Mjlab-Velocity-Flat-Backlash-OpenMD
EXP=openmd_velocity
TAG=$(date +%Y%m%d_%H%M%S)
train "$TASK" --env.scene.num-envs 64 --agent.max-iterations 5 \
  --agent.seed 42 --agent.logger tensorboard --agent.run-name "smoke_m6_$TAG"
```

确认日志能更新、奖励和损失有限、`nan_state` 为零且生成 PT，再开正式训练。下面 4,096 环境、10,000 次更新是走路示例预算。每次每环境采 24 步，共 98,304 条状态转移。有显存不足时调小环境数，记录实际吞吐、每次更新时间和每次采样量。可在 tmux 会话启动以便断线重连。

```bash
train "$TASK" --env.scene.num-envs 4096 --agent.max-iterations 10000 \
  --agent.seed 42 --agent.save-interval 250 --agent.logger tensorboard \
  --agent.run-name "walk_m6_$TAG"
```

在另一云端终端进入相同仓库并激活 venv，运行 `tensorboard --logdir logs/rsl_rl/openmd_velocity --host 127.0.0.1 --port 6006`。本机通过 SSH 转发端口，例如 `ssh -N -L 6006:127.0.0.1:6006 -p 云端SSH端口 用户名@云端主机`，浏览器打开 `http://127.0.0.1:6006`。在 Scalars 选本轮 `walk_m6_...`，横轴用 Step。

| 一起看的曲线 | 怎么判断 |
|---|---|
| `Episode_Termination/nan_state`、`fell_over`、`time_out` 和 `Train/mean_episode_length` | 先排数值错误，然后区分跌倒与正常到时。走路单回合 20 秒，在 50 Hz 下上限约 1,000 控制步。终止项是聚合计数，百分比需用评估中实际倒下数除以试验数。 |
| `Train/mean_reward`、`Episode_Reward/track_linear_velocity`、`Metrics/twist/error_vel_xy` | 总回报、带权跟踪贡献、速度误差要合看。把新权重打开时的跳变与课程表对齐，带权分项的绝对值不能直接当物理误差。 |
| `Curriculum/action_rate_weight`、`Episode_Reward/action_rate_l2` | 同时看惩罚权重和分项变化。动作更平滑时仍要查有没有牺牲步幅、抗扰或转向。 |
| `Loss/value`、`Loss/surrogate`、`Policy/mean_std`、`Perf/total_fps` | 看持续不稳定、探索是否过早变窄以及采样吞吐。候选策略达到要求要靠控制任务指标和回放判断。 |

以走路策略 m9999 为例，分别取第 4900～5099、9800～9999 次更新的 200 个记录点。平均总回报从 117.90 升到 123.28，平面速度误差却从 0.421 变为 0.423。此时应让中、后期候选执行同一组速度、转向和受扰试验。连续几个保存周期已经满足预先约定的指标，而继续训练只提高总分时，可以结束本轮。若后期某项退化，就保留表现更好的较早检查点。

## 6. 回放指定检查点，导出独立策略

回到训练终端，沿用同一 `TASK`、`EXP`、`TAG`。从本次输出中选一个真实存在的 PT，下面示例取第 5000 号，再用同样指令对照 9999。若通过 viewer 手动发指令，同时记录指令、试验环境数和初始条件。

```bash
RUN=$(find "logs/rsl_rl/$EXP" -mindepth 1 -maxdepth 1 -type d -name "*_walk_m6_$TAG" -print -quit)
ls -v "${RUN:?未找到本轮目录}"/model_*.pt | tail
CK=5000
PT="$RUN/model_$CK.pt"
play "$TASK" --checkpoint-file "$PT" --num-envs 1 --viewer viser
```

用可视化观察零指令、持续前进、定点转向、一次可重复的侧向扰动和解除指令后的停稳。云端无图形桌面时用 Viser；桌面复核时，把选中的 `.pt` 和对应任务名复制到装有同版本训练环境、能打开 OpenGL 窗口的工作站。设置 `PT` 为桌面上的实际文件路径。下面的命令使用一个 CPU 环境和原生 MuJoCo 窗口，加载本教程的带回差行走任务：

```bash
TASK=Mjlab-Velocity-Flat-Backlash-OpenMD
play "$TASK" --checkpoint-file "${PT:?请先设置桌面检查点路径}" --viewer native --device cpu --num-envs 1
```

在 Windows 桌面上，按 [Windows PowerShell 回放](autodl_standup_retrain_guide.md#windows-powershell-回放)运行对应命令。回放普通 OpenMD 行走检查点时，沿用它训练时的 `Mjlab-Velocity-Flat-OpenMD`。只比较显示方式时保持 PT、任务、设备与环境数一致；网页入口可用 `python scripts/play_web.py "$TASK" --checkpoint-file "$PT" --device cpu --num-envs 1`，默认端口 8081，可用 `PLAY_VISER_PORT` 调整。两种 viewer 都由 mjlab 推进同一类训练环境。对照时记录指令、初始条件和实际仿真时间；跨仿真器检查见第 7 节。[网页回放入口](../scripts/play_web.py)

**回放用 PT，部署用包含观测归一化的 ONNX。** 用 Ctrl+C 关闭 viewer 后，在同一终端运行下面的导出和校验命令。导出器将策略写入指定文件。

```bash
mkdir -p exports
OUT="exports/walk_m6_${TAG}_${CK}.onnx"
python scripts/export.py "$TASK" --checkpoint-file "$PT" --num-envs 1 --onnx-file "$OUT"
python scripts/check_openmd_training.py --onnx "$OUT"
sha256sum "$PT" "$OUT"
```

检查脚本用零观测和固定扰动探针验证单输入 `[1,61]`、单输出 `[1,14]`、值有限并打印哈希。导出器把训练时的观测归一化嵌入网络。需要逐项验证导出数值时，固定同一批原始 61 维观测，以推理模式分别送入带归一化的 PyTorch actor 和 ONNX，对照各输出关节并记录最大绝对误差。保存对应配置、PT、ONNX 和校验记录。训练 PT 另含优化器等状态，继续 PPO 时保留它。[导出入口](../scripts/export.py) · [导出实现](../src/mjlab_microduck/export.py)

要把多个行走候选放在相同条件下比较，可在训练侧运行下面的固定指令测试。在配有 CUDA 的服务器上运行。仅有 CPU 时显式传 `--device cpu` 并先调小环境数，验证流程。换另一个 ONNX 时沿用同一组种子，查看 JSON 中各环境的实际测量时长、前进速度和航向。另做带扰动的测试时记录施加时间、方向和幅度。该直行评估器移除了推扰事件。[走路评估脚本](../scripts/eval_straightness.py)

```bash
python scripts/eval_straightness.py --task "$TASK" --onnx "$OUT" \
  --lin-vel-x 0.2 --ang-vel-z 0 --freeze-pose-cmd --seed 42 \
  --num-envs 64 --seconds 10 --json "exports/walk_m6_${TAG}_${CK}_seed42.json"
```

## 7. 在 Sim2Sim 中验证输入映射与动作

这里要对比的是 **训练侧 mjlab/MuJoCo Warp** 与**独立推进的 CPU MuJoCo 或另一目标仿真器**。同一 PT 在两个 viewer 中动作相似，确认的是回放入口。把同一 ONNX 放入另一个物理后端，才会检验控制循环、执行器和传感器实现是否一致。

标准 OpenMD 起立策略可通过任务名选择配套的独立场景，从俯卧、仰卧或坐姿复核跌倒恢复。行走、带齿隙或 VelStand 权重应按各自的训练任务配置。[OpenMD 起立 ONNX 回放命令](autodl_standup_retrain_guide.md#在独立-cpu-mujoco-中复核-openmd-起立)

1. **锁住两端输入。** 保存同一 ONNX 的 SHA‑256、对应的 XML/碰撞族、m6 参数与控制律、HOME 和实际动作顺序。匹配物理步长、50 Hz 控制、动作倍率、延迟/滤波和指令。先固定初态、种子及标称物理，关闭或固定噪声/随机化，确保比较的是同一条件。
2. **在同一状态逐项对数。** 两端分别设定已知基座倾角、非零关节位置/速度、上一动作和非零命令，分别输出 61 维 actor 输入、14 维原始动作、缩放后的目标角。61 维包含 48 维本体观测及 13 维指令（速度 3、头部 4、身体 6）。检查姿态/角速度坐标系、重力方向和关节顺序。带回差时还要单独给一个被动轴施加位移和速度，核对编码器读数与舵机位置反馈。
3. **明确现有工具的边界。** 通用 `scripts/infer_policy.py` 可独立运行 CPU MuJoCo，但其默认场景、HOME 和间隙反馈需逐项对齐。可将主动轴固定在 HOME、速度清零，再将左踝的被动轴单独设为 +0.010 rad 和 +0.10 rad/s。比较训练端和 CPU 端的编码器变化。如果 CPU 端仍读到零，先补齐间隙映射再测闭环。切换场景和 `--new-cmd-obs` 只覆盖部分输入配置。[通用回放器](../scripts/infer_policy.py) · [间隙编码器](../src/mjlab_microduck/actuator/friction_dr_bam.py)
4. **通过接口对数后再跑闭环。** 两端跑同样的“静止→直行→转向→停步”和固定时刻的扰动，记录实际存活时长、前进速度、航向漂移、足端滑移和跌倒。SitStand 分别测坐下和起身。VelStand 分直立、前后倒、侧倒。踢球和翻滚要把球、接触面与结束条件带上。比较多组相同种子。很快跌倒的短轨迹也可能显得很直。初始动作就分叉先查观测和动作映射，若初始一致而逐步分叉，再查接触、执行器、延迟和积分设置。

## 8. 在 Sim2Real 中加载候选策略并分步验收

| 阶段 | 操作与放行依据 |
|---|---|
| 建立参照 | 拍下结构版本和实机 HOME，记录 robotd 版本、原槽位、ONNX 哈希、控制频率、动作倍率、滤波、切换条件。保存旧配置和权重，候选使用新文件名。先确认停机、卸力和恢复旧配置的方法。 |
| 核对接口 | 核对单位、14 个主动关节次序（左腿 5 → 头颈 4 → 右腿 5）、IMU 方向、61 维命令语义。区分训练倍率、板端 walk 倍率与 stand 倍率。从目标机器配置确认 walk 和 stand 各自倍率，以及零指令是否触发槽位切换。记录每个槽位实际生效的参数。 |
| 加载与低风险检查 | 按该镜像实际支持的加载入口选择目标槽位。使用 TOML 的镜像应核对 `/etc/robot/robotd.toml` 的 `[policy]` 路径。服务重新加载后用启动日志和哈希确认实际模型。先在支撑下核对 HOME、零指令、极小指令的关节方向与异常振动，确保可随时卸力。 |
| 分项落地 | 在平整、防滑、有保护的区域从小幅前进和停止开始，再做转向、坐站和恢复，最后测试踢球、翻滚。记录指令、动作视频或遥测、跌倒和异常发热。出现持续顶限、明显反向、异常冲击或连续跌倒时停测并恢复已验证配置，带着该段观测回仿真复核。 |

改变打印件、轴承或质量分布后，先用同一 ONNX、同一指令和种子在原装配与新装配场景中比较，再做受控实机测试。达标的技能可以沿用。退化的技能优先检查物理模型并单独微调。若关节数、顺序、命令或 HOME 语义变化，先处理训练和运行时接口，再评估是否适合从旧 PT 初始化。归档时把结构版本、BAM 参数指纹、运行目录、检查点号、导出哈希、两端评估和实机结论关联到同一个候选。
