# Microduck RL —— 环境搭建与跑通流程指南

> 本文档面向第一次接触本仓库的人：从零装好环境，把"训练 → 导出 → 演练"整条
> 流水线跑通。内容基于仓库的 README / AGENTS.md / 代码实测整理，并在文末标注了
> 哪些步骤在本机已验证、哪些需要你的 GPU 环境。

## 0. 一分钟看懂全流程

本仓库（microduck_rl）只负责**训练和导出**，真机部署在另一个仓库
（pollen-robotics/microduck，含机载运行时）。完整链路：

```
[本仓库] 训练 PPO 策略 (50 Hz, MuJoCo Warp)
   │  uv run train <TASK_ID> --env.scene.num-envs 4096
   ▼
[本仓库] 导出 ONNX（烘焙 obs 归一化，必须走 scripts/export.py）
   │  uv run scripts/export.py <TASK_ID> --wandb-run-path <...>
   ▼
[本仓库] CPU MuJoCo 演练（键盘驱动，模拟真机部署）
   │  uv run scripts/infer_policy.py --walking output.onnx
   ▼
[另一个仓库] 真机运行时加载 ONNX，50 Hz 下发关节目标位置
```

关键约定（改代码前必读 AGENTS.md）：

- 观测是 **61 维**（48 维本体感受 + 13 维指令块 `[twist(3), head_pose(4), body_pose(6)]`），
  全策略族共享，真机运行时靠它热插拔策略。
- 动作是 **14 维关节目标位置**（14 个舵机，位置闭环模式）。
- 执行器是 **BAM 电压控制模型**（Dynamixel XL330 的 kt/R/摩擦参数），不是理想 PD。
- 控制频率 50 Hz；`NUM_STEPS_PER_ENV = 24`，curriculum 的 step = `iteration × 24`。

## 1. 环境要求

### 1.1 硬件

| 项目 | 要求 | 说明 |
|---|---|---|
| GPU | **NVIDIA CUDA GPU，必需** | 训练跑在 MuJoCo Warp（GPU 并行物理），没有 GPU 训练起不来（`select_gpus()` 会 IndexError）。 |
| 显存 | 冒烟测试任意卡都行；4096 envs 建议 ≥16 GB | 显存不够就减小 `--env.scene.num-envs`，训练仍有效只是更慢。HF Jobs 默认 flavor 是 L4（24 GB）。 |
| CPU / 内存 | 普通即可 | 训练瓶颈在 GPU；导出/演练是 CPU 的。 |
| 磁盘 | 预留 ≥20 GB | `uv sync` 的 venv + uv 缓存（torch、mujoco-warp 等 CUDA wheel）约 5–10 GB，训练日志另算。 |
| 无 GPU 的机器 | 用 HF Jobs（`--hf-jobs`） | 见 §3.6。 |

### 1.2 操作系统

- 开发/训练主力环境是 **Linux**（本仓库在 Ubuntu 上开发和验证）。
- **ARM Linux（DGX Spark / GB10 / Jetson）**：官方支持，但有两个坑，仓库已处理
  （见 §4 FAQ）：torch 必须从 PyTorch 的 cu129 index 装（`[tool.uv.sources]` 已配置），
  首次 `uv sync` 会拉 ~2 GB CUDA wheel，务必 `UV_HTTP_TIMEOUT=600`。
- macOS / Windows 未验证：MuJoCo Warp 在 CPU 上慢到没法训练，不建议。

### 1.3 软件

| 软件 | 版本要求 | 说明 |
|---|---|---|
| Python | **3.12**（`>=3.12,<3.13`，pyproject 锁定） | 别用 3.11/3.13，bam 依赖与 HF Jobs 解释器都按 3.12 对齐。 |
| [uv](https://docs.astral.sh/uv/) | 新版即可 | 唯一的包管理器；`uv run` / `uv sync` 全流程依赖它。 |
| NVIDIA 驱动 | 能跑 CUDA 12.x 即可 | mujoco-warp 自带 CUDA toolkit，系统只需驱动。 |
| wandb | 账号 + 登录 | 训练日志/checkpoint 走 wandb（`wandb login`，或 `~/.netrc`）。 |
| Hugging Face | 可选 | 只有用 `--hf-jobs` 才需要（`hf auth login` 或 `HF_TOKEN`）。 |
| 网络 | 首次同步需访问 PyPI / GitHub / download.pytorch.org | 首次 `uv sync` 下载量大。 |

## 2. 安装步骤（从零开始）

```bash
# 1) 安装 uv（Linux/macOS 一行命令；其余方式见 uv 官方文档）
curl -LsSf https://astral.sh/uv/install.sh | sh
#    重开终端或 source 一下 PATH

# 2) 拿到代码
git clone https://github.com/pollen-robotics/microduck_rl
cd microduck_rl

# 3) 装依赖（创建 .venv，拉 torch / mujoco-warp / bam(git源) 等）
#    ARM 机器（DGX Spark/GB10/Jetson）务必加超时环境变量：
export UV_HTTP_TIMEOUT=600
uv sync
#    如果 uv 报缓存目录权限错误（如 ~/.cache/uv Permission denied），
#    把缓存指到可写目录再跑：
#    export UV_CACHE_DIR=/tmp/uv-cache && uv sync
```

> 注意：`better-actuator-models`（bam）是 git 依赖（Rhoban/bam 的
> `mjlab_frictionloss` 分支），首次同步需要能访问 GitHub。

### 2.1 装完立刻验证

```bash
# (a) 环境里的 torch 是否带 CUDA（训练前必须 True）
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
#    期望: 2.9.1 True 1  （ARM 上必须是 2.9.1+cu129 之类，不能是 +cpu）

# (b) 任务注册表能列出来（能 import 全部任务 = 配置没被改坏）
uv run list-envs | head

# (c) 单元测试（CPU 即可，锁关节映射/奖励符号/NaN 防护等不变量）
uv run --with pytest pytest tests/
```

## 3. 跑通流程（标准路径）

### 3.1 冒烟测试 —— 永远先跑这个

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5
```

5 个 iteration × 64 envs，1–2 分钟内跑完。它能拦住 ~95% 的配置错误
（模型/观测维度/NaN/导出）。**任何长训练之前必须先冒烟**。

### 3.2 正式训练

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096
```

- 默认 `max_iterations=50000`（velocity 任务）；预算参考：简单技巧类 ~1000 iter，
  走姿/恢复类 4000–6000 iter；README 说 4096 envs 下 ~1–2 h 能出一个可用的走姿。
- 产物：
  - 本地 checkpoint：`logs/rsl_rl/<experiment_name>/model_<iter>.pt`
    （velocity 任务的 experiment_name 是 `velocity`，可在 cfg 的
    `RslRl...RunnerCfg.experiment_name` 里改）
  - wandb 项目 `mjlab_microduck`：每个训练 term 的 `Episode_Reward/<term>`，
    所有 penalty 必须 ≤ 0；看总奖励上升 + episode length 符合任务预期。
- 常用训练参数覆盖（tyro 风格，`--` 开头）：
  - `--agent.max_iterations 2000`、`--agent.run-name myrun`
  - `--env.scene.num-envs 1024`
  - `--agent.load-checkpoint model_1234.pt --agent.resume True`（断点续训）

### 3.3 看训练效果 / 播放

```bash
# 用 wandb 上的 run 播放（会自动下载 checkpoint）
uv run play Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <entity/project/run_id>
```

本地 checkpoint 直接播可以看 `scripts/play_latest.py`。

### 3.4 导出 ONNX（部署的唯一合法路径）

```bash
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <entity/project/run_id>
# 输出 output.onnx；也可 --onnx-file xxx.onnx --checkpoint <iter> 指定
```

**为什么必须走这个脚本**：观测归一化（EmpiricalNormalization）会被烘焙进 ONNX
计算图。手转 checkpoint 部署 = 策略拿到未归一化的观测，在 viewer 里看不出来
（训练环境内部也会归一化），到真机上才炸。也支持 `--agent zero/random` 输出空策略。

### 3.5 CPU 部署演练

```bash
uv run scripts/infer_policy.py --walking output.onnx
# 键盘：方向键给线速度/角速度指令，模拟真机运行时写指令块
# 可选：--standing stand.onnx --sitstand ... --new-cmd-obs 等多策略热切换演练
#      --record / --save-csv 供 sim2real 对比
```

这是"真机前的最后彩排"：确认 ONNX 的 obs 契约、指令块写入、策略切换都正常。

### 3.6 （可选）没有本地 GPU：Hugging Face Jobs

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 --hf-jobs
```

详见 `scripts/hf/README.md`：需要 `hf auth login`；可用 `--namespace <org>`、
`--flavor l4x1|a10g-large|a100-large`、`--timeout 12h`、`--detach` 等。

## 4. 常见问题（FAQ）

| 症状 | 原因 / 解法 |
|---|---|
| `uv sync` 下载中途失败/超时 | 首次下载量大（~2 GB+）；`export UV_HTTP_TIMEOUT=600` 重跑。 |
| `uv sync` 报缓存目录 Permission denied | uv 缓存目录权限问题（本机遇到过）。`export UV_CACHE_DIR=<可写目录>` 再跑。 |
| ARM 上 `torch.cuda.device_count()==0`，训练 `IndexError: list index out of range` | PyPI 的 torch 在 aarch64 是 CPU-only。仓库已用 `[tool.uv.sources]` 指向 cu129 index，但前提是：**torch 保持直接依赖且 pin `==2.9.1`**（删掉 pin 或降级为传递依赖，路由就失效）。由 `tests/test_aarch64_cuda_torch.py` 锁定。 |
| `nvidia-smi` 报错 / 训练一开始就 GPU 相关异常 | 驱动或容器没把 GPU 透传进来。训练必须真实 CUDA GPU；沙箱/容器里先修 GPU 或改用 `--hf-jobs`。 |
| `wandb` 报未登录 | `wandb login`。没有 wandb 也能训（本地 logs 仍在），但看不到曲线、`--wandb-run-path` 系列命令用不了。 |
| 训练中 `Episode_Reward/<penalty>` 出现正值 | 奖励符号约定被改坏了（penalty 必须 ≤0），回查 mdp.py 的符号约定。 |
| 训练 NaN 崩溃 | `nan_state` 终止事件兜底；一般伴随接触求解不稳定，看 sim 配置（nconmax/iterations）。 |
| `train` 命令不存在或跑出别的程序 | pyproject 里 `train` 与 mjlab 同名脚本是刻意保留的（见 train_cli.py 注释），`uv sync` 后必须存在，不要删。 |

## 5. 任务与关键文件速查

### 任务（`uv run list-envs` 为准，此处为常用几个）

| Task id | 说明 |
|---|---|
| `Mjlab-Velocity-{Flat,Rough}-MicroDuck` | 主任务：行走 + 头部姿态指令 |
| `Mjlab-VelStand-{Flat,Rough}-MicroDuck` | 行走 + 跌倒恢复 |
| `Mjlab-StandUp-{Flat,Rough}-MicroDuck` | 从趴/躺/坐站起 |
| `Mjlab-SitStand-{Flat,Rough}-MicroDuck` | 坐 ↔ 站 |
| `Mjlab-Roulade-Flat-MicroDuck` | 前滚翻 |
| `Mjlab-Velocity-Flat-MicroDuck-Rollers` | 轮滑速度跟踪 |
| `Mjlab-Velocity-Flat-Backlash-MicroDuck` | 带 ±1° 齿轮回差的孪生任务 |

（任意任务 id 中间插 `-Backlash-` 就是回差变体，obs/动作维度不变。）

### 关键文件

| 文件 | 作用 |
|---|---|
| `src/mjlab_microduck/robot/microduck_constants.py` | 机器人 cfg、HOME 姿态、**BAM 执行器参数（`motor_name="xl330"`、`vin_range`、`kp_fw`、延迟）** |
| `src/mjlab_microduck/actuator/friction_dr_bam.py` | BAM 执行器 + 摩擦 DR + 回差编码器反馈 |
| `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py` | 主行走任务配置（奖励/DR/指令/curriculum），也是其他任务的基座 |
| `src/mjlab_microduck/tasks/mdp.py` | 全部自定义 MDP 函数 |
| `src/mjlab_microduck/tasks/__init__.py` | 任务注册表 |
| `src/mjlab_microduck/robot/microduck/robot_*.xml` | MJCF 模型（Onshape 导出） |
| `scripts/export.py` / `scripts/infer_policy.py` | 导出 ONNX / CPU 演练 |

## 6. 本文档的验证情况

- 已验证：代码结构与命令与 README/AGENTS.md 一致；BAM/执行器参数、任务注册、
  export 流程均已读码核对。
- 本机沙箱无 CUDA GPU 且 uv 缓存目录有权限问题，**未实际跑通训练**；
  `uv sync` 与 `uv run list-envs` 的实测结果见会话记录（如 GPU 就绪即可按 §3 顺序执行）。

## 7. 真机"走路姿态"悬空回放(可选玩法)

场景: 拿到一支训练好的行走策略,但只想先在真机上"悬空迈腿"看效果 —— 不跑
策略网络,而是把它在仿真里走路时的关节角度录下来,再按 50 Hz 写回 15 个
HL-2909-C001 舵机的目标位置寄存器,循环播放。

```bash
# 1) 录制(CPU MuJoCo, 默认 walk.onnx 即 9-02 那版 XL330 行走模型)
uv run python scripts/record_walk_angles.py --onnx walk.onnx \
    --lin-vel-x 0.15 --warmup 3 --duration 8 --out walk_angles.csv
#    → walk_angles.csv (50Hz: time + target_<14关节> + measured_<14关节> + trunk_z_m)
#    → walk_angles.json (元数据: 默认姿态/关节名/舵机ID/无缝循环窗口)

# 2) 回放(先离线校验帧格式, 不碰串口)
uv run --with pyserial python scripts/replay_walk_angles.py \
    --csv walk_angles.csv --dry-run

# 3) 真机回放 —— 鸭子拿稳/悬空后回车; 回车即上电开扭力; Ctrl+C 随时回 HOME
uv run --with pyserial python scripts/replay_walk_angles.py \
    --csv walk_angles.csv              # 默认 1.0x 循环, 遇 Ctrl+C 回到 HOME
```

要点:

- **回放的是 `target_*` 列**(策略下发的目标角 = HOME 姿态 + action),与运行时
  `robotd` 写入舵机的是同一量;`measured_*` 列是仿真实际达到的角度,只作对照。
- 角度约定与运行时 `duck-control/src/ftbus.rs` 完全一致:
  `tick = round((rad + π)·4096/2π)`,即 舵机中位 2048 = **关节机械零位**(装配/CAD
  零位),**不是站立姿势**;全部关节同向 (+1)。站立姿势(HOME)下舵机应读到非 2048
  的值:左 hip_roll 1991、左 hip_pitch **1749**、左 ankle **2343**、颈/头 pitch
  2276、右 hip_pitch **2347**、右 ankle **1753** —— 若站立时读回来全为 2048,
  说明是在站立姿势校的中位,每个关节偏了一个 HOME 角(最大 26.3°),回放会整体歪。
  回放脚本启动时(开扭力前)会打印整张 `present vs home` 核对表,自检零位;
  `--verify-home` 还会在回到 HOME 后再核对一次,`--apply-home-offset` 可将
  实测常数偏差临时修正(治标;治本还是回机械位重做 0x0B 标定)。
- 循环窗口: 录制端的 `loop_window_steps` 已按步态周期(自相关)取整,回放默认
  用它,循环接缝无跳变。
- 安全: 默认扭矩上电前有回车确认;全部目标被夹在 0..4095 单圈范围内(不会绕圈);
  `--no-loop`/`--finish off` 可改为单次播放后松扭力。

### 7.1 零位(中位)标定流程(换装/换舵机后必做)

**零位是什么**: 每个关节在其模型/CAD 框架下角度=0 的姿势(仿真模型编译默认
qpos 就是全 0,即 CAD 零位就是模型的自然零位)。站立姿势不是零位——站立时
整条腿绕 hip_pitch 后倾 26.2°、脚踝前抬 25.95°(HOME 姿态)。

1. **看参考图**: `uv run python scripts/render_zero_pose.py --out docs/zeronotes`
   → `docs/zeronotes_zero_{side,front,quarter,leg_side}.png`(零位姿势)与
   `_home_*`(站立姿势,同机位对照)。零位姿势的要点: 两腿完全竖直、脚掌水平、
   躯干/头按图中几何。
2. **逐关节摆姿势**: 从站立姿态把每个关节转到表中零位(或对着图摆):
   每关节转动量 = −HOME 角(如 left_hip_pitch +26.24°、left_ankle −25.95°、
   neck_pitch −20°;hip_yaw/knee/mouth ≈0)。
3. **逐台校中位**(runtime 仓库, 单机连接、禁广播):
   `python3 scripts/provision-hls.py --port /dev/ttyACM0 --home`
   (0x0B 位置校准, 无参数=当前位置校到中位; HLS 固件 3.45 支持)。
4. **复核**: 鸭子站直后 `uv run --with pyserial python scripts/check_hls_zero.py`
   —— 每关节偏差应 ≈0(站姿读数 == 下表 home ticks,不是 2048)。
   若全 2048: 是在站立姿势校的(错), 需按 1-3 重做; 需要临时演示可先
   `replay --apply-home-offset`(常数修正, 效果等价于重标)。
5. **回放**: `uv run --with pyserial python scripts/replay_walk_angles.py --csv walk_angles.csv`
   (启动时同样打印零位核对表)。

### 7.2 坐起立(替代走路)版本

坐起立比走路好训练、演示也更好看(慢速大幅姿态,伺服零压力)。录制时用
`--sitstand` 调度模式(指令 flag 0=站/1=坐,按时间表切换):

```bash
# 录制 8 s 循环: 站 2 s → 蹲 3 s → 站 3 s(50 Hz, 含无缝循环窗口)
uv run python scripts/record_walk_angles.py \
    --onnx /home/joyandai/microduck/policies/alpha_sitstand.onnx \
    --sitstand /home/joyandai/microduck/policies/alpha_sitstand.onnx \
    --sit-schedule "0:2.0,1:3.0,0:3.0" --out sitstand_angles.csv

# 回放与走路完全相同(换 CSV 即可); 坐↔站转换瞬时步进大(>20 rad/s 需求),
# 真机固件会自动限速平滑; 觉得快就加 --stretch 1.5
uv run --with pyserial python scripts/replay_walk_angles.py --csv sitstand_angles.csv
```

已知点: 现有 alpha_sitstand 是 XL330 时代的模型,CPU 部署场景实测坐/起完整
(站 z≈116mm、坐 z≈59mm、倾斜≤6°);零位/映射约定与走路完全一致。
