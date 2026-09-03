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
