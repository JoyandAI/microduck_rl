# AutoDL 训练部署指南（RTX 5090）—— 重训"起立"策略

> 场景：AutoDL 上租一台 **RTX 5090**，把**已经上传好的本仓库**装好依赖，重训
> `Mjlab-StandUp-Flat-MicroDuck`（坐姿 → 站立）策略。
>
> 配套脚本：`scripts/autodl_setup.sh` + 依赖清单 `requirements-autodl.txt`
> —— 就是"要装什么库、按什么顺序一个个装"的一份清单（仓库本身你自己装，见 §2.4）。
> 相关文档：`docs/policy_failure_analysis.md`（为什么必须重训：kd=192.68 把舵机变成刹车）、
> `docs/training_config_velocity_hd1910.md`（HD-1910 执行器参数 / 延迟单位陷阱）。

---

## 0. 选什么：RTX 5090 的版本组合

| 项目 | 选这个 | 依据 |
|---|---|---|
| **GPU** | **RTX 5090 32 GB**（Blackwell，算力 sm_120） | 32 GB 显存 → 默认 4096 envs 很宽松，可试 8192 |
| **镜像** | **Ubuntu 22.04 + CUDA 12.8（或更高，如 13.0）** 的基础镜像；PyTorch/Minconda 镜像都行 | Blackwell 需要 CUDA 12.8+ 才认识 sm_120；镜像里预装的 torch **不用管**（我们自建 venv） |
| **驱动** | `nvidia-smi` 里 `CUDA Version ≥ 12.8`（即驱动 ≥ 570） | RTX 50 系首发驱动就是 570 系列；驱动太老会报 `no kernel image is available` |
| **Python** | **3.12**（`pyproject.toml` 锁 `>=3.12,<3.13`） | bam 与 HF Jobs 都按 3.12 对齐；脚本用 `uv venv --python 3.12` |
| **CUDA toolkit** | **不用装** | 三方各自带自己的 CUDA，见 §0.1 |
| **磁盘** | 仓库 / venv / uv 缓存都放 `/root/autodl-tmp`（数据盘） | 系统盘常只有 30 GB，venv 就 ~8 GB |
| **代码上传** | **从 Codeup clone**（`feat/feetech-hls2909`，HD-1910 那批文件已提交），或 rsync 工作区 | 见 §1.1 / §1.2 |

### 0.1 为什么"CUDA 12.8 镜像 + 5090"这条链路是通的

三件事各自带自己的 CUDA，系统里**不需要装 CUDA toolkit**：

| 组件 | 自带什么 | 实测证据 |
|---|---|---|
| `torch==2.9.1`（PyPI x86_64 wheel） | CUDA **12.8** runtime | `torch.__version__ = 2.9.1+cu128`，`torch.version.cuda = '12.8'`；`torch._C._cuda_getArchFlags()` = `sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 **sm_120**`（`libtorch_cuda.so` 里也能搜到 `sm_120`） |
| `warp-lang==1.12.0`（物理引擎） | CUDA Toolkit **12.9**（NVRTC 静态链接，运行时 JIT 编译 kernel），**只依赖驱动** | `wp.context.runtime.toolkit_version = (12, 9)`，`min_driver_version = (12, 0)`，`is_cuda_compatibility_enabled = True`；NVRTC 支持 arch 列表含 **120、121** |
| `mujoco-warp==3.8.1` / `mujoco==3.10.0` | 纯 Python + warp，无自己的 CUDA | 由 `mjlab==1.3.0` 带进来 |

结论：

- **5090（sm_120）能跑**：torch 的轮子里有 sm_120 的 fatbin，warp 的 NVRTC 能编译 sm_120。
- **镜像必须 CUDA 12.8+（驱动 ≥ 570）**：这是 Blackwell 的硬门槛。CUDA 12.4/12.1 的老镜像配 5090 会报
  `CUDA error: no kernel image is available for execution on the device`。
- 驱动更新（如 CUDA 13.0 的 580 驱动）也没问题，向下兼容 12.x。
- 镜像里预装的 conda torch、CUDA 版本号都**不影响训练**：本项目自建 `.venv`，版本由脚本/`uv.lock` 钉死。

### 0.2 显存 ↔ 环境数 ↔ 时间

| 参考点 | 数值 |
|---|---|
| 已验证配置（本地 RTX 4080 Laptop 12 GB，1024 envs） | `Steps per second: 23515`，≈ **1.05 s/iter**；15000 iter 实跑 **4 h 23 min** |
| 1 iter 的步数 | `envs × 24`（`num_steps_per_env=24`）：4096 envs → 98304 步/iter |
| 5090 32 GB | 默认 **4096 envs**；想吃满 GPU 可以试 `--envs 8192`（先看显存和 steps/s） |
| 时间怎么估 | 日志里 `Steps per second` ÷ 98304 = 每 iter 秒数；4090 级预期 ≥ 8 万 steps/s，明显低就要查瓶颈 |
| 显存不够 | envs 减半（4096 → 2048 → 1024），算法不变 |

### 0.3 2026-09-11 实测的 AutoDL 5090 实例（照着抄能避开几个坑）

| 项 | 实测值 / 结论 |
|---|---|
| GPU | `NVIDIA GeForce RTX 5090, 32607 MiB, driver 580.105.08, compute_cap 12.0`；`nvidia-smi` 显示 `CUDA Version: 13.0` |
| 磁盘 | `/`（系统盘）30 GB；`/root/autodl-tmp`（数据盘）50 GB —— venv + 缓存必须放数据盘 |
| Python | 镜像自带 **conda Python 3.12.3**（`/root/miniconda3/bin/python`），PATH 里**没有** `python3` / `python3.12`，只有 `python` |
| uv 托管解释器 | `uv venv --python 3.12` 默认会去 GitHub 下托管 Python。已让脚本优先复用系统 3.12；万一要下，先 `source /etc/network_turbo` 再 `uv python install 3.12`，并把它装进 `UV_PYTHON_INSTALL_DIR`（脚本默认 `/root/autodl-tmp/.microduck/uv-python`） |
| 镜像源速度 | 清华 tuna **~1 MB/s（可用）**；`pypi.org` 直连 **56 KB/s（别用）** |
| GitHub | 直连不稳，`source /etc/network_turbo` 后 200 ✓ |
| 代理的副作用 | `/etc/network_turbo` 自己写明"对其他资源（如 pip 源）会更慢" ⇒ **只在装 uv / 下 Python 时开代理，装依赖时关掉** |
| 安装耗时 | torch 的 CUDA 轮子合计约 4 GB，按 ~1 MB/s 全程 **约 1 小时** |
| wandb | 不登录就 `WANDB_MODE=offline`（脚本 `--train` 会自动判断） |

---

## 1. 开工前：把代码传上去

### 1.1 推荐：直接从 Codeup clone

HD-1910 换装那一批文件**已经提交并推送到 Codeup**（分支 `feat/feetech-hls2909`），
原先 untracked / 未提交的那批也一并进去了：`vendor/bam/bam/params/hd1910/`、
`hd1910_calibration/`、`tests/test_hd1910_cfg.py`、`docs/*.md`、新增脚本。
所以实例上直接 clone 就是完整可训练的一份：

```bash
cd /root/autodl-tmp
git clone -b feat/feetech-hls2909 \
  https://codeup.aliyun.com/69692c5e66d410a0f264fd91/mircoduck_rl.git microduck_rl
cd microduck_rl
```

- Codeup 在国内/AutoDL 上速度正常；私有库需要 Codeup 的账号或访问令牌（克隆时提示输入）。
- 想用 SSH 而不是 HTTPS，就在 Codeup 控制台加公钥后用 `git@codeup.aliyun.com:...` 那个地址。

**上传后确认这几个文件在**（缺 `vendor/bam/bam/params/hd1910/` 就一定训不了）：

```bash
ls vendor/bam/bam/params/hd1910/m5.json \
   src/mjlab_microduck/robot/microduck/robot_allcollisions.xml \
   hd1910_calibration/params_recommended_m5.json \
   requirements-autodl.txt pyproject.toml
```

### 1.2 备选：rsync 本地工作区（还没提交的改动）

如果你本地又改了代码还没提交，或者不想走 Codeup，就用 rsync 传工作区
（在本地仓库根目录执行，端口/地址用 AutoDL 控制台给的 SSH 信息）：

```bash
rsync -avzP -e "ssh -p <端口>" \
  --exclude '.venv' --exclude '.uv-cache' --exclude 'logs' --exclude 'wandb' \
  --exclude '*.log' --exclude '*.onnx' --exclude '.git' \
  --exclude 'hd1910_calibration/bench_*' --exclude 'hd1910_calibration/fit*' \
  ./ root@<地址>:/root/autodl-tmp/microduck_rl/
```

- 两条 `hd1910_calibration/*` 的 exclude 挡掉 427 MB 台架原始数据（`fit_logs` 215 MB 等），只留 0.6 MB 参数文件。
- 上传量约 **190 MB**（`vendor/` 23 MB + `src/` 29 MB + 其余）。
- 没有 rsync 就 `scp -P <端口> -r ./ root@<地址>:/root/autodl-tmp/microduck_rl/`。
- **别只 `git clone` 一份老的干净提交**：那样会缺上面那批 untracked 文件，训练直接失败。

### 1.3 磁盘：大件放数据盘

```bash
df -h          # 看 / 和 /root/autodl-tmp
```

脚本会自动把 `UV_CACHE_DIR`、`UV_PYTHON_INSTALL_DIR` 指到 `/root/autodl-tmp/.microduck/`；
如果在别处建仓库，注意 venv 也要放数据盘（`uv venv` 默认建在当前目录）。

---

## 2. 装依赖：一张清单，一个个装

依赖清单就在仓库里：**`requirements-autodl.txt`**（只管第三方库 + 仓库内的 `vendor/bam`，
**不含仓库本身** —— 仓库你自己装，见 §2.4）。

### 2.1 跑脚本（逐行读清单、逐行装）

```bash
cd /root/autodl-tmp/microduck_rl
bash scripts/autodl_setup.sh --list      # 先看要装什么（不真装）
bash scripts/autodl_setup.sh             # 开始装（默认清华 PyPI 镜像）
bash scripts/autodl_setup.sh --upstream  # 镜像缺包时改用官方 PyPI
```

脚本做的事很直白：设缓存目录 → 装 uv → `uv venv --python 3.12 .venv` → **读
`requirements-autodl.txt` 一行一个 `uv pip install`** → 装完把关键版本再钉一遍 →
自检（能不能看到 GPU、关键包能不能 import）。哪一行挂了就停在哪一行，单独重跑那一条即可。

### 2.2 清单内容（`requirements-autodl.txt`，照抄即可）

| # | 库 | 版本 | 作用 / 备注 |
|---|---|---|---|
| 1 | `torch` | `==2.9.1` | 训练框架底座；PyPI x86_64 轮子自带 CUDA 12.8、含 sm_120（5090 直接用） |
| 2 | `warp-lang` | `==1.12.0` | MuJoCo Warp 的物理引擎。**必须 1.12.x**：1.15 删了 `wp.context`，mjlab 1.3 会崩 |
| 3 | `mujoco` | `==3.10.0` | CPU 侧 MuJoCo（导出 / 彩排脚本） |
| 4 | `mujoco-warp` | `==3.8.1` | GPU 并行物理，训练算力所在（已验证搭配；别升 3.9+，`ls_parallel` 被移除） |
| 5 | `mjlab` | `==1.3.0` | 训练框架；会顺带装 `rsl-rl-lib==5.0.1`、`tyro`、`wandb`、`tensorboard`、`viser`、`trimesh`、`tensordict` 等 |
| 6 | `numpy` | `==2.4.1` | 数值 |
| 7 | `scipy` | `==1.18.0` | **必须显式装**：mjlab 1.3.0 用了它却没声明（terrains 模块），漏了连 `import mjlab` 都失败 |
| 8 | `onnx` | `==1.22.0` | ONNX 导出 |
| 9 | `onnxscript` | `==0.5.7` | torch → ONNX 导出后端 |
| 10 | `onnxruntime` | `==1.24.4` | CPU 部署彩排 `scripts/infer_policy.py` |
| 11 | `matplotlib` | `>=3.10.9` | 画图脚本 |
| 12 | `huggingface_hub` | `>=0.27.0` | `--hf-jobs` / 上传（纯本地训练可省） |
| 13 | `rustypot` | `>=1.4.2` | 舵机总线（真机脚本用，纯训练可省） |
| 14 | `better-actuator-models`（`-e ./vendor/bam`） | 本地 | 仓库内的 BAM 执行器模型 + HD-1910 参数。**只装本体，千万别加 `[identification]` / `[all]` extra** |

一个已经绕过的坑：**bam 不要装 extra** —— `identification` extra 里的 `zmq==0.0.0` 是缺
`.dist-info` 的坏轮子，装上还会把 `protobuf` 拽回 `<4.0`，连带把 `onnx` 降到 1.17（没有 wheel、
要现场编译）。我们只用 `bam.model` / `bam.actuator` / `bam.actuators` / `bam.mjlab`，
本体依赖只有 `numpy + colorama`。

### 2.3 不想用脚本

```bash
cd /root/autodl-tmp/microduck_rl
export UV_CACHE_DIR=/root/autodl-tmp/.microduck/uv-cache UV_HTTP_TIMEOUT=600
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 .venv

# 一次性装（清单里已带 -e ./vendor/bam）
uv pip install --python .venv/bin/python \
   --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
   -r requirements-autodl.txt

# 想一个个看得清楚，就照清单手敲：
P="uv pip install --python .venv/bin/python --index-url https://pypi.tuna.tsinghua.edu.cn/simple"
$P torch==2.9.1
$P warp-lang==1.12.0
$P mujoco==3.10.0 mujoco-warp==3.8.1
$P mjlab==1.3.0
$P numpy==2.4.1 scipy==1.18.0
$P onnx==1.22.0 onnxscript==0.5.7 onnxruntime==1.24.4
$P "matplotlib>=3.10.9" "huggingface_hub>=0.27.0" "rustypot>=1.4.2"
$P -e ./vendor/bam
```

### 2.4 装仓库本身（你自己来）

依赖装完后，进仓库根目录把**仓库自己**装进 venv：

```bash
cd /root/autodl-tmp/microduck_rl
uv pip install --python .venv/bin/python --no-deps -e .
```

- **`--no-deps` 不能省**：否则它会去 PyPI 拉一个**未定制版** `better-actuator-models`
  覆盖 `vendor/bam`（PyPI 那份没有 hd1910 参数），等于把执行器换了。
- 这一步干两件事：把 `src/mjlab_microduck` 放进 venv、注册 `mjlab.tasks` 入口点与 `train` 命令。
- 不装它，`list-envs` 里就没有 `Mjlab-*-MicroDuck`，训练也无从下手。
  （或者用仓库自带的正规路径 `uv sync`，它会把仓库和依赖一起按 `uv.lock` 装好——但与 §2.1 的手工路二选一，别混用。）

### 2.5 装完确认（三条命令）

```bash
# 1) GPU 与关键包（5090 应该打印 sm_120 与 32.x GiB）
uv run --no-sync python -c "
import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())
[print(' ', torch.cuda.get_device_properties(i).name, 'sm_%d%d' % (torch.cuda.get_device_properties(i).major, torch.cuda.get_device_properties(i).minor)) for i in range(torch.cuda.device_count())]"

# 2) 任务注册表里有我们的任务
uv run --no-sync list-envs | grep MicroDuck

# 3) HD-1910 参数是修复版（kd 必须是 0.346，不是 192.68）
grep -o '"kd": *[0-9.]*' vendor/bam/bam/params/hd1910/m5.json
#    如果打出来是 192.68： cp hd1910_calibration/params_recommended_m5.json vendor/bam/bam/params/hd1910/m5.json
```

> 第 3 条为什么重要：`kd=192.68` 会让 D 项在 **0.127 rad/s** 就把占空比打满，仿真里舵机退化成
> **刹车**，策略必然训废（实测走路跟踪 99.5% → 3%、8 秒摔 6 次）。详见 `docs/policy_failure_analysis.md`。

> 关于仓库自带的 `uv sync`：那是**按 `uv.lock` 全量锁定**的正规路径（可复现性最好），
> 但它依赖 astral 的解析与官方索引。本脚本这条"手动清单"路更透明、国内更快；
> **两条路二选一，别混着用**——混用会让 uv 反复重装。

---

## 3. 训练"起立"策略

> 前提：依赖装完（§2.1）+ 仓库本身装完（§2.4）。
> 用 `uv run` 时请带 `--no-sync`（本环境是手工装的，不然 uv 会照 `uv.lock` 再装一遍）；
> 或者先 `source .venv/bin/activate`，之后直接敲 `train` / `python scripts/...`。

```bash
# 冒烟测试（必跑，1-2 分钟，64 envs × 5 iter）
uv run --no-sync train Mjlab-StandUp-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5

# 正式训练
uv run --no-sync train Mjlab-StandUp-Flat-MicroDuck --env.scene.num-envs 4096 --agent.run-name standup-r1
# 或后台跑（日志落 logs/autodl/，setsid 脱离终端，SSH 断了不停）
bash scripts/autodl_setup.sh --train --envs 4096
```

起立任务的配置（`src/mjlab_microduck/tasks/microduck_standup_env_cfg.py`）：

| 项 | 值 |
|---|---|
| 任务 id | `Mjlab-StandUp-Flat-MicroDuck`（rough 地形版 `-Rough-`） |
| 机器人模型 | `robot_allcollisions.xml`（能从躺着站起来） |
| `experiment_name` | `microduck_stand` → checkpoint 在 `logs/rsl_rl/microduck_stand/<时间戳>_<run名>/` |
| `max_iterations` | **15000**（`--agent.max_iterations N` 可覆盖） |
| `save_interval` | 每 250 iter 存 `model_<iter>.pt` |
| episode | 6.0 s（= 300 个 50 Hz 控制步） |
| 目标 | 坐姿（`SIT_Z≈0.060`）→ 站姿（`STAND_Z=0.115`，HOME 关节角），站住后还要跟 body pose 指令 |
| obs / action | 61 维 / 14 维（全策略族共享，别改） |

**先短后长**：第一次建议 `--agent.max_iterations 2000` 验证整条流水线（存盘/日志/续训），
没问题再跑满 15000。坐↔起立同理，把 `--task` 换成 `Mjlab-SitStand-Flat-MicroDuck`。

### 3.1 盯训练

```bash
tail -f logs/autodl/<时间戳>_Mjlab-StandUp-Flat-MicroDuck.log     # --train 启动时打印的路径
grep -m5 "Learning iteration" <日志>
grep -m1 -A12 "Learning iteration 10/" <日志>                     # 看前 10 iter 的速度与奖励
```

看什么（口径来自 `AGENTS.md`）：

- `Mean reward` 上升，而且**主任务项**（`standing_composite` / `posture_height*` / `upright*`）自己在涨。
- `Mean episode length` 符合预期（上限 300 步；学不会会早退，学会了会打满）。
- **每个 `Episode_Reward/<penalty>` 必须 ≤ 0**（符号约定；出正值说明奖励被 hack）。
- `Steps per second` 正常（§0.2），`Mean action std` 在收敛。

### 3.2 断点续训

```bash
ls logs/rsl_rl/microduck_stand/                    # 找 run 目录名
uv run --no-sync train Mjlab-StandUp-Flat-MicroDuck --env.scene.num-envs 4096 \
    --agent.resume True --agent.load-run <run目录名> --agent.load-checkpoint model_7500.pt
```

- `--agent.load-run` 是**正则**匹配 run 目录名，不传就取最新那个。
- 续训会**新建**一个 run 目录（时间戳必然不同），权重和 iter 计数器从 checkpoint 接着走。

---

## 4. 训练之后：导出 ONNX + 彩排

```bash
# 1) 导出（必须走这个脚本：obs 归一化会被烘进 ONNX 计算图）
uv run --no-sync scripts/export.py Mjlab-StandUp-Flat-MicroDuck \
    --checkpoint-file logs/rsl_rl/microduck_stand/<run>/model_15000.pt --onnx-file standup.onnx

# 2) headless 忠实评测：用任务自己的出生分布（坐姿 + 地面状态混合），看起身成功率/姿态
uv run --no-sync python scripts/eval_onnx_bam.py --onnx standup.onnx \
    --task Mjlab-StandUp-Flat-MicroDuck --lin-vel-x 0 --seconds 8

# 3) CPU MuJoCo 部署彩排（键盘驱动；起立策略挂在 standing 槽位）
uv run --no-sync scripts/infer_policy.py --standing standup.onnx --new-cmd-obs --motor hd1910
```

- `--new-cmd-obs` **必加**：本策略族用统一 13D 指令块（`twist(3)+head_pose(4)+body_pose(6)`），
  不加会按旧 3D 布局喂 obs＝策略看错输入。
- `eval_onnx_bam.py` 要显式 `--lin-vel-x 0`：它默认 0.2，而起立任务的 twist 槽只是保活用的零填充槽。
- `infer_policy.py` 从**站立关键帧**出生，验证的是 obs 契约 / 指令槽 / 多策略热切换；
  "从坐姿起身"这个过程看 `eval_onnx_bam.py`。
- **别手转 checkpoint**：`obs_normalization=True`，不烘归一化在 viewer 里正常、上真机必炸。

---

## 5. 运维

| 事项 | 做法 |
|---|---|
| 配环境省钱 | 先用**无卡模式**装依赖（`uv sync`/下载都不需要 GPU），装完再切正常模式开机 |
| 训练期间 | 保持开机；`setsid` 后台跑，SSH 断了不中断 |
| 产物位置 | 日志 `logs/autodl/*.log`；checkpoint `logs/rsl_rl/microduck_stand/<run>/model_*.pt`；wandb 离线数据 `wandb/offline-run-*` |
| 取回本地 | `scp -P <端口> root@<地址>:/root/autodl-tmp/microduck_rl/standup.onnx ./` |
| 数据盘 | `/root/autodl-tmp` 随实例保留；**释放实例会清空**，checkpoint 先传走 |

---

## 6. FAQ（5090 + 手工装库版）

| 症状 | 原因 / 解法 |
|---|---|
| `CUDA error: no kernel image is available for execution on the device` | 镜像/驱动太老认不了 sm_120：换 **CUDA 12.8+（驱动 ≥ 570）** 的镜像 |
| `torch.cuda.device_count() == 0`，训练 `IndexError: list index out of range` | 同上，或 GPU 没透传；无卡模式属正常 |
| 训练跑到一半被 uv 重新装依赖 / 卡在 Resolving | 你敲的是 `uv run` 而不是 `uv run --no-sync`：手工装的环境会被 `uv.lock` 覆盖重装 |
| `ModuleNotFoundError: No module named 'scipy'` | 漏了清单第 7 条（mjlab 自己没声明 scipy） |
| `list-envs` 里没有 `Mjlab-*-MicroDuck` | 仓库本身没装进 venv：`uv pip install --python .venv/bin/python --no-deps -e .`（注册 `mjlab.tasks` 入口点） |
| `AttributeError: module 'warp' has no attribute 'context'` | warp 被装成 1.13+ 了：重新 `uv pip install warp-lang==1.12.0` |
| 装 bam 时 `zmq` 报错 / onnx 要现场编译 | 装成了 `vendor/bam[identification]` 或没加 `--no-deps`：卸掉重装本体 |
| 训练时舵机像"刹车"、策略学不动 | `m5.json` 的 `kd` 是 192.68：`cp hd1910_calibration/params_recommended_m5.json vendor/bam/bam/params/hd1910/m5.json` |
| `uv pip install` 下载慢/超时 | `UV_HTTP_TIMEOUT=600`（脚本已设）+ 清华镜像；仍失败用 `--upstream` 或 `source /etc/network_turbo` |
| 镜像里没有 python3.12，`uv venv` 卡住 | 脚本会优先复用机器上已有的 3.12（如 conda 的 `/root/miniconda3/bin/python`）。真要下托管解释器：先 `source /etc/network_turbo`，再 `UV_PYTHON_INSTALL_DIR=/root/autodl-tmp/.microduck/uv-python uv python install 3.12`，然后**关掉代理**重跑脚本 |
| `uv pip install` 下载只有几十 KB/s | 大概率开了学术代理（它自己声明会让 pip 源变慢）：关掉代理/重开一个 SSH 会话，用清华镜像（实测 ~1 MB/s，而 pypi.org 直连只有 56 KB/s） |
| CUDA out of memory | `--env.scene.num-envs` 减半 |
| wandb 卡在登录提示 | `export WANDB_MODE=offline`（脚本 --train 时自动判断） |
| 磁盘满 | 清 `logs/`、`wandb/`；确认 `UV_CACHE_DIR` 在 `/root/autodl-tmp` |

---

## 7. 训练前请确认（本仓库红线）

- `vendor/bam/bam/params/hd1910/m5.json` 的 **kd=0.346**（§2.5 第 3 条）。
- 别改的不变量：61 维观测（`[twist(3), head_pose(4), body_pose(6)]` 顺序）、14 关节序、
  `passive_*` 命名、BAM 执行器、obs 归一化必须由 `export.py` 烘焙、训练不加动作低通滤波。详见 `AGENTS.md`。
- 延迟单位：`delay_*_lag` 以**物理子步**（0.005 s）计，不是 env step（`docs/training_config_velocity_hd1910.md` §2）。
- 可选：`mdp.py` 的 head-pose-bias 门控有重复乘门（`gate²`），起立/坐起立都受影响（P1-1，见 `docs/policy_failure_analysis.md`）。
  要一并修就**先改再训**。

---

## 8. 事实来源

| 结论 | 出处 |
|---|---|
| torch 2.9.1 = cu128，fatbin 含 sm_120 | 本机 `.venv` 实测 `torch._C._cuda_getArchFlags()`、`libtorch_cuda.so` 里搜到 `sm_120` |
| warp 1.12.0 toolkit 12.9 / min_driver (12,0) / NVRTC 支持 120、121 | 本机实测 `wp.context.runtime` 与 `nvrtc_supported_archs` |
| 依赖清单版本 | 本机已验证环境的 `importlib.metadata` + `uv.lock` |
| bam 只装本体、别加 extra | `vendor/bam/pyproject.toml`（本体依赖仅 numpy+colorama；`zmq` 在 identification extra） |
| `-e . --no-deps`（仓库本体）注册 `mjlab.tasks` 入口点 | 本机实测安装后 `dist-info/entry_points.txt` 里确有 `mjlab.tasks` 与 `train` |
| 1024 envs → 23515 steps/s、15000 iter = 4 h 23 min | `train_sitstand_hd1910.log`、`docs/policy_failure_analysis.md` |
| 坏 kd 让舵机变刹车 | `docs/policy_failure_analysis.md`、`tests/test_hd1910_cfg.py` |
| AutoDL 数据盘 / 学术加速 | [基础配置](https://api.autodl.com/docs/base_config/)、[学术资源加速](https://api.autodl.com/docs/network_turbo/) |

> 说明：本文的软件链路（sm_120 支持、依赖版本、入口点注册、脚本流程）都在本机核对/跑通过；
> RTX 5090 真机与 AutoDL 实例本身未实测，第一次上去请按 §2.5 的三条命令确认一遍。
