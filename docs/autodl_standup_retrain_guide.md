# 在 AutoDL 上训练 OpenMD 起立策略

本指南适用于使用飞特 HD‑1910 舵机的 OpenMD。在一台 Linux GPU 实例上，从坐姿起立任务 `Mjlab-StandUp-Flat-OpenMD` 开始，完成安装、短训练、正式训练、回放和 ONNX 导出。如果要训练走路、坐站、右踢或翻滚，先按[任务与实机模型对照](OpenMicroDuck_retrain_guide.md)选择任务。

StandUp 用于训练起立和站稳，课程包含坐姿、俯卧和仰卧等不同初态。公开策略里的 `stand` 槽使用 VelStand 17500，`walk` 槽使用带关节间隙的行走策略 m9999。训练行走时可选择 `Mjlab-Velocity-Flat-Backlash-OpenMD`，并把后续命令中的实验目录改为 `openmd_velocity`。[公开策略与任务对照](OpenMicroDuck_retrain_guide.md#实机部署参考)

## 1. 准备实例和源码

选择带 NVIDIA GPU 的 Linux x86_64 实例，挂载关机后仍保留的可写数据卷。项目、虚拟环境、下载缓存和训练输出都放在这个卷上。以下命令在 Bash 中运行。以 Ubuntu 为例，若缺少基础工具，用系统管理员权限运行 `apt-get update` 和 `apt-get install -y git curl ca-certificates`。

```bash
uname -m
nvidia-smi
export DATA_ROOT=/mnt/data   # 改为实例的持久卷路径；AutoDL 常用 /root/autodl-tmp
df -h "$DATA_ROOT"
test -w "$DATA_ROOT"
export UV_CACHE_DIR="$DATA_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$DATA_ROOT/.local/share/uv/python"
```

确认架构为 `x86_64`，并能从 `nvidia-smi` 看到所选 GPU 和宿主驱动。训练环境需要驱动支持 GPU 架构以及 PyTorch 和 Warp 使用的 CUDA 运行时。RTX 5090 示例使用支持 Blackwell 和 CUDA 12.8 的驱动；最终以第 2 节的 GPU 运算和第 3 节的物理短训来核验。`nvidia-smi` 的 CUDA Version 表示驱动支持的版本，Python 环境使用的版本看 `torch.version.cuda`。若容器里看不到 GPU，先在云平台确认挂载和驱动。[NVIDIA 驱动兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html)

在空的数据卷首次获取源码时运行下列命令。已有完整工程的用户可直接在含 `pyproject.toml` 和 `uv.lock` 的目录执行后面的检查。

```bash
cd "$DATA_ROOT"
git clone --single-branch --branch openmd-training https://github.com/JoyandAI/microduck_rl.git microduck_rl
cd microduck_rl
ls scripts/check_openmd_training.py src/mjlab_microduck/tasks/openmd_env_cfg.py
ls src/mjlab_microduck/robot/openmd/{robot_stand.xml,robot_walk_backlash.xml,meshes/stl/sole_left.stl}
ls vendor/bam/bam/params/hd1910/m6.json
```

## 2. 安装并确认训练环境

先准备 uv 和 Python 3.12。新机器可以用 uv 官方安装程序，再由 uv 获取 Python。若 `python3.12 --version` 已显示 3.12.x，可跳过 `uv python install 3.12`。若镜像已提供 pip，也可在已有的工具虚拟环境中用 `python -m pip install uv` 安装 uv。[uv 安装说明](https://docs.astral.sh/uv/getting-started/installation/)

```bash
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  curl -fLsS https://astral.sh/uv/install.sh -o "$DATA_ROOT/uv-install.sh" &&
    sh "$DATA_ROOT/uv-install.sh"
fi
uv --version
if ! uv python find --no-python-downloads 3.12 >/dev/null 2>&1; then
  uv python install 3.12
fi
uv python find 3.12
```

选定一个实例可访问的 HTTPS Python 包源。下例使用华为云 HTTPS 镜像；也可切换至官方 PyPI，以当前实例能够实际下载 wheel 的地址为准。在同一终端导出变量，第三方依赖和随后 editable 安装的构建依赖便使用同一源。脚本也支持显式传入 `--index "$UV_DEFAULT_INDEX"`。如果能浏览包目录，下载时却报 403 或超时，可先用下面的可选命令下载训练清单里的一个小 wheel，检查镜像的文件下载服务。确认报错包的实际下载也恢复后，再重试安装。[依赖清单](../requirements-autodl.txt) · [uv 包源设置](https://docs.astral.sh/uv/concepts/indexes/)

```bash
export UV_DEFAULT_INDEX=https://repo.huaweicloud.com/repository/pypi/simple
# 改用官方源时运行：export UV_DEFAULT_INDEX=https://pypi.org/simple
# 可选：实际取回一个约 0.7 MB 的 wheel，无需另下载一份 PyTorch。
uvx --python 3.12 --from pip pip download --index-url "$UV_DEFAULT_INDEX" \
  --only-binary=:all: --no-deps --dest "$DATA_ROOT/.cache/wheel-check" onnxscript==0.5.7
```

在仓库根目录运行安装脚本。脚本从项目的 `uv.lock` 自动生成版本约束，创建 `.venv` 并安装配套依赖与仓内 BAM。接着把本项目以 editable 模式安装到同一个虚拟环境，以注册训练任务。`--no-deps` 让它沿用刚装好的依赖。

```bash
bash scripts/autodl_setup.sh --task Mjlab-StandUp-Flat-OpenMD
uv pip install --python .venv/bin/python --no-deps -e .
source .venv/bin/activate
uv pip check --python .venv/bin/python
python - <<'PY'
import bam, mjlab, mjlab_microduck, mjviser, mujoco, mujoco_warp, onnxruntime, rsl_rl, torch, warp
from rsl_rl.runners import OnPolicyRunner
assert torch.cuda.is_available(), "CUDA unavailable"
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
print("CUDA tensor:", torch.ones(1, device="cuda").item())
PY
python scripts/check_openmd_training.py
```

`uv pip check` 核对依赖声明。上面的导入和一次 CUDA 运算核对运行时；最后的预检核对实际导入的源码、BAM 路径和 m6 参数文件，编译 7 个代表性任务的机器人模型和网格。看到 `HD-1910 m6 OK` 和对应的 `get_openmd_*_spec` 后再开始训练。可在交互终端用 `list-envs | grep -F OpenMD` 查看全部 14 个基础任务；这个版本的 `list-envs` 以任务数作为退出码。自动化或启用 `set -o pipefail` 时，用 `python scripts/check_openmd_training.py` 判断成功。若只能看到 `MicroDuck`，运行 `which python list-envs` 核对当前虚拟环境与 editable 安装位置。[检查脚本](../scripts/check_openmd_training.py)

### 注册表里找不到 OpenMD

在刚安装的仓库根目录运行下面的命令。最后一行会加载已安装的任务插件并检查指定任务。若插件导入失败，终端会显示具体的配置文件和缺少的 Python 名称。

```bash
git branch --show-current
git rev-parse --short HEAD
.venv/bin/python scripts/check_openmd_training.py --registered-task Mjlab-Velocity-Flat-Backlash-OpenMD
```

反馈时请附上分支、提交号和完整的 Python 异常。若 uv 提示 `Failed to hardlink files; falling back to full copy`，表示缓存和安装目录之间改用复制，安装可以继续。用上面的任务检查确认注册结果。

### 旧环境报错时怎样重装

如果 `uv pip check` 通过，训练入口仍报找不到 `libmujoco.so.3.10.0`，或在导入 `mjviser` 时报 API、模块或符号错误，先在仓库根目录检查正在使用的解释器、已装版本、共享库和真实导入。当前锁文件对应 `mujoco==3.10.0`、`mjviser==0.0.14`；包的声明可以满足而原生库仍无法载入。

```bash
.venv/bin/python - <<'PY'
from importlib.metadata import distribution
import sys
print(sys.executable)
for name in ("mujoco", "mjviser"):
    package = distribution(name)
    print(name, package.version, package.locate_file(""))
    if name == "mujoco":
        print(list(package.locate_file("mujoco").glob("libmujoco.so*")))
import mujoco, mjviser
from mjlab.scripts import play
PY
```

若发现版本偏离或环境经过多次混装，先停止使用这个环境的任务，在仓库根目录用下面的命令保留旧目录。随后按本节的**安装、editable 安装、导入和预检**顺序重新执行。工程中的 `logs/`、`exports/` 和检查点可继续使用。

```bash
if declare -F deactivate >/dev/null; then deactivate; fi
mv -T -- .venv ".venv.before-reinstall-$(date +%Y%m%d-%H%M%S)-$$"
```

## 3. 先跑 64 个环境、5 次 PPO 更新

以下命令在同一个已激活 venv 的 Bash 会话执行，日志用本地 TensorBoard。第一次使用独立的运行名，便于将冒烟与正式训练的权重分开。

```bash
TASK=Mjlab-StandUp-Flat-OpenMD
EXP=openmd_stand
TAG=$(date +%Y%m%d_%H%M%S)
train "$TASK" --env.scene.num-envs 64 --agent.max-iterations 5 \
  --agent.seed 42 --agent.logger tensorboard --agent.run-name "smoke_${TAG}"
```

确认日志正常完成 5 次更新、奖励和损失为有限值，并生成检查点。用下面的命令导出这次短训练的最后一个检查点，先走通部署文件格式。短训练本身只验证流程。

```bash
SMOKE_DIR=$(find "logs/rsl_rl/$EXP" -maxdepth 1 -type d -name "*_smoke_${TAG}" -print -quit)
SMOKE_PT=$(find "${SMOKE_DIR:?未找到本次冒烟目录}" -maxdepth 1 -name 'model_*.pt' | sort -V | tail -1)
mkdir -p exports
python scripts/export.py "$TASK" --checkpoint-file "${SMOKE_PT:?未生成冒烟检查点}" \
  --num-envs 1 --onnx-file "exports/smoke_${TAG}.onnx"
python scripts/check_openmd_training.py --onnx "exports/smoke_${TAG}.onnx"
```

## 4. 正式训练与回放

起立基础配置的预算为 15,000 次 PPO 更新，每次每个并行环境采样 24 步。示例使用 2,048 个环境，显存充足可尝试 4,096。`--env.scene.num-envs` 控制并行采样数，`--agent.max-iterations` 控制更新次数。训练默认无头运行。可以在第二个终端打开本地 TensorBoard，借助实例端口转发查看学习曲线。

```bash
cd "${DATA_ROOT:?请在新终端重新指定持久卷路径}/microduck_rl"
source .venv/bin/activate
tensorboard --logdir logs/rsl_rl/openmd_stand --host 127.0.0.1 --port 6006
```

在原训练终端继续执行。根据主任务奖励和回合长度挑选几份检查点，再用相同输入回放，比较起身、双脚支撑、头部姿态和站稳后的保持时间。最新检查点可用来确认训练正常收尾。

```bash
train "$TASK" --env.scene.num-envs 2048 --agent.max-iterations 15000 \
  --agent.seed 42 --agent.logger tensorboard --agent.run-name "train_${TAG}"
RUN=$(find "logs/rsl_rl/$EXP" -maxdepth 1 -type d -name "*_train_${TAG}" -print -quit)
ls -v "${RUN:?未找到本次训练目录}"/model_*.pt | tail
PT=$(find "$RUN" -maxdepth 1 -name 'model_*.pt' | sort -V | tail -1)
play "$TASK" --checkpoint-file "${PT:?未找到检查点}" --num-envs 8 --viewer viser
```

通过实例的端口转发打开 Viser 输出的地址。可把 `PT` 改成同次运行中其他检查点的实际路径，对照多次起身过程。结束回放后继续导出。保留本次运行的 `params/env.yaml`、`params/agent.yaml`、选中的 `.pt`、任务名及源码版本，以便以后继续训练或追溯。

### 网页回放与 MuJoCo 窗口表现不同，怎么对照？

先用同一环境中的 mjlab 打开同一个 `.pt`。沿用上文的 `TASK` 和 `PT`，保持任务、检查点和环境数一致。在有 OpenGL 图形桌面的工作站上，使用同版本训练环境，将 `PT` 设为复制过来的检查点路径。起立用 `Mjlab-StandUp-Flat-OpenMD`；走路按训练时选择普通或带回差的 OpenMD 任务。依次在 CPU 上运行下列命令，关闭第一个窗口后再启动第二个。

```bash
python scripts/play_web.py "$TASK" --checkpoint-file "$PT" --device cpu --num-envs 1
play "$TASK" --checkpoint-file "$PT" --viewer native --device cpu --num-envs 1
```

网页脚本默认使用 Viser 和端口 8081。两个入口都由 mjlab 根据任务 ID 构造回放环境、载入策略，再选择显示后端。对照时记录前进、侧移、转向指令，以及网页 Commands 面板是否启用手动控制。两次独立启动可能抽到不同的初始状态或随机化参数。单次观感不足以定位偏差。[网页回放入口](../scripts/play_web.py)

如果另一条命令是 `scripts/infer_policy.py`，它在独立的 CPU MuJoCo 场景运行导出的 ONNX。标准 OpenMD 起立策略的命令见下文。反馈两种回放的差异时，请附任务名、两条完整命令、源码版本、起始姿态及权重文件名或哈希。

#### Windows PowerShell 回放

在有图形桌面、已装好同版本训练工程和 `.venv` 的 Windows 工作站上，用 PowerShell 进入仓库根目录。下面的命令回放带回差的 OpenMD 行走检查点。按提示输入已下载的 `.pt` 完整路径，路径中可以包含空格，输入时无需加引号。

```powershell
$PT = Read-Host '请输入已下载的 .pt 检查点完整路径'
.\.venv\Scripts\python.exe -m mjlab.scripts.play Mjlab-Velocity-Flat-Backlash-OpenMD --checkpoint-file $PT --num-envs 1 --device cpu --viewer native
```

若要回放本指南的起立检查点，将命令中的任务名改为 `Mjlab-StandUp-Flat-OpenMD`；普通平地行走使用 `Mjlab-Velocity-Flat-OpenMD`。保持任务名与生成该检查点时一致。

## 5. 导出并核对上机接口

```bash
OUT="exports/${TASK}_${TAG}_$(basename "$PT" .pt).onnx"
python scripts/export.py "$TASK" --checkpoint-file "$PT" --num-envs 1 --onnx-file "$OUT"
python scripts/check_openmd_training.py --onnx "$OUT"
```

使用仓库的[导出入口](../scripts/export.py)，观测归一化会随策略一起写入 ONNX。检查脚本使用 ONNX Runtime 在 CPU 上输入零观测和一组固定种子的扰动观测，确认输入 `[1,61]`、输出 `[1,14]`、动作值有限，并打印文件 SHA‑256。

### 在独立 CPU MuJoCo 中复核 OpenMD 起立

在有图形桌面的工作站上，使用同版本工程，将 `OUT` 指向上一步导出的起立 ONNX。下面的命令从俯卧开始回放标准任务 `Mjlab-StandUp-Flat-OpenMD`。旧 ONNX 需显式指定 `--task`。新版导出会把训练任务名写入 ONNX，单独用 `--standing` 加载该策略时也能自动选中配套场景。

```bash
python scripts/infer_policy.py --task Mjlab-StandUp-Flat-OpenMD \
  --standing "$OUT" --start-pose prone
```

启动时核对任务名和 `openmd/scene_stand.xml`。把 `--start-pose` 改为 `supine`（仰卧）或 `sit`（坐姿），观察头部何时离地、双脚何时开始承重，以及起身后能否保持站稳。[独立回放入口](../scripts/infer_policy.py)

旧版通用回放默认选择 MicroDuck 场景。即使手动选择旧 OpenMD 起立 XML，独立 MuJoCo 也读不到训练时另外设置的自碰撞和脚底接触参数。训练和独立回放现在共用 `robot_stand.xml` 中的碰撞设置，保留已有检查点训练时的头颈接触关系。回放按该任务设置关节零位、61 维观测和执行器延迟，并读取 ONNX 记录的动作倍率。与训练端比较时，使用同一检查点、起始姿态和命令，另行记录扰动与随机化条件。[起立机器人 XML](../src/mjlab_microduck/robot/openmd/robot_stand.xml)

上机前核对运行时的 50 Hz 控制频率、关节顺序、HOME 和命令语义，按槽位核对动作倍率。保留原策略和对应设置，再将新模型作为候选单独测试。零输入和扰动输入能验证接口，起立动作仍需在不同初态回放并在有支撑的条件下实测。[实机部署参考](OpenMicroDuck_retrain_guide.md#实机部署参考) · [OpenMicroDuck PPO 训练教程](OpenMicroDuck_PPO训练教程.md)
