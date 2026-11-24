# 在 AutoDL 上训练 OpenMD 起立策略

本指南适用于使用飞特 HD‑1910 舵机的 OpenMD。在一台 Linux GPU 实例上，从坐姿起立任务 `Mjlab-StandUp-Flat-OpenMD` 开始，完成安装、短训练、正式训练、回放和 ONNX 导出。如果要训练走路、坐站、右踢或翻滚，先按[任务与实机模型对照](OpenMicroDuck_retrain_guide.md)选择任务。

StandUp 用于训练起立和站稳，课程包含坐姿、俯卧和仰卧等不同初态。公开策略里的 `stand` 槽使用 VelStand 17500，`walk` 槽使用带关节间隙的行走策略 m9999。训练行走时可选择 `Mjlab-Velocity-Flat-Backlash-OpenMD`，并把后续命令中的实验目录改为 `openmd_velocity`。[公开策略与任务对照](OpenMicroDuck_retrain_guide.md#实机部署参考)

## 1. 准备实例和源码

使用 Linux x86_64、Python 3.12 和 NVIDIA GPU。配套依赖清单固定了 PyTorch 2.9.1 和 mjlab 1.3.0。RTX 5090 使用支持 CUDA 12.8 运行时的驱动镜像。将完整训练源码放到 AutoDL 数据盘，例如 `/root/autodl-tmp/microduck_rl`。进入包含 `pyproject.toml` 的那层目录。

已有本指南对应的完整训练工程时，可直接进入含 `pyproject.toml` 的目录。在空的数据盘工作目录首次获取公开源码时，运行以下命令。

```bash
cd /root/autodl-tmp
git clone --single-branch --branch openmd-training https://github.com/JoyandAI/microduck_rl.git microduck_rl
```

开始安装前，核对下面的任务、预检脚本和资产路径。产品仓的 `policies/` 提供用于回放对照的部署策略。[训练工程配置](../pyproject.toml) · [机器人与公开权重](https://github.com/JoyandAI/OpenMicroDuck)

```bash
cd /root/autodl-tmp/microduck_rl
nvidia-smi
ls scripts/check_openmd_training.py src/mjlab_microduck/tasks/openmd_env_cfg.py
ls src/mjlab_microduck/robot/openmd/{robot_stand.xml,robot_walk_backlash.xml,meshes/stl/sole_left.stl}
ls vendor/bam/bam/params/hd1910/m6.json
```

## 2. 安装并确认训练环境

在仓库根目录运行下面的命令。安装脚本按[依赖清单](../requirements-autodl.txt)创建 `.venv` 并安装仓库内的 BAM。随后把本项目安装到同一环境，使 mjlab 能发现 OpenMD 任务。此处只调用安装功能。

```bash
bash scripts/autodl_setup.sh --task Mjlab-StandUp-Flat-OpenMD
export PATH="$HOME/.local/bin:$PATH"
uv pip install --python .venv/bin/python --no-deps -e .
source .venv/bin/activate
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0), torch.version.cuda)'
list-envs | grep -F OpenMD
python scripts/check_openmd_training.py
```

最后一条应打印 `HD-1910 m6 OK` 和所检查任务使用的模型。它会核对实际导入的源码、BAM 路径和 m6 参数文件，以 7 个代表性任务检查模型工厂，并用 CPU 编译相应模型和网格。完整的 14 个基础任务可在上一步的 `list-envs` 输出中逐项核对。若任务列表只有 `MicroDuck`，先核对第一节的文件，再用 `which python list-envs` 检查两个入口均在本次安装的 `.venv/bin/`。m5 和 m6 各有与模型配套的标定，切换时需同时核对 BAM 模型、参数文件和增益。检查某个 m5 参数值无法确认训练正在使用 m6。[检查脚本](../scripts/check_openmd_training.py)

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
cd /root/autodl-tmp/microduck_rl
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

先用同一环境中的 mjlab 打开同一个 `.pt`。沿用上文的 `TASK` 和 `PT`，保持任务、检查点和环境数一致。在有图形桌面的机器上可依次运行下面两条命令，结束第一个窗口后再启动第二个。

```bash
python scripts/play_web.py "$TASK" --checkpoint-file "$PT" --num-envs 1
play "$TASK" --checkpoint-file "$PT" --num-envs 1 --viewer native
```

网页脚本默认使用 Viser 和端口 8081。两个入口都由 mjlab 根据任务 ID 构造回放环境、载入策略，再选择显示后端。对照时记录前进、侧移、转向指令，以及网页 Commands 面板是否启用手动控制。两次独立启动可能抽到不同的初始状态或随机化参数。单次观感不足以定位偏差。[网页回放入口](../scripts/play_web.py)

如果另一条命令是 `scripts/infer_policy.py`，它运行的是导出的 ONNX 和独立的 CPU MuJoCo 场景。先核对 ONNX 来自同一检查点，再核对 61 维观测模式、场景 XML（含是否带关节间隙）、HOME、舵机模型、动作倍率和输入指令。反馈问题时请附两条完整命令、源码版本及权重文件名或哈希，注明是否已上机。

## 5. 导出并核对上机接口

```bash
OUT="exports/${TASK}_${TAG}_$(basename "$PT" .pt).onnx"
python scripts/export.py "$TASK" --checkpoint-file "$PT" --num-envs 1 --onnx-file "$OUT"
python scripts/check_openmd_training.py --onnx "$OUT"
```

使用仓库的[导出入口](../scripts/export.py)，观测归一化会随策略一起写入 ONNX。检查脚本使用 ONNX Runtime 在 CPU 上输入零观测和一组固定种子的扰动观测，确认输入 `[1,61]`、输出 `[1,14]`、动作值有限，并打印文件 SHA‑256。

上机前核对运行时的 50 Hz 控制频率、关节顺序、HOME 和命令语义，按槽位核对动作倍率。保留原策略和对应设置，再将新模型作为候选单独测试。零输入和扰动输入能验证接口，起立动作仍需在不同初态回放并在有支撑的条件下实测。[实机部署参考](OpenMicroDuck_retrain_guide.md#实机部署参考) · [OpenMicroDuck PPO 训练教程](OpenMicroDuck_PPO训练教程.md)
