# OpenMicroDuck 强化学习训练

本项目使用 mjlab、MuJoCo Warp 和 rsl_rl PPO，为采用飞特 HD‑1910 舵机的 OpenMicroDuck 训练动作策略。任务名中的 `OpenMD` 标识这套机器人模型。首次训练建议从平地起立任务开始，先用 64 个并行环境跑通 5 次 PPO 更新，再按目标选择行走、坐站、触地拾取、踢球或翻滚。

| 我想做什么 | 从哪里开始 |
|---|---|
| 在云服务器上完成一次训练、回放和 ONNX 导出 | [AutoDL 起立训练指南](docs/autodl_standup_retrain_guide.md) |
| 准备 URDF、MJCF 与网格，理解 HD‑1910 执行器、PPO 奖励和训练曲线，验证策略并上机 | [OpenMicroDuck PPO 训练教程](docs/OpenMicroDuck_PPO训练教程.md) |
| 查注册任务、训练使用的 XML、已发布动作与部署槽位 | [OpenMD 任务与公开策略对照](docs/OpenMicroDuck_retrain_guide.md) |
| 下载已有策略和配套部署参数 | [OpenMicroDuck 公开策略目录](https://github.com/JoyandAI/OpenMicroDuck/tree/main/policies) |

安装配套训练源码并激活环境后，运行以下命令。第一条列出已安装的 OpenMD 任务，第二条检查 HD‑1910 执行器参数并编译机器人模型。

```bash
list-envs | grep -F OpenMD
python scripts/check_openmd_training.py
```

如果只看到以 `MicroDuck` 命名的任务，先确认当前源码中存在 `src/mjlab_microduck/tasks/openmd_env_cfg.py` 和 `src/mjlab_microduck/robot/openmd/`，再检查当前虚拟环境中的可编辑安装。原版 Microduck 的型号、任务和训练示例见[英文简介](README.md)。硬件图纸、装配资料和部署说明见 [OpenMicroDuck 项目主页](https://github.com/JoyandAI/OpenMicroDuck)。
