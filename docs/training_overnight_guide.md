# 今晚训模型 —— 速查指令流

> 已验证环境（2026-09-02）：
> - GPU：NVIDIA GeForce RTX 4080 Laptop，12.4 GB 显存 ✓
> - torch 2.9.1+cu128，MuJoCo Warp / mjlab 1.3.0 ✓
> - 冒烟测试（64 envs × 5 iters）通过，无 NaN，全部 penalty ≤ 0 ✓
> - 2048 envs：1.36 s/iter；4096 envs：2.06 s/iter（单 GPU 实测）
> - checkpoint 目录：`logs/rsl_rl/velocity/<run-name>/model_<iter>.pt`
> - 训练日志自带 tensorboard 事件文件（无需 wandb 账号即可看曲线）

## 一、今晚（一条命令）

```bash
cd /home/joyandai/microduck_rl

# 4096 envs：整晚跑 6000 迭代 ≈ 3.5 h（走姿预算 4000-6000 迭代）
# 建议 22:00 开跑，凌晨 1:30 左右就完整跑完；想稳一点/显存担心小波动就用 2048
UV_CACHE_DIR=/tmp/uv-cache WANDB_MODE=offline \
  uv run train Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num-envs 4096 \
    --agent.max_iterations 6000 \
    --agent.run-name overnight1
```

- `WANDB_MODE=offline`：不登录 wandb 也能训；曲线记录在本地。
- 想实时看 web 曲线？先 `uv run wandb login`（需要账号），然后去掉 `WANDB_MODE=offline` 再跑。
- 中断了怎么办：重新跑并加 `--agent.load-checkpoint logs/rsl_rl/velocity/<run>/model_<iter>.pt --agent.resume True`。

**中途看进度**（另一个终端）：`tail -f` 训练终端输出，或看迭代数；
每 250 迭代存一个 checkpoint 到 `logs/rsl_rl/velocity/overnight1/`。

## 二、明早（3 步出模型）

```bash
cd /home/joyandai/microduck_rl

# 1) 看训练曲线（本地 tensorboard，浏览器开 http://localhost:6006）
UV_CACHE_DIR=/tmp/uv-cache uv run tensorboard --logdir logs/rsl_rl/velocity/ --port 6006

# 2) 导出 ONNX（烘焙观测归一化——部署唯一合法路径）
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file logs/rsl_rl/velocity/overnight1/model_6000.pt \
  --onnx-file walk.onnx

# 3) CPU 演练看效果（键盘控制：方向键走，G/Y/R 等触发其他策略）
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/infer_policy.py --walking walk.onnx
```

## 三、预期与判断标准

| 时间点 | 预期 |
|---|---|
| 训练 30 min（~700 iter） | 奖励明显上升（>2），episode 长度在涨，能站着 |
| 训练 1.5 h（~2500 iter） | 开始**走**（air_time>0、track 误差下降）；可能还笨拙 |
| 3.5 h（6000 iter） | 达到 velocity 任务推荐预算；跑得快慢看命令范围（±0.4 m/s 是设计目标） |
| 看曲线判据 | 每个 `Episode_Reward/<penalty>` ≤ 0；主项 air_time/track_* 上涨；`Mean reward` 持续升或平台 |

> 明早导出如果还"走得像醉汉"：正常——AGENTS.md 说 25 cm 小机器人走姿
> 需要 4000–6000 迭代，明早继续 resume 训练即可（用第一节的 resume 命令）。

## 四、注意事项

1. **先跑 64 envs 冒烟**（没跑过或改动过代码后再跑）：已由我验证通过；
   你自己新开终端第一次跑正式训练前，如果环境变量 `UV_CACHE_DIR` 没设置，
   在机器上直接 `uv run ...` 即可（冒烟时缓存权限问题已用 `/tmp/uv-cache` 规避，
   正式跑请保持一致，方便命中 uv 缓存）。
2. 训练时 GPU 风扇会拉满，散热注意；笔记本插电。
3. 导出后把 `walk.onnx` 拿到 `pollen-robotics/microduck` 运行时即可部署
   （61D 观测契约不变，与现有运行时兼容）。
4. 换 HL-2909 舵机的模型改动**不在本次范围内**——等机械件 + 标定完成后，
   按 `docs/feetech_hls2915_servo_swap.md` 与 `docs/mechanical_swap_requirement.md` 执行。
