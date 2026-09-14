# 走路任务训练配置 —— Mjlab-Velocity-Flat-MicroDuck (HD-1910)

> 本文档记录 **走路 (velocity)** 任务在换成 **Feetech HD-1910-C001** 舵机后的
> 完整训练配置，用于复现一次训练跑。所有数值都从 cfg 实测导出，末尾附 `file:line` 来源。
>
> 生成时间：2026-09-10 · 舵机：HD-1910-C001（21 g，5.0 V 稳压轨）
>
> ---
> ## ⚠️ 训练前必读（2026-09-11 更新）
>
> **当前磁盘上的 `vendor/bam/bam/params/hd1910/m5.json` 含一个已知有害参数：`kd = 192.68`。**
> 它使 D 项占空比斜率 = **7.85 duty/(rad/s)**，仅 **0.127 rad/s** 的关节角速度就把占空比打满
> ⇒ 仿真里舵机退化成**刹车**，训练必然失败（实测：走路策略跟踪从 99.5% 崩到 3%、8 秒摔 6 次）。
> 该参数来自自由拟合，而台架数据**无法辨识 D 强度**（`characterization.json` 自己写明了这一点）。
>
> **训练前先安装已验证的修复参数**：
> ```bash
> cp hd1910_calibration/params_recommended_m5.json vendor/bam/bam/params/hd1910/m5.json
> cp hd1910_calibration/params_recommended_m5.json /home/joyandai/bam/bam/params/hd1910/m5.json
> ```
> （`kt=0.692` 台架实测 + `kd=0.346` 与台架静态回归一致；实测走路恢复到 0.202/0.20 跟踪、0 摔、tilt 3.1°。）
>
> **完整分析见 `docs/policy_failure_analysis.md`**；另见 `tests/test_hd1910_cfg.py` 里的
> `test_d_channel_cannot_turn_the_servo_into_a_brake`（现为 xfail，装入修复后应去掉 marker）。

---

## 1. 一句话流程

```
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs <N> --agent.max_iterations 15000
   ↓  checkpoint: logs/rsl_rl/velocity/<ts>/model_<iter>.pt
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <...>   # 烤入 obs 归一化
   ↓  output.onnx
uv run scripts/infer_policy.py --walking output.onnx --motor hd1910             # CPU 部署彩排
```

**必须走 `scripts/export.py`**：观测归一化 (`obs_normalization=True`) 是烘进 ONNX 计算图的。
手转 checkpoint 会得到一个拿到未归一化观测的策略 —— 在仿真 viewer 里看不出来（训练环境
内部也会归一化），到真机才炸。

---

## 2. 执行器 / 舵机参数（BAM）

配置位置：`src/mjlab_microduck/robot/microduck_constants.py` → `_BAM_ACTUATOR_KWARGS`

| 项 | 值 | 说明 |
|---|---|---|
| `motor_name` | `hd1910` | BAM 执行器律 `HD1910Actuator`（已注册） |
| `model` | `m5` | 6 档全 PASS；m5 = 方向分离负载摩擦（物理最完整），比数值最优的 m4 差 ~4% |
| `kp_fw` | `32.0` | 真机 reg50 回读；**不是** XL330 的 200（会硬 6 倍） |
| `vin_range` | `(4.75, 5.25)` | 整机 **5.0 V 稳压轨 ±5%**。辨识数据也是在 5.0–5.2 V 采的 ⇒ 无需电压换算 |
| `vin_drop_resistance_range` | `(0.0, 0.069)` Ω | `V_drop = R·I_bat`；沿用 XL330 时代已证的线阻上限 |
| `vin_min` | `4.0` | 舵机自身低压告警 reg15=40 |
| `delay_min_lag` / `delay_max_lag` | `4` / `7` | ⚠️ **单位是物理子步 (0.005 s)**，见下 |

### ⚠️ 延迟单位陷阱（很容易搞错 4 倍）

`delay_*_lag` **不是** env step，而是 **物理子步**。证据（mjlab 1.3.0）：

```python
# mjlab/envs/manager_based_rl_env.py:414-419
for _ in range(self.cfg.decimation):          # decimation = 4
    self.action_manager.apply_action()
    self.scene.write_data_to_sim()            # -> _apply_actuator_controls -> act.compute()
    self.sim.step()                           #    -> DelayBuffer.append()
    self.scene.update(dt=self.physics_dt)
```
`mjlab/entity/entity.py:830` 注释原文：*"Called before each `sim.step()` **within the decimation
loop**"*。所以一个 env step 内 `compute()` 被调用 4 次 ⇒ lag 计数 = `sim.mujoco.timestep`。

代入本环境：辨识出的 `command_delay = 0.023–0.031 s` ÷ 0.005 s = **4.6–6.2 lag** ⇒ 取 `(4, 7)`
（20–35 ms）包住实测区间并留 DR 余量。若按 0.02 s/env step 读，会得到 (1,3)=5–15 ms，**把
执行器延迟少建模 2–4 倍**。

> 另注：m5 档的 `command_delay` 在 2026-09-10 18:51 的重拟合中由 ≈0（4.9e-06 s）改为
> **0.02534 s**，仍落在实测区间 0.023–0.031 s 内，与 `(4, 7)` 一致。
> 这里保守按“存在延迟”建模 —— 延迟建模不足比过度建模更危险。

---

## 3. 仿真与 env 设置

| 项 | 值 |
|---|---|
| task id | `Mjlab-Velocity-Flat-MicroDuck` |
| 机器人模型 | `MICRODUCK_WALK_ROBOT_CFG`（`robot_walk.xml`，全碰撞体） |
| 物理子步 `sim.mujoco.timestep` | `0.005 s` (200 Hz) |
| `decimation` | `4` |
| **控制频率** | **50 Hz**（env step = 0.02 s） |
| `episode_length_s` | `20.0`（= 1000 env steps） |
| `sim.nconmax` | `35` |
| `sim.mujoco.iterations` / `ls_iterations` | `10` / `20` |
| 地形 | `plane`（Flat 变体） |
| 关节模型质量 | 整机 **779.2 g**（XL330 基线 737.2 g + 14×3 g HD-1910） |

---

## 4. 观测契约（全策略族共享，61D）

命令块固定为 `[twist(3), head_pose(4), body_pose(6)]`，**不用的槽位补零、绝不删槽**。

- **actor**（61D = 3 + 3 + 14 + 14 + 14 + 13）
  `base_ang_vel`, `projected_gravity`, `joint_pos`, `joint_vel`, `actions`,
  `command`(3), `head_command`(4), `body_command`(6)
- **critic**（特权）：另含 `base_lin_vel`, `foot_height`, `foot_air_time`,
  `foot_contact`, `foot_contact_forces`

---

## 5. 指令项

| 指令 | 类型 | 重采样 | 范围 |
|---|---|---|---|
| `twist` | VelocityCommandCommandOnly | 3–8 s | vx ±0.4、vy ±0.3 m/s、wz ±1.0 rad/s |
| `head_pose` | UniformPoseCommand | 2–5 s | 初始 (neck ±0.05, head_pitch ±0.05, yaw ±0.07, roll ±0.015)，由课程放宽至 100% 可达行程 |
| `body_pose` | UniformPoseCommand | 2–5 s | 初始 ±0.005（xyz/m、rad），由课程放宽 |

> 约定：**全零指令 = 站立**（部署空闲态），所以零指令行为必须显式训练。

---

## 6. 奖励表（16 项）

正权重 = 任务项；**负权重 = penalty，其 `Episode_Reward/<name>` 必须 ≤ 0**。
（有例外约定：`mdp.py` 里自否定的 `*_penalty` / `*_l1` 返回 ≤0，配 **正** 权重 —— 加负号会
double-negate 成“奖励违规行为”。）

| 项 | 权重 | 函数 |
|---|---:|---|
| `track_linear_velocity` | **+2.0** | 线速度跟踪（主任务） |
| `track_angular_velocity` | **+2.0** | 角速度跟踪（主任务） |
| `air_time` | +3.0 | 抬脚时长（驱动迈步） |
| `upright` | +2.0 | 躯干直立 |
| `pose` | +1.0 | `variable_posture`（姿态） |
| `head_pose_tracking` | +2.0 | 头部指令跟踪 |
| `body_pose_tracking` | 0.0 | 躯干姿态指令（权重 0，保留槽位/梯度） |
| `head_pose_bias` | 0.0 → **3.0** | 头 DC 下垂惩罚（课程引入，见 §7） |
| `body_ang_vel` | −0.05 | 躯干角速度 |
| `angular_momentum` | −0.02 | 角动量 |
| `action_rate_l2` | −0.1 → **−1.0** | 动作平滑（课程） |
| `foot_clearance` | −2.0 | 足端离地高度偏差 |
| `foot_swing_height` | −0.25 | 摆动高度 |
| `foot_slip` | −0.1 | 足端打滑 |
| `dof_pos_limits` | −1.0 | 关节限位 |
| `self_collisions` | −1.0 | 自碰撞 |

**看 run 的规矩**：总奖励上升 + episode length 符合任务预期 + **每个 penalty 项 ≤ 0** +
主任务项（`track_linear_velocity`）确实在涨（总奖励可能靠 regularizer 上涨而步态没学会）。

---

## 7. 课程（7 条；step = iteration × 24）

| 课程 | 作用 | 阶段 |
|---|---|---|
| `action_rate_weight` | 动作平滑权重 | 0→−0.1, 12k→−0.2, 18k→−0.4, 24k→−0.6, 30k→−0.8, 36k→−1.0 |
| `head_pose_bias_weight` | 头下垂惩罚 | 0→0.0, 14400→1.0, 24000→2.0, 36000→3.0 |
| `standing_envs` | 站立指令占比 | `standing_stages` |
| `head_pose_range` | 头指令范围 5%→100% | `range_stages` |
| `body_pose_range` | 躯干指令范围 | `range_stages` |
| `com_range` | 躯干 CoM DR ±3→±15 mm | `range_stages` |
| `head_com_range` | 头部 CoM DR | `range_stages` |

> 阶段必须与策略实际学会的东西对齐：技能还在探索期就上“动作税”会让“什么都不做”变成最优解。

---

## 8. 域随机化（DR）

| 开关 | 值 | 范围 |
|---|---|---|
| `ENABLE_COM_RANDOMIZATION` | ✅ | ±3 mm → 课程 ±15 mm |
| `ENABLE_HEAD_COM_RANDOMIZATION` | ✅ | ±3 mm → 课程 ±10 mm |
| `ENABLE_MASS_INERTIA_RANDOMIZATION` | ✅ | ×(0.95, 1.05)（质量+惯量同步） |
| `ENABLE_JOINT_FRICTION_RANDOMIZATION` | ✅ | ×(0.9, 1.1)（scale `FrictionDRBamActuator.friction_scale`） |
| `ENABLE_ARMATURE_RANDOMIZATION` | ✅ | ×(0.9, 1.1) |
| `ENABLE_VELOCITY_PUSHES` | ✅ | ±0.3 m/s，每 3–6 s |
| `ENABLE_IMU_ORIENTATION_RANDOMIZATION` | ✅ | 零中心 ±6°（**补偿不了系统性安装偏置**，那是真机标定） |
| `ENABLE_ENCODER_BIAS` | ✅ | ±0.015 rad（≈0.86°）每环境常数 |
| `ENABLE_KP_RANDOMIZATION` / `KD` / `JOINT_DAMPING` / `BASE_ORIENTATION` | ❌ | 关闭 |
| `ENABLE_SYMMETRY`（镜像对称 loss） | ❌ | 关闭 |

> DR 不可跨 reset 累积 —— 自定义 DR 函数必须“先还原再施加”。

---

## 9. Runner / PPO 超参

| 项 | 值 |
|---|---|
| `experiment_name` / `run_name` | `velocity` |
| `max_iterations` | **50000**（cfg 默认；本次目标 15000，用 CLI 覆盖） |
| `save_interval` | 250 |
| `num_steps_per_env` | 24 |
| `wandb_project` | `mjlab_microduck` |
| actor / critic | `(512, 256, 128)`, `elu`, `obs_normalization=True` |
| learning_rate | 1e-3（`schedule="adaptive"`, `desired_kl=0.01`） |
| entropy_coef / clip / gamma / lam | 0.01 / 0.2 / 0.99 / 0.95 |
| num_learning_epochs / num_mini_batches | 5 / 4 |

---

## 10. 训练命令

> ⚠️ 需要 **CUDA GPU**。4096 envs 建议 ≥16 GB 空闲显存（显存不够就减小 `--env.scene.num-envs`，
> 训练仍有效，只是更慢）。velocity 任务比 sitstand 轻（`nconmax=35` vs `200`）。

```bash
# 0) 冒烟测试（必须先跑；1–2 分钟，拦 ~95% 配置错误）
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5

# 1) 正式训练到 15000 iterations
uv run train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 1024 \
  --agent.max_iterations 15000

# 2) 续训（从某个 checkpoint 接着跑）
uv run train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 1024 \
  --agent.load-checkpoint logs/rsl_rl/velocity/<run>/model_15000.pt \
  --agent.resume True \
  --agent.max_iterations 30000
```

**num_envs 选择**：≥16 GB → 4096；8–12 GB → 1024–2048；4–8 GB → 512–1024；<4 GB → 用
`--hf-jobs` 跑在 L4(24 GB)。多卡机器可 `CUDA_VISIBLE_DEVICES=1 uv run train ...`。

**产物**：`logs/rsl_rl/velocity/<timestamp>/model_<iter>.pt`；曲线在 wandb 项目
`mjlab_microduck`（本机是离线 run，`wandb/offline-run-*`）。

**导出**：
```bash
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <entity/project/run_id>
# 或本地 checkpoint：--onnx-file walk_hd1910.onnx --checkpoint <iter>
```

**CPU 彩排**（真机前最后一次；`--motor` 默认已是 `hd1910`）：
```bash
uv run scripts/infer_policy.py --walking walk_hd1910.onnx --motor hd1910
```

---

## 11. 本次换装的连带影响（重要）

- 机器人模型质量从 737.2 g → **779.2 g**（+3 g/舵机）。**所有旧策略都是在错误质量下训的**
  （含昨天那只 800.2 g 的走姿），因此走路必须重训。
- 换舵机同时改了执行器物理（XL330 → HD-1910 的 kt/R/摩擦/kp），旧 ONNX 不可继续用。

---

## 12. 来源（file:line）

- 执行器参数：`src/mjlab_microduck/robot/microduck_constants.py:122-155`
- 延迟单位证据：`.venv/.../mjlab/envs/manager_based_rl_env.py:414-419`、`.venv/.../mjlab/entity/entity.py:830`
- env 设置 / 奖励 / 课程 / 指令：`src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py`
- DR 开关与范围：`src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py:59-145`
- runner cfg：`src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py:1368`（`MicroduckRlCfg`）
- 舵机辨识：`hd1910_calibration/fit_kpfix/mae_report_kpfix.md`、`vendor/bam/bam/params/hd1910/characterization.json`
- 回归测试：`tests/test_hd1910_cfg.py`（锁执行器接线 / 质量 / 延迟单位 / “绝不 12 V”）
