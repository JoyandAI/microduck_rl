# HD1910 腿摆标定 MAE 报告（72 条合并集 · kp 修正后重跑）

## 1. 数据

- 总集 `hd1910_calibration/bench_combined72_processed`：**72 条** = arm15 36 + arm10 36
  （4 轨迹 × 3 配重 × 3 重复 × 2 臂长）
- 划分：**拟合 48 条**（rep0+rep1）/ **独立验证 24 条**（rep2 留出，每组最后一次重复）
- 完整性：无碰撞、无越限（|θ|≤78.8°，限位 +96.7/−88.4°）；vin 逐条实测 5.0–5.2 V
- ⚠️ 修正：`log["kp"]` 从 10272 改为 **32**（见第 4 节）

## 2. 钉定的参数（全部来自独立实测，不参与拟合）

| 参数 | 值 | 依据 |
|---|---:|---|
| `kp` | 32 | 寄存器 reg50 回读（FDdebug 交叉验证） |
| `error_gain` | 0.163 | 实测 P 斜率 5.18 duty/rad ÷ kp 32 |
| `kd` | 0.346 | 实测 D 斜率 0.0141 duty/(rad/s) × d_scale 4 ÷ 0.163（等效值，见第 4 节） |
| `kt` / `R` | 0.7358 / 3.75 | 规格书（⚠️ 台架标定仍为 TODO） |
| `max_velocity` | 8.03 | 5 V 供电下按电压折算 |
| `max_current` | 3.25 | 寄存器 reg28=500 回读 |

放开拟合：摩擦全套 + `armature` + `q_offset` + `command_delay`

## 3. 结果（门限：独立验证 MAE < 0.157 rad）

| 模型 | 拟合 MAE(rad) | 验证 MAE(rad) | 判定 | 仿真饱和% | 真机饱和% |
|---|---:|---:|:--:|---:|---:|
| m5 | 0.00811 | 0.00823 | PASS | 21.3 | 4.2 |
| m6 | 0.00831 | 0.00847 | PASS | 21.3 | 4.2 |
| m3 | 0.00857 | 0.00869 | PASS | 21.3 | 4.2 |
| m4 | 0.00885 | 0.00897 | PASS | 21.3 | 4.2 |
| m1 | 0.01329 | 0.01341 | PASS | 21.3 | 4.2 |
| m2 | 0.01329 | 0.01342 | PASS | 21.3 | 4.2 |

**6/6 PASS**，train≈validation 无过拟合。推荐 **m5**（验证 MAE 最小）；
追求简洁可选 **m3**（3 个摩擦参数，MAE 仅差 5%）。

## 4. 相对旧拟合（kp=10272 bug）的改善

| 模型 | 旧·同步 | 旧·自由回放 | 新·同步 | 新·自由回放 | 同步提升 |
|---|---:|---:|---:|---:|---:|
| m1 | 0.02484 | 0.03211 | 0.01341 | 0.01927 | **1.85×** |
| m2 | 0.02769 | 0.03282 | 0.01342 | 0.01926 | **2.06×** |
| m3 | 0.02538 | 0.03258 | 0.00869 | 0.01525 | **2.92×** |
| m4 | 0.02446 | 0.02978 | 0.00897 | 0.01357 | **2.73×** |
| m5 | 0.02537 | 0.03106 | 0.00823 | 0.01378 | **3.08×** |
| m6 | 0.02665 | 0.03187 | 0.00847 | 0.01432 | **3.15×** |

（旧拟合自身报告的 score 0.0289~0.0298 与我实测的自由回放 0.0298~0.0328 一致 ✓）

## 5. ⚠️ 必须记录的结构性问题

**bug**：`scripts/record_pendulum_bench.py:479` 用 `r16(50)` 读 Kp，但 reg50=Kp、reg51=Kd 是
两个独立 8 位寄存器 → `log["kp"] = 32 + 40×256 = 10272`。而 `bam/feetech/actuator.py:213`
是 `self.kp = log["kp"]`（日志值直接覆盖模型默认 32），使模型 P 增益
= 10272 × error_gain，实测真值 5.18 duty/rad 的 **323 倍** → 控制器全程 bang-bang（占空比饱和 94%）。

**D 项**：`d_scale=4` 沿用 HLS 惯例、从未标定。实测 `duty = 5.18·Δq − 0.0141·dq`
（36 条日志回归，R²=0.92~0.996），而模型给出 D_eff = kd·eg/d_scale = 1.63 → **116 倍**。
本次用等效 `kd=0.346` 复现实测 D 项；固件寄存器 kd=40 不变，问题在 `d_scale`。

**其它未标定/未建模**：`kt`/`R` 用规格书值；`arm_mass=0`（摆杆自身质量忽略，
arm10 与 arm15 的偏差不同，可能污染摩擦）；仿真饱和率 21% 仍高于真机 4.2%。

**Stribeck 项不被数据支持**：m2 的 `friction_stribeck`=4.8e-18≈0（m2 退化为 m1，
两者 MAE 相同），m4/m5/m6 的 Stribeck 项同样收敛到 ~0。

## 6. 各模型摩擦参数

| 模型 | friction_base | friction_viscous | 其它 |
|---|---:|---:|---|
| m1 | 0.01718 | 0.00744 |  |
| m2 | 0.01712 | 0.00746 | friction_stribeck=4.755e-18, dtheta_stribeck=0.3038, alpha=4.553 |
| m3 | 0.01297 | 0.00404 | load_friction_base=0.2826 |
| m4 | 0.01061 | 0.00795 | friction_stribeck=6.921e-07, load_friction_base=0.07218, load_friction_stribeck=0.2568, dtheta_stribeck=1, alpha=9.34 |
| m5 | 0.01492 | 0.00563 | friction_stribeck=8.535e-07, load_friction_motor=0.194, load_friction_external=0.1285, load_friction_motor_stribeck=0.2557, load_friction_external_stribeck=4.829e-05, dtheta_stribeck=0.9995, alpha=1.963 |
| m6 | 0.01535 | 0.00511 | friction_stribeck=8.56e-06, load_friction_motor=0.3092, load_friction_external=0.1423, load_friction_motor_stribeck=0.1199, load_friction_external_stribeck=0.0007551, load_friction_motor_quad=0.005892, load_friction_external_quad=0.005573, dtheta_stribeck=0.5464, alpha=5.648 |
