# HL-2909 摆锤台架辨识实验 —— 实验步骤 × 仓库代码对照

> **文档定位**：为本仓库（`Rhoban/bam` 工作副本，已新增 `HLS2909Actuator`）设计
> **HL-2909-C001 的完整辨识实验流程**。流程依据
> `microduck_rl/docs/pendulum_bench_runbook.html`（速查手册，唯一参数表）
> 与 `make_pendulum_experiment_design_html.py`（详细实验设计）设计，并
> **逐步骤映射到两边的仓库代码**：本仓库（bam 模型库）与 `microduck_rl`
> （台架采集/接线脚本 + vendor 部署）。
>
> 本文档不改变 runbook 的取值与安全红线；runbook 与本文冲突时**以 runbook 为准**。

---

## 0. 现状与目标

### 0.1 已完成（本仓库工作树）

| 项 | 位置 | 说明 |
|---|---|---|
| 执行器模型 | `bam/feetech/actuator.py` `HLS2909Actuator` | 二阶限幅目标爬坡（速度+加速度）、P(1/8)+D(1/4)、固件限流、`stateful=True` |
| 注册 | `bam/actuators.py` `actuators["hls2909"]` | `load_model` / `bam.fit --actuator hls2909` / mjlab 入口 |
| 文档条目 | `docs/usage/actuators.rst`、`README.md` | 已挂 "mujoco_cpu / mjlab_gpu" 按钮 |
| 临时参数 | `bam/params/hls2909/m1.json` | **非摆锤标定值**，仅行为级校准（见 0.2） |

### 0.2 为什么要做这个实验

`bam/params/hls2909/m1.json` 当前值：

```json
{ "kt": 1.454, "R": 20.0, "armature": 0.001, "q_offset": 0.0,
  "friction_base": 0.08, "friction_viscous": 0.012, "error_gain": 0.166,
  "kd": 32, "max_velocity": 19.1637, "max_acceleration": 500.0,
  "max_current": 1.95, "model": "m1", "actuator": "hls2909" }
```

- `kt`/`R`/`max_velocity`/`max_current`/`kp`/`kd`/`error_gain` 来自
  数据手册 + 探针实测（内部一致组，基本可信）；
- **`friction_base=0.08` / `friction_viscous=0.012` / `armature=0.001` 是
  "已知良好策略在真机站立"的行为级反推**，未知误差有多大——这正是
  `HLS2909Actuator` docstring 中"still await a proper pendulum identification"的含义；
- microduck 侧 `src/mjlab_microduck/robot/microduck_constants.py` 的
  `_BAM_ACTUATOR_KWARGS`（`model="m1"`）同样标注 **PLACEHOLDER，标定后升级**。

**目标**：用 BAM 论文（[arXiv 2410.08650](https://arxiv.org/pdf/2410.08650v1)）的
摆锤辨识协议，对单只 HL-2909 拟合 M1–M6 六个摩擦模型，以独立验证 MAE 选档，
落地 `bam/params/hls2909/mN.json`，同步 vendor + microduck 常量，冒烟后长训。

> 命令约定：§3 中 `python3 microduck_rl/scripts/...` 以仓库列表目录
> （`/home/joyandai`）为 cwd；`cd microduck_rl` 后也可去掉前缀。
> 涉及"本仓库"（bam 模型库）的命令以 `bam/` 为前缀，在该仓库根目录执行。

---

## 1. 原理速记（辨识对象与代码路径）

摆锤台架提供**已知惯量**与**已知重力矩**，是辨识以下量的标准装置：

```
待辨识：kt, R, armature, q_offset, error_gain, kd
        + 摩擦预算 friction_base / friction_viscous / friction_stribeck /
          dtheta_stribeck / alpha / load_friction_{motor,external}[_stribeck|_quad]
固件已知（固定不优化）：kp=32, kd=32(缩放1/4), error_gain=0.166(初值),
        max_velocity=19.1637, max_acceleration=500.0, max_current=1.95
```

M1–M6 定义（`bam/model.py` `models` 注册表）：

| 模型 | 特征 | 相对 M1 增加的可辨识摩擦项 |
|---|---|---|
| m1 | Coulomb | — |
| m2 | Stribeck | `friction_stribeck`, `dtheta_stribeck` |
| m3 | Load-dependent | `load_friction_motor/external` |
| m4 | Load-dependent + Stribeck | + `*_stribeck`, `dtheta_stribeck`, `alpha` |
| m5 | 上 + Directional | 同 m4（拆分驱动侧/负载侧） |
| m6 | 上 + Quadratic | + `*_quad` |

### 1.1 台架几何 → 标准 Pendulum 的等价化（关键：bam 侧零改动）

BAM 的回放与拟合统一使用 `bam/testbench.py` 的 `Pendulum`
（`mass`/`arm_mass`/`length` 三点 + `compute_bias` 的 `sin(q)` 重力矩）。
台架真实几何（端部点质量 m_tip + 均匀杆 m_arm + 近轴 m_hub）与标准 Pendulum
动力学恒等的等价参数（`scripts/record_pendulum_bench.py::pendulum_equiv`）：

```
M     = m_tip·L² + m_arm·L²/3 + m_hub·r_hub²            (绕轴惯量)
B_max = (m_tip + m_arm/2)·g·L + m_hub·g·r_hub            (最大重力矩)
L_eq  = M·g / B_max      m_eq = B_max²/(M·g²)            (arm_mass=0)
```

每条 log 只写 `mass=m_eq, arm_mass=0.0, length=L_eq`，
`bam.testbench.Pendulum` 无需任何修改即可复现同一动力学。

### 1.2 bam 拟合回路（下一步 E 关的引擎）

```
bam/fit.py          :python -m bam.fit  → optuna/CMA-ES 最小化位置 MAE
  ├─ bam.model.Model(m1..m6)            摩擦模型 + 参数
  ├─ bam.actuators["hls2909"]           HLS2909Actuator（compute_control/compute_torque）
  ├─ bam.testbench.Pendulum             由 log["mass"/"arm_mass"/"length"] 构造
  ├─ bam.simulate.Simulator.rollout_log 回放（simulate_control=True 用固件律）
  └─ bam.logs.Logs                      读 log 目录（JSON 格式见 §5）
```

---

## 2. 实验台与环境（摘要）

机械 5 条硬约束、BOM、接线详见 runbook §1–§2（`cad/pendulum_bench/` 可打印参考）。
要点：

1. 输出轴**水平**，摆臂在竖直平面内 ±90° 自由摆动；
2. 轴心高 ≥ 臂长 + 50 mm；台架固定牢靠（C 夹 ×2），晃动 < 1°；
3. **全部转动件可拆下称重**（秤 0.1 g，卡尺 ±1 mm），摆长 L 按
   "轴心 → 端部总质量**质心**" 实测；
4. 总线**只接台架这一只**（建议 ID=1，鸭总线断开），12 V 电源经 3 A 保险丝 + 急停；
5. `m_tip·g·L ≤ 0.35 N·m`（runbook 红线，脚本 `--torque-budget` 硬检查）。

**本次唯一参数表**（runbook §3，各值在每次装夹后实测后填写）：

| 量 | 建议值 | 精度 |
|---|---|---|
| L | 0.150 m | ±1 mm |
| m_tip（三档） | 50 / 100 / 150 g | 0.1 g（含托盘+砝码+夹紧件整套称） |
| m_arm | 12–25 g（碳纤 Ø6，越轻越好） | 0.1 g |
| m_hub | 15–30 g | 0.1 g |
| r_hub | ≈0.01 m | ±2 mm |

> 版本说明：`make_pendulum_experiment_design_html.py`（详细版设计文档）中为
> 50/100/160 g、3 轨迹、5 重复、45 条；**runbook 为 50/100/150 g、4 轨迹、
> 3 重复、36 条——本文档以 runbook 为准**（用户指定）。差异不影响流程，
> 仅影响样本量。

---

## 3. 实验过程步骤（逐关放行）—— 每步 × 对应仓库代码

总流程：**A 机械验收 → B 只读检查 → C 试跑 → D 36 条正式采集 → E M1–M6 拟合 → F 验收落地**。
任一关失败停在当前关，不带病采完整批。

### A. 机械验收与称量

| 项 | 内容 |
|---|---|
| 动作 | 单独称 m_arm、m_hub（+r_hub）、三档 m_tip、装好后测 L；断电手动摆过 ±90° |
| 通过 | C 夹锁死；轴水平；全部数值有记录；最大档重力矩 ≤ 0.35 N·m |
| 对应代码 | 无（人工）。几何 → 等价参数在 **D 关入口**由 `record_pendulum_bench.py::pendulum_equiv()` 计算，并做 `--torque-budget` 拒绝检查（超限直接报错退出） |

### B. 接线与只读检查

```bash
# 电：+12V → 3A 保险丝 → 急停 → 舵机；GND 共地（电源/USB-TTL/舵机）；TTL DATA 半双工
# ① 扫描/改 ID/确认位置模式（只读 + 写 ID + 模式，不动扭矩）
python3 microduck_rl/scripts/setup_bench_servo.py --port /dev/ttyACM0 --find-id 23 --set-id 1
```

| 项 | 内容 |
|---|---|
| 检查项 | 只发现 1 只舵机；11.5–12.6 V；温度 < 45 °C；reg33 mode=0；无错误状态 |
| 对应代码 | `microduck_rl/scripts/setup_bench_servo.py`（FT-SCS 扫描=ping reg62、改 ID=reg5、回读 Kp/Kd/Ki=reg21/22/23、电流保护=reg28、限位=reg9/11、电压=reg62、温度=reg63） |
| 对应 bam 初值 | `bam/feetech/actuator.py::HLS2909Actuator.__init__/initialize` 中的 `kp=32 / kd=32 / error_gain=0.166 / max_current=1.95 / max_velocity=19.1637 / max_acceleration=500.0` 即来自同一批固件寄存器，回读值与模型默认值必须一致 |

### C. 50 g 单条试跑

```bash
python3 microduck_rl/scripts/record_pendulum_bench.py --port /dev/ttyACM0 --id 1 \
  --tip-mass 0.05 --arm-mass 0.018 --arm-length 0.15 --hub-mass 0.020 --hub-radius 0.010 \
  --trajectory up_and_down --reps 1 --out microduck_rl/hls2909_calibration/pilot
```

| 项 | 内容 |
|---|---|
| 脚本自动做 | 临时限流 reg44=150（0.975 A，结束恢复原值）→ 位置伺服配置（reg33=0、reg41=0、reg46=32767）→ **零位 q_zero**（断电扭矩 30 读中位，reg56）→ **符号检测**（+0.2 rad 试步，Δ 符号 → sign）→ 逐拍写目标 reg42 / 读 reg56/60/62/69/63 |
| 观察/通过 | 正角方向正确；0 rad=自然下垂；无滑动/碰撞；> 50 样本；温升 < 5 °C；采样中位 ≤ 20 ms |
| 对应代码（轨迹） | `microduck_rl/scripts/record_pendulum_bench.py`（内置 `_traj` 副本，**无 bam 依赖**，轨迹定义与 `bam/trajectory.py` 一致：`sin_time_square`/`sin_sin`/`lift_and_drop`/`up_and_down`，每条 6 s） |
| 对应代码（角度） | 12-bit 编码器 `LSB_RAD = 2π/4096`；速度 = 相邻拍位置差分（**不用 reg58**，量化 0.077 rad/s 太粗） |

### D. 正式 36 条（3 档 × 4 轨迹 × 3 重复，约 40–60 min）

```bash
python3 microduck_rl/scripts/record_pendulum_bench.py --port /dev/ttyACM0 --id 1 \
  --tip-mass 0.05 --tip-mass 0.10 --tip-mass 0.15 \
  --arm-mass 0.018 --arm-length 0.15 --hub-mass 0.020 --hub-radius 0.010 \
  --trajectory sin_time_square --trajectory sin_sin --trajectory lift_and_drop --trajectory up_and_down \
  --reps 3 --limit-a 0.975 --torque-budget 0.35 --out microduck_rl/hls2909_calibration/bench
```

| 项 | 内容 |
|---|---|
| 执行顺序 | 外层 m_tip（50→100→150）→ 中层轨迹（4 种）→ 内层 rep（0..2）；每进一档质量：**关扭矩 → 打印"换砝码" → 等操作者输 `START`** → 才继续 |
| 安全链（内置） | 温度 > 50 °C / \|I\|>1.0 A×0.7 s / 单条 30 s 看门狗 → 立即 reg46=0,40=0,55=0 断扭矩 + 恢复原限流 + 退出 |
| 中途异常 | 松动/碰撞/异响/线束拉扯 → 急停，**整组 3 次重录**（该组数据作废） |
| 通过 | `manifest.json` 中 records=36；每条 > 50 点；无安全中止；电压/采样周期无明显漂移 |
| 对应代码 | `record_pendulum_bench.py` 全部 + `microduck_rl/scripts/read_hls_registers.py`（FT-SCS 帧：`FF FF|ID|LEN|INSTR|PARA|CHECK`，`read_regs`/`write_regs`/`sm11`/`sm16`/`u16_le`）；安全 = `record_pendulum_bench.py::Safety` |
| 输出格式 | 每条 log 直接兼容 bam 的 `rollout_log`（字段见 §5）；另存 `telemetry`（duty/vin/current/temp，体检用，不参与拟合） |

### E. M1–M6 拟合 + 独立验证

```bash
cd microduck_rl
.venv/bin/python scripts/fit_leg_pendulum.py --logdir hls2909_calibration/bench \
  --actuator hls2909 --models m1 m2 m3 m4 m5 m6 --trials 20000 \
  --out hls2909_calibration/fit
```

| 项 | 内容 |
|---|---|
| 分组 | 每个 "轨迹×质量" 的**最后一次重复**留出（3 档 × 4 轨迹 = 12 条独立验证），其余 24 条拟合 |
| 拟合引擎 | `fit_leg_pendulum.py` 子进程调 `bam.fit`（`python -m bam.fit --logdir … --actuator hls2909 --model mN --method cmaes --trials 20000 --output fit_mN.json`），即 `bam/fit.py`：optuna（CMA-ES/BIPOP）最小化 `compute_score` = 各 log 位置 MAE 均值（`bam.simulate.Simulator.rollout_log(simulate_control=True)`） |
| 独立验证 | `fit_leg_pendulum.py::evaluate`：`bam.model.load_model(fit_mN.json)` + `rollout_log(reset_period=0.5, simulate_control=True)` 逐 log 求 MAE —— 与 `bam.fit.compute_score` 同口径 |
| 输出 | `fit/` 下 `fit_m1.json … fit_m6.json` + `mae_report.md`（train/val MAE 表 + PASS/FAIL） |
| 通过 | 12 条留出数据验证 MAE < 0.157 rad（≈ 10% × π/2）；train/validation 无分叉；同等通过时**选 m1** |
| 配套（可选） | `bam/plot.py --sim --params fit_m1.json --params fit_m6.json` 画回放对比图；`bam.fit --eval` 重评 |

**等价直跑（本仓库，不用 microduck 包装）**：

```bash
cd bam
uv sync --extra identification
uv run python -m bam.fit --actuator hls2909 --model m1 \
  --logdir ../microduck_rl/hls2909_calibration/bench \
  --output bam/params/hls2909/m1.json --trials 100000 \
  --set "{'max_velocity':19.1637,'max_acceleration':500.0,'max_current':1.95}"
```

> `--set` 把**固件出厂限幅**（限速/限加速度/限流）固定为真值，其余
> （kt/R/armature/q_offset/error_gain/kd/摩擦全套）交给优化——与
> runbook §2.3 的"限幅（出厂值固定）"一致；否则 CMA-ES 会把已知量当自由度漂移，
> 容易产生假性低 MAE。`fit_leg_pendulum.py` 未暴露 `--set`；
> 若要固定，用上面这条直跑命令或在 wrapper 中加透传。

**M1–M6 全部拟合完再判档**（`bam.fit --validation_kp` 不适用于本协议：
我们不扫 kp，验证集按"最后一次重复"留出）。

### F. 验收与落地

1. **判档**：`mae_report.md` 中 PASS（validation MAE < 0.157 rad）且 train≈val → 选参数最少的模型（默认 m1）；若简单模型明显差而 m6 显著更优且无分叉，选 m6 并写理由。
2. **落地 bam 库**：`fit_mN.json` → `bam/params/hls2909/mN.json`（建议至少 m1 + m6 两档），
   `mN.json` 的 `"model"`/`"actuator"` 字段自动满足 `load_model` 约定。
3. **落地 microduck**：
   - `vendor/bam/bam/params/hls2909/mN.json`（editable install，与 2 保持同步）；
   - `src/mjlab_microduck/robot/microduck_constants.py :: _BAM_ACTUATOR_KWARGS`：
     `model="m1"` 升级为判档结果（**需你点头**，见该文件注释），
     `kp_fw=32.0` 已为出厂真值，`vin_range` 等占位值按实测更新。
4. **冒烟 → 长训**：`scripts/eval_sim_vs_real_step.py`（单舵机阶跃交叉验证，备用）、
   `scripts/validate_bam_testbench.py` / `scripts/testbench_sim2real.py`（microduck）通过后开长训。
5. **回归**：`hls2909_calibration/` 数据目录 zip 备份；提交前跑 `uv run pytest`（若识别流水线测试存在）。

---

## 4. 变量控制表

**自变量（3 维正交网格，36 条）**：

| 维 | 水平 | 作用 |
|---|---|---|
| m_tip | 50 / 100 / 150 g | 负载相关摩擦项（m3–m6）可辨识 |
| 轨迹 | sin_time_square（速度扫频 ±57°）、sin_sin（多频复合 ±90°）、lift_and_drop（抬升→断扭矩自由落体）、up_and_down（低速 0→+90°→+72°） | 全速度谱/谐波/背驱 Stribeck/静态摩擦 |
| 重复 | 3 | 噪声平均；最后 1 次留作独立验证 |

**控制变量（全程恒定，逐条记录）**：

| 变量 | 设定 | 保证方式 |
|---|---|---|
| Kp / Kd | 32 / 32 | 不写寄存器；B 关回读确认（reg21/22） |
| 电流限幅 | 0.975 A（reg44=150） | 脚本写入；每条结束/异常恢复开始前读到的值 |
| 轨迹时长 | 6 s | `trajectory.duration`；每条后回零 + 0.5 s |
| 采样 | 目标 ~6 ms/拍 | 循环计时；实际时间戳如实记录（期望 6–15 ms） |
| 几何 | L、r_hub、杆/臂不换 | 单次实验不拆装；换档只换砝码且重称重填 `--tip-mass` |
| q_zero / sign | 每次运行重测 | 脚本自动（30 读中位 / ±0.2 rad 试步）并写 manifest |
| 温度上限 | 50 °C | 每拍监控，越限即停 |
| 舵机状态 | mode=0；reg41=0（最大加速度）；reg46=32767 | record 统一写 |
| vin | 12.0 V 稳压电源 | 逐拍记录 reg62；每条 log 写实测中位数 |

**干扰对策**（runbook §4.3 摘）：温度漂移（每 10 条打印；>45 °C 暂停 2 min）、
总线抖动（dt 中位 3–15 ms；丢拍 >30% 判废）、人手触碰/振动（无命令跳变 >5° 判废该条）、
台架共振（垫胶；某档全部异常 → 检装配）、砝码松动（M6 螺纹胶；每档前查）、
电压漂移（<11.5 V 暂停）。

---

## 5. 数据格式 ↔ bam 拟合管线（零改动兼容性）

`record_pendulum_bench.py` 每条约 1100 点，文件名
`{trajectory}_tip{m}g_rep{r}.json`，顶层：

```json
{ "motor": "hls2909", "kp": 32, "vin": 12.5, "dt": 0.006,
  "mass": 0.1140, "arm_mass": 0.0, "length": 0.1928,
  "trajectory": "up_and_down", "tip_mass_kg": 0.1, "rep": 0,
  "entries": [
    { "position": 0.0, "speed": 0.0, "goal_position": 0.0, "torque_enable": true }, …
  ],
  "telemetry": [ { "t": 0.0, "duty": 0.0, "vin": 12.4, "current_A": 0.1, "temp": 31 }, … ] }
```

| 字段 | 含义 | bam 侧消费方 |
|---|---|---|
| `motor`/`kp`/`vin` | 执行器注册名 / 固件 P / 实测电压中位数 | `bam/actuator.py::DCMotorActuator.load_log`（kp、vin 覆盖默认值） |
| `dt` | 相邻拍时间戳**中位数** | `Simulator.rollout_log` 的步进 |
| `mass`/`arm_mass`/`length` | m_eq / 0.0 / L_eq | `bam/testbench.py::Pendulum`（惯量 + 重力矩） |
| `entries[].position` | θ（reg56 − q_zero）×LSB_RAD×sign | **拟合目标**（MAE 对象） |
| `entries[].speed` | 前向差分速度 | 回放起点/`simulate_control` 输入 |
| `entries[].goal_position` | 轨迹命令 | 回放时注入固件控制律 |
| `entries[].torque_enable` | t<2 s 断扭矩等 | 背驱段（只重力+摩擦） |
| `telemetry` | duty/vin/current/temp | 仅体检/安全，**不参与拟合** |

兼容性要点：

- `bam.logs.Logs` 读目录下全部 `*.json`。**注意**：`record_pendulum_bench` 输出目录里的
  `manifest.json` 没有 `entries` 字段，`bam.fit` 直接指到 raw 目录会失败——
  `fit_leg_pendulum.py` 已把 manifest 排除（只拷贝轨迹 log 到临时目录再拟合）；
  直跑 `python -m bam.fit` 时请指到 `bench_processed/`（`process_bench_logs.py` 输出）。
- `Simulator.rollout_log` 只要求 `dt` 标量 + `entries` 四键，**不需要**
  `timestamp`；record 使用变间隔，`fit_leg_pendulum` 直接用 `dt` 中位数拟合。
  若要均匀网格（更稳）：`scripts/process_bench_logs.py --in bench --out bench_processed --dt 0.005`
  （注意它按 `entries[].t` 插值，与 `bam/process.py` 的 `timestamp` 键不同，两者不可混用）。
- **stateful 执行器在拟合中的行为**：`HLS2909Actuator` 的
  `q_target_smooth/v_target_smooth` 在每条 `rollout_log` 开始时经
  `Simulator.reset → Model.reset → Actuator.reset()` 清空，首次
  `compute_control` 从当拍 `q` 初始化（= 舵机上电行为）→ 每条 log 独立、
  确定性回放，批处理（`Logs.make_batch` 向量化）同样成立。

---

## 6. HLS-2909 特有约束与已知问题

1. **固件二阶目标限幅**：内部目标先限速（`max_velocity`，reg 84=250 LSB=183 RPM≈19.16 rad/s）
   再限加速度（`max_acceleration`；实验写 reg41=0=固件最大，模型默认 500 rad/s² 用于训练）。
   实验中 reg41=0 与模型 500.0 的差异由拟合吸收？（`max_acceleration` 是模型参数，
   建议 E 关 `--set` 固定 500 或允许优化并记录结果；**不要**同时固定与训练策略不一致的假设）。
2. **位置环 P(1/8) + D(1/4) 缩放**：`duty = Δq·kp·error_gain/8 + (−dq)·kd·error_gain/4`，
   Ki 位置模式无效；`kd` 初始 32（出厂 reg22/51 实测）。
3. **限流模型**：固件 reg28/44 = 300×6.5 mA = 1.95 A（出厂保护值）；
   实验临时降为 150（0.975 A）仅为防堵转发热——**拟合与部署用 1.95 A**。
4. **编码器 12-bit**（0.087°/LSB = 0.001534 rad）：零位展开用
   `((pos − q_zero + 2048) % 4096) − 2048`；速度用相邻拍差分，不用 reg58。
5. **已修复的兼容性 bug（本次改动）**：新 `HLS2909Actuator.compute_control` 原有
   `if dt > 0:` 在 **bam.fit 批处理回放**（`Logs.make_batch` 后 dt 是 n_logs 向量）下抛
   "ambiguous truth value"。已改为 `if np.any(np.asarray(dt) > 0):`
   （`bam/feetech/actuator.py`），标量单条与 mjlab 路径不受影响；
   若直接以本仓库跑 `python -m bam.fit`，此修复为前置条件。
6. **vendor/老版本差异**：`microduck_rl/vendor/bam` 里的旧 `HLS2909Actuator`
   （kp=200 占位、max_velocity=8.06、armature=1e-4、kd=0、目标从 0 起爬）
   与本仓库新版（kp=32、19.16、1e-3、kd=32、从当前 q 起爬）**数值语义不同**，
   拟合/训练必须统一使用新版；vendor 同步时注意带 `bam/feetech/actuator.py`
   全文 + `bam/params/hls2909/`。

---

## 7. 验收决策规则

```
独立验证 MAE < 0.157 rad ?
 ├─ 是 且 train ≈ validation  → ✓ 落地（同等通过选 m1；m6 显著更优且无分叉则选 m6+说明）
 ├─ 是 但 train ≪ validation  → 过拟合：减小自由度（--set 固定已知量）/ 加重复
 └─ 否
      ├─ 看 overlay：落体段不重合 → 摩擦/Stribeck（考虑 m6）；
      │   驱动段相位平移 → 零位/符号/q_offset；饱和段好 → 限流模型正确
      ├─ 回查数据：摆幅/样本数/装配/换砝码重称重填是否执行
      └─ 仍失败 → 提高 trials、检查固定参数、或补充 5 重复采样
```

判档后**必须**做的事：更新 `bam/params/hls2909/mN.json`（本仓库）与
`vendor/bam/bam/params/hls2909/` + `microduck_constants.py` 模型号，做冒烟，
并确认没有把"实验临时限流 0.975 A"带进模型（模型应保持 1.95 A）。

---

## 8. 安全红线（执行时）

- `m_tip·g·L ≤ 0.35 N·m`（L=0.15 m、忽略杆/轮毂力矩时 m_tip ≤ ~238 g，计入后约
  ≤ ~225 g；**禁止加大砝码**）；
- 摆动平面内手勿入/头勿探；砝码必须 M6 锁紧螺母 + 螺纹胶机械防松（甩出是最大风险）；
- 全程人在场（`lift_and_drop` 第 2 s 起断扭矩自由落体）；
- 异常断电顺序：**先断 12 V，再拔 TTL**；数据目录即时 zip 备份；
- 红线触发（T>50 °C / |I|>1.0 A×0.7 s / 看门狗 30 s）由脚本自动断扭退出，人工复核并修复。

---

## 9. 交付清单

| 文件/目录 | 作用 |
|---|---|
| `microduck_rl/hls2909_calibration/bench/` | 36 条原始 JSON + `manifest.json`（勿手工改） |
| `microduck_rl/hls2909_calibration/fit/` | `fit_m1..m6.json` + `mae_report.md`（+ overlay 图） |
| `bam/params/hls2909/m1..m6.json` | **本仓库最终模型**（至少 m1 + m6） |
| 同步 | `vendor/bam/bam/params/hls2909/`；`microduck_constants.py::_BAM_ACTUATOR_KWARGS`（model 档位，需确认） |
| 可提交上游 | `bam/feetech/actuator.py`（新版 HLS2909Actuator）、`bam/actuators.py` 注册、`bam/params/hls2909/`、`docs/usage/actuators.rst`（补照片）、`README.md`、本文档（可英文化） |

---

## 10. 附录

### 10.1 关键文件对照表

| 环节 | 本仓库（bam 模型库） | microduck_rl（台架/部署） |
|---|---|---|
| 执行器模型 | `bam/feetech/actuator.py::HLS2909Actuator`（新） | vendor 同文件（旧版，需同步） |
| 注册/加载 | `bam/actuators.py`；`bam/model.py::load_model` | `vendor/bam/bam/actuators.py` |
| 台架/采集 | —（`bam/feetech/record.py`、`all_record.py` 是 pypot/STS3215 7.4V 旧管线：无 HLS 12V 寄存器语义、无安全链、kp 扫描协议也与此不同，**本协议不使用**） | `scripts/setup_bench_servo.py`、`scripts/record_pendulum_bench.py`、`scripts/read_hls_registers.py` |
| 后处理 | `bam/process.py`（timestamp 键，与本协议不同） | `scripts/process_bench_logs.py`（t 键插值，本协议使用） |
| 拟合 | `bam/fit.py` + `bam/simulate.py` + `bam/testbench.py` + `bam/logs.py` | `scripts/fit_leg_pendulum.py`（bam.fit 包装 + 独立验证 + maep_report） |
| 回放对比 | `bam/plot.py` | `scripts/eval_sim_vs_real_step.py`、`scripts/validate_bam_testbench.py` |
| 部署常量 | — | `src/mjlab_microduck/robot/microduck_constants.py::_BAM_ACTUATOR_KWARGS` |

### 10.2 寄存器速查（FT-SCS，1 Mbps；详见 `microduck_rl/docs/feetech_hls_memtable.md`）

| reg | 含义 | 实验中 |
|---|---|---|
| 5 | ID | 1 |
| 9/11 | 角度限位 | 0/4095（多圈） |
| 21/22/23 | 位置环 P/D/I（EPROM） | 32/32/0 |
| 28(44) | 保护电流 6.5 mA/LSB | 300→**150**（0.975 A）临时；结束恢复 300 |
| 33 | 模式 | 0（位置伺服） |
| 41(85/86) | 加速度限幅 | 0（固件最大） |
| 42 | 目标位置 | 逐拍写入 |
| 46(47) | 运行速度/扭矩开关 | 32767（最大）；安全/中断时 0 |
| 56 | 当前位置（LSB） | 读数 → θ |
| 60 | 占空比 0.1% | telemetry |
| 62 | 电压 0.1 V | telemetry / log vin |
| 63 | 温度 °C | 安全监控 |
| 69 | 电流 6.5 mA | 安全监控 |

### 10.3 文档对照

- **速查手册**：`microduck_rl/docs/pendulum_bench_runbook.html`（唯一参数表/命令/安全）
- **详细设计**：`microduck_rl/scripts/make_pendulum_experiment_design_html.py`
  生成的 `pendulum_bench_experiment_design.html`（参数与 runbook 有出入，见 §2 说明）
- **上游辨识文档**：本仓库 `docs/identification/{setup,actuator_modeling,acquisition,fitting,contributing}.rst`
- **论文**：[arXiv 2410.08650](https://arxiv.org/pdf/2410.08650v1)
