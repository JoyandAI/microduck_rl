# 摩擦辨识实验:为什么用这 4 条轨迹,以及 M1–M6 摩擦参数是怎么算出来的

> **文档定位**:面向 microduck_rl 更换舵机(Dynamixel XL330 → Feetech HL-2909)后的**单自由度摆锤辨识实验**。用 BAM 论文([arXiv 2410.08650](https://arxiv.org/pdf/2410.08650v1)、*Extended Friction Models for the Physics Simulation of Servo Actuators*)**摆锤协议**,对**单只 HL-2909** 拟合 M1–M6 六个摩擦模型。本文回答两个问题:
> **① 为什么实验选这 4 条轨迹、3 档砝码、3 次重复这些具体值?** **② M1–M6 的摩擦参数到底是怎么算出来的?**
>
> **权威性与来源(冲突时以此为准)**
> - **唯一参数表**:`microduck_rl/docs/pendulum_bench_runbook.html`(速查手册)。**与它冲突处以它为准。**
> - 实验步骤 × 代码对照:`/home/joyandai/bam/docs/identification/hls2909_pendulum_experiment.md`。
> - 公式权威(模型定义):`bam/docs/theory/models.rst`;实现权威:`bam/model.py`、`bam/mjlab.py`。
> - 本文所有代码引用均以**实际读到的文件(含行号)**为准;所有数值/公式均与**所读代码**一致,未发明。
>
> **路径约定**:`bam/…` 相对于 bam 模型库根目录 `/home/joyandai/bam`(其包在 `bam/bam/` 下,故 `bam/model.py` 即 `/home/joyandai/bam/bam/model.py`);`scripts/…` 相对于 `/home/joyandai/microduck_rl`;`microduck_rl/docs/…` 同仓库根。

---

## 0. 一句话总结

- **为什么用这 4 条轨迹?** 因为要拟合的是"**摩擦–速度(与负载)**关系",激励必须同时扫过四种互补工况:**全速度谱**(`sin_time_square`)、**多频复合/谐波**(`sin_sin`)、**背驱+断扭矩自由落体**(`lift_and_drop`)、**低速爬行+静摩擦+负载**(`up_and_down`)。配合 **3 档砝码**让负载相关项(M3–M6)可辨识、**3 次重复**(最后 1 次留作独立验证)防过拟合,构成 **3×4×3 = 36 条**正交网格。
- **M1–M6 参数怎么算出来?** 不是抄数据手册,而是"**让仿真轨迹尽量贴合真机轨迹**"的反推:36 条日志 → **几何等价**转成 bam 的 `Pendulum` → `Simulator.rollout_log` 用**固件控制律**重算整条轨迹 → 以**位置 MAE** 作代价 → **optuna CMA-ES/BIPOP** 最小化 → 输出 `fit_m1..m6.json` → 用 **12 条留出数据**独立回放验证 `MAE < 0.157 rad` → **判档**(同等通过选参数最少者,默认 m1)。

---

## 1. 为什么要这些实验值

### 1.1 辨识对象与"摩擦预算"总览

`/home/joyandai/bam/docs/identification/hls2909_pendulum_experiment.md` §1(第 59–64 行)给出待辨识量:

```
待辨识:kt, R, armature, q_offset, error_gain, kd
      + 摩擦预算 friction_base / friction_viscous / friction_stribeck /
        dtheta_stribeck / alpha / load_friction_{motor,external}[_stribeck|_quad]
固件已知(固定不优化):kp=32, kd=32(缩放1/4), error_gain=0.166(初值),
      max_velocity=19.1637, max_acceleration=500.0, max_current=1.95
```

M1–M6 的差别**只**在摩擦项里(`bam/model.py::models` 注册表,第 256–281 行):

| 模型 | 特征 | 相对 M1 **新增的可辨识摩擦项** |
|---|---|---|
| **m1** | Coulomb | —(只有 `friction_base`, `friction_viscous`) |
| **m2** | Stribeck | `friction_stribeck`, `dtheta_stribeck`(另含 `alpha`) |
| **m3** | Load-dependent | `load_friction_base`(**非方向**,见 §2 说明) |
| **m4** | Load-dependent + Stribeck | m3 + `friction_stribeck`, `dtheta_stribeck`, `alpha`, `load_friction_stribeck` |
| **m5** | 上 + Directional | `load_friction_motor` / `load_friction_external` 及各自 `_stribeck`(拆分电机侧/负载侧) |
| **m6** | 上 + Quadratic | + `load_friction_motor_quad`, `load_friction_external_quad` |

**关键**:m3/m4/m5/m6 都引入了**负载相关项 `load_friction_*`**——这是本实验**必须换砝码**的根本原因(见 §1.4)。

### 1.2 需要覆盖的四个速度/工况区域

| 工况 | 含义 | 对应可辨识项 | 为什么必须覆盖 |
|---|---|---|---|
| **全速度谱** | 从 ~0 到高速的连续扫频 | m1 Coulomb / m2 粘性 + Stribeck | 摩擦–速度曲线主体靠它在一条轨迹里"一次扫完" |
| **低速 / Stribeck** | 极低速下的摩擦陡降 → 爬行 | m2 Stribeck(`friction_stribeck`/`dtheta_stribeck`) | Stribeck 只在低速处体现,高速激励看不到 |
| **背驱(断扭矩自由落体)** | 电机不驱动,仅重力 + 摩擦 | m5 电机侧 vs 负载侧分离;粘性 vs 反电动势阻尼解耦 | 只有断扭矩才能把**背驱摩擦**从驱动摩擦中 dissociate |
| **静态 / 负载** | 近零速度下的静摩擦 + 与负载有关的项 | m3+ `load_friction_*`;静摩擦(启动爬行) | 静摩擦与负载项都只在"接近停 + 负载变化"里显形 |

### 1.3 逐条解释 4 条轨迹"为什么是这些值"

轨迹数学定义见 `bam/trajectory.py`,并在 `scripts/record_pendulum_bench.py` 第 76–82 行以**内置副本**(`_traj`,无 bam 依赖)复现,两者一致。`bam/trajectory.py` 的注册表 `trajectories`(第 162–170 行)注册全部四种轨迹(`"sin_sin"` 也在其中,第 166 行)。

#### 1.3.1 `sin_time_square` — 速度扫频(单条覆盖全速度谱)

**代码** `bam/trajectory.py` 第 77–87 行:

```python
def __call__(self, t):
    return np.sin(t**2), True            # duration = 6.0
```

**数学**:`θ(t) = sin(t²)`。瞬时相位 `φ(t)=t²`,角频率 `dφ/dt = 2t`。

| 为什么是这个值 | 推导 / 依据 |
|---|---|
| **幅值 ±1 rad ≈ ±57°** | `sin` 本身限幅 ±1 rad = ±57.3°(runbook §4 轨迹表;"幅值恒定 ±1 rad(≈±57°)",第 147–148 行)。摆幅始终在水平线以内,不撞台架,安全。 |
| **频率随时间单调变快** | `dφ/dt = 2t`,从 0 单调升到末段——这是"**速度扫频**"的本质:单条 6 s 内把速度从 ~0 一路扫到最大。 |
| **"约 5.7 Hz"(runbook 第 148 行)** | 相位在 6 s 内扫过 `t² = 36 rad`,即 `36/(2π) ≈ 5.73` 个完整往复周期;更严格说,**末段瞬时频率**为 `dφ/dt = 2×6 = 12 rad/s ≈ 1.91 Hz`。 |
| **单条覆盖全速度谱** | 频率单调从 ~0 起,正好把"极低速(Stribeck)→中速→高速(粘性/Coulomb)"一次串起来;`trajectory.py` 第 81 行注明"Recommended as the primary identification trajectory"(首选/主激励)。 |

> 为什么需要它:摩擦–速度曲线的主体必须是**连续扫过**的。若用固定频率正弦只能得到若干孤立速度点;`sin(t²)` 在一条 6 s 内连续扫遍,拟合器能拿到**连续的 (v, τ_friction) 采样**,m1/m2 的形状才可靠锁定。

#### 1.3.2 `sin_sin` — 多频复合(谐波丰富)

**代码** `bam/trajectory.py` 第 135–145 行:

```python
def __call__(self, t):
    angle = np.sin(t) * np.pi / 2 + np.sin(5.0 * t) * 0.5 * np.sin(t * 2.0)
    return angle, True            # duration = 6.0
```

**数学**:`θ(t) = sin(t)·(π/2) + sin(5t)·0.5·sin(2t)`。runbook(第 152–153 行)记为"主摆 ±90° 叠加 3~4 倍频的小幅调制"。

| 为什么是这个值 | 推导 / 依据 |
|---|---|
| **基频项 `sin(t)·π/2`** | 幅值 π/2 ≈ 1.57 rad = **±90°(到水平)**,角频率 1 rad/s(周期 2π ≈ 6.28 s)。这是"慢的宽摆",6 s 内几乎走一个主往复,**扫过大幅度的负载变化**(重力矩随 sin θ 变化)。 |
| **调制项 `sin(5t)·0.5·sin(2t)`** | 积化和差 `0.5·sin(5t)·sin(2t) = 0.25·[cos(3t) − cos(7t)]`——等价于 **3 倍频与 7 倍频**两个小正弦(各幅值 0.25 rad),即"3~4 倍频的小幅调制"。 |
| **最大摆幅 ~±90°** | 主摆 ±90°,叠加的小调制幅值 ≤0.5 rad(28.6°),但因主摆到 ±90° 的瞬间调制项相位偏 0,故单次最大摆幅为 ~±90°(runbook 第 155 行)。 |
| **为什么"谐波丰富"** | 同时含 1、3、7 倍频 → 激励谱在**多个速度/加速度区间同时叠加**,产生反复的快速变速与换向,对 m2–m6 里随速度/方向/负载变化的项是"富信息"激励。 |

> 为什么需要它:`sin_time_square` 频率虽连续但**同一时刻只有单一瞬时频率**;`sin_sin` 在同一时刻把**快、慢两套运动叠加**,同时激励"慢速大负载"与"快速小调制"两个通道,暴露**齿轮间隙/非线性**与**负载相关项**,是对纯扫频的互补。

#### 1.3.3 `lift_and_drop` — 抬升 → 断扭矩自由落体(背驱 + 低速 Stribeck)

**代码** `bam/trajectory.py` 第 61–74 行:

```python
def __call__(self, t):
    keyframes = [[0.0, 0.0, 0.0], [2.0, -np.pi / 2, 0.0]]   # 三次样条 0 → −π/2 用 2 s
    angle = cubic_interpolate(keyframes, t)
    enable = t < 2.0                                          # t=2 s 断扭矩
    return angle, enable
```

**数学**:前 2 s 用三次样条把杆从下垂(0°)抬到 **−π/2(对侧水平)**,`torque_enable = t < 2.0`,即 **2 s 时刻自动断扭矩**,接着自由落体摆回并自然衰减直到 6 s 结束。

| 为什么是这个值 | 推导 / 依据 |
|---|---|
| **三次样条 0 → −π/2(2 s)** | 端点「值 + 一阶导=0」双约束(`cubic_interpolate`,`trajectory.py` 第 12–41 行):起点 (0,0,0)、终点 (2, −π/2, 0),两端**速度都为 0** → 平滑无冲击,且杆在 2 s 时刻**以零速度到达 −90°**,为干净的断扭矩交接做准备。 |
| **断扭矩在 t=2 s** | 2 s 是"抬升动作"与"自由落体"的分界。断扭矩后电机**不再驱动**,只有**重力 + 摩擦**作用 → 这才是真正的**背驱**观测段。`enable = t<2.0` 写入 log 的 `torque_enable` 字段,拟合器据此把这段当**背驱段(只受重力+摩擦)**,从而把**背驱摩擦**与**驱动摩擦**分开,也让**粘性摩擦与反电动势阻尼**解耦(`docs/identification/acquisition.rst` 第 43–45 行明确此目的)。 |
| **为什么用 −90°(对侧水平)** | 起点是垂直下垂(0°),抬到**对侧水平(−90°)** 使重力势能最大 → 断扭矩后自由落体的速度范围、摆幅最大,可把**从高到低速、穿过近零速度**的摩擦–速度区都采到;同时定义一致(始终同侧 −90°),符号/零位约定与递增方向自洽(keyframe 即 `-np.pi/2`)。 |
| **为什么时长 6 s** | 断扭矩后需足够时间让摆锤大幅摆动并**自然衰减**(等效摆周期 ≈2π√(L_eq/g)≈0.75 s,L_eq≈0.14 m,6 s ≈ **8 个周期**,可看到阻尼包络)。 |

> 为什么需要它:前三条都在"**电机驱动**",只有这条有"**电机松开、杆自由摆**"的观测。**背驱摩擦(尤其 m5 电机侧/负载侧分离)与低速 Stribeck** 只在这种"被动"状态下显形。runbook 特别强调这条"最需要清空摆动区""人站侧面别在摆动平面"(第 158–159 行)。

#### 1.3.4 `up_and_down` — 低速爬行 + 静摩擦 + 负载

**代码** `bam/trajectory.py` 第 90–105 行:

```python
def __call__(self, t):
    keyframes = [
        [0.0, 0.0, 0.0],
        [3.0, np.pi / 2, 0.0],
        [6.0, 0.8 * np.pi / 2, 0.0],
    ]
    return cubic_interpolate(keyframes, t), True   # duration = 6.0
```

**数学**:三次样条 **0 → π/2 → 0.8·π/2**,即 **0° → +90° → +72°**,扭矩全程开启。前 3 s 从下垂升到水平(+90°),后 3 s 平滑降到 +72° 并保持到结束。

| 为什么是这个值 | 推导 / 依据 |
|---|---|
| **0° → +90°(3 s)** | 低速、大摆幅。从**垂直下垂(0°、静止)**开始必须克服**静摩擦**才能启动——这正是观察"**爬行/起步抖动**"的窗口(是"匀滑慢动"还是"一顿一顿",后者就是静摩擦的量;runbook 第 162–163 行)。 |
| **+90° → +72°(3 s)** | 降到 +72° 并**保持**:接近停时速度趋近 0,同样是**静摩擦/爬行 + 负载**的观测段;+72° 使杆处于稳定的**重力矩非零**姿态,能激励负载相关项。 |
| **三次样条(端点速度=0)** | 端点 (0°,0),(90°,0),(72°,0) 都指定一阶导=0 → 起步/停留/转向都从零速度进行,**平滑无冲击**,并"在零速度附近多停留",把 Stribeck/静摩擦区放大。 |
| **为什么是 6 s** | 慢速爬行 + 两次端点停留需足够时间;6 s 恰好走完"低速上、低速下、末段停留"完整过程。 |
| **为什么扭矩全程开** | 这条**不测背驱**,而是测"电机主动驱动去克服静摩擦、并把负载相关项暴露出来"——`enable=True` 全程,与 `lift_and_drop` 的断扭矩形成两个极端互补。 |

> 为什么需要它:`sin_time_square`/`sin_sin` 的速度都不低,**静摩擦(近零速)与负载相关项**被快速运动淹没;`up_and_down` 刻意把**平均速度压到很低、并在端点停留**,专门照亮"静摩擦/爬行 + 负载"。runbook 把它定义为"低速爬行/静摩擦 与负载相关摩擦辨识"(第 161 行)。

#### 1.3.5 为什么统一 6 s(时长)

`duration = 6.0` 在四条轨迹里**全部相同**;`scripts/record_pendulum_bench.py` 的 `--duration` 循环以 `t > traj.duration` 为界。

| 理由 | 说明 |
|---|---|
| **BAM 协议统一** | 上游 `docs/identification/acquisition.rst`(第 27–28 行)明确"Each trajectory runs for 6 seconds"。 |
| **足够扫全物理过程** | 6 s 内:`sin_time_square` 扫完全速度谱;`lift_and_drop` 完成自由落体 + 阻尼衰减;`up_and_down` 走完低速往返 + 端点停留。等效摆周期 ≈0.75 s,6 s ≈ 8 个周期,足以观察阻尼包络。 |
| **样本量与公平性** | 6 s ÷ 6 ms ≈ **约 1000 拍/条**(runbook"目标 ~1100",第 296 行;脚本最低门槛 `>50` 样本)。**统一时长**使每条样本量可比,MAE 跨轨迹、跨档比较才公平。 |
| **批次可控** | 36 条 × 6 s + 换砝码/回零/间隔 ≈ 40–60 min(runbook 第 121 行),一次性可完成,舵机温升可控。 |

#### 1.3.6 为什么用三次样条

所有"点到点移动"的轨迹(`lift_and_drop`, `up_and_down`)都用 `cubic_interpolate`(`trajectory.py` 第 12–41 行):

| 特性 | 为什么需要 |
|---|---|
| **端点指定一阶导** | 每个 keyframe 是 `[t, x, x']` 三元组,显式给出**目标速度 x'**;全部设 0 → 起点/终点**零速度**。 |
| **平滑无冲击(C1/C2)** | 三次多项式强制位置、速度(进而加速度)连续,无阶跃,不产生机械冲击或方向跳变。 |
| **确定性/可复现** | `np.linalg.solve` 求解固定插值矩阵,对同一 `t` 输出恒定值——采集与 bam 回放使用**同一轨迹公式**,保证"命令注入 = 命令复现"。 |
| **放慢端点附近** | 端点速度 0,等效于**在端点附近减速停留**,把 Stribeck/静摩擦区放大采样——正是 `up_and_down` 想要的。 |

> `record_pendulum_bench.py` 第 53–66 行内置了一个**无 numpy 依赖的 `_cubic` 三阶 Hermite 版本**(与 `cubic_interpolate` 数值等价),作为 bam 不可用时的回退;两者结果一致。

### 1.4 3 档砝码(50/100/150 g)为什么必要

#### 1.4.1 负载相关摩擦项只有在负载变化时才可辨识

回顾 M3–M6:它们**相对 m1/m2 增加的全是 `load_friction_*`(及 `_stribeck`/`_quad`)**。这类项的数学形式是**与当前负载(重力矩/电流/负载)成正比**。

- 若只有**一档质量**:同一轨迹下重力矩 `B(θ)=(m_tip+m_arm/2)·g·L·sinθ + …` 只随 θ 变化。**负载相关项与常数 Coulomb 项在数据里不可分**——拟合器无法判断"这部分力矩是常数摩擦,还是随负载线性变化的摩擦"。
- 有了 **50/100/150 g 三档**:同一轨迹、同一几何下,负载的**幅度整体被放大 1×/2×/3×**,而常数摩擦(Coulomb/Stribeck)**不随负载变**。这样拟合器才能把"常数项"与"负载比例项"从数据中**分离**出来,`load_friction_*` 系数才**可辨识**。

| 档位目的 | 说明 |
|---|---|
| 提供**负载对比度** | 50/100/150 g 呈 1:2:3,覆盖负载变化的整段比例范围,利于线性(及二次)负载项回归。 |
| **换档不换几何** | 只换端部砝码(runbook 强调"换档重称重填 `--tip-mass`"),杆/臂/几何不变 → 载荷是**唯一**被改的自变量,分离干净。 |
| **正交网格** | 质量 × 轨迹 × 重复构成 3 维正交网格(每格只动一个因素),可分离度好。 |

#### 1.4.2 0.35 N·m 安全红线 → m_tip ≈ 238 g 上限推算

runbook 与 `hls2909_pendulum_experiment.md` 都定死 **`m_tip·g·L ≤ 0.35 N·m`** 红线(脚本 `--torque-budget` 默认 0.35,硬检查)。推算:

**① 忽略摆杆/轮毂(只算端部点质量)**:`m_tip ≤ 0.35/(g·L) = 0.35/(9.81×0.15) = 0.2379 kg ≈ 238 g`(L 取 runbook 参数表 `0.150 m`;g=9.81,脚本 `G=9.81`)。

**② 计入摆杆 m_arm 与近轴 m_hub**:脚本 `pendulum_equiv`(`record_pendulum_bench.py` 第 94–99 行)算的完整最大重力矩为 `B_max = (m_tip + m_arm/2)·g·L + m_hub·g·r_hub`。以 runbook 示例几何(m_arm=0.018, m_hub=0.020, r_hub=0.010, L=0.15),由 `B_max ≤ 0.35` 解 `m_tip`:

```
m_tip ≤ (0.35 − m_hub·g·r_hub)/(g·L) − m_arm/2 ≈ 0.2275 kg ≈ 227.5 g
```

所以文档 §8 写"忽略杆/轮毂力矩约 ≤ ~238 g,计入后约 ≤ ~225 g;**禁止加大砝码**"。

**③ 为什么 150 g 是安全顶档**(上面几何,g=9.81,L=0.15):

```
m_tip = 50 g  → B_max ≈ 0.0888 N·m
m_tip = 100 g → B_max ≈ 0.1624 N·m
m_tip = 150 g → B_max ≈ 0.2359 N·m   ← 仍远低于 0.35 N·m 红线
```

150 g 这一档只有红线的 ~67%,留足裕量,同时已把负载放大到 1.5×,足够覆盖 M3–M6 负载项。

> **红线由脚本强制执行**:`record_pendulum_bench.py` 用 `--torque-budget`(默认 0.35)在**运行前**对每档 `pendulum_equiv` 的 `B_max` 检查,超限直接 `ap.error` 拒绝运行(第 166–173 行)。所以红线不是口头约定,而是代码级硬闸。

#### 1.4.3 为什么是 50/100/150 而不是更多档

- **3 档足够分离**:负载相关项对 M3–M6 是有限的待拟合自由度,3 档(1:2:3)已给出足够回归所需的负载对比度;更多档性价比低(每条 6 s、需换砝码、需人工确认)。
- **档位落在安全窗内**:三档都远低于红线上限(~238 g),最高档 150 g 仍有裕量。
- **覆盖但不冗余**:50→100→150 均匀倍率,便于把 `load_friction_*` 与其二次项(`_quad`)一起回归辨识。

### 1.5 3 次重复的作用

#### 1.5.1 噪声平均

36 条中的"重复"轴 = 3。物理上每条记录都有**传感器量化噪声、总线抖动、微小装配误差、温升漂移**等;做 3 次取平均/让拟合器在多条上最小化残差,可**压低随机噪声对参数估计的方差**,让 m1–m6 的 `friction_*` 估计更稳。

#### 1.5.2 最后 1 次留作独立验证(防过拟合)

这是设计关键:**M1–M6 越靠后的模型参数越多**(m6 有最多自由度)。若只用同批数据**拟合+评优**,参数多的模型会靠"拟合噪声"取得假性低误差——即过拟合。

`hls2909_pendulum_experiment.md` §3.E 与 runbook §2.3(第 96 行)明确:每个"轨迹×质量"的**最后一次重复**留出(**12 条独立验证**),其余 **24 条**拟合。

- **训练集**:36 − 12 = **24 条**(每个"轨迹×质量"取前 2 次重复,12 组合 × 2 = 24)。
- **验证集**:**12 条**(每个"轨迹×质量"第 3 次重复,从没进过拟合)。
- **验收门**:`独立验证 MAE < 0.157 rad`(≈ 10% × π/2)。**训练/验证无分叉**才落地;同等通过时**选 m1**(参数最少)。

> 这样选档只看"**未见过的数据**上回放得好不好",能诚实地区分"模型真的好"还是"参数多所以拟合好"。若验证明显差于训练 → 判为过拟合。

#### 1.5.3 36 = 3 × 4 × 3

```
36 条 = 3(砝码档 m_tip: 50/100/150) × 4(轨迹) × 3(重复)
```

每档质量 × 每条轨迹各测 3 次。每条 6 s,加换砝码/回零/间隔,整批约 40–60 min。

### 1.6 这些"值"的工程边界(`record_pendulum_bench.py` 常量)

#### 1.6.1 12-bit 编码器分辨率 `LSB_RAD = 2π/4096 ≈ 0.001534 rad/LSB`

```python
# record_pendulum_bench.py 第 84 行
LSB_RAD = 2 * math.pi / 4096.0      # = 0.001534 rad/LSB = 0.0879°/LSB
```

- 输出轴编码器 **12 位**,4096 LSB/圈 → `0.0879°/LSB`(`0.001534 rad`),这是**角度测量分辨率的物理上限**。
- **零位展开**:`((pos − q_zero + 2048) % 4096) − 2048`,把跨零位 LSB 展开成有符号角度。
- **速度用相邻拍差分,不用 reg58**:`speed = (θ_now − θ_prev)/Δt`。reg58 出厂量化 **0.077 rad/s 太粗**(runbook 第 85 行),无法分辨低速 Stribeck 的细微速度差;前向差分可到 LSB 级分辨率下限。
- **为什么分辨率要够**:m2 Stribeck / m3–m6 负载项都在**低速**显形,量化太粗会抹掉细节。

#### 1.6.2 ~6 ms 拍周期(`dt_loop = 0.006`)

```python
# record_pendulum_bench.py 第 268 行
dt_loop = 0.006
```

- 固件 DTs = 3 ms;总线 1 Mbps 下一轮"写目标(reg42) + 读 reg56/60/62/69/63"的往返约 4–6 ms。
- 用 **6 ms** 恰好"**一拍控制 + 一拍测量**"同拍(先写后读,控制与测量对齐),既不超固件能力,又尽量提高采样密度。
- 实际时间戳含抖动,**以实测为准**(期望 6–15 ms,runbook 第 84 行);每条 log 写 `dt` = 相邻拍时间戳**中位数**。

#### 1.6.3 临时限流 0.975 A(`reg44=150`)

```python
# record_pendulum_bench.py 第 180 行
limit_lsb = int(round(args.limit_a / 0.0065))     # 0.975 / 0.0065 = 150
```

- 固件保护电流寄存器 `reg28/44`,单位 6.5 mA/LSB。**出厂保护值 300 → 300×6.5 mA = 1.95 A**。
- 实验**临时降为 150 → 150×6.5 mA = 0.975 A**(`--limit-a 0.975`)。原因:摆锤在**重力矩下近于堵转**(尤其保持姿态时电流大),临时限流**防堵转发热**。
- 每条结束/异常会**恢复运行前读到的原始值**(`original_limit_lsb`),安全退出。
- **重要**:0.975 A 只是**实验期临时防护**;**拟合与部署的模型应保持 1.95 A**(`hls2909_pendulum_experiment.md` §6.3)——**别把临时限流带进模型**(§7 明确警告)。

#### 1.6.4 温度 50 °C 红线

```python
# record_pendulum_bench.py 第 86 行
TEMP_ABORT_C = 50.0
```

- 舵机温度 `reg63` 每拍读取并监控,**`> 50 °C` 立即断扭矩 → 恢复限流 → 退出**(`Safety.emergency`)。
- 配套电流红线:`|I| > 1.0 A` 持续 `0.7 s`(`CURRENT_LIMIT_A=1.0`, `CURRENT_BURST_S=0.7`)也会触发紧急停机。
- **为什么 50 °C**:HL-2909 为空心杯/无刷+减速器结构,温升过高会退磁、损伤线圈与减速器油脂。50 °C 是"留安全裕量的停机红点";runbook 还建议 >45 °C 暂停冷却 2 min。**安全性优先于数据完整性**——越红线整批报废。

#### 1.6.5 看门狗 30 s

```python
# record_pendulum_bench.py 第 89 行
WATCHDOG_S = 30.0
```

- 单条记录超过 **30 s** 未完成 → `Safety.emergency("看门狗 30s")` → 立即断扭矩 + 恢复限流 + 退出。
- **为什么 30 s**:每条轨迹 6 s + 回零 + 0.5 s 稳定(`SETTLE_S`),正常远小于 30 s。若单条跑不完 30 s,基本是**总线挂死/舵机无响应/丢拍过多**——是"卡死"征兆。脚本层 `RECORD_DURATION_S = 7.0` 是每条硬上限(配合 `t > traj.duration` 的 6 s),30 s 是更大的兜底看门狗。触发即停机,避免长时间堵转发热。

#### 1.6.6 附:数据格式与拟合管线的"值"

- 每条 log 的 `mass/arm_mass/length` 写**等效 Pendulum 参数** `m_eq / 0.0 / L_eq`,由 `pendulum_equiv` 计算(`L_eq = M·g/B_max`, `m_eq = B_max²/(g²·M)`)。这样 bam 内部 `testbench.Pendulum` 无需改动即可复现同一动力学。
- `entries` 只保留 `position/speed/goal_position/torque_enable` 四键,与 `bam.logs` 的 `rollout_log` 兼容,bam.fit 直接可读。

### 1.7 与详细版设计文档的差异(以 runbook 为准)

`make_pendulum_experiment_design_html.py`(详细版设计)与 `pendulum_bench_runbook.html`(速查/唯一参数表)存在下列出入。按 `hls2909_pendulum_experiment.md` §0 的约定,**冲突处一律以 runbook 为准**。

| 维度 | 详细版设计 | runbook(唯一参数表) | 以谁为准 |
|---|---|---|---|
| 端部质量 m_tip | 50 / **100 / 160** g | 50 / 100 / **150** g | **runbook 为准** |
| 轨迹数 | **3 种**(lift_and_drop / sin_time_square / up_and_down,**无 sin_sin**) | **4 种**(+ **sin_sin**) | **runbook 为准** |
| 重复 n | **5** | **3** | **runbook 为准** |
| 总条数 | 3×3×5 = **45** | 3×4×3 = **36** | **runbook 为准** |
| 摆长 L | **0.20 m** | **0.150 m** | **runbook 为准**(几何不同 → 力矩预算也不同) |
| 力矩预算上限 | `L=0.2 m → m_tip ≤ 170 g` | `L=0.15 m → m_tip ≈ 238 g(忽略杆)/≈225 g(计入)` | 上限随 L 变;红线始终是 0.35 N·m |

> 详细版是"早期/更理想化"草案(更多重复抗噪、加一档 160 g、舍 sin_sin、用更长摆臂)。最终 runbook 为**工程可执行**收敛:3 次重复足够(因为留出验证防过拟合)、加 sin_sin 补谐波激励、砝码顶档降到 150 g、摆长定为 0.150 m **以把各档完整重力矩稳压在 0.35 N·m 红线内**。本文全部"为什么"以 runbook 的 50/100/150 g、4 轨迹、3 重复、36 条为基准。

---

## 2. M1–M6 六个摩擦模型

### 2.1 一句话总览

BAM 用六档**阶梯式**摩擦模型(M1→M6)表达舵机减速器的不完美传递:从最简单的**库仑+粘性**,逐步叠加 **Stribeck(低速静摩擦)**、**负载相关(随传递扭矩增长)**、**方向拆分(电机侧 vs 外部负载侧不对称)**、以及**二次耦合项**。

最终作用于仿真的是"最大抵抗摩擦扭矩"(摩擦预算)$\tau_{fm}$。MuJoCo / mjlab 把它拆成两块写入两个仿真字段(实现里拆开,但数学上等价于 $\tau_{fm}$):

- **速度无关项**(Coulomb + Stribeck + 负载项)→ `dof_frictionloss`;
- **速度相关项**(粘性)→ `dof_damping`。

即总摩擦预算 $\tau_{fm} = \text{frictionloss} + \text{damping}\cdot|\dot\theta|$,仿真把制动力矩裁剪在 $[-\tau_{fm}, \tau_{fm}]$ 内。

### 2.2 六档一览表(flag 组合)

注册表见 `bam/model.py` 第 256–281 行(`models` dict)。每档是 `Model(...)` 上若干 **flag** 的组合:`stribeck`、`load_dependent`、`directional`、`quadratic`(`bam/model.py` 第 38–55 行)。

| 档 | 特征(flag 组合) | 相对 M1 **增加**的可辨识摩擦参数 | 捕捉的物理现象 |
|---|---|---|---|
| **m1** | 无 flag(仅 Coulomb + Viscous) | —(只有 `friction_base` + `friction_viscous`) | 基底库仑摩擦 + 线性粘性阻尼,多数物理引擎的默认定标 |
| **m2** | `stribeck=True` | `friction_stribeck`, `dtheta_stribeck`, `alpha` | **Stribeck**:低速时静摩擦 > 滑动摩擦,随速度平滑过渡(预滑动/静摩擦峰) |
| **m3** | `load_dependent=True`(**非方向**) | `load_friction_base` | **Load-dependent**:摩擦随传递的减速箱负载(\|$\tau_m-\tau_e$\|)线性增长 |
| **m4** | `load_dependent=True` + `stribeck=True`(非方向) | m3 + `friction_stribeck`, `dtheta_stribeck`, `alpha`, `load_friction_stribeck` | m3 的负载项本身也带 **Stribeck 软化**(负载越小、启动越"粘") |
| **m5** | `load_dependent=True` + `stribeck=True` + `directional=True` | 同 m4(`load_friction_{motor,external}` + `load_friction_{motor,external}_stribeck`) | **Directional**:电机侧 / 外部负载侧系数不同,建模驱动与背驱效率不对称 |
| **m6** | m5 + `quadratic=True` | `load_friction_motor_quad`, `load_friction_external_quad` | m5 + 负载/电机扭矩的**二次耦合项**(谐波减速器类非线性) |

对每档,`Model.set_actuator`(`bam/model.py` 第 71–123 行)按 flag 精确决定创建哪个 `Parameter` 对象,所以"该档有多少参数"是**代码决定的**。§2.4 逐档列全。

> **注意(与 `identification` 文档 §1 简表的一处出入)**:`docs/identification/hls2909_pendulum_experiment.md` §1 的 M1–M6 简表把 **m3** 写成 `load_friction_motor/external`。但 `bam/model.py` 第 97–104 行里,**非方向**(m3/m4)创建的是 `load_friction_base`,只有**方向**(m5/m6)才创建 `load_friction_motor` + `load_friction_external`。本文档以**代码与 `models.rst`** 为准:m3 用 `load_friction_base`。

### 2.3 全部摩擦参数清单

参数类见 `bam/parameter.py`:每个 `Parameter(value, min, max, optimize=True)`,`optimize=True` 表示参加拟合(默认全部摩擦参数都参与)。

#### 2.3.1 通用基座(每个模型都有)

| 参数名 | 符号 | 单位 | 默认值 | 边界 | 物理含义 |
|---|---|---|---|---|---|
| `friction_base` | $K_c$ | **Nm** | 0.05 | [0, 0.2] | 库仑摩擦常数,与速度/负载无关的恒定抵抗扭矩(`bam/model.py` 第 92 行) |
| `friction_viscous` | $K_v$ | **Nm·s/rad** | 0.1 | [0, 1.0] | 线性粘性阻尼系数,乘以 \|$\dot\theta$\| 得速度相关摩擦(第 123 行) |

#### 2.3.2 Stribeck(m2 起)

| 参数名 | 符号 | 单位 | 默认值 | 边界 | 物理含义 |
|---|---|---|---|---|---|
| `friction_stribeck` | $K_{cs}$ | **Nm** | 0.05 | [0, 0.2] | Stribeck 幅值:低速/静摩擦在 `friction_base` 之上**多出**的部分(第 94 行) |
| `dtheta_stribeck` | $\dot\theta_s$ | **rad/s** | 0.2 | [0.10, 1.0] | Stribeck 特征速度:静摩擦过渡到滑动摩擦的"拐点"速度(第 119 行) |
| `alpha` | $\alpha$ | **无单位(指数)** | 1.35 | [1.0, 10.0] | Stribeck 曲线陡度:越大过渡越陡(越接近"硬"静-动摩擦切换)(第 120 行) |

#### 2.3.3 负载相关(m3 起)

系数作用于减速箱传递扭矩 $\tau_{\text{gb}}$(见正文),单位**无单位**(扭矩比系数)——因为它实际是"负载扭矩的倍数"。

| 参数名 | 符号 | 单位 | 默认值 | 边界 | 物理含义 |
|---|---|---|---|---|---|
| `load_friction_base` | $K_l$ | 无单位 | 0.05 | [0, 0.5] | (**非方向**,m3/m4)负载项系数,乘 \|$\tau_m-\tau_e$\|(第 104 行) |
| `load_friction_motor` | $K_m$ | 无单位 | 0.05 | [0, 0.5] | (**方向**,m5/m6)**电机侧**负载系数,乘 $\tau_m$(第 99 行) |
| `load_friction_external` | $K_e$ | 无单位 | 0.05 | [0, 0.5] | (**方向**,m5/m6)**外部负载侧**系数,乘 $\tau_e$(第 100 行) |

#### 2.3.4 负载项的 Stribeck 软化(m4 起)

| 参数名 | 符号 | 单位 | 默认值 | 边界 | 物理含义 |
|---|---|---|---|---|---|
| `load_friction_stribeck` | $K_{ls}$ | 无单位 | 0.05 | [0, 1.0] | (**非方向**,m4)负载项的 Stribeck 幅值系数(第 111 行) |
| `load_friction_motor_stribeck` | $K_{ms}$ | 无单位 | 0.05 | [0, 1.0] | (**方向**,m5/m6)电机侧负载项 Stribeck 系数(第 108 行) |
| `load_friction_external_stribeck` | $K_{es}$ | 无单位 | 0.05 | [0, 1.0] | (**方向**,m5/m6)外部负载侧负载项 Stribeck 系数(第 109 行) |

#### 2.3.5 二次耦合项(m6)

| 参数名 | 符号 | 单位 | 默认值 | 边界 | 物理含义 |
|---|---|---|---|---|---|
| `load_friction_motor_quad` | $K_{mq}$ | **1/Nm** | 0.0 | [0, 0.01] | 背驱(\|$\tau_e$\|>\|$\tau_m$\|)时乘 $\tau_m^2$ 的二次系数(第 114 行) |
| `load_friction_external_quad` | $K_{eq}$ | **1/Nm** | 0.0 | [0, 0.01] | 驱动(\|$\tau_m$\|>\|$\tau_e$\|)时乘 $\tau_e^2$ 的二次系数(第 115 行) |

#### 2.3.6 同时被辨识的"非摩擦"干扰参数(`set_actuator` 恒创建)

单位均为物理单位,虽属摩擦预算之外的台架级"妨害"参数,但拟合时一并放开。

| 参数名 | 单位 | 默认值 | 边界 | 含义 |
|---|---|---|---|---|
| `q_offset` | rad | 0.0 | [-0.1, 0.1] | 电机零位 / 台架装配误差 |
| `command_delay` | s | 0.0 | [0.0, 0.05] | 命令传输/固件响应延迟,在回放里对目标序列做分数移位 |

#### 2.3.7 执行器 / 固件参数(非摩擦,但参与回放)

- DC 电机:`kt`(Nm/A)、`R`(Ω)、`armature`(H)。
- HLS-2909 固件律:`error_gain`(P 缩放 1/8)、`kd`(D 缩放 1/4)、`kp`、`max_velocity`(rad/s)、`max_acceleration`(rad/s²)、`max_current`(A)。其中**限幅(`max_velocity`/`max_acceleration`/`max_current`)在拟合中用 `--set` 固定为出厂真值**,不作为自由度。

### 2.4 每档模型的可辨识摩擦参数全集(代码决定)

> 依据 `bam/model.py::set_actuator` 的 flag 分支(第 92–123 行)逐一列出。

| 档 | 摩擦参数全集 | 计数 |
|---|---|---|
| **m1** | `friction_base`, `friction_viscous` | **2** |
| **m2** | m1 + `friction_stribeck`, `dtheta_stribeck`, `alpha` | **5** |
| **m3** | m1 + `load_friction_base` | **3** |
| **m4** | m3 + `friction_stribeck`, `dtheta_stribeck`, `alpha`, `load_friction_stribeck` | **7** |
| **m5** | m1 + `load_friction_motor`, `load_friction_external`, `friction_stribeck`, `dtheta_stribeck`, `alpha`, `load_friction_motor_stribeck`, `load_friction_external_stribeck` | **9** |
| **m6** | m5 + `load_friction_motor_quad`, `load_friction_external_quad` | **11** |

> 加计"非摩擦"参数(`q_offset`, `command_delay`)与执行器/固件参数后,每档可优化自由度会多出——这正是 §2.6 要控制自由度的原因。

### 2.5 摩擦预算公式(总摩擦力矩)

符号约定(`docs/theory/models.rst` 记法):$\dot\theta$ 关节速度,$\tau_m$ 电机扭矩,$\tau_e$ 外部/负载扭矩,$\tau_{fm}$ 最大抵抗摩擦扭矩(摩擦预算)。

**实现拆分**(`bam/mjlab.py::_compute_friction_budget` 第 470–548 行):

$$
\tau_{fm} = \text{frictionloss} + \underbrace{K_v |\dot\theta|}_{\to\ \text{dof\_damping}},\qquad \text{frictionloss} \to \text{dof\_frictionloss}
$$

Stribeck 系数(所有带 Stribeck 的模型共用,`bam/mjlab.py` 第 734 行):

$$
s(\dot\theta)=\exp\!\Big(-\Big|\frac{\dot\theta}{\dot\theta_s}\Big|^{\alpha}\Big)
$$

**M1 — Coulomb-Viscous**:$\ \tau_{fm}=K_v|\dot\theta|+K_c$。代码:`frictionloss = friction_base`;`damping = friction_viscous`。

**M2 — Stribeck**:$\ \tau_{fm}=K_v|\dot\theta|+K_c+s(\dot\theta)\,K_{cs}$。代码:`frictionloss += s·friction_stribeck`。

**M3 — Load-dependent**:$\ \tau_{fm}=K_v|\dot\theta|+K_c+K_l|\tau_m-\tau_e|$。代码(非方向):`gearbox = |external_torque − motor_torque|`;`frictionloss += load_friction_base·gearbox`。

**M4 — Stribeck + Load-dependent**:

$$
\tau_{fm}=K_v|\dot\theta|+K_c+K_l|\tau_m-\tau_e|+s(\dot\theta)\big(K_{cs}+K_{ls}|\tau_m-\tau_e|\big)
$$

代码:`frictionloss += load_friction_base·gb + s·(friction_stribeck + load_friction_stribeck·gb)`。

**M5 — Directional load-dependent**:

$$
\tau_{fm}=K_v|\dot\theta|+K_c+\big|K_m\tau_m-K_e\tau_e\big|+s(\dot\theta)\,\big(K_{cs}+\big|K_{ms}\tau_m-K_{es}\tau_e\big|\big)
$$

代码(方向):$\tau_{\text{gb}}=\big|K_e\tau_e-K_m\tau_m\big|$、$\tau_{\text{gb,s}}=\big|K_{es}\tau_e-K_{ms}\tau_m\big|$,取绝对值后等价于 $|K_m\tau_m-K_e\tau_e|$。`frictionloss += τ_gb + s·(friction_stribeck + τ_gb,s)`。

**M6 — Quadratic directional**:

$$
\tau_{fm}=K_v|\dot\theta|+K_c+\big|K_m\tau_m-K_e\tau_e\big|+s(\dot\theta)\Big(K_{cs}+\big|K_{ms}\tau_m-K_{es}\tau_e\big|+Q\Big)
$$

$$
Q=\begin{cases} K_{eq}\tau_e^2, & |\tau_m|>|\tau_e|\ (\text{驱动:电机扭矩占优})\\ K_{mq}\tau_m^2, & \text{否则}\ (|\tau_e|>|\tau_m|,\ \text{背驱})\end{cases}
$$

代码(`bam/mjlab.py` 第 519–529 行):`drive_mask = |τ_m| > |τ_e|`;`quad_term = drive_mask·load_friction_external_quad·τ_e² + backdrive_mask·load_friction_motor_quad·τ_m²`;`frictionloss += s·quad_term`。

> **实现一处差异(提醒判档脚本核对)**:`bam/model.py` 的**标量版** `compute_frictions`(第 184–203 行)在二次项上还乘了一个 `enable_quadratic = sign(τ_e) ≠ sign(τ_m)` 的门控(只有负载与电机**反向**时才计二次项);而 `bam/mjlab.py` 的**向量版**(第 519–529 行)没有这个符号门控,只用 `drive_mask`。bam 拟合(`bam.fit`)走**标量版**,mjlab 部署走**向量版**,二者在"负载与电机同向"时的二次项取值可能不同——若 m6 判档/落地出现分量偏差,先查这一处。

#### 2.5.1 负载相关项在摆锤台架里实际取什么

摆锤台架提供**已知重力矩**(`bam/testbench.py::Pendulum` 的 `compute_bias` = $\propto\sin q$)。因此外部负载扭矩 $\tau_e$ 主要由**重力矩分量**充当(随摆角 $\theta$ 变化),这正是让 **m3/m4/m5/m6** 的负载项**可辨识**的激励来源:3 档砝码(50/100/150 g)× 4 条轨迹就是在扫 $\tau_e$ 的幅值与方向。

### 2.6 为什么参数要"阶梯式"增加,而不是直接用 m6

#### 2.6.1 参数越少越稳、越多越易过拟合

- 由 §2.4,m1 只有 2 个摩擦参数,m6 有 11 个;加上 `q_offset`/`command_delay`/执行器固件参数后,m6 的可优化自由度显著多于 m1。
- 每次拟合用**同一批数据**做 6 次**软件**拟合(不是 6 次硬件实验),候选模型共享同一条训练集。参数越多越能"记住"训练集噪声 → 训练 MAE 低、独立验证 MAE 高(train ≪ validation 的分叉),即过拟合。
- 越复杂的模型需要**越多、越充分覆盖激励空间的数据**才能稳定辨识。摆锤协议只有 3 档负载 × 4 条轨迹 × 3 重复 = 36 条,其中仅 12 条留作独立验证;对 m6 这类高自由度模型,样本量偏紧。

#### 2.6.2 判档规则

1. 用每个"轨迹×质量"的**最后一次重复**(3 档 × 4 轨迹 = 12 条)作**独立验证**,其余 24 条拟合。
2. 验收门:独立验证 **MAE < 0.157 rad**(≈ 10% × π/2)。
3. **同等通过(多个模型都达标且 train ≈ validation)→ 选参数最少的模型 = 默认 m1**。
4. 仅当简单模型明显差、而 m6 显著更优且无分叉时,才选 m6 并写明理由。

即:**参数最少者优先**,除非更高档位在独立验证上带来"值得用额外自由度换"的实打实误差下降。这是 BAM 论文 "fits all candidates, selects best trade-off from validation error" 在本协议的落地。

#### 2.6.3 对 HL-2909 拟合的落地提示

- `bam/params/hls2909/m1.json` 当前 `friction_base=0.08`、`friction_viscous=0.012` 是**行为级反推**("已知良好策略在真机站立"),**非摆锤标定值**。本次实验目的就是把它们替换为摆锤辨识真值并**判档**(默认 m1,若 m6 实打实更好则选 m6)。
- 拟合时用 `--set` 固定固件限幅(`max_velocity=19.1637`、`max_acceleration=500.0`、`max_current=1.95`),否则 CMA-ES 会把已知量当自由度漂移、产生假性低 MAE。

### 2.7 量级示例:`bam/params/xl330/m6.json`(仅对照)

用作对照基准,说明 m6 各参数的实际量级。实际数值(`bam/params/xl330/m6.json`):

```json
{ "kt": 0.34597, "R": 2.5019, "armature": 0.001573,
  "q_offset": 0.014997, "command_delay": 0.010221,
  "friction_base": 0.011920,      // Nm
  "friction_stribeck": 0.000851,  // Nm
  "load_friction_motor": 0.227817,   // [-]
  "load_friction_external": 0.106512,// [-]
  "load_friction_motor_stribeck": 1.47e-08,  // [-]
  "load_friction_external_stribeck": 0.142018,
  "load_friction_motor_quad": 0.005266,    // 1/Nm
  "load_friction_external_quad": 0.002986, // 1/Nm
  "dtheta_stribeck": 0.260667, "alpha": 8.5288,
  "friction_viscous": 0.005788,   // Nm·s/rad
  "model": "m6", "actuator": "xl330" }
```

**量级感受**:

- 库仑项很小:`friction_base ≈ 0.012 Nm`,相对 XL330 工作扭矩量级(同档 ~0.3 Nm)约 4%,是常数抵抗。
- 粘性项很小:`friction_viscous ≈ 0.0058 Nm·s/rad`,在 $\dot\theta=1$ rad/s 时贡献约 0.0058 Nm。
- Stribeck 只在低速起作用:`alpha=8.53` 很陡,`dtheta_stribeck=0.26 rad/s`。计算 $s(\dot\theta)$:

  | $\dot\theta$ (rad/s) | $s(\dot\theta)$ |
  |---|---|
  | 0.00 | 1.000 |
  | 0.05 | 1.000 |
  | 0.20 | 0.901 |
  | 0.2607 | 0.367 |
  | 0.30 | 0.036 |
  | 0.50 | ≈0 |

  即速度超过约 0.3 rad/s 后 Stribeck 基本消失,幅值 `friction_stribeck ≈ 0.00085 Nm` 也小。

- **负载项是 m6 的主力**:`load_friction_motor=0.228`、`load_friction_external=0.107`。在 $\tau_m\approx0.3$ Nm、$\tau_e\approx-0.2$ Nm 的驱动工况下:
  $$
  \tau_{\text{gb}}=\big|0.107\cdot(-0.2)-0.228\cdot0.3\big|\approx 0.0897\,\text{Nm}
  $$
  远大于库仑项;配上 `load_friction_external_stribeck=0.142` 的 Stribeck 负载项(约 0.028 Nm),合计约 0.13 Nm。
- 二次项很小:`load_friction_external_quad≈0.003`、`load_friction_motor_quad≈0.005`,在 0.3 Nm 处只有 ~0.0003 Nm,可视为对高负载端的小幅修正。

> **注意**:XL330 是新舵机、m6 参数量级**只作参考**,不能直接搬给 HL-2909。HL-2909 的 `kt=1.454`、`R=20.0`、`max_current=1.95`、`max_velocity=19.16` 完全不同;其摩擦参数须由本次摆锤辨识重新拟合。

---

## 3. 参数是怎么算出来的

### 3.1 完整链路(ASCII 流程)

```
36 条 JSON 日志 (24 拟合 + 12 留出验证)
      │  bam.logs.Logs 逐条读入 (logs.py:28)
      │  mass/arm_mass/length 已是 pendulum_equiv 换好的等价参数 (arm_mass=0)
      ▼
bam.simulate.Simulator.rollout_log (simulate.py:106)
      │  在每个时间步:
      │     ① 固件控制律 HLS2909Actuator.compute_control  → 电压 (feetech/actuator.py:298)
      │     ② DC 电机力矩 compute_torque                    → τ_m (actuator.py:317)
      │     ③ Pendulum.compute_bias                         → 重力矩 τ_e (testbench.py:68)
      │        + compute_mass + armature                    → 惯量 M (testbench.py:58 / actuator.py:234)
      │     ④ Model.compute_frictions                       → (frictionloss,damping) (model.py:125)
      │     ⑤ 停止扭矩削顶 tau_stop (dof_frictionloss 语义) → 净力矩 (simulate.py:92)
      │     ⑥ 欧拉积分 q += dq·dt                           (simulate.py:99)
      ▼
预测轨迹 q_hat(t)   vs  日志真值 q_log(t)  →  位置 MAE
      ▼
optuna CMA-ES / BIPOP 最小化 "各 log 位置 MAE 均值" (fit.py:99,215)
      → 最优参数写进 fit_m1.json … fit_m6.json
      → 12 条留出数据 load_model + 再回放 → 独立验证 MAE
      → MAE < 0.157 rad 且 train≈val → 落地 (等同通过选 m1)
```

**核心问题**:为什么每个摩擦/电机参数是从"让仿真轨迹尽量贴合真机轨迹"这个目标**反推**出来的,而不是抄数据手册?因为数据手册只给 `kt/R` 的初始可信值;`friction_base/friction_viscous/armature` 在 `m1.json` 里目前是"行为级反推占位值",只有跑这套拟合才能变成"摆锤实测值"。

### 3.2 输入:36 条 JSON 日志

#### 3.2.1 采集来源与命名

采集脚本 `scripts/record_pendulum_bench.py`(`--reps 3`,每条约 6 s,标称 ~6 ms/拍)产生 **36 条**,文件名 `{trajectory}_tip{m}g_rep{r}.json`,外加 `manifest.json`(元数据,**无 `entries` 字段**)。

```
4 轨迹 × 3 重复 = 12 条/档  ×  3 档(m_tip) = 36 条
轨迹(record_pendulum_bench.py:76-82 内置副本 / bam/trajectory.py:162 同义):
  sin_time_square : θ = sin(t²)                          (速度扫频 ±57°)
  sin_sin         : θ = sin(t)·π/2 + sin(5t)·0.5·sin(2t) (多频复合 ±90°)
  lift_and_drop   : 0 → −π/2 (2s 三次样条) 后断扭矩自由落体  (背驱/Stribeck)
  up_and_down     : 0 → π/2 → 0.8·π/2 (低速三次样条)        (静态/负载摩擦)
```

#### 3.2.2 顶层字段

每条 log(`record_pendulum_bench.py` 第 317–327 行)顶层:

```json
{
  "motor": "hls2909",          // 执行器注册名 → bam.actuators["hls2909"]
  "kp": 32,                     // 固件位置环 P;bam 用 log 值覆盖默认
  "vin": 12.5,                  // 实测电压中位数;覆盖默认
  "dt": 0.006,                  // 相邻拍时间戳中位数
  "mass": 0.11475,              // m_eq  [kg]   ← pendulum_equiv 换算
  "arm_mass": 0.0,              // 置 0       ← 已换算成"点质量"等价
  "length": 0.14423,            // L_eq  [m]   ← pendulum_equiv 换算
  "trajectory": "up_and_down",
  "tip_mass_kg": 0.10,          // 原始端部总质量,仅记录用
  "rep": 0,                     // 本条是第几次重复
  "entries": [
    { "position": 0.0, "speed": 0.0, "goal_position": 0.0, "torque_enable": true },
    ...                          // 约 1000–1200 点(6 s @ ~6 ms)
  ],
  "telemetry": [ { "t": 0.0, "duty": 0.0, "vin": 12.4, "current_A": 0.1, "temp": 31 }, ... ]
}
```

#### 3.2.3 `entries` 各键与 bam 消费方

| 字段 | 单位 | 来源/含义 | bam 侧消费方 |
|---|---|---|---|
| `position` | rad | `(reg56 − q_zero)·LSB_RAD·sign`,12-bit,0.001534 rad/LSB | **拟合目标**(MAE 对象) |
| `speed` | rad/s | 相邻拍前向差分(不读 reg58,量化太粗) | 回放初态 + 固件律输入 |
| `goal_position` | rad | 轨迹命令值 | 注入固件控制律 |
| `torque_enable` | bool | `lift_and_drop` 第 2 s 后为假(自由落体) | 背驱段只受重力+摩擦 |
| `telemetry`(duty/vin/current/temp) | — | 体检/安全用 | **不参与拟合**(明确剔除) |

关键点:`rollout_log` **只要求** `dt` 标量 + `entries` 四键,不要求 `timestamp`,也不要求统一网格。record 用变间隔真值 `dt`(中位数),`fit_leg_pendulum` 直接用该标量拟合。

> ⚠ 目录里混有 `manifest.json`(无 `entries`),`bam.fit` 直接指回 raw 目录会崩。`fit_leg_pendulum.py` 已排除;直跑 `python -m bam.fit` 时请指向 `bench_processed/`(`scripts/process_bench_logs.py` 输出)。

### 3.3 几何等价:`m_tip/m_arm/L/m_hub/r_hub → M/B_max/m_eq/L_eq`

#### 3.3.1 为什么要换算

BAM 的回放/拟合统一使用 `bam/testbench.py` 的 `Pendulum`,它只认三个数 `mass`/`arm_mass`/`length`(`testbench.py` 第 53–56 行),且把"质量"当**端部点质量**:
- `compute_mass` = `mass·length² + (arm_mass/3)·length²`(`testbench.py` 第 58–66 行)
- `compute_bias` = `(mass + arm_mass/2)·g·length·sin(q)`,`g = −9.80665`(`testbench.py` 第 68–75 行)

而真实台架几何是"端部点质量 m_tip + 均匀杆 m_arm + 近轴轮毂 m_hub"三件套。二者**动力学恒等**的条件由 `record_pendulum_bench.py::pendulum_equiv`(第 94–99 行)给出。

#### 3.3.2 公式

```
M     = m_tip·L² + m_arm·L²/3 + m_hub·r_hub²                 (绕轴惯量 [kg·m²])
B_max = (m_tip + m_arm/2)·g·L + m_hub·g·r_hub                (最大重力矩 [N·m], q=±90°)
L_eq  = M·g / B_max                                          (点质量摆长 [m])
m_eq  = B_max² / (g²·M)                                      (点质量 [kg])
```

推导(等价性验证):要求 `Pendulum` 复现同样惯量 M 和最大重力矩 B_max。令 `arm_mass=0`,则
- 惯量 `m_eq·L_eq² = (B_max²/(g²M))·(M²g²/B_max²) = M` ✔
- 重力矩 `m_eq·g·L_eq = (B_max²/(g²M))·g·(M·g/B_max) = B_max` ✔

#### 3.3.3 record 脚本怎么用

每条 log 只写 `mass=m_eq, arm_mass=0.0, length=L_eq`,因此 `Pendulum` **零改动**即复现同一动力学。

用 runbook 几何示例(`L=0.150, m_arm=0.018, m_hub=0.020, r_hub=0.010`)实测验证:

| m_tip [g] | M [kg·m²] | B_max [N·m] | L_eq [m] | m_eq [kg] | Pendulum 惯量比 | Pendulum 重力矩比 |
|---|---:|---:|---:|---:|---:|---:|
| 50 | 0.001262 | 0.08878 | 0.13945 | 0.06490 | 1.0000 | 0.9997 |
| 100 | 0.002387 | 0.16236 | 0.14423 | 0.11475 | 1.0000 | 0.9997 |
| 150 | 0.003512 | 0.23593 | 0.14603 | 0.16469 | 1.0000 | 0.9997 |

- 惯量**严格**复现(比值 1.0000)。
- 重力矩比值 0.9997:record 里 `G=9.81`,而 `Pendulum` 用 `g=9.80665`,相差 ≈0.03%,工程可忽略;语义上,log 里 `m_eq/L_eq` 是拿 9.81 算的,回放时 `Pendulum` 用 9.80665 的 `sin(q)` 力矩多乘了 0.9997。

### 3.4 正演回放:`Simulator.rollout_log`

`bam/fit.py` 用的仿真引擎是 `bam/simulate.py`。回放时 `simulate_control=True`,即**不用日志里的控制量,而是用固件控制律实时重算**(这正是"标定固件+摩擦"的意义)。

#### 3.4.1 初始化(`simulate.py` 第 106–133 行)

```
reset(q=first_position, dq=first_speed)     # 从日志第一条真值起播
  → Model.reset() → Actuator.reset()        # 清空固件内部目标 (stateful 复位)
  → actuator.load_log(log)                  # 用 log 的 mass/length/kp/vin 建 Pendulum
  → 若 command_delay>0: 预计算延迟后的 goal 序列 (fractional_delay_shift)
```

每时间步循环 `for k, entry in enumerate(log["entries"])`:
- 先 `positions.append(q); velocities.append(dq)`(**步进前**状态,位置 k 对齐日志 k);
- 算控制量 → `controls.append(control)`;
- `self.step(control, entry["torque_enable"], dt)`。
- 若 `reset_period` 设了:每 `reset_period` 秒把仿真状态重置回当前日志真值(`simulate.py` 第 153–155 行),用于**抑制长时程误差累积**(`fit_leg_pendulum` 用 0.5 s)。

#### 3.4.2 固件控制律:`HLS2909Actuator.compute_control`(`bam/feetech/actuator.py` 第 298–391 行)

HLS-2909 位置伺服的**内部目标**不是直接跟 `goal_position`,而是走一个**二阶限幅的目标爬坡**(速度 + 加速度),然后 P(1/8)+D(1/4) 力矩。stateful,每拍一次:

```
① 内部目标(位置/速度)初始化或复位:
   q_target_smooth[0] = q,  v_target_smooth[0] = 0      (上电行为:从当前角开始, 静止)
② 目标速度爬坡:
   v_target = (q_target − q_target_smooth) / dt
   v_target = clamp(v_target, ±v_max)                   # 限速 (max_velocity)
   v_target = clamp(v_target, v_target_smooth ± a_max·dt)  # 限加速度 (max_acceleration)
   remaining = |q_target − q_target_smooth| / dt
   v_target = clamp(v_target, ±remaining)               # 软停车:剩余平移恰好停在目标, 无二阶过冲
   q_target_smooth += v_target·dt
③ 位置环 P(1/8) + D(1/4):
   duty = (q_target_smooth − q)·kp·error_gain/8
          + (−dq)·kd·error_gain/4                        # D 项作用在实际角速度上, Ki 无效
④ 固件限流 (register 28/44 = max_current):
   back_emf  = kt·dq
   duty_span = R·max_current / vin
   duty_center = back_emf / vin
   duty = clamp(duty, duty_center − duty_span, duty_center + duty_span)
        # 使 I = (duty·vin − kt·dq)/R 保持 |I| ≤ max_current
        # 只是"尝试": 高转速时反电动势使窗口落在物理范围外, 实际到不了 max_current
⑤ 物理 PWM 限幅 (max_pwm = 0.97) ← 最后施加
   返回控制电压 v = vin · duty
```

默认/出厂值(`feetech/actuator.py`):`kp=32, kd=32(缩1/4), error_gain=0.166, vin=12.0, max_pwm=0.97, max_current=1.95, max_velocity=19.1637, max_acceleration=500.0`。`kp`/`vin` 在 `load_log` 里被日志值覆盖。

#### 3.4.3 电机力矩:`VoltageControlledActuator.compute_torque`(`actuator.py` 第 317–342 行)

DBDC 电机含反电动势方程:

```
τ_m = kt·v/R − kt²·dq/R        (控制电压 v = vin·duty)
     = kt·(vin·duty)/R − kt²·dq/R
τ_m ← τ_m · torque_enable       (断扭矩 = 0, 只重力+摩擦)
```

注意这里**不重复**做限流:限流已在 `compute_control` 以 duty 窗口形式处理,电压已反映饱和,力矩直接由 DC 方程得出。

#### 3.4.4 重力矩与惯量:`Pendulum`(`testbench.py` 第 58–75 行)

```
τ_e(q)  = (mass + arm_mass/2)·g·length·sin(q)   // arm_mass=0 → m_eq·g·L_eq·sin(q)
M       = mass·length² + (arm_mass/3)·length² + armature   // ← compute_mass + get_extra_inertia
```

其中 `get_extra_inertia = model.armature`(电机转子/齿轮箱的**反射视在惯量**,默认 1e-3 kg·m²)。

注意所有用到 `q` 的地方都加了 `q_offset`:`bias_torque = compute_bias(q + q_offset, dq)`、`motor_torque = compute_torque(control, torque_enable, q + q_offset, dq)`。`q_offset` 是台架零位残差的"干扰参数",默认 0.0。

#### 3.4.5 摩擦预算与停止扭矩削顶(`Simulator.step`,`simulate.py` 第 65–104 行)

这是 BAM 最与众不同处:**不**直接把 Coulomb/Stribeck 摩擦当静态摩擦加进去,而是用 MuJoCo `dof_frictionloss` 的语义——**预算封顶的停止扭矩**:

```
① 算摩擦预算:
   (frictionloss, damping) = model.compute_frictions(τ_m, τ_e, dq)   # model.py:125
     frictionloss : Coulomb/Stribeck/负载项之和 [N·m]   ← 静态摩擦"预算"
     damping      : 粘性项 [N·m/(rad/s)]                 = friction_viscous

② 停止扭矩 tau_stop = (M/dt)·dq + (τ_m + τ_e)
   # 这是"若要让 dq 在一拍后归零"所需的扭矩

③ 静态摩擦 = −sign(tau_stop)·min( |tau_stop|, frictionloss + damping·|dq| )
   net_torque += 静态摩擦
   # 摩擦总是与"会使速度过零的那个净扭矩"反向, 但最多抵消到零(不反向), 封顶为预算

④ α = net_torque / M
   dq += α·dt ;  dq = clamp(dq, ±100)     # 速度硬限
   q  += dq·dt
```

`compute_frictions`(`model.py` 第 125–208 行)随模型 M1–M6 变复杂:

| 模型 | 特征 | `frictionloss` 构成 |
|---|---|---|
| m1 | Coulomb | `friction_base` |
| m2 | +Stribeck | `friction_base` + `stribeck_coeff·friction_stribeck`;`stribeck_coeff=exp(−(\|dq\|/dtheta)^alpha)` |
| m3 | +负载 | `friction_base` + `load_friction_base·\|τ_e−τ_m\|` |
| m4 | 负载+Stribeck | m3 + `stribeck_coeff·friction_stribeck` + `stribeck_coeff·load_friction_stribeck·\|τ_e−τ_m\|` |
| m5 | 上+directional | `friction_base` + `\|τ_e·load_friction_external − τ_m·load_friction_motor\|`(+Stribeck 项) |
| m6 | 上+quadratic | m5 + `stribeck_coeff·(方向项·二次项)` |

`damping`(粘性)**一律等于 `friction_viscous`**。

> **语义关键**:`frictionloss` 是"最大可施加的静摩擦扭矩",不是"一定会加的摩擦"。静止段(`tau_stop` 落在预算内)它把速度钉在 0;运动段(预算被 `damping·|dq|` 放大)它算作与速度反向的阻力。这就是论文里 `τ_fm` 的"摩擦力预算 + 停止扭矩削顶"。

#### 3.4.6 完整单步伪代码

```
tau_e     = Pendulum.compute_bias(q + q_offset, dq)                # 重力
tau_m     = Actuator.compute_torque(v, torque_enable, q+q_offset, dq)  # 电机
floss,damp = Model.compute_frictions(tau_m, tau_e, dq)             # 摩擦预算
M         = Pendulum.compute_mass(q+q_offset, dq) + armature       # 惯量
net = tau_m + tau_e
tau_stop = (M/dt)·dq + net
fric = −sign(tau_stop)·min(|tau_stop|, floss + damp·|dq|)
net += fric
dq += (net/M)·dt ; dq=clip(dq,±100)
q  += dq·dt
```

### 3.5 代价函数与优化器

#### 3.5.1 `compute_score`:位置 MAE(`fit.py` 第 62–70 行)

```
def compute_score(model, log):
    result = Simulator(model).rollout_log(log, reset_period, simulate_control=True)
    positions      = result[0]                        # 预测轨迹 q_hat(t)
    log_positions  = [e["position"] for e in log]     # 真机轨迹 q_log(t)
    return np.mean(np.abs(positions − log_positions)) # 单条 MAE
```

#### 3.5.2 目标函数:各 log 位置 MAE 均值(`fit.py` 第 73–82 行)

```
compute_scores(model, logs) = mean_{log}[ compute_score(model, log) ]
```

- 训练时 `objective()` 用 `logs.make_batch()` 一次向量化回放整批(所有 log 的 `mass/length/dt/kp/...` 与 `entries` 向量化,截断到最小长度),得到"所有 log、所有拍"的 MAE 均值。
- 评估/验证时 `compute_scores` 逐条回放再平均(口径一致)。

即:**位置 MAE(所有 log、所有时间点平均)**。

#### 3.5.3 优化算法:optuna CMA-ES / BIPOP(`fit.py` 第 99–108, 215–236 行)

```
objective(trial):
    model = make_model()                       # 重建 m1..m6 + hls2909
    for name,param in model.get_parameters():
        if param.optimize:
            param.value = trial.suggest_float(name, param.min, param.max)
    return compute_score(model, logs_batch)

sampler = optuna.samplers.CmaEsSampler(restart_strategy="bipop")   # 默认 --method cmaes
study.optimize(objective, n_trials=args.trials)                    # 默认 trials=100000
```

- `--trials`:`bam.fit` 默认 100 000;`fit_leg_pendulum.py` 默认 20 000(`fit_leg_pendulum.py:85`),命令可覆盖;复杂模型(M5/M6)建议加 trials。
- 每个 `trial` 建一个新模型、跑一次整批复放。CMA-ES 的 `restart_strategy="bipop"` 在收敛后重启,用于多模态/高维任务的鲁棒搜索。
- `--load-study` 可"续跑";`--workers>1` 用共享 SQLite 研究库并行。

#### 3.5.4 哪些量交给优化、哪些固定(`fit.py` 第 40, 85–96 行)

`make_model` 读 `--set`:一个 Python dict 字符串,被 `eval` 后**逐个把参数 `.value` 覆写为真值,且 `.optimize=False`**——即"固定为真值,不漂移"。

m1 全部可辨识参数(`model.py` 的 Model + `feetech/actuator.py` 的 initialize):

| 参数 | 默认 | 优化区间 | 说明 |
|---|---|---|---|
| `kt` | 1.454 | (0.2, 3.0) | 力矩常数,交给优化 |
| `R` | 20.0 | (2.0, 50.0) | 电机电阻,交给优化 |
| `armature` | 1e-3 | (1e-6, 0.05) | 反射惯量,交给优化 |
| `q_offset` | 0.0 | (−0.1, 0.1) | 零位残差,交给优化 |
| `command_delay` | 0.0 | (0.0, 0.05) | 通讯延迟,交给优化($q_{goal}$ 序列分数延迟位移) |
| `error_gain` | 0.166 | (0.01, 2.0) | 位置环增益映射,交给优化 |
| `kd` | 32.0 | (0.0, 254.0) | 位置环 D 增益,交给优化 |
| `friction_base` | 0.05 | (0.0, 0.2) | 摩擦预算,交给优化 |
| `friction_viscous` | 0.1 | (0.0, 1.0) | 粘性摩擦,交给优化 |
| `max_velocity` | 19.1637 | (1.92, 191.6) | 出厂限速,--set 固定 |
| `max_acceleration` | 500.0 | (50, 5000) | 出厂限加速度,--set 固定 |
| `max_current` | 1.95 | (0.1, 5.0) | 出厂限流,--set 固定 |

**`--set` 的用意**:把**固件出厂限幅**(`max_velocity`/`max_acceleration`/`max_current`)固定为真值,**防止 CMA-ES 把已知量当自由度漂移、制造假性低 MAE**。否则优化器会让这些限幅去迁就轨迹,得到一组物理上不可能的参数,MAE 却好看。

推荐直跑(不含 `--set` 也可,但 `fit_leg_pendulum` wrapper 未透传 `--set`):

```bash
cd bam
uv sync --extra identification
uv run python -m bam.fit --actuator hls2909 --model m1 \
  --logdir ../microduck_rl/hls2909_calibration/bench \
  --output bam/params/hls2909/m1.json --trials 100000 \
  --set "{'max_velocity':19.1637,'max_acceleration':500.0,'max_current':1.95}"
```

> ⚠ `fit_leg_pendulum.py` 的 `run_fit`(`fit_leg_pendulum.py` 第 43–60 行)**没有** `--set` 透传,所以它拟合时 `max_velocity/max_acceleration/max_current` 连同其他参数一起漂移。因此严格做 `--set` 固定限幅的版本,请用上面这条直跑命令(或在 wrapper 中加透传)。这是本实验文档标注的**取舍/待改点**。

### 3.6 独立验证与判档

#### 3.6.1 留出分组(`fit_leg_pendulum.py` 第 97–115 行)

```
36 条按 "轨迹×质量" 分组,每个 (trajectory, m_tip) 组取最后一次重复(rep 最大) 留出:
  4 轨迹 × 3 档 = 12 条独立验证
  其余 24 条 → 拟合集(拷到临时目录交给 bam.fit)
```
分组键:文件名 `{traj}_tip{mg}_rep{r}.json` 去掉 `_rep` 后缀。

#### 3.6.2 独立验证回放(`fit_leg_pendulum.py::evaluate`)

```
model = load_model(fit_mN.json)                    # bam.model.load_model
for each 验证log:
    res = Simulator(model).rollout_log(log, reset_period=0.5, simulate_control=True)
    pos_pred = res[0];  pos_real = log entries position
    MAE = mean(|pos_pred − pos_real|)
validation_MAE = mean over 12 条
```

与 `bam.fit.compute_score` **同口径**(位置 MAE),但用 `reset_period=0.5`(每 0.5 s 同步一次真值)抑制长时程发散。

#### 3.6.3 判档规则

```
门:独立验证 MAE < 0.157 rad (≈ 0.1 × π/2)
 ├─ 是 且 train ≈ validation  → 落地;同等通过选 m1
 │     (m6 显著更优且无分叉 → 选 m6 + 写理由)
 ├─ 是 但 train ≪ validation  → 过拟合:--set 固定已知量 / 加重复
 └─ 否
     ├─ 看 overlay: 落体段不重合→摩擦(试 m6);
     │   驱动段相位平移→零位/符号/q_offset; 饱和段好→限流模型正确
     └─ 回查数据/提高 trials/补 5 重复
```

报告写入 `mae_report.md`(`fit_leg_pendulum.py` 第 133–147 行)。`train_mae` 用同一模型在 24 条拟合数据上 `evaluate` 得到,用于比较 train/val 是否分叉。

### 3.7 落地:`bam/params/hls2909/m1.json` 当前值与来源标注

`/home/joyandai/bam/bam/params/hls2909/m1.json` 全文:

```json
{
  "kt": 1.454, "R": 20.0, "armature": 0.001, "q_offset": 0.0,
  "friction_base": 0.08, "friction_viscous": 0.012,
  "error_gain": 0.166, "kd": 32,
  "max_velocity": 19.1637, "max_acceleration": 500.0, "max_current": 1.95,
  "model": "m1", "actuator": "hls2909"
}
```

#### 3.7.1 来源分级

| 参数 | 值 | 来源类别 | 依据 |
|---|---|---|---|
| `kt` | 1.454 | **出厂/实测**(可信) | 数据手册 KT=14.83 kg·cm/A(=1.454 N·m/A,输出侧),齿轮比 320:1;`feetech/actuator.py` |
| `R` | 20.0 | **出厂/实测**(内部一致组) | 12 V/0.6 A 堵转等价;数据手册组 `kt/堵转/ R` 内部自洽(探针回归得 6.8 Ω、堵转 3.5 Ω,均标不可靠) |
| `error_gain` | 0.166 | **出厂/标定** | 把 `kp·Δq` 映射到 duty;STS3215 示波器实测,HLS 固件经行为探针确认 `duty=Δq·kp·error_gain/8` |
| `kd` | 32 | **出厂/实测** | 寄存器 22/51,固件缩放 1/4 |
| `max_velocity` | 19.1637 | **出厂/实测** | 寄存器 84=250 LSB=183 RPM;物理解释的 8.06 rad/s 是从 kt/R 涌现,非限速 |
| `max_acceleration` | 500.0 | **模型语义值(行为/训练用)** | 固件 reg41=0=最大加速度(无穷),500 是"可数值求解的大值";microduck 训练用 500 |
| `max_current` | 1.95 | **出厂/实测** | 寄存器 28/44=300×6.5 mA=1.95 A;**实验临时降到 0.975 A(reg44=150)仅为防堵转**,拟合/部署用 1.95 A |
| `armature` | 0.001 | **行为级反推占位** | 校准自"已知良好策略在真机站立";docstring 明言"still await a proper pendulum identification" |
| `friction_base` | 0.08 | **行为级反推占位** | m1 摩擦预算,非摆锤标定值 |
| `friction_viscous` | 0.012 | **行为级反推占位** | 同上 |
| `q_offset` | 0.0 | **模型默认/拟合项** | 台架零位残差干扰参数,默认 0;实际由优化给定 |

#### 3.7.2 标识

- **出厂/实测**(`kt/R/error_gain/kd/max_velocity/max_current`):数据手册 + 固件寄存器回读 + 示波器/探针实测,**内部自洽、基本可信**,拟合时是初值(`--set` 把限幅固定为真值)。
- **行为级反推占位**(`friction_base=0.08 / friction_viscous=0.012 / armature=0.001`):由"已知良好策略真机站得住"反推,**未知误差**——这正是要跑摆锤拟合的动机。
- **模型语义值**(`max_acceleration=500.0`):不是严格出厂数,是"可数值求解的固件最大值"代理,与 microduck 训练口径一致。

#### 3.7.3 拟合后应如何变化

拟合输出 `fit_m1.json` 会**覆写**:`kt/R/armature/friction_base/friction_viscous/error_gain/kd`(+ m3–m6 的负载/Stribeck/二次项)+ `q_offset/command_delay`。落地时把 `fit_m1.json` → `bam/params/hls2909/m1.json`(`"model"`/`"actuator"` 字段已满足 `load_model` 约定)。**不要**把实验临时限流 0.975 A 带进模型(保持 1.95 A)。

### 3.8 一致性提醒(易踩的坑)

1. **`arm_mass=0` 是等价已完成的标志**:`Pendulum.compute_mass`/`compute_bias` 只认 `(mass, arm_mass, length)`。若某条 log 的 `arm_mass` 不是 0,说明没走 `pendulum_equiv`,惯量/重力矩会算错。
2. **`manifest.json` 需排除**:它没有 `entries`,`bam.fit` 指到 raw 目录必挂。`fit_leg_pendulum.py` 已排除;直跑 `bam.fit` 指到 `bench_processed/`。
3. **`command_delay` 会在 `--set` 缺省下自由漂移**:`q_offset` 与 `command_delay` 都是"干扰参数",拟合结果里的它们代表"模型自洽的台架零位/通讯延迟",不一定是物理真值。
4. **stateful 执行器需按拍顺序调用**:`HLS2909Actuator` 的 `q_target_smooth/v_target_smooth` 在每条 `rollout_log` 开头经 `Simulator.reset → Model.reset → Actuator.reset()` 清空,首次 `compute_control` 从当拍 `q` 初始化(= 舵机上电)→ 每条 log 独立、确定性回放;批处理(`Logs.make_batch` 向量化)同样成立。
5. **批处理下的 dt 是向量**:`HLS2909.compute_control` 里 `if np.any(np.asarray(dt) > 0)` 是修复过的(避免 bam.fit 批处理时 `if dt>0` 的 "ambiguous truth value");若用旧 vendor 版跑 `python -m bam.fit`,此修复为前置条件。
6. **`G=9.81` vs `g=9.80665`**:等价换算用 9.81,`Pendulum` 用 9.80665,重力矩比 ≈0.9997,工程可忽略,但严格逐位核对时注意。
7. **批处理截断**:`Logs.make_batch` 把所有 log 截到最小长度,若各条长度差 >1 会打警告——不同轨迹/重复的点数应接近,否则检查采样。
8. **`evaluate` 用 `reset_period=0.5`**:独立验证每 0.5 s 与真机轨迹同步一次;这会让长时程 MAE 偏乐观,与 `bam.fit --eval`(无 reset_period)略有差别,判档以 wrapper 口径为准。

---

## 4. 快速参考

### 4.1 关键文件位置表

| 内容 | 位置 |
|---|---|
| **唯一参数表**(速查手册,冲突以它为准) | `microduck_rl/docs/pendulum_bench_runbook.html` |
| **实验步骤 × 代码对照** | `/home/joyandai/bam/docs/identification/hls2909_pendulum_experiment.md` |
| 轨迹定义 + 注册表(4 条轨迹 + `sin_sin`) | `bam/trajectory.py`(注册表第 162–170 行) |
| `Model` 类(flag + 参数对象 + `compute_frictions`) | `bam/model.py` 第 17–208 行 |
| `models` 注册表(m1–m6 flag 组合) | `bam/model.py` 第 256–281 行 |
| `set_actuator`(按 flag 创建参数) | `bam/model.py` 第 71–123 行 |
| 摩擦参数默认值/边界 | `bam/model.py` 第 92–123 行 |
| `compute_frictions`(标量版摩擦预算) | `bam/model.py` 第 125–208 行 |
| `Parameter` 类(value/min/max/optimize) | `bam/parameter.py` |
| 六个模型表达式(公式权威) | `bam/docs/theory/models.rst` |
| `_compute_friction_budget`(向量版摩擦预算) | `bam/mjlab.py` 第 470–548 行 |
| `compute`(Stribeck 系数、写字段) | `bam/mjlab.py` 第 639–748 行(Stribeck 第 734 行) |
| 固件控制律 `HLS2909Actuator` | `bam/feetech/actuator.py`(`compute_control` 第 298–391 行、`initialize` 第 266–296 行) |
| 电机力矩 `compute_torque` | `bam/actuator.py`(`VoltageControlledActuator` 第 317–342 行) |
| 台架动力学 `Pendulum` | `bam/testbench.py`(`compute_mass` 第 58–66 行、`compute_bias` 第 68–75 行) |
| 回放引擎 | `bam/simulate.py`(`rollout_log` 第 106–180 行、`step` 第 65–104 行) |
| 目标函数/优化器 | `bam/fit.py`(`compute_score` 62–70、`compute_scores` 73–82、`objective` 99–108、`make_model` 85–96、CMA-ES 215–219) |
| 批量读取 | `bam/logs.py`(`make_batch` 76–106) |
| 采集脚本(内置 `_traj` 副本、`pendulum_equiv`、安全链) | `microduck_rl/scripts/record_pendulum_bench.py`(`_traj` 76–82、`LSB_RAD` 84、`G` 85、安全常量 86–91、`pendulum_equiv` 94–99) |
| 拟合/验收包装 | `microduck_rl/scripts/fit_leg_pendulum.py`(`run_fit` 43–60、`evaluate` 63–77、留出分组 97–115、`mae_report` 133–147) |
| 后处理(均匀网格) | `microduck_rl/scripts/process_bench_logs.py` |
| 只读检查/接线 | `microduck_rl/scripts/setup_bench_servo.py`、`microduck_rl/scripts/read_hls_registers.py` |
| 详细版设计(与 runbook 有出入) | `microduck_rl/scripts/make_pendulum_experiment_design_html.py` |
| XL330 m6 参数量级示例 | `bam/params/xl330/m6.json` |
| **HLS-2909 m1 现值(占位/行为级)** | `bam/params/hls2909/m1.json` |
| 部署常量(判档后同步) | `microduck_rl/src/mjlab_microduck/robot/microduck_constants.py::_BAM_ACTUATOR_KWARGS`(当前 `model="m1"`,标注 PLACEHOLDER) |
| 上游采集建议 | `bam/docs/identification/acquisition.rst` |
| BAM 论文 | [arXiv 2410.08650](https://arxiv.org/pdf/2410.08650v1) |

### 4.2 关键命令

**只读检查/B 关**(`cwd=/home/joyandai`):
```bash
python3 microduck_rl/scripts/setup_bench_servo.py --port /dev/ttyACM0 --find-id 23 --set-id 1
```

**C 关 50 g 单条试跑**:
```bash
python3 microduck_rl/scripts/record_pendulum_bench.py --port /dev/ttyACM0 --id 1 \
  --tip-mass 0.05 --arm-mass 0.018 --arm-length 0.15 --hub-mass 0.020 --hub-radius 0.010 \
  --trajectory up_and_down --reps 1 --out microduck_rl/hls2909_calibration/pilot
```

**D 关正式 36 条(约 40–60 min)**:
```bash
python3 microduck_rl/scripts/record_pendulum_bench.py --port /dev/ttyACM0 --id 1 \
  --tip-mass 0.05 --tip-mass 0.10 --tip-mass 0.15 \
  --arm-mass 0.018 --arm-length 0.15 --hub-mass 0.020 --hub-radius 0.010 \
  --trajectory sin_time_square --trajectory sin_sin --trajectory lift_and_drop --trajectory up_and_down \
  --reps 3 --limit-a 0.975 --torque-budget 0.35 --out microduck_rl/hls2909_calibration/bench
```

**E 关 M1–M6 拟合 + 独立验证**(`cwd=/home/joyandai/microduck_rl`):
```bash
.venv/bin/python scripts/fit_leg_pendulum.py --logdir hls2909_calibration/bench \
  --actuator hls2909 --models m1 m2 m3 m4 m5 m6 --trials 20000 \
  --out hls2909_calibration/fit
```

**等价直跑(本仓库,可 `--set` 固定限幅)**:
```bash
cd bam
uv sync --extra identification
uv run python -m bam.fit --actuator hls2909 --model m1 \
  --logdir ../microduck_rl/hls2909_calibration/bench \
  --output bam/params/hls2909/m1.json --trials 100000 \
  --set "{'max_velocity':19.1637,'max_acceleration':500.0,'max_current':1.95}"
```

### 4.3 当前 `hls2909 m1.json` 值

`/home/joyandai/bam/bam/params/hls2909/m1.json`(当前为**行为级占位**,待摆锤辨识替换):

```json
{
  "kt": 1.454, "R": 20.0, "armature": 0.001, "q_offset": 0.0,
  "friction_base": 0.08, "friction_viscous": 0.012,
  "error_gain": 0.166, "kd": 32,
  "max_velocity": 19.1637, "max_acceleration": 500.0, "max_current": 1.95,
  "model": "m1", "actuator": "hls2909"
}
```

**判档落地要点**:
1. 拟合输出 `fit_m1..m6.json` + `mae_report.md`;选档门 `独立验证 MAE < 0.157 rad` 且 train≈val;同等通过选 **m1**。
2. 落地到 `bam/params/hls2909/mN.json`(本仓库)+ `vendor/bam/bam/params/hls2909/` + `microduck_constants.py` 的 `model="m1"` 升级为判档结果(需确认)。
3. **保持模型中 `max_current = 1.95 A`**(实验临时限流 0.975 A 不要带进模型)。
4. 冒烟:单舵机阶跃交叉验证 / `validate_bam_testbench.py` / `testbench_sim2real.py`,通过后开长训。
5. 回归:`hls2909_calibration/` 数据目录 zip 备份;提交前跑 `uv run pytest`。

---

## 5. 参考文献

- BAM 论文(摆锤辨识协议 / M1–M6):arXiv [2410.08650](https://arxiv.org/pdf/2410.08650v1)
- 唯一参数表 / 速查手册:`microduck_rl/docs/pendulum_bench_runbook.html`
- 实验步骤 × 代码对照:`/home/joyandai/bam/docs/identification/hls2909_pendulum_experiment.md`
- 上游采集建议:`bam/docs/identification/acquisition.rst`
- 模型公式权威:`bam/docs/theory/models.rst`
- 模型/参数/实现:`bam/model.py`、`bam/parameter.py`、`bam/mjlab.py`
- 轨迹定义 + 注册表:`bam/trajectory.py`
- 采集脚本:`microduck_rl/scripts/record_pendulum_bench.py`;拟合包装:`microduck_rl/scripts/fit_leg_pendulum.py`;详细版设计:`microduck_rl/scripts/make_pendulum_experiment_design_html.py`

> **一致性声明**:本文所有数值(LSB 分辨率、m_tip 上限、B_max、L_eq、m_eq、摆周期、reg44 换算、`s(θ)` 表格等)均已用脚本/公式复核,与所引代码一致;凡 runbook 与详细版冲突处均标注"**以 runbook 为准**"。m3 参数符号(m3/m4 用 `load_friction_base`,`m5/m6` 才用 `load_friction_motor/external`)以**代码与 `models.rst`** 为准;`sin_sin` 在 `bam/trajectory.py` 注册表(第 166 行)中,`record_pendulum_bench.py` 内置 `_traj` 副本亦有。
