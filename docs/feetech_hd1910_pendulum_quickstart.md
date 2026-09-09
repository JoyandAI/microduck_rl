# HD-1910-C001 摆锤实验 · 无脑操作手册（照抄版）

> 与 `hls2909_pendulum_quickstart.md` 同构，**HD 差异全部用 ⚠️HD 标注**；
> 共享部分（台架几何称量、四轨迹说明、拟合流程、安全红线）直接沿用，本文不再重复展开。
>
> 当前 HD-1910 真机状态（2026-09-09 回读）：**ID=14**，固件 3.46，reg3=7946，
> 模式 **4 = 纯位置PD（Sim2Real）**，增益 **Kp=32 / Kd=40 / Ki=0**，保护电流 3.25A，
> 供电实测 5.1V（正式实验**调到 6.0V**）。详细寄存器与行为结论见
> `hd1910_servo_notes.md`（三仓库均有副本）。

---

## ⚠️HD 0. 前置：采集脚本适配清单（HD 必做，HLS 不用）

**✅ 已适配（2026-09-09）**：`microduck_rl/scripts/record_pendulum_bench.py` 已按下表全部改完
（B 式写序/保持模式/零位守卫/电流安全说明），直接用本文命令即可。以下为改动说明与旧版核对要点：

| # | 改动 | 原因（真机实测） |
|---|---|---|
| 1 | 写序改 **B 式**：配置时 `46=max` 一次，之后**每拍只写** `42=目标 → 40=1`（正文实验命令不变，只改脚本内部） | B 式连做 16/16 全动最稳；HLS 原序 `46=0→42→40=1→46=32767` 在 mode 4 下**不动**（2026-09-09 实测） |
| 2 | 配置段**不写 reg33=0**；脚本现在自动保持当前模式（读到 4 或 0 就不动） | 写 0 会切到模式 0（角度伺服+限力），与部署控制律不一致，识别结果不能用 |
| 3 | 目标写入做**零位守卫**：q_zero 落在 300–3796 之外直接拒绝运行（配合"误差不回绕"） | mode 4 **有效目标被钳位 [0,4095] 且误差不回绕**：q_zero 靠近 0/4095 时小目标会冲向限位（实测 4094→338 走 -3755 LSB 长路径；写 6349 停在 4094） |
| 4 | 采集命令 `--id 14`（或先用 `setup_bench_servo.py --find-id 14 --set-id 1` 改成 1） | 当前舵机 ID=14，并非 1 |
| 5 | ⚠️ **电流安全检查失效**：HD 的 reg69 反馈在运动中读 0（reg73 偏置 2040）→ 脚本的 `|I|>1.0A×0.7s` 自动中止在 HD 上不起作用 | 安全链只剩：温度(reg63) + 30s 看门狗 + 3A 物理保险丝 + 急停 —— 人为关注度要提高 |

上电后先用 `tools/check_servo_bench.py --port /dev/ttyACM0 --id 14 --yes` 复检到 **PASS**（含空转），
再进第 2 节正式采集。

> 台架：HD-1910 与 HL-2909 同尺寸（34×20×23 mm）、同 25T 舵盘、双轴，现有夹具可直接用；
> 外壳是 PA66+GF43% 塑料，C 夹/压板力度别过大。回差 ≤0.5°（比 HLS 的 ≤1° 更好，有利拟合）。

---

## 1. 称量参数表（只做一次，和 HLS 完全一样，填下表）

| 参数 | 含义 | 怎么量 | 示例 | 你的实测值 |
|---|---|---|---|---|
| `m_arm` | 摆杆质量（只称杆） | 电子秤 0.1 g | **10/15/20cm 杆已称：0.0109 / 0.0124 / 0.0145 kg** | 用哪根填哪根 |
| `m_hub` | 舵机臂 + 杆夹 + 近轴螺丝 | 电子秤 0.1 g | 0.020 kg | `____` |
| `r_hub` | 轴心 → m_hub 质心 | 卡尺 ±2 mm | 0.010 m | `____` |
| `L` | 轴心 → 端部总质量**质心** | 卡尺 ±1 mm | 0.150 m | `____` |
| `m_tip` 50 g | 托盘 + 砝码 + 夹紧件**整套** | 电子秤 0.1 g | 0.050 kg | `____` |
| `m_tip` 100 g | 同上一整套 | 电子秤 0.1 g | 0.100 kg | `____` |
| `m_tip` 150 g | 同上一整套 | 电子秤 0.1 g | 0.150 kg | `____` |

⚠️ 红线：`m_tip × g × L ≤ 0.35 N·m`。HD-1910 @6V 额定 3 kg·cm(0.294 N·m)，
150 g + 0.15 m 的完整静态重力矩 ≈ 0.236 N·m → **工作在额定以内，不会发热**，比 HLS 更从容。

**现场填写速查**（抄命令前 2 分钟搞定，下划线处填值）：

- ✅ 已确定：`--arm-mass 0.0124`（**15 cm 杆**已称；10/20 cm 杆改用 0.0109 / 0.0145）
- ❓ 现场称/量（电子秤 0.1 g + 卡尺 ±1 mm）：
  `--tip-mass` 50g 档整套 `____` kg（100g 档 `____`、150g 档 `____`，**整套含托盘+夹头+螺丝**）
  `--hub-mass` `____` kg；`--hub-radius` `____` m；`--arm-length` = 实测 **L** `____` m
  （15 cm 杆名义 0.150，装好端部堆叠后实测，可能略大于 0.150）

> 其余命令参数均为脚本自动处理（临时限流/零位/符号检测/安全链），无需填写。

### ⚠️HD 模型初值（已写入 `bam/params/hd1910/m1.json`，与真机一致）

```text
kt = 0.736 N·m/A（规格书 7.5 kg·cm/A）      R ≈ 3.75 Ω（6V/1.6A 堵转）
kp = 32 / kd = 40 / ki = 0（真机回读）       max_current = 3.25 A（reg28/44=500）
max_velocity = 9.63 rad/s @6V（规格 92RPM）  max_acceleration = 500.0（限幅默认关，仅备用）
供电 6.0 V；模型 vin=6.0；编码器 12-bit 0.088°/LSB；减速比 320:1
```

---

## 2. 正式实验命令（HD 版：三档各一条，只改 `--tip-mass`）

> 命令里的 `0.018 / 0.15 / 0.020 / 0.010` 换成第 1 节实测值；
> **`--tip-mass` 必须填你称的整套质量**；输出目录换成 `hd1910_calibration/bench`（与 HLS 分开）。

### 2.1 台架就位（同 HLS）

- [x] 装摆臂（先不装砝码）；C 夹 ×2 锁死；轴水平；手动摆 ±90° 无碰撞；摆动平面清空
- [x] ⚠️HD **电源 = 6.0 V 稳压**（HD-1910 是 6V 级，4–8.4V；**不是 12V！** 12V 会触发过压保护）
- [x] ⚠️HD 舵机总线 ID=14；确认只有这一只在线（`--list-ports` + 扫描）

### 2.2 第 1 档：装 50 g 整套 → 先试跑 1 条

```bash
/usr/bin/python3 scripts/record_pendulum_bench.py --port /dev/ttyACM0 --id 14 \
  --tip-mass 0.05 --arm-mass 0.0124 --arm-length 0.15 --hub-mass 0.020 --hub-radius 0.010 \
  --trajectory up_and_down --reps 1 --out hd1910_calibration/pilot
```

✅ 标准：输入 START 后，杆从下垂 0° 匀抬到 +90°（正方向正确）、无碰撞、`OK` 且样本 > 50。
→ 通过后跑全量：

```bash
/usr/bin/python3 scripts/record_pendulum_bench.py --port /dev/ttyACM0 --id 14 \
  --tip-mass 0.05 --arm-mass 0.0124 --arm-length 0.15 --hub-mass 0.020 --hub-radius 0.010 \
  --trajectory sin_time_square --trajectory sin_sin --trajectory lift_and_drop --trajectory up_and_down \
  --reps 3 --limit-a 0.975 --torque-budget 0.35 --out hd1910_calibration/bench
```

脚本会要求输入 `START`（大写，确认手已离开摆动平面）。共 12 条（4 轨迹 × 3 次），约 12–15 分钟。

### 2.3 第 2 档：换 100 g → 上一条命令只改 `--tip-mass 0.10`

### 2.4 第 3 档：换 150 g → 再改 `--tip-mass 0.15`

换砝码要点（同 HLS）：重新整套称重 → M6 螺母锁紧 + 螺纹胶 → 手动摆过 ±90° → 输入 START。

### 2.5 数据检查

```bash
ls hd1910_calibration/bench/*.json | grep -v manifest | wc -l   # 应 = 36
```

### 2.6 拟合（同 HLS 流程，仅目录与 models 相同）

```bash
/usr/bin/python3 scripts/process_bench_logs.py \
  --in hd1910_calibration/bench --out hd1910_calibration/bench_processed --dt 0.005

.venv/bin/python scripts/fit_leg_pendulum.py \
  --logdir hd1910_calibration/bench_processed \
  --actuator hd1910 --models m1 m2 m3 m4 m5 m6 \
  --trials 20000 --out hd1910_calibration/fit
```

> ⚠️HD 拟合时用 `--actuator hd1910`（bam 已注册）；bam 侧版本要求 = 本仓库新版
> （`HD1910Actuator` 在 `/home/joyandai/bam/bam/feetech/actuator.py`，editable install 已指向）。

### 2.7 看结果

```bash
cat hd1910_calibration/fit/mae_report.md
```

判定规则同 HLS：PASS = 独立验证 MAE < 0.157 rad；全过且无分叉 → 选 PASS 中 MAE 最小档，同等选 m1。
落地路径：`bam/params/hd1910/mN.json`（当前只有 m1.json 初值档）。

---

## 3. 命令行参数全解（record_pendulum_bench.py 为主）

### 3.1 `record_pendulum_bench.py`（采集，最重要）

```bash
/usr/bin/python3 scripts/record_pendulum_bench.py \
  --port /dev/ttyACM0 --id 14 \
  --tip-mass 0.05 --tip-mass 0.10 --tip-mass 0.15 \
  --arm-mass 0.0124 --arm-length 0.15 --hub-mass 0.020 --hub-radius 0.010 \
  --trajectory sin_time_square --trajectory sin_sin --trajectory lift_and_drop --trajectory up_and_down \
  --reps 3 --limit-a 0.975 --torque-budget 0.35 --out hd1910_calibration/bench
```

| 参数 | 含义 | 取值/单位 | 怎么填 | 填错了会怎样 |
|---|---|---|---|---|
| `--port` | USB-TTL 串口设备 | `/dev/ttyACM0` 等 | `ls /dev/ttyACM*` 看到哪个填哪个 | 打不开直接报错 |
| `--id` | 舵机总线 ID | 0–253 | ⚠️HD 当前=**14**（HLS 默认是 1）；可先用 setup 脚本改成 1 | 写错 ID → 采集全程无响应/扫不到 |
| `--tip-mass`（可重复） | 端部总质量，**每档一个**，脚本每档暂停等你换砝码 | 千克（kg），0.1 g 秤**整套**称 | 50/100/150 g 档分别填 `0.05 / 0.10 / 0.15`（用实测值） | 填太小 → 重力矩档位错、负载相关摩擦（m3–m6）辨识不出来；填超红线 → 脚本拒绝运行 |
| `--arm-mass` | 摆杆质量（只称杆，不含端部托） | kg | 实测，如 `0.018` | 影响惯量 M 与重力矩 B_max 的换算，误差会进拟合 |
| `--arm-length` | **L**：轴心 → 端部总质量质心 | 米（m），卡尺 ±1 mm | 实测，如 `0.150` | 摆长是全部动力学换算的基准，务必实测不是杆长 |
| `--hub-mass` | 舵机臂 + 杆夹 + 近轴螺丝总质量 | kg | 实测，如 `0.020` | 偏小 → 惯量/重力矩少算（量级小，影响有限） |
| `--hub-radius` | 轴心 → m_hub 质心 | 米（m） | 实测，如 `0.010` | 同上 |
| `--trajectory`（可重复） | 激励轨迹，每条 6 s | 四个：`sin_time_square`（速度扫频 ±57°）、`sin_sin`（多频复合 ±90°）、`lift_and_drop`（抬升→断扭自由落体）、`up_and_down`（低速 0→+90°→+72°） | **四个都写** | 少一个 → 该激励缺失，对应摩擦项（背驱/静摩擦）不可辨识；写错名 → 报错退出 |
| `--reps` | 每"质量×轨迹"的重复次数 | 整数 | `3`（**最后一次重复留作独立验证**，拟合脚本自动切分） | 只填 1 → 无法切独立验证集，拟合脚本报错 |
| `--out` | 输出目录（原始 log + manifest） | 路径 | HLS 用 `hls2909_calibration/bench`；⚠️HD 用 `hd1910_calibration/bench` 分开存 | 与 HLS 混用会污染 36 条的目录 |
| `--limit-a` | **临时限流**（A）：运行期间写 reg44，结束恢复运行前的值 | 安培 | `0.975`（与 HLS 相同，防堵转发热） | 太大 → 堵转发热风险；太小 → 正常摆锤被限流（峰值约 0.4A，0.975 有余量） |
| `--torque-budget` | 最大静态重力矩红线检查 | N·m | `0.35`（**禁止加大**） | 超了 → 脚本直接拒绝运行 |

**脚本自动做的事**（不用管，但要了解）：临时限流 → 位置伺服配置 → 零位 q_zero（断电 30 读中位）→
符号检测（+0.2 rad 试步定 sign）→ 逐拍写目标/读反馈 → 安全链（温度 >50°C / |I|>1.0A×0.7s /
30s 看门狗 → 立即断扭退出）。

**交互**：每进一档质量，脚本先关扭矩、打印 `[换砝码] …`，等你输入 **`START`** 大写回车才继续。

### 3.2 `setup_bench_servo.py`（扫描 / 改 ID / 只读体检）

| 参数 | 含义 | 说明 |
|---|---|---|
| `--port` | 串口 | 同 3.1 |
| `--find-id` | 指定要配置的舵机 ID | ⚠️HD 当前 = `14` |
| `--set-id` | 把舵机 ID 改成该值 | 改成 `1` 后，采集命令可用 `--id 1`；EPROM 写入，断电不丢 |

```bash
/usr/bin/python3 scripts/setup_bench_servo.py --port /dev/ttyACM0 --find-id 14 --set-id 1
```

### 3.3 `process_bench_logs.py`（重采样）

| 参数 | 含义 | 说明 |
|---|---|---|
| `--in` | 原始 log 目录 | `hd1910_calibration/bench` |
| `--out` | 输出目录 | `hd1910_calibration/bench_processed` |
| `--dt` | 均匀网格步长（s） | `0.005`（5 ms），bam 回放按固定 dt 步进 |

### 3.4 `fit_leg_pendulum.py`（拟合 + 独立验证）

| 参数 | 含义 | 说明 |
|---|---|---|
| `--logdir` | log 目录 | 用 `bench_processed`（均匀网格） |
| `--actuator` | bam 注册名 | ⚠️HD 用 **`hd1910`**（HLS 用 `hls2909`） |
| `--models` | 拟合哪些模型 | `m1 m2 m3 m4 m5 m6`；想省时间先 `m1 m6` |
| `--trials` | 每模型优化次数 | `20000`（CMA-ES） |
| `--out` | 输出目录 | `hd1910_calibration/fit`（产 `fit_mN.json` + `mae_report.md`） |

### 3.5 `check_servo_bench.py`（舵机体检，跑实验前必过）

| 参数 | 含义 | 说明 |
|---|---|---|
| `--port` / `--id` | 串口 / 舵机 ID | 当前 `--id 14` |
| `--no-move` | 只读体检（不动） | 快速检查用 |
| `--move` | 空转步数（LSB） | 默认 150 ≈ 13.2° |
| `--yes` | 跳过交互确认 | 无人值守用 |
| `--out` | 报告 JSON | 默认 `bench_servo_check.json` |

---

## 4. 安全红线（同 HLS 5 条 + HD 3 条）

1. `m_tip·g·L ≤ 0.35 N·m`，禁止加大砝码；
2. 摆动平面内手勿入/头勿探，操作站侧面；
3. 砝码必须 M6 锁紧螺母 + 螺纹胶（甩出是最大风险）；
4. 异常断电顺序：**先断 6 V，再拔 TTL**；
5. 全程人在场（`lift_and_drop` 一半时间是断扭矩自由落体）；
6. ⚠️HD `_tip-mass` 挂载前**先断电源**；HD-1910 最大 8.4V，**严禁 12V 电源**；
7. ⚠️HD 实验期间**不要**手动切模式（保持 33=4），否则与控制律假设不符；
8. ⚠️HD 若发现"转了一整圈以上"（长路径）→ 立即停，检查 q_zero 是否靠近 0/4095。

---

## 5. HD 特有常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 采集时舵机不动 | 写序还是 HLS 原序（见 ⚠️HD 0-1） | 改成 A/B 写序后重测 |
| 位置跳变 > 180° | mode 4 误差不回绕，q_zero 在 0/4095 附近 | 重新装摆臂使垂线避开回绕区，或脚本加环绕感知 |
| 拟合用 `--actuator hls2909` 却想评估 HD | 名字填错 | 改 `hd1910` |
| 电压报 5.1 V | 电源档不到位 | 调到 6.0 V（实验建议值） |
| Kd 读出来是 40 | HD 出厂增益就是 40（不是 HLS 的 32） | 正常；模型默认已同步 |
| 空转正常但跑不过 A/B 写序 | 没按 ⚠️HD 0-1 改代码 | 先适配，再体检到 PASS |

---

> 关联文档：`hls2909_pendulum_quickstart.md`（HLS 版，串口/命令格式完全一样）、
> `hd1910_servo_notes.md`（寄存器/实测/写序结论，三仓库副本）、
> `hd1910_servo_check_quickstart.md`（体检工具速查）、
> `microduck_rl/docs/pendulum_bench_runbook.html`（runbook：参数唯一表/安全）。
