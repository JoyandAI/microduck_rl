# HL-2909 摆锤实验 · 无脑操作手册（照抄版）

> 一句话：**今天**做 §1 空转测试（验证舵机 + 串口链路），**明天**做 §2 正式实验
> （50 g → 100 g → 150 g 三档，每档录入一条命令）→ §3 跑拟合。
>
> 所有命令都在 `microduck_rl` 仓库根目录执行：
> ```bash
> cd /home/joyandai/microduck_rl
> ```
> 文中 `0.018 / 0.15 / 0.020 / 0.010` 只是**示例值**，必须换成你自己实称实量的数字。
> 与 runbook（`docs/pendulum_bench_runbook.html`）冲突时，以 runbook 为准。

---

## 0. 开始之前：把台架参数称好、量好（只做一次，填下表）

| 参数 | 含义 | 怎么量 | 示例 | 你的实测值 |
|---|---|---|---|---|
| `m_arm` | 摆杆质量（**纯杆**，不含固定螺栓；10/15/20cm=8.5/10.0/12.1g） | 电子秤 0.1 g | **0.0085 / 0.0100 / 0.0121 kg** | 用哪根填哪根 |
| `m_hub` | 舵机臂 + 杆夹 + 近轴螺丝的总质量 | 电子秤 0.1 g | 0.020 kg | `____` |
| `r_hub` | 轴心 → m_hub 质心距离 | 卡尺 ±2 mm | 0.010 m | `____` |
| `L` | 轴心 → 端部总质量**质心** | 卡尺 ±1 mm | 0.150 m | `____` |
| `m_tip` 50 g | 托盘 + 砝码 + 杆端夹 + 螺丝**整套**称 | 电子秤 0.1 g | 0.050 kg | `____` |
| `m_tip` 100 g | 同上一整套 | 电子秤 0.1 g | 0.100 kg | `____` |
| `m_tip` 150 g | 同上一整套 | 电子秤 0.1 g | 0.150 kg | `____` |

⚠️ 红线：`m_tip × g × L ≤ 0.35 N·m`。150 g + L=0.15 m 是安全的（约 0.23 N·m），
**不要加大砝码或加长摆臂**。

---

**现场填写速查**（抄命令前 2 分钟搞定，下划线处填值）：

- ✅ 已确定：`--arm-mass 0.0124`（**15 cm 杆**已称；10/20 cm 杆改用 0.0109 / 0.0145）
- ❓ 现场称/量（电子秤 0.1 g + 卡尺 ±1 mm）：
  `--tip-mass` 50g 档整套 `____` kg（100g 档 `____`、150g 档 `____`，**整套含托盘+夹头+螺丝**）
  `--hub-mass` `____` kg；`--hub-radius` `____` m；`--arm-length` = 实测 **L** `____` m
  （15 cm 杆名义 0.150，装好端部堆叠后实测，可能略大于 0.150）

> 其余命令参数均为脚本自动处理（临时限流/零位/符号检测/安全链），无需填写。

## 1. 今天先做：空转测试（10 分钟）

**目的**：不装摆臂，只验证 电脑 → USB-TTL → 舵机 全链路通、舵机真的会转。
装完台架再测就晚了，先测通再说。

### 1.1 接线（对照 runbook 针脚表逐根核对，别凭颜色猜）

```
电源 +12V ── 3A 保险丝 ── 急停开关 ── 舵机电源脚
电源 GND ── USB-TTL GND ── 舵机 GND   （三者共地 !）
USB-TTL DATA（半双工）── 舵机 TTL 信号脚
```
顺序：**先接完所有线，最后才开 12 V**。
⚠️ 这条总线上**只允许接台架这一只舵机**（鸭子的总线拔掉），避免误写别的舵机。

### 1.2 找串口

```bash
ls /dev/ttyACM* /dev/ttyUSB*
```
- 有输出（一般是 `/dev/ttyACM0`）→ 记下名字，下面命令都用它。
- 没有 → USB-TTL 没插好 / 换根线 / `lsusb` 看是否识别。**修好再往下走。**

### 1.3 命令 A：扫描 + 改 ID + 只读体检（不会转动，放心）

```bash
/usr/bin/python3 scripts/setup_bench_servo.py --port /dev/ttyACM0
```

✅ 看到 `发现舵机 ID: [N]` 且只有 1 个 → 直接继续。
✅ 看到多个 ID，或想把台架舵机固定成 `ID = 1`：

```bash
/usr/bin/python3 scripts/setup_bench_servo.py --port /dev/ttyACM0 --find-id <扫描到的编号> --set-id 1
```

**通过标准**（脚本会打印 reg 表）：
- 电压 reg62 = 11.5 ~ 12.6 V（0.1 V 一格）
- 温度 reg63 < 45 °C
- 运行模式 reg33 = 0
- Kp/Kd/Ki = 32 / 32 / 0

### 1.4 命令 B：空转转一下（舵机不带负载，转约 13° 然后回位）

```bash
/usr/bin/python3 scripts/diag_hls_goal.py --port /dev/ttyACM0 --id 1 --single
```

✅ 看到一串 `[ok]`，最后是 `[OK] 跟踪目标`，舵机转了一点又回到原位 → **空转测试通过**。
❌ 看到 `[!!] 不动` → 检查 供电 / ID / 接线 / 急停是否合上，重试；没修好别进正式实验。

### 1.5 测完收尾

- **先断 12 V，再拔 TTL**（顺序不能反）。
- 明天正式实验前**重新上电一次**，让舵机寄存器回到出厂状态（空转测试改过会话寄存器）。

---

## 2. 明天正式实验（约 1.5 小时）

### 2.1 台架就位

- [x] 装上摆臂（**先不装砝码**）
- [x] C 夹 ×2 把台架锁死在桌上，晃动 < 1°
- [x] 输出轴水平（小水平仪）
- [x] 断电手动摆过 ±90°，无碰撞、线束不挂
- [x] 摆动平面清空：操作者站侧面，别站摆动的正前方

### 2.2 第 1 档：装 50 g 整套 → 先试跑 1 条，再跑全量

**试跑 1 条**（`up_and_down`，1 次重复，检查方向/零位/碰撞）：

```bash
/usr/bin/python3 scripts/record_pendulum_bench.py --port /dev/ttyACM0 --id 1 \
  --tip-mass 0.05 --arm-mass 0.0124 --arm-length 0.15 --hub-mass 0.020 --hub-radius 0.010 \
  --trajectory up_and_down --reps 1 --out hls2909_calibration/pilot
```

✅ 标准：输入 START 后，杆从下垂 0° 匀速抬到 +90°（正方向正确）、没有碰撞、
输出 `OK` 且样本 > 50 → 试跑通过，接着跑全量：

```bash
/usr/bin/python3 scripts/record_pendulum_bench.py --port /dev/ttyACM0 --id 1 \
  --tip-mass 0.05 --arm-mass 0.0124 --arm-length 0.15 --hub-mass 0.020 --hub-radius 0.010 \
  --trajectory sin_time_square --trajectory sin_sin --trajectory lift_and_drop --trajectory up_and_down \
  --reps 3 --limit-a 0.975 --torque-budget 0.35 --out hls2909_calibration/bench
```

> 命令里的质量/长度换成 §0 表格实测值；**`--tip-mass` 必须填你称的整套质量**。

脚本会打印 `[换砝码] 请安装端部总质量 50.0 g …`，并要求输入：

```
START
```

（大写 START，回车。这是"手已离开摆动平面"的确认，脚本才会上扭矩。）

**这一档要跑 12 条**（4 种轨迹 × 3 次重复），每条约 7 s，共约 12–15 分钟。
✅ 看到 12 个 `[record] ... OK n=... dt=...ms`，且 `dt` 在 6–15 ms、样本 n>1000 → 完成。
每条结束杆会回到下垂零位停 0.5 s，正常现象。

### 2.3 第 2 档：换 100 g 整套 → 上一条命令，只改一个数字

```bash
/usr/bin/python3 scripts/record_pendulum_bench.py --port /dev/ttyACM0 --id 1 \
  --tip-mass 0.10 --arm-mass 0.0124 --arm-length 0.15 --hub-mass 0.020 --hub-radius 0.010 \
  --trajectory sin_time_square --trajectory sin_sin --trajectory lift_and_drop --trajectory up_and_down \
  --reps 3 --limit-a 0.975 --torque-budget 0.35 --out hls2909_calibration/bench
```

**换砝码要点**：重新整套称重 → M6 螺母锁紧 + 螺纹胶 → 手动摆过 ±90° → 输入 START。

### 2.4 第 3 档：换 150 g 整套 → 再改一个数字

```bash
/usr/bin/python3 scripts/record_pendulum_bench.py --port /dev/ttyACM0 --id 1 \
  --tip-mass 0.15 --arm-mass 0.0124 --arm-length 0.15 --hub-mass 0.020 --hub-radius 0.010 \
  --trajectory sin_time_square --trajectory sin_sin --trajectory lift_and_drop --trajectory up_and_down \
  --reps 3 --limit-a 0.975 --torque-budget 0.35 --out hls2909_calibration/bench
```

### 2.5 检查数据

```bash
ls hls2909_calibration/bench/*.json | grep -v manifest | wc -l   # 应 = 36（36 条 log）
```

（`manifest.json` 是每档/每次命令单独写的汇总，拟合时会自动跳过它，不用管；
36 条 log 文件都在就行。）如果不是 36 → 看缺哪个文件，对应那档重跑一遍
（用同一输出目录即可，log 会累积，不重复）。

### 2.6 拟合（可以后台挂着跑，约 1–2 小时）

```bash
# ① 重采样到均匀时间网格（更稳）
/usr/bin/python3 scripts/process_bench_logs.py \
  --in hls2909_calibration/bench --out hls2909_calibration/bench_processed --dt 0.005

# ② 六个模型 M1–M6 各拟合一次 + 独立验证（每组第 3 条留出验证）
.venv/bin/python scripts/fit_leg_pendulum.py \
  --logdir hls2909_calibration/bench_processed \
  --actuator hls2909 --models m1 m2 m3 m4 m5 m6 \
  --trials 20000 --out hls2909_calibration/fit
```

> 想省时间可以先只跑 `--models m1 m6` 看效果，全量六个模型之后再补。
> 注意：采集脚本用 `/usr/bin/python3`（装了 pyserial），拟合脚本用 `.venv/bin/python`
> （装了 optuna，且 bam 指向本仓库新版 `HLS2909Actuator`）——**别混用**。

### 2.7 看结果

```bash
cat hls2909_calibration/fit/mae_report.md
```

- 每行一个模型，`判定` 列 **PASS** = 独立验证 MAE < 0.157 rad。
- 全部 PASS 且训练/验证无明显分叉 → 选 PASS 里 MAE 最小的档；**同等通过选 m1**。
- 落地到 `bam/params/hls2909/mN.json` + vendor 同步的步骤，见
  `hls2909_pendulum_experiment.md` §F（明天做完实验再说，现在不用看）。

---

## 3. 备选：一条命令跑完全部三档（不想分三次输命令就用这个）

脚本每进一档质量会**自动暂停**并等你在终端输入 `START`，剩下的只是换砝码：

```bash
/usr/bin/python3 scripts/record_pendulum_bench.py --port /dev/ttyACM0 --id 1 \
  --tip-mass 0.05 --tip-mass 0.10 --tip-mass 0.15 \
  --arm-mass 0.0124 --arm-length 0.15 --hub-mass 0.020 --hub-radius 0.010 \
  --trajectory sin_time_square --trajectory sin_sin --trajectory lift_and_drop --trajectory up_and_down \
  --reps 3 --limit-a 0.975 --torque-budget 0.35 --out hls2909_calibration/bench
```

即：装好 50 g → START → 换 100 g → START → 换 150 g → START → 36 条一次收工。

---

## 4. 安全红线（别看太多，就这 5 条）

1. `m_tip·g·L ≤ 0.35 N·m`，**禁止加大砝码**；
2. 摆动平面内手勿入、头勿探，操作站侧面；
3. 砝码必须 M6 锁紧螺母 + 螺纹胶（甩出是最大风险）；
4. 异常断电顺序：**先断 12 V，再拔 TTL**；
5. 全程人在场（`lift_and_drop` 一半时间是断扭矩自由落体）。

出错就按急停（直接断 12 V），脚本自带的温度/电流/看门狗安全链也会自动断扭退出。
任何一条数据中途异常（碰撞、异响、松动）→ 那组的 3 次重复作废，修复后整组重录。

---

## 5. 用到的代码速查（出问题时看这里）

| 文件（都在 `microduck_rl/scripts/`） | 干什么 |
|---|---|
| `setup_bench_servo.py` | §1.3：扫描 / 改 ID / 只读体检 |
| `diag_hls_goal.py` | §1.4：空转测试（上电转 13° 回位） |
| `record_pendulum_bench.py` | §2：摆锤采集（含安全链、换档 START、36 条） |
| `process_bench_logs.py` | §2.6：重采样到均匀 5 ms 网格 |
| `fit_leg_pendulum.py` | §2.6：bam.fit 包装 + M1–M6 拟合 + 独立验证 + MAE 报告 |

详细设计见 `docs/identification/hls2909_pendulum_experiment.md`（本仓库内）与
`docs/pendulum_bench_runbook.html`（microduck_rl 内）。
