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
| 3 | 目标写入做**零位守卫**：q_zero 落在安全带（=按所选轨迹实测峰值 + 64 LSB 余量，脚本自动算出；四轨迹限幅 80° 后 = **[975,3120]**）之外直接拒绝运行；脚本先尝试 reg31 位置偏移自动居中——**2026-09-10 实测更正：mode 4 控制环确实使用 reg31（原"不使用"结论错误）**，但**仍不能用它居中**：它只改报数，编码器原始读数与目标钳位 [0,4095] 不变，会把守卫骗过去而实际走不完轨迹（详见 `hd1910_servo_notes.md` §6.3）；失败后脚本打印本机**物理对准建议**（转舵机本体 α° 或拨舵盘 k 齿，按当前 q_zero 动态计算） | mode 4 **有效目标被钳位 [0,4095] 且误差不回绕**：q_zero 靠近 0/4095 时小目标会冲向限位（实测 4094→338 走 -3755 LSB 长路径；写 6349 停在 4094） |
| 4 | 采集命令 `--id 1`（已通过 `setup_bench_servo.py --set-id 1` 改好） | 台架舵机 ID 已设为 1 |
| 5 | ⚠️ **电流中止改为"HD 默认关闭" + 新增卡死检测（2026-09-10 修订）**：原 `\|I\|>1.0A×0.7s` 在 HD 上是**误报源**——mode 4 起步瞬态恒读 ≈2.1A（317 LSB）、单条峰值可达 4.06A，均超 5.1V/3.75Ω≈1.36A 的物理堵转上限，实测**一条轨迹都没跑完就误中止**（2.03A）。现脚本按型号自动选阈值：**HD-1910 → `--current-abort 0`（关闭）**，HLS 仍 1.0A；同时新增**位置误差卡死检测**（`--jam-err-rad 0.8` × `--jam-hold-s 1.0`：扭矩开启期间 `\|q−q_goal\|` 持续超限即断扭）。安全链 = 温度(reg63)>50°C + 30s 看门狗 + 卡死检测 + 3A 物理保险丝 + 人在场 | `hd1910_servo_notes.md` §6.5 T5 / §6.6⑤；2026-09-10 实测 2.03A 误中止，探针实测 reg69 运动中甚至出现负值（−3.0A） |
| 6 | ⚠️ **串口短帧崩溃已修**：`read_hls_registers.read_frame` 遇 `LEN=0` 噪声帧会放行、随后在 `body[0]` 抛 `IndexError`，**一条坏帧杀掉整档采集**（2026-09-10 第 6 条记录处实测崩掉） | 现 `LEN<2` 按超时丢弃，采集可连续跑完 12 条 |
| 7 | ⚠️ **每条记录改为"主动稳位"起跑（逐拍重写目标）**：原"目标回零 + 断扭 + 睡 0.5s"会让摆杆带余摆进入下一条，而日志首拍速度写 0 → 仿真以"静止"起跑与真机不符（实测 `pos@t0` 达 −0.635 rad、首秒误差 0.654 rad ≈ PASS 预算 0.157 rad 的 12%）。现记录间隙保持上扭、**逐拍重写目标**回 q_zero，连续满足 `\|q\|<0.02 rad` 且 `\|dq\|<0.25 rad/s` 达 0.25s 才起跑（实测 0.27–0.6s 收敛，残余 ≤0.006 rad）。⚠️ 目标**必须逐拍重写**：只写一次的"单发目标"实测不生效（摆杆 6s 停在旧目标处） | 2026-09-10 修复后 12 条 `pos@t0` = +0.003…+0.009 rad、首秒误差 ≤0.099 rad、rep 间位置范围重合 |
| 8 | ⚠️ **摆角上限 + 防撞网（2026-09-10 实测：本台架摆动平面两侧都有硬障碍）**：**正侧 +96.7°**、**负侧 −88.4°**（负侧被顶住时占空比打到 **+1.00 = 100% 饱和**）。默认 `--amp-max-deg 80`：峰值超 80° 的轨迹**按比例整体缩放**（保波形、不削顶）；`--hard-limit-deg 86`：运行中任一采样 `\|q\|>86°` 立即断扭。缩放只改幅值、不改频率成分，**拟合按日志 `goal_position` 回放 → 不影响辨识口径**。⚠️ 首轮 100 g 的 12 条因 `sin_sin`(±100.8°) 两侧撞停、`lift_and_drop`(−90°) 距负侧障碍仅 2.4° 而**作废重录**，旧数据存 `hd1910_calibration/bench_collision_uncapped/` | 首轮实测：3 条 rep 在 +96.59/+96.68/+96.77° 被顶住 6–9 拍且指令继续到 +100.8°；负侧 −87.0～−88.5° 被顶住 19–21 拍、占空比 +1.00 |


上电后先用 `tools/check_servo_bench.py --port /dev/ttyACM0 --id 1 --yes` 复检到 **PASS**（含空转），
再进第 2 节正式采集。

> 台架：HD-1910 与 HL-2909 同尺寸（34×20×23 mm）、同 25T 舵盘、双轴，现有夹具可直接用；
> 外壳是 PA66+GF43% 塑料，C 夹/压板力度别过大。回差 ≤0.5°（比 HLS 的 ≤1° 更好，有利拟合）。

---

## 1. 称量参数表（只做一次，和 HLS 完全一样，填下表）

| 参数 | 含义 | 本次实测值（15cm 杆，已验证） |
|---|---|---|
| `m_arm` | 纯杆质量（不含固定螺栓） | **0.0085 / 0.0100 / 0.0121 kg**（对应 10 / 15 / 20 cm 杆；均 = 整套 − 2.4g 螺栓） |
| `m_hub` | 近轴转件：2 舵盘 + 8 螺丝（无杆夹/无夹杆螺丝，螺栓已计入 tip） | **0.0021 kg** |
| `r_hub` | 轴心 → m_hub 质心 | **0.005 m**（估，hub 仅 2.1g 影响可忽略） |
| `L` | 轴心 → 砝码片中心（卡尺实测；现取值 0.150） | **0.150 m** |
| `m_tip` 50 g | 砝码片 + 固定螺栓(2.4g) **整套** | **0.0524 kg** |
| `m_tip` 100 g | 同上（当前档） | **0.1024 kg** |
| `m_tip` 150 g | 同上 | **0.1524 kg** |

⚠️ 红线：`m_tip × g × L ≤ 0.35 N·m`。150g 档完整静态重力矩 ≈ 0.24 N·m → 安全，
且 HD-1910 额定(3 kg·cm=0.294 N·m)以上才算重载——摆锤全程额定以内。

> **杆长 ↔ 参数对应**：`--arm-length` 0.10 / 0.15 / 0.20 与纯杆 0.0085 / 0.0100 / 0.0121 kg 一一对应；
> 20cm 杆 + 150g 档的时刻矩 ≈0.31 N·m（< 0.35 红线，但余量小——实测 L 别超过 0.21 m）。
>
> 机械说明（已与现场核对）：杆末端用**螺栓（2.4g）直接固定砝码片**，无托盘、无夹杆块；
> 螺栓在杆末端 → 计入 m_tip（与砝码整套称）；2.1g = 2 个舵盘 + 8 颗舵盘螺丝 = m_hub；
> 其余命令参数均为脚本自动处理（临时限流/零位/符号检测/安全链），无需填写。

### ⚠️HD 模型初值（已写入 `bam/params/hd1910/m1.json`，与真机一致）

```text
kt = 0.736 N·m/A（规格书 7.5 kg·cm/A）      R ≈ 3.75 Ω（6V/1.6A 堵转）
kp = 32 / kd = 40 / ki = 0（真机回读）       max_current = 3.25 A（reg28/44=500）
max_velocity = 8.03 rad/s @5.0V（= @6V 规格 92RPM 的 9.63 按电压折算；限幅默认关，仅备用）
max_acceleration = 500.0（仅备用）
供电 **5.0 V**（2026-09-10 起台架与整机统一）；模型 vin=5.0；编码器 12-bit 0.088°/LSB；减速比 320:1
```

---

## 2. 正式实验命令（HD 版：三档各一条，只改 `--tip-mass`）

> 命令里的 `0.018 / 0.15 / 0.020 / 0.010` 换成第 1 节实测值；
> **`--tip-mass` 必须填你称的整套质量**；输出目录换成 `hd1910_calibration/bench`（与 HLS 分开）。

### 2.1 台架就位（同 HLS）

- [x] 装摆臂（先不装砝码）；C 夹 ×2 锁死；轴水平；手动摆 **±101°** 无碰撞（默认轨迹集峰值 sin_sin ±100.8°）；摆动平面清空
- [x] ⚠️HD **电源 = 5.0 V 稳压**（统一 5 V；4–8.4V 档内，堵转力矩 0.98 N·m、空载 8.03 rad/s，对台架最大重力矩 0.239 N·m 仍有 **4.1× 余量**）。⚠️ 必须是**独立电源**，不得用电脑 USB 口 5 V 给舵机供电（掉线重枚举的根因）
- [x] ⚠️HD 供电电压会在每条 log 记录（reg62 中位数），拟合自动使用实测 `vin`，无需改模型
- [x] ⚠️HD 舵机总线 ID=**1**（已改）；确认只有这一只在线（`--list-ports` + 扫描）

### 2.2 第 1 条：先试跑（100g 档，已核对参数）

```bash
/usr/bin/python3 scripts/record_pendulum_bench.py --port /dev/ttyACM0 --id 1 \
  --tip-mass 0.1024 --arm-mass 0.0100 --arm-length 0.15 \
  --hub-mass 0.0021 --hub-radius 0.005 \
  --trajectory up_and_down --reps 1 --out hd1910_calibration/pilot
```

✅ 标准：输入 START 后，杆从下垂 0° 匀抬到 +80°（正方向正确）、无碰撞、`OK` 且样本 > 1000。
→ 通过后按 50 → 100 → 150 g 顺序跑全量（每档一条命令，**只改 `--tip-mass` 和 `--out`**）：

**50 g 档：**
```bash
/usr/bin/python3 scripts/record_pendulum_bench.py --port /dev/ttyACM0 --id 1 \
  --tip-mass 0.0524 --arm-mass 0.0100 --arm-length 0.15 \
  --hub-mass 0.0021 --hub-radius 0.005 \
  --trajectory sin_time_square --trajectory sin_sin --trajectory lift_and_drop --trajectory up_and_down \
  --reps 3 --limit-a 0.975 --torque-budget 0.35 --out hd1910_calibration/bench_wegiht50_arm15
```

**100 g 档：** 同 50 g 档命令，改 `--tip-mass 0.1024`、`--out .../bench_weight100g_arm15`

**150 g 档：** 同 50 g 档命令，改 `--tip-mass 0.1524`、`--out .../bench_wegiht150_arm15`

⚠️ **每档必须换独立 `--out` 目录**：文件名只由「轨迹+tip质量+rep」决定，同 `--out` 会静默覆盖；而且档名进目录便于核对实物重量
（2026-09-10 踩过：三档都写同一个 `--out` 时后一档直接覆盖前一档，且 `--tip-mass` 忘了改 → 记录质量与实物不符）。
**重量核对法**：三档的准静态保持占空比（`up_and_down` 在 70–80° 区间、|dq|<0.1）应正比于 `m_eq·g·L_eq`，实测 0.123/0.239/0.381 ↔ 记录质量比 0.53/1.00/1.51 吻合即正确。

脚本要求输入 `START`（大写，确认手已离开摆动平面）。每档 12 条（4 轨迹 × 3 次），约 2–3 分钟（含每条起跑前主动稳位）。
换砝码要点：重称"砝码片+螺栓"整套（2.4g 螺栓不变，砝码片 50/100/150g）→ 手动摆过 **±85°**（限幅后峰值 80°）→ 输入 START。

### 2.5 数据检查

```bash
for d in 50 100g 150; do ls hd1910_calibration/bench_wegiht${d}_arm15/*.json 2>/dev/null | wc -l; done   # 各 12
```

### 2.6 拟合（三步：合并 → 重采样 → 拟合）

```bash
cd /home/joyandai/microduck_rl
# ① 三档合并到一个目录（文件名的 tip 质量不同 → 不会重名；manifest 忽略）
mkdir -p hd1910_calibration/bench_all
for d in bench_wegiht50_arm15 bench_weight100g_arm15 bench_wegiht150_arm15; do
  cp hd1910_calibration/$d/*.json hd1910_calibration/bench_all/
done
rm -f hd1910_calibration/bench_all/manifest.json

# ② 重采样到统一 5ms 网格
/usr/bin/python3 scripts/process_bench_logs.py \
  --in hd1910_calibration/bench_all --out hd1910_calibration/bench_all_processed --dt 0.005

# ③ 拟合（依赖: optuna + cmaes；缺 cmaes 会报 ModuleNotFoundError）
uv pip install cmaes -p .venv/bin/python
for m in m1 m2 m3 m4 m5 m6; do
  ( WANDB_MODE=offline .venv/bin/python scripts/fit_leg_pendulum.py \
      --logdir hd1910_calibration/bench_all_processed \
      --actuator hd1910 --models $m --trials 20000 --out hd1910_calibration/fit_$m \
      > hd1910_calibration/fit_logs/$m.log 2>&1 ) &
done; wait
```

> ⚠️HD 拟合时用 `--actuator hd1910`（bam 已注册）。`bam` 解析到 `microduck_rl/vendor/bam`（与
> `/home/joyandai/bam` 功能等价，仅 `kt` 差 0.0002 与注释差异）。
> 实测速度：**≈0.04 s/trial**（24 条训练 log 回放）→ 20000 trial ≈ 13 分钟/模型，32 核可 6 进程并行。
> 各模型独立 `--out` 是为了避免并行时 `mae_report.md` 互相覆盖；汇总见 2.7。

### 2.7 看结果

```bash
cat hd1910_calibration/fit/mae_report.md
```

判定规则同 HLS：PASS = 独立验证 MAE < 0.157 rad；全过且无分叉 → 选 PASS 中 MAE 最小档，同等选 m1。
落地路径：`bam/params/hd1910/mN.json`。

> ⚠️ **2026-09-10 实测补充：必须用"钉死物理参数"的口径判档**。自由拟合会把 `error_gain/kt/R/max_current`
> 一起优化，结果离真机很远（`error_gain` 2.1–6.5 vs 台架实测 0.163），摩擦项在替控制模型代偿、**参数不可解释**；
> 用 `bam.fit --set "{...}"` 把物理已知量钉死后，验证 MAE 只从 0.022 涨到 **0.028 rad**，且判档**反转**为
> **m1（参数最少）最优**。完整结果、落地参数、复现命令见 **`hd1910_pendulum_results.md`**。

---

## 3. 命令行参数全解（record_pendulum_bench.py 为主）

### 3.1 `record_pendulum_bench.py`（采集，最重要）

```bash
/usr/bin/python3 scripts/record_pendulum_bench.py \
  --port /dev/ttyACM0 --id 1 \
  --tip-mass 0.0524 --tip-mass 0.1024 --tip-mass 0.1524 \
  --arm-mass 0.0100 --arm-length 0.15 --hub-mass 0.0021 --hub-radius 0.005 \
  --trajectory sin_time_square --trajectory sin_sin --trajectory lift_and_drop --trajectory up_and_down \
  --reps 3 --limit-a 0.975 --torque-budget 0.35 --out hd1910_calibration/bench
```

| 参数 | 含义 | 取值/单位 | 怎么填 | 填错了会怎样 |
|---|---|---|---|---|
| `--port` | USB-TTL 串口设备 | `/dev/ttyACM0` 等 | `ls /dev/ttyACM*` 看到哪个填哪个 | 打不开直接报错 |
| `--id` | 舵机总线 ID | 0–253 | ⚠️HD 当前=**1**（已用 setup 脚本改好） | 写错 ID → 采集全程无响应/扫不到 |
| `--tip-mass`（可重复） | 端部总质量，**每档一个**，脚本每档暂停等你换砝码 | 千克（kg），0.1 g 秤**整套**称 | 本次实测：`0.0524 / 0.1024 / 0.1524`（砝码片+固定螺栓 2.4g 整套） | 填太小 → 重力矩档位错、负载相关摩擦（m3–m6）辨识不出来；填超红线 → 脚本拒绝运行 |
| `--arm-mass` | 摆杆质量（只称杆，不含端部托） | kg | 本次实测：`0.0100`（15cm 纯杆） | 影响惯量 M 与重力矩 B_max 的换算，误差会进拟合 |
| `--arm-length` | **L**：轴心 → 端部总质量质心 | 米（m），卡尺 ±1 mm | 本次取值：`0.150`；10/15/20cm 杆对应 `0.10 / 0.15 / 0.20`（装好后实测更准） | 摆长是全部动力学换算的基准，务必实测不是杆长 |
| `--hub-mass` | 舵机臂 + 杆夹 + 近轴螺丝总质量 | kg | 本次实测：`0.0021`（2 舵盘+8 螺丝；无杆夹） | 偏小 → 惯量/重力矩少算（量级小，影响有限） |
| `--hub-radius` | 轴心 → m_hub 质心 | 米（m） | 本次取值：`0.005`（hub 仅 2.1g 影响可忽略） | 同上 |
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
/usr/bin/python3 scripts/setup_bench_servo.py --port /dev/ttyACM0 --find-id 14 --set-id 1   # 已完成：ID 现为 1
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
| `--port` / `--id` | 串口 / 舵机 ID | 当前 `--id 1`（已改好） |
| `--no-move` | 只读体检（不动） | 快速检查用 |
| `--move` | 空转步数（LSB） | 默认 150 ≈ 13.2° |
| `--yes` | 跳过交互确认 | 无人值守用 |
| `--out` | 报告 JSON | 默认 `bench_servo_check.json` |

---

## 4. 安全红线（同 HLS 5 条 + HD 3 条）

1. `m_tip·g·L ≤ 0.35 N·m`，禁止加大砝码；
2. 摆动平面内手勿入/头勿探，操作站侧面；
3. 砝码必须 M6 锁紧螺母 + 螺纹胶（甩出是最大风险）；
4. 异常断电顺序：**先断舵机电源（5 V），再拔 TTL**；
5. 全程人在场（`lift_and_drop` 一半时间是断扭矩自由落体）；
6. ⚠️HD `_tip-mass` 挂载前**先断电源**；HD-1910 最大 8.4V，**严禁 12V 电源**；
7. ⚠️HD 实验期间**不要**手动切模式（保持 33=4），否则与控制律假设不符；
8. ⚠️HD 若发现"转了一整圈以上"（长路径）→ 立即停；正常情况零位守卫已先把安全带外的 q_zero 拒绝运行（脚本会打印物理对准建议：转舵机本体 α° 或拨舵盘 k 齿）。

---

## 5. HD 特有常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 采集时舵机不动 | 写序还是 HLS 原序（见 ⚠️HD 0-1） | 改成 A/B 写序后重测 |
| 位置跳变 > 180° | mode 4 误差不回绕，q_zero 在 0/4095 附近 | 超出安全带 [1211,2884] 时脚本直接拒绝并打印**物理对准建议**（转舵机本体 α°/拨舵盘 k 齿，动态计算）；reg31 居中已实测无效（mode 4 控制环不用它） |
| 拟合用 `--actuator hls2909` 却想评估 HD | 名字填错 | 改 `hd1910` |
| 电压报 5.1 V | 正常（统一 5 V 档） | 无需调整；若掉到 4.x V 说明电源带载能力不足或接线压降大 |
| Kd 读出来是 40 | HD 出厂增益就是 40（不是 HLS 的 32） | 正常；模型默认已同步 |
| 空转正常但跑不过 A/B 写序 | 没按 ⚠️HD 0-1 改代码 | 先适配，再体检到 PASS |

---

> 关联文档：`hls2909_pendulum_quickstart.md`（HLS 版，串口/命令格式完全一样）、
> `hd1910_servo_notes.md`（寄存器/实测/写序结论，三仓库副本）、
> `hd1910_servo_check_quickstart.md`（体检工具速查）、
> `microduck_rl/docs/pendulum_bench_runbook.html`（runbook：参数唯一表/安全）。
