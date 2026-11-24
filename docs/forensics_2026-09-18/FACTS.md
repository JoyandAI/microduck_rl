# 事实底本（2026-09-17 / 09-18 两天）—— agent 分析的唯一共享输入

> **本文件只写"实测到了什么 + 在哪个文件里"，不写推测。** 任何结论都必须能引用到下面的数字或原始文件。
> 分析目标（用户 2026-09-18 定调）：
> **① 直走时的偏航"缩小一点"即可，不要求为零；② 尽量对齐官方；③ 左手/右手信号要能转弯。**

---

## 0. 判据（先定死，防止事后找解释）

> ⚠️ **2026-09-18 更正（agent C 指出，已核实）**：本节早先版本把**我们的均值**与**官方的中位数**混着比
> （"净偏航 +8.5° vs +2.8°"、"路径效率 0.373 vs 0.43"）—— 这正是 AGENTS.md 明令禁止的"一个指标两个定义"。
> 下表全部改用**同一口径**（中位 + 占比），逐格可从 `logs/axis/j_*.json` 与 `logs/queue_align/*_walk.json` 复算。

**同口径对照（全部 P64，vx=0.2，seed 0，20 s 窗口存活率 1.00）：**

| 策略 | 偏航 DC（跨 env 均值） | **20 s 净偏航中位** | **frac(\|net\|<30°)** | 路径效率中位 | 沿向速度 | 判读 |
|---|---:|---:|---:|---:|---:|---|
| **官方（参照）** | −0.19 | **+2.8°** | **0.72** | 0.430 | 0.038 | 目标 |
| align_official 终件 | −21.11 | +430° | 0.00 | 0.080 | −0.005 | 不走 + 乱转 |
| align_full 终件 | −11.37 | +206° | 0.11 | 0.160 | 0.022 | 差 |
| align_fwd 终件 | **+0.40** | **+78°** | **0.28** | 0.463 | 0.095 | DC 好看但**每条腿仍在漂** |
| ck7000（旧交付件） | −0.70 | +118° | 0.08 | 0.340 | 0.091 | 同上 |

> **关键更正**：**"偏航 DC ≈ 0"不等于"走直"。** 偏航冲量的符号在 env 之间是**自发对称破缺**（同一次训练 12 个走路
> 检查点六正六负），所以跨 env 平均会互相抵消 —— 平均值好看，**单条腿 20 s 仍漂 78–118°，只有 8–28% 的 env 能留在 30° 以内**。
> ⇒ 真正的目标是缩小**每步偏航冲量的摆幅（AC）**，而不是把均值压到 0。官方单支撑相 AC 只有 **−1.04 deg/s**，我们是 **−31.3**。

**判据（修订版，四项必须同报，缺一项就会把坏策略判成好）**

| # | 指标 | 官方参照 | 目标 | 为什么必须报 |
|---|---|---:|---:|---|
| ① | 偏航 DC（均值） | −0.19 deg/s | 参考 | 单看它会被"env 间正负抵消"骗 |
| ② | **20 s 净偏航中位** | **+2.8°** | **越小越好** | 单腿真实漂移 |
| ③ | **frac(\|net\|<30°)** | **0.72** | **↑** | 分布而非中心 |
| ④ | 沿向速度（门：≥0.08 m/s） | 0.038 | ≥0.08 | 防"不走就不漂"的别名（`ck00000` 未训练 DC +0.02 是因为速度 0） |
| ⑤ | 零命令离地 | 0.00 mm | < 8 mm | 踏步 |
| ⑥ | 外八 / 躯干高 / hip_yaw 限位占用 | 0.4° / 119.8 mm / ~0 | 见验收表 | 防"焊死一个关节换偏航"（`align_full/fwd` 就是这么干的，at-limit 1.000/0.995） |
| ⑦ | 转向 `track_angular_velocity` | 官方能跟 +18.5 deg/s | ≥0.5 | 用户第二诉求 |

**实验规模（agent C 的精确功效计算）**：同配方跨 seed SD = **13.5 deg/s**（signed bias）。
`n/arm = 15.7·(σ/Δ)²`：Δ=20 deg/s → **9 seed**；Δ=10 → 30；Δ=5 → **117**。
`MDD(n=8) = 20.4 deg/s`（path_eff 0.288）。⇒ **判一条臂至少要 8 seed；3 seed 只能看到 ~42 deg/s 的差。**
**三臂现状：都只有 1 个 seed（seed 42 = mjlab 默认）⇒ 臂间不可分。**

协议统一为 **P64**：64 环境 × 20 s × seed 0 × DR 开 × 冻结指令（`--freeze-pose-cmd`）。

> ⚠️ **2026-09-18 第三处更正（agent D 指出）**：本文件早先版本的"官方 0.2 / 我们 +0.40"取的是
> `yaw_rate_bias_deg_s`（**欧拉差分**口径）；**同一批文件里还有 `yaw_dc_body_deg_s`（机体角速度口径），
> 两者差 5 倍**（官方 −0.163 vs −0.188；我们 +0.816 vs +0.395）。而且 `logs/axis/` 里**混了两套协议**：
> `policy_axis_matrix.py:77` **硬编码 `num_envs=1`**，而 `eval_straightness.py` 跑的是 64 环境。
> **⇒ 引用任何偏航数字前先写清：口径（rate_bias / dc_body）+ 协议（1 env / 64 env）+ 是否按路程归一化。**
> 另：**P64 s0 下官方 30/64 个环境在 vx=0.2 下几乎不前进**（made-good 0.038 m/s，5 seed 0.042±0.006）
> ⇒ "官方很直"里有一部分是"官方没在走"；**跨策略比较必须同时报沿向速度，或按每米归一化**
> （agent D：ck7000 **2.9** vs 官方 **11.0 deg/m**）。
AGENTS.md 的两条纪律：**单环境绝对值不可引用**（可差 20–44×、会翻符号）；**比值必须带中位窗口与存活比例一起报**。

---

## 1. 三臂训练结果（09-17 夜 → 09-18 01:16，全部 rc=0、指纹 `7ec8bdc54393e1b1` 未变）

配置（三臂都把所有旧偏差钉在官方值，只差头部三项开关）：

| 臂 | 头部三项 | run 目录 | ONNX |
|---|---|---|---|
| `align_official` | 全关（逐字官方） | `logs/rsl_rl/velocity/align_official` | `logs/queue_align/align_official.onnx` |
| `align_full` | 乱动 + 低头 | `logs/rsl_rl/velocity/align_full` | `logs/queue_align/align_full.onnx` |
| `align_fwd` | 乱动 + 低头 + 一直向前 | `logs/rsl_rl/velocity/align_fwd` | `logs/queue_align/align_fwd.onnx` |

**最终交付检查点，P64 vx=0.2**（原始：`logs/queue_align/align_*_walk.log`，逐检查点表 `logs/queue_align/sweep/straightness.tsv`）：

| 臂 | 偏航 DC | 20 s 净偏航 | 路径效率 | 实测 vx | 存活 | 摔 |
|---|---:|---:|---:|---:|---:|---:|
| align_official | **−21.1 ± 6.1** | −422° | **−0.023** | +0.181 | 64/64 | 0 |
| align_full | −11.4 ± 9.3 | −228° | 0.089 | +0.179 | 64/64 | 0 |
| **align_fwd** | **+0.40 ± 7.9** | **+8.5°** | **0.373** | +0.166 | 64/64 | 0 |

**逐检查点（`straightness.tsv`，|偏航 DC| 升序前 8）**：

```
align_full_ck03000   +0.931  净+99.5   eff 0.411  win 20.00  surv 1.00  falls 0  vx 0.085  tilt 4.31
align_fwd_ck08000    -1.455  净+93.3   eff 0.451  win 20.00  surv 1.00  falls 0  vx 0.081  tilt 4.43
align_full_ck01000   +2.292  净+138.5  eff 0.107  ...
align_full_ck05000   -4.792  净+106.3  eff 0.313
align_fwd_ck06000    -6.637  净+126.2  eff 0.249
...
align_official_ck01000  -137.132 净+2761.9 eff 0.035
```

**⇒ 同一臂内部逐检查点跨度极大**（官方臂 −13 → −137 deg/s），且**每臂只有 1 个 seed**（命令未传 `--agent.seed`）。
AGENTS.md 的实测：要分开这种量级的臂需要 **~52 个 seed/臂**。**⇒ 臂间排序目前不可分离**，"align_fwd 最好"只能说与噪声带外的官方臂差距（+0.4 vs −21）可信。

## 2. 头部三项的效果（`logs/queue_align/align_fwd_acceptance.log`，17 项 8 通过）

| 行 | 值 | 门槛 | 官方 | 修复前 | 判定 |
|---|---:|---:|---:|---:|---|
| 头-躯干点头 std（走） | **3.06°** | 6.0 | 1.7 | 21.4 | **PASS** |
| 头-躯干点头 p2p（走） | **12.5°** | 20.0 | 6.9 | 68.9 | **PASS** |
| 颈部步频幅度 | **0.91°** | 3.0 | 0.75 | 16.4 | **PASS** |
| 头世界俯仰 std（走） | **3.56°** | 6.0 | n/a | 11.5 | **PASS** |
| 零命令 head_roll（站立） | 4.24° | — | 2.0 | **6.8** | 改善 |
| 零命令 head_yaw（站立） | 3.65° | — | −3.07 | −0.006 | — |

**失败的 9 行里 8 行是同一个病：hip_yaw 站姿**
`hip_yaw dev L 25.8° / R 12.0°`（门槛 8，官方 4.6/5.5）、**`hip_yaw 顶限位 99.5%`**（门槛 5%，官方 ~0）、`abduction 18.7°`（门槛 3，官方 0.4）、`raw|a| 2.67`（门槛 1.5）、站立 `raw max|a| 8.53`、`stance twist hip_yaw L 25.7°`、`stance trunk z 86 mm`（门槛 100）。
（第 9 行 `falls in 20 s` 是**判据口径 bug**：它数的是 `resets`，而 20 s 窗口 = episode 长度 ⇒ 每条 rollout 必然收 1 次超时重置。AGENTS.md 有明确记录。）

## 3. 零命令静止（原地踏步）—— `logs/queue_align/sweep/*_stand_forensics.json`

metric 键：`gait.foot_clearance_body_z_m`（**米**）、`trunk.z_mean/tilt_mean_deg`、`joints_deg[<joint>].mean_dev_from_home`。
**注意 `gait_forensics.py` 硬编码 `num_envs = 1`**（脚本第 211 行）⇒ 绝对值按 AGENTS.md 不可直接引用，只有 25× 量级的分离可当信号。

| 检查点 | 离地 mm | 躯干 mm | tilt | 步频 Hz |
|---|---:|---:|---:|---:|
| align_official_ck8999（终件） | **0.7** | 83.2 | 1.87 | 1.05 |
| align_official_ck8000 | 17.3 | 82.3 | 2.12 | 3.20 |
| align_full_ck8999 | 20.5 | 102.4 | 2.22 | 2.75 |
| align_fwd_ck8000 | 18.1 | 95.6 | 4.36 | 2.75 |
| （旧交付件 ck7000） | 0.23 | — | — | — |
| （踏步缺陷参照） | 19.4 | — | — | 3.58 |

## 4. 转弯（P1，从未解决）

- `track_angular_velocity`（权重 2.0，参数与官方逐字相同）长期只拿 **0.119/1.0** ⇒ 策略白扔 **~23% 总奖励**也不转。
- ±34 deg/s 偏航指令 ⇒ 实测偏航率变化 **<2%**，**原地时连符号都不对**；`obs_cmd_wz` 证明指令确实送达。
- 官方策略同指令跟到 **+18.5 deg/s** ⇒ 是**我们的解**的问题，不是任务不可行、也不是指令没进去。
- **2026-09-18 第四处更正（agent A）**：本节早先引用的 **"策略白扔 23% 总奖励也不转"口径错**。
  该算式 `(1−0.119)×2/7.55` 假设 raw 上限为 1.0，但 `track_angular_velocity` 实际是
  `exp(−[(cmd_z−ω_z)² + ω_x² + ω_y²]/std²)`（`mjlab/tasks/velocity/mdp/rewards.py:61-64`）——
  **未指令的 roll/pitch 角速度把上限压到 0.27–0.30**，所以实测"可让渡"的部分只有 **4.3–5.8%**。
  **同时这也意味着：单看偏航，这条项的定价被 ω_xy 稀释了约 4.2 倍。**
- **可观测性（agent A，结构性结论）**：actor 61D **无航向、无机体线速度**；**critic 76D 也没有航向**；
  actor 是纯前馈 MLP（无记忆）⇒ **任何"航向角"奖励在策略能区分的状态上都不存在可利用方向**
  （±120° 观测相同、梯度反号），优势函数也表达不了。**策略唯一能看到的偏航量是 `base_ang_vel[:,2]`。**
  ⇒ 这解释了为什么 `heading_hold` 反复被启用却从未起作用（agent E 实测 rho=+0.13，方向还反）。
- **偏航定价定量（agent A，在实测工作点）**：唯一给偏航定价的还是 `track_angular_velocity`，
  对持续偏置 b 是**二次型 −0.543·b²/步**；实测每-env 偏置 |b|=0.1001 rad/s ⇒ 只收 **−0.0054/步
  = 总奖励 6.749 的 0.080%**。其余项 ∂/∂ω_z 全 ≤3e-4（`body_ang_vel` 构造上排除 z；`pose` 对 hip_yaw
  只有 1.2e-4/rad ⇒ **左 hip_yaw 顶限位是免费的**）。且 |ω_z|(0.873) 是偏置的 8.7× ⇒ **梯度逐点翻符号、
  只有 15.2% 的样本存在可用梯度**（线速度通道 67.4%），按 std 归一后弱 **6.1 倍**。
- **2026-09-18 第五处更正（agent F，P64 三档配对，冻结 `src/` 后重跑 22 次）**：本节早先的
  "**±34 deg/s 指令只换来 <2% 实际偏航变化、原地时符号都错**"**混淆了两条策略**。精确数字
  （`wz = ±1.0 rad/s = ±57.3 deg/s`，`ODD=(v(+1)−v(−1))/2`）：

  | 策略 | 走路 vx=0.2 的 ODD | 增益 | 原地 vx=0 的 ODD | 增益 | 同向 env |
  |---|---:|---:|---:|---:|---:|
  | 官方 alpha_walking | **+33.0 deg/s** | **57.6%** | +16.8 | 29.3% | 64/64 |
  | 我们 align_fwd | +6.45 | **11.3%** | +4.59 | 8.0% | 64/64 |
  | 我们 ck7000 | −1.09 | **−1.9%** | **−3.62** | **−6.3%** | 5/64 |
  | 我们 align_official | +3.25 | 5.7% | — | — | 63/64 |

  ⇒ **通道不是"死的"，是"增益只有官方的 1/5"**；"原地符号错"只在 ck7000 上成立（5/64 同向）。
  F 的机制解释：`track_angular_velocity` 把**步态自己甩出的 ω_z 摆动（我们 AC rms 80 deg/s，
  官方 20）和 ω_xy（我们 56–63，官方 27–40）**放进同一个高斯 ⇒ 学会转向给 PPO 的总收益只有
  **0.13–0.25/步**（官方 0.79，差 3.2–6 倍），且**天花板与指令大小无关**（只付"转或不转"）。
  ⇒ **权重是公共比例因子，改权重不改变信噪比；能改的只有盆地** ⇒ 处方 = 转向优先课程。
- **2026-09-18 新增（agent F + 我复核）**：**头部三项与 `hip_yaw` 被焊死共变** ——
  验收表 `hip_yaw parked at limit (walk)`：无头部项 **46%**（align_official）→ 有头部项 **99.5%**
  （align_full/align_fwd），官方参照 ~0%。**焊死的关节转向增益恰好为 0** ⇒ 转向臂**不应**带头部三项。
- **测量可信度（agent F）**：同一 ONNX、同 seed、同 64 个 DR 抽样跑 3 次，64 环境平均偏航率 DC
  = +0.479/+0.080/+0.276 deg/s（SEM **0.16**）；单环境跨次 SD 中位 **1.16 deg/s**。
  **连 `eval_straightness.py` 自己重跑都与它存下的 json 差 0.30 deg/s** ⇒
  **仓库里所有 ±0.3 deg/s 级别的"改善/恶化"结论都必须用同抽样配对差值复核。**
- 机制（`mdp.py:6198-6207` 自己的原话）：*"the turn channel is not under-priced, it is **UNREACHABLE from the gait basin** the policy settled in: a forward shuffle whose foot placement is decoupled from the yaw command."*
- 已实现的处方、**从未被证伪、只被中止**：`MICRODUCK_TURN_FIRST`（前 1200 迭代只发转向指令，前进量保留 ±0.05 保神经元）、`MICRODUCK_YAW_DC`（净偏航率定价，门控在直线指令）。
- 副作用线索：ck7000 把 **`head_yaw` 当备用转向通道**（head_yaw 指令 −1.40 → 身体偏航 **−22.93 deg/s**，+0.60 → −0.25 平衡点，+1.40 → −4.95；步速不变 0.255–0.268）。

## 5. 配方（现在的默认 = 官方骨架 + 头部三项）

- **19 项奖励** = 官方 16 + 头部 3。旧偏差（姿态卫生 7 项 / per-joint posture / air-time 进度门 / heading_hold / hip_roll std 0.12 / 姿态零命令桶）**默认全部关闭**，各自一个环境变量可恢复。
- 头部三项：`head_motion` −0.08（颈关节速度 deadband **2.0 rad/s**）、`head_droop` +2.0（单侧、相对命令 pitch L1，db **0.15**）、`head_forward` +2.0（yaw/roll 双侧、相对 HOME、按命令门控，db yaw **0.12** / roll **0.10**）。三项都走官方 `head_pose_bias` 的时间线（600 前 0，1500 满权）。
- **死区标定（实测，`scripts/compare_walk_angles.py`，1 env × seed 0 × 12 s）**：官方走路 head_yaw −0.014±0.045 / head_roll −0.024±0.032；站立 head_yaw +0.018±0.023 / head_roll −0.035±0.004。我们 ck7000 站立 head_roll **+0.119**。⇒ 三个死区都 ≥ 官方峰值（0.104 / 0.088 / 0.177 rad/s）⇒ **官方步态躲不掉的摆动免费**。
- BAM/舵机层（**刻意不动**）：HD-1910-C001、`kp_fw=32`、7.5 V（DR 7.1–7.9）、m5 参数、指令延迟 4–7 子步（20–35 ms）。

## 6. 硬件/模型事实（用于排除法）

- 质量链：官方 XL330 版 **737.24 g** → HLS-2909（+4.5 g/舵机）**800.24 g** → HD-1910（**−1.5 g/舵机**）**779.24 g**；净 **+3 g/舵机 = +42 g**。口径出处：`microduck_sitstand_env_cfg.py:356`（"XL330 18g → HD-1910 **21g** = +3g"）。
- **`AGENTS.md` 写的 874 g 是错的**：874 = 737 + 14 × 9.8 g，而 9.8 g 是 `scripts/update_servo_mass.py` 的**默认参数** `--delta-m 0.0098`，从未被实际使用。**用户 2026-09-18 确认"真机和仿真差不多"⇒ 质量线关闭。**
- 头部链 288.9 g = 整机 **37.1%**，力臂 75.6 mm；HOME 整机质心 x=+0.4 mm，脚跟 x=−20.5 mm ⇒ 向后裕度 20.9 mm。
- 模型精确可镜像（质量差 0.000000 g、FK 镜像误差 0.0000 mm）⇒ 左右不对称是**策略属性**，不是模型缺陷。

## 7. 已实测证伪的处方（**不要再提**）

| 处方 | 实测结果 |
|---|---|
| `hip_roll` pose std 收紧到 0.05（作为治外八手段） | 外八 0.454 → **0.583（更糟）** |
| 任何形式的**单关节** `hip_roll` L1（db 0.10 / 0.03、权重 −0.8 / −3.0） | 外八降到 0.16–0.22，但**零命令存活 100% → 0%、tilt 47°**（从膝 ±0.43/踝 ±0.58 逃走） |
| sitstand `head_pose_l1`（`MICRODUCK_HEAD_L1=1`） | 脖子 −1.934 没动、**坐下 0/8** |
| sitstand `head_ground` + `neck_limit_proximity` | 承重 3.4 N → 0，脖子不动，坐下 20/32（对照 24/24） |
| 收紧 `head_pose_tracking` 的 std（fine_std=0.1） | 300 迭代内步态崩（air_time 1.01→0.02） |
| 提高 `kp_fw`（32→64/100/200）修走歪 | 偏航 −2.15 → −2.86 → **+2.29** → +5.05（符号乱跳）、速度 −83% |
| 镜像损失当可靠杠杆 | ck2000–5000 偏航效果**符号翻转** |
| 加权重压偏航（heading_hold 权重 2.5 / yaw 率惩罚） | 高斯在 240° 误差处梯度为 0；`e6b4003` 记录 yaw-rate 惩罚冻结偏航、漂移更糟 |
| 运行时用 wz 偏置抵消漂移 | wz 0 → +0.2115 rad/s 只把偏航从 −12.25 推到 −12.57 deg/s |
| 单环境绝对数字 | 可差 20–44× 且翻符号 |

## 8. 工具清单（分析时可用的现成测量）

| 脚本 | 作用 | 协议 |
|---|---|---|
| `scripts/eval_straightness.py` | 偏航 DC、净偏航、路径效率、存活、首次重置 | **P64**（可配 `--num-envs`） |
| `scripts/gait_forensics.py` | 每关节均值/std、离地、步频、头/躯干、splay、限位占用 | **1 env**（硬编码） |
| `scripts/compare_walk_angles.py` | 14 关节相对 HOME 的 mean/std、L/R hip、头 4 关节 | 1 env |
| `scripts/policy_axis_matrix.py` | 三轴（走直 / 不外八 / 静态稳）一次跑完，可带 `:preswap` | 1 env |
| `scripts/check_posture_acceptance.py` | 17 行 PASS/FAIL（含官方参照列） | 跑两次 forensics |
| `scripts/eval_reward_terms.py` | 在当前工作点逐项算奖励值/梯度 | 可配 |
| `scripts/check_lr_symmetry.py` | 模型镜像审计 | CPU |
| `scripts/q1_straightness_matrix.sh` | 5 seed 矩阵 | P64 |

## 9. 这两天的关键原始产物（分析请直接读这些）

```
logs/queue_align/sweep/straightness.tsv      # 27 个检查点 × P64 走直
logs/queue_align/sweep/marching.tsv          # 零命令（键名有 bug，真值见 *_stand_forensics.json）
logs/queue_align/sweep/*_stand_forensics.json# 零命令逐关节 + 头 + 离地
logs/queue_align/align_*_walk.log|_stand.log # 三臂终件 P64
logs/queue_align/align_*_acceptance.log      # 17 行验收表（含官方列）
logs/axis/j_{official,ck7000,ck12000,R3hrl1,R4db03}_{0.2,0.0}.json  # 三轴矩阵
logs/axis/THREE_AXIS_SUMMARY.txt   ⚠️ **provenance 不可靠，2026-09-18 起停用**：官方行与同日同 seed 同 64env 的 ev_*.log 及逐帧 CSV 重算都对不上（详见 agent_B 报告）
logs/axis/csv_<policy>_<vx>.csv    # eval_straightness.py --csv 的 P64 逐帧落盘（64env×1000步×27列，含 phase/wz/逐关节）⇒ **绝对值可引用**
logs/head_baseline/cmp_vx{0.2,0.0}.log       # 头 4 关节基线（官方 vs ck7000）
logs/*_train.log                             # 各臂训练日志（含 fell_over/error_vel_yaw 曲线）
docs/official_alignment_2026-09-17.md        # 对齐决策与台账
docs/experiment_report_2026-09-17.md         # 三轴对照 + 十条被证伪处方
docs/q1_q2_root_cause_2026-09-16.md          # 走歪/头下垂的完整因果链
```

## 10. 正在跑的训练（09-18 14:57 起，串行）

| 臂 | 配方 | 进度 |
|---|---|---|
| `ours_head` | 我们旧配方 12 处偏差全开 + 头部三项 | 2086/9000 |
| `ours_turn` | = ours_head + `MICRODUCK_TURN_FIRST=1` | 排队 |

（冒烟已逐项验证开关生效：`stance_hip_yaw / stillness_zero_cmd / yaw_hold_zero_cmd / standing_leg_vel / heading_hold / head_motion / head_droop / head_forward` 全部出现。）
