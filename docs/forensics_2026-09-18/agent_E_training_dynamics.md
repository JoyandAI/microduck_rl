# Agent E — 训练动态与日志归因：哪个训练期可观测量能预测最终走直程度？

> **只读分析**，未改任何源码、未启动任何训练。所有数字来自本仓库日志；解析脚本在 `/tmp/edyn/`
> （`parse_logs.py` / `corr.py` / `within.py` / `decision.py` / `night_pred.py` / `consistency.py` / `night_cmp.py`）。
> 解析器已与 tfevents 逐点对齐验证：9000 个 iter 的 `Train/mean_reward`、`Metrics/twist/error_vel_yaw`、
> `Metrics/peak_height_mean`、`Metrics/angular_momentum_mean`、`Episode_Reward/foot_clearance` 全部一致
> （差异仅为文本日志的打印舍入：mean reward 5e-3、其余 5e-5）⇒ **文本日志 = tfevents，没有额外信息**，
> tag 数 50 与文本日志完全同集（`logs/rsl_rl/velocity/align_fwd/events.out.tfevents.1789655826.joyandai.522767.0`）。

---

## 结论(3 行内)

1. **没有**任何一个训练期日志量能在训练中可靠预测该臂的最终 P64 走直：三个独立 A/B 集（align 三臂、
   night_queue 四臂×3seed、round3 三臂）里，几乎所有候选指标的方向都会**翻符号**
   （见"训练指标 vs 走直 相关表"的 `FLIPS` 列）。**`Metrics/twist/error_vel_yaw` 不是那个量**——它
   终值 ~3.3–4.0 rad/s 而偏航指令上界只有 1.0 rad/s，量级说明它被**步内偏航振荡**主导；在
   night_queue 里终值最低的 D 臂（2.68–2.97）恰恰是走直最差的臂（yaw DC +19.0 deg/s）。
2. 在三个数据集里方向**一致**的那批指标，**全部指向反方向**：`Episode_Reward/track_linear_velocity`
   vs 最终 path_eff = align **−0.85**、round3 **−0.80**、night_queue **−0.50…−0.67**（全为负）；
   `upright`（−0.75 / −0.80 / −0.30…−0.57）、`Mean episode length`（−0.65 / −0.80 / −0.20…−0.54）
   同样为负，而 `fell_over` 为正（+0.65 / +0.80 / +0.10…+0.48）。
   **即"奖励/跟踪奖励越高、越稳越不摔"的那条臂走得越歪**——奖励最优解**就是**那个前向蹭步的绕圈步态
   ⇒ "盯奖励"会把人带反。同权重同参数的 `Episode_Reward/track_linear_velocity` 在 align it8000：
   official **1.137**（走直最差 eff 0.125）vs fwd **0.970**（最好 0.451）。
3. 走直好的检查点**不对应任何课程阶段边界**，而是全部落在 **iter ≥ 3000（课程在 it2000 结束之后）**；
   且 **3 条臂里有 2 条最后 1000 迭代走直变差而奖励仍在涨**（official eff 0.125→0.076、full 0.226→0.128）
   ⇒ **应当训练到 it8000 后扫检查点选件，而不是信终件**。

---

## 训练指标 vs 走直 相关表

### 协议与数据来源

| 数据集 | 臂 | 物理 | 评测协议 | 源 |
|---|---|---|---|---|
| **D1 align** | 3（official / full / fwd） | **当前** `friction_base=0.0067` | **P64** 20 s vx=0.2 seed0 | `logs/queue_align/align_*_train.log`、`logs/queue_align/sweep/straightness.tsv` |
| **D2 night_queue** | 4（A/B/C/D）× 3 seed = **12** | **旧** `friction_base=0.0136` | **N32** 20 s vx=0.2 seed0 | `logs/night_queue/{A,B,C,D}_s4{2,3,4}.{log,json}` |
| **D3 round3** | 3（splay_std05 / both / hrl1） | **当前** `friction_base=0.0067` | **P64** | `logs/r3_train_splay_*.log`、`logs/queue_round3/*_walk02.json` |

物理差异的出处：`grep -m1 "friction_base" logs/night_queue/A_s42.log` → `friction_base=0.0136`；
`.../r3_train_splay_both.log`、`.../align_fwd_train.log` → `friction_base=0.0067`。
与 `AGENTS.md`「2026-09-16 HD-1910 重新辨识」一致。

**D1 的对齐方式**：`straightness.tsv` 的 `align_<arm>_ck<NNNNN>` → 训练日志里同一 iteration 的指标，
用 **W=201 居中滑动均值**（脚本 `/tmp/edyn/corr.py`）。ck8999 对应日志最后一个 iter（8999）。

### 表 1：D1 align 三臂，同一 iteration 跨臂秩相关（决定性最强的一张）

`rho` = 在**同一个迭代** `it`，分别用「该指标」和「该检查点的 P64 走直」给 3 条臂排序，
算这两组排名的 Spearman，再对 it ∈ {1000,2000,…,8000} 取平均；
`perfect` = 8 个迭代里完全同序（\|rho\|=1.0）的个数（`/tmp/edyn/decision.py` A 节）。
**这一列回答的是"此刻能不能给在飞的三条臂排序"，是最贴近决策问题的统计量。**

| 指标 | 目标 `path_eff_med`（高=直） | 目标 `\|yaw_dc\|`（低=直） |
|---|---|---|
| `Episode_Reward/foot_swing_height` | **+0.81**，perfect 5/8 | **−0.87**，perfect 6/8 |
| `Episode_Reward/foot_clearance` | **+0.81**，perfect 5/8 | **−0.87**，perfect 6/8 |
| `Metrics/peak_height_mean` | −0.81，perfect 5/8 | **+0.87**，perfect 6/8 |
| `Metrics/angular_momentum_mean` | −0.81，perfect 5/8 | **+0.87**，perfect 6/8 |
| `Episode_Reward/dof_pos_limits` | −0.81，perfect 5/8 | **+0.87**，perfect 6/8 |
| `Episode_Reward/air_time` | −0.81，perfect 5/8 | **+0.87**，perfect 6/8 |
| `Mean reward` | **−0.81**，perfect 5/8 | +0.75，perfect 4/8 |
| `Metrics/air_time_mean` | −0.75，perfect 4/8 | **+0.81**，perfect 5/8 |
| `Episode_Reward/body_ang_vel` | +0.69，perfect 5/8 | −0.81，perfect 5/8 |
| `Metrics/twist/error_vel_yaw` | −0.69，perfect 5/8（it8000 翻 +0.5） | +0.75，perfect 6/8（**it8000 翻 −0.5**） |
| `Episode_Reward/track_angular_velocity` | +0.56，perfect 5/8（it8000 翻 −0.5） | −0.62，perfect 6/8（**it8000 翻 +0.5**） |
| `Episode_Reward/track_linear_velocity` | **−0.50**，perfect **仅 2/8** | **+0.44**，perfect 仅 3/8 |
| `Episode_Reward/upright` | −0.62，perfect **仅 2/8** | +0.50，perfect 仅 2/8 |

完整逐迭代值可由 `/tmp/edyn/decision.py` 重跑得到（上面每行末尾的 ±0.5 是 n=3 时"两同一异"）。

⇒ 方向读法（用加权后值的符号，注意权重课程）：
`foot_clearance`(−2.0) / `foot_swing_height`(−0.25) / `air_time`(+3.0) / `dof_pos_limits`(−1.0)
全是**加权后**值 ⇒ 「foot 抬升惩罚**越小**（抬脚越接近 0.02 m 目标）、`air_time` 奖励**越低**、
关节限位惩罚**越大**」的那条臂走得越直。

**⚠ 表 1 与表 2 不是同一个统计量，不要混读。** 表 1 = "该迭代的指标 vs 该迭代的 P64"；
表 2 = "该迭代的指标 vs **最终** P64"。所以 `track_linear_velocity` 在表 1 里只有 −0.50/perfect 2/8
（同迭代几乎分不开臂），在表 2 里的 −0.85 是"**拿早期的它去预测最终**"的成绩。

### 表 2：三数据集方向一致性（`/tmp/edyn/consistency.py`）

对 D1 和 D3，各算「训练指标@it → **最终** P64 `path_eff_med`」的跨臂秩相关，对 it 取平均：

| 指标 | D1 align (n=3臂) | D3 round3 (n=3臂) | 一致？ |
|---|---:|---:|---|
| `Episode_Reward/track_linear_velocity` | −0.85 | −0.80 | **SAME SIGN** |
| `Episode_Reward/upright` | −0.75 | −0.80 | **SAME SIGN** |
| `Episode_Termination/fell_over` | +0.65 | +0.80 | **SAME SIGN** |
| `Mean episode length` | −0.65 | −0.80 | **SAME SIGN** |
| `Episode_Reward/self_collisions` | +0.70 | +0.70 | **SAME SIGN** |
| `Episode_Reward/action_rate_l2` | +0.75 | +0.40 | **SAME SIGN** |
| `Episode_Reward/dof_pos_limits` | −0.50 | −0.60 | **SAME SIGN** |
| `Metrics/twist/error_vel_yaw` | −0.40 | −0.55 | **SAME SIGN（但都是"反"的）** |
| `Metrics/twist/error_vel_xy` | +1.00 | −0.30 | **FLIPS** |
| `Metrics/peak_height_mean` | −0.55 | **+0.60** | **FLIPS** |
| `Metrics/air_time_mean` | −0.70 | **+0.60** | **FLIPS** |
| `Metrics/angular_momentum_mean` | −0.50 | **+0.60** | **FLIPS** |
| `Episode_Reward/track_angular_velocity` | +0.30 | −0.75 | **FLIPS** |
| `Episode_Reward/foot_clearance` | +0.50 | −0.30 | **FLIPS** |
| `Mean reward` | −0.65 | +0.05 | **FLIPS** |

**⇒ 表 1 里 SNR 最大的那几个量（`angular_momentum_mean`、`peak_height_mean`、`air_time_mean`）
在 D3 里方向全反。** 这说明它们刻画的是"**头部三项有没有生效**"（official 臂没有头部三项，
步态是慢节奏高抬腿的长 air_time 步；两条有头部三项的臂都变成快节奏低抬腿），
而不是"**走直程度**"的因果量。用它们当"走直预言机"会在换一批配方时给出相反的建议。

### 表 3：D2 night_queue n=12（同一物理、同一协议）——为什么 `error_vel_yaw` 不能当判据

`Spearman(指标@it, 最终 P32 path_eff_med)`，12 条 run 池化（`/tmp/edyn/night_pred.py`）：

| 指标 | it1000 | it1500 | it2000 | it2500 | it2900 |
|---|---:|---:|---:|---:|---:|
| `Metrics/twist/error_vel_yaw` | −0.44 | −0.06 | −0.22 | **+0.22** | **+0.15** |
| `Episode_Reward/track_linear_velocity` | **−0.67** | −0.50 | −0.53 | −0.59 | **−0.62** |
| `Episode_Reward/track_angular_velocity` | −0.62 | −0.41 | −0.48 | −0.59 | −0.62 |
| `Episode_Reward/body_ang_vel` | −0.08 | −0.61 | −0.43 | **−0.73** | −0.54 |
| `Episode_Reward/upright` | −0.57 | −0.30 | −0.33 | −0.49 | −0.58 |
| `Metrics/angular_momentum_mean` | +0.20 | +0.53 | +0.27 | +0.43 | +0.42 |
| `Metrics/peak_height_mean` | −0.07 | −0.14 | −0.35 | −0.11 | −0.05 |

**终值表（last-100-iter 训练均值 vs P32 评测）**：

| run | `error_vel_yaw` | `track_ang` | `Mean reward` | **yaw DC (deg/s)** | eff_med | vx |
|---|---:|---:|---:|---:|---:|---:|
| A_s42 | 3.266 | 0.0605 | 57.52 | −22.806 | 0.473 | 0.218 |
| A_s43 | 3.303 | 0.0585 | 57.53 | +3.669 | 0.646 | 0.212 |
| A_s44 | 3.267 | 0.0663 | 62.34 | +13.897 | 0.574 | 0.216 |
| B_s42 | 3.177 | 0.0365 | 39.17 | −3.395 | 0.709 | 0.151 |
| B_s43 | 3.284 | 0.0399 | 42.71 | +8.765 | 0.681 | 0.197 |
| B_s44 | 3.249 | 0.0375 | 41.00 | +15.049 | 0.684 | 0.154 |
| C_s42 | 3.412 | 0.0528 | 58.28 | +1.467 | 0.674 | 0.141 |
| C_s43 | 3.344 | 0.0530 | 58.09 | +16.477 | 0.499 | 0.146 |
| C_s44 | 3.423 | 0.0509 | 57.96 | −9.706 | 0.726 | 0.140 |
| **D_s42** | **2.972** | 0.0627 | 52.48 | **+22.418** | 0.692 | 0.129 |
| **D_s43** | **2.678** | 0.0800 | 54.75 | +7.072 | 0.525 | 0.129 |
| **D_s44** | **2.958** | 0.0670 | 53.76 | **+27.578** | 0.508 | 0.093 |

**逐字官方臂 D 的 `error_vel_yaw` 最低（2.68–2.97），走直最差（yaw DC 均值 +19.0）；
heading_hold 臂 C 的 `error_vel_yaw` 最高（3.34–3.42），走直却比 D 好。**
`Spearman(error_vel_yaw, |yaw DC|) = −0.32`（n=12，**符号为反**）；
`Spearman(track_angular_velocity, |yaw DC|) = +0.41`（也是反的）。

**机制**：`Metrics/twist/error_vel_yaw` = 指令区间内 `mean|wz_cmd − wz_body|`（rad/s），
定义在 `.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/mdp/velocity_command.py:70-74`。
偏航指令范围 `ang_vel_z = (-1.0, 1.0)`（`src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py:1786`）
⇒ `|wz_cmd| ≤ 1.0`，而实测终值 **3.3–4.0 rad/s**（`logs/queue_align/align_fwd_train.log:522324`
= `3.9521`；`align_official_train.log:468318` = `3.8737`）。**误差是命令上界的 3–4 倍
⇒ 这个数几乎全是步内偏航振荡，DC 漂移只占很小一部分。** 它测的不是我们关心的东西。

### 表 4：同迭代原始值（D1，it8000，窗口 [8000,8200) 的均值±窗内 SD，`/tmp/edyn/decision.py`）

| 指标 | align_official | align_full | align_fwd | SNR(最好−最差) |
|---|---:|---:|---:|---:|
| P64 `path_eff_med`（ck8000） | 0.125 | 0.226 | **0.451** | — |
| `Metrics/angular_momentum_mean` | 0.0112±0.0001 | 0.0065±0.0001 | 0.0067±0.0001 | **43.1** |
| `Metrics/air_time_mean` | 0.1262±0.0010 | 0.1132±0.0012 | 0.1123±0.0009 | 15.3 |
| `Episode_Reward/pose` | 0.0079±0.0007 | 0.0018±0.0002 | 0.0019±0.0001 | 12.7 |
| `Mean reward` | 98.95±1.01 | 83.09±1.30 | 82.91±1.42 | 13.0 |
| `Metrics/peak_height_mean` | 0.0298±0.0007 | 0.0241±0.0004 | 0.0240±0.0003 | 10.5 |
| `Episode_Reward/dof_pos_limits` | −0.0953±0.0056 | −0.1988±0.0048 | −0.1516±0.0057 | 9.9 |
| `Metrics/twist/error_vel_xy` | 0.4544±0.0134 | 0.5731±0.0151 | 0.5802±0.0189 | 7.7 |
| `Episode_Reward/track_linear_velocity` | **1.1374** | 1.0081 | 0.9704 | 5.5 |
| `Metrics/twist/error_vel_yaw` | 3.9839±0.1036 | 3.3949±0.0867 | **4.0833±0.1064** | **0.9 ← 塌** |

**注意最后两行**：it8000 时 `error_vel_yaw` 的 SNR 塌到 0.9（fwd 反而是三者里最高），
而 `track_linear_velocity` 却把 official 和另两条臂分开——**分离方向与"走直"相反**。
`track_linear_velocity` 权重三臂相同（2.0，`grep -m1 weight logs/rsl_rl/velocity/align_*/params/env.yaml`），
所以这个比较是**同权重同参数**的合法比较：**逐字官方臂前向跟踪最好（1.137），走直最差（eff 0.125）。**

---

## 阶段边界与解耦区间

### 课程边界清单（`/tmp/edyn/decision.py` B 节；三臂完全相同的调度，除头部三项）

| step（=env step，iteration×24） | it | 变化 |
|---|---:|---|
| `action_rate_weight` | 500 / 750 / 1000 / 1250 / 1500 | −0.1 → −0.2 → −0.4 → −0.6 → −0.8 → −1.0 |
| `com_range` | 500 / 1000 / 1500 | 0.003 → 0.005 → 0.01 → 0.015 |
| `head_com_range` | 500 / 1000 | 0.003 → 0.005 → 0.01 |
| `head_pose_bias_weight` | **600** / 1000 / 1500 | 0 → 1 → 2 → 3 |
| `head_stability_weight`（仅 full/fwd） | **600** / 1000 / 1500 | 0 → −0.02 → −0.04 → −0.08 |
| `head_droop_weight`（仅 full/fwd） | **600** / 1000 / 1500 | 0 → 0.5 → 1.0 → 2.0 |
| `head_forward_weight`（仅 fwd） | **600** / 1000 / 1500 | 0 → 0.5 → 1.0 → 2.0 |
| `head_pose_range` | 500 / 1000 / 1500 / **2000** | 0.07 → 0.21 → 0.49 → 0.91 → **1.4** |
| `standing_envs` | 500 / 750 / 1000 / 1500 / **2000** | 0.02 → 0.05 → 0.1 → 0.15 → 0.2 → **0.25** |

night_queue 另有 `heading_hold_weight`（A/B/C）：it300 0→0.575、it611 →1.25、it950 →1.925、it1300 →2.5
（`/tmp/edyn/decision.py` 输出；源 `logs/night_queue/C_s42.log`）。A/B 还多一个从 it0 起线性爬的
`stance_tax_weight`（0.0017 → 1.0）。

### 边界处的 step-down：有，但都是"难度上升"，而且**都恢复了**

以 z = 前后各 100 迭代均值差 / 前置窗标准误 度量（z<−2 记为 drop）：

| 边界 | align_official 最大 z<−2 的项 | align_full | align_fwd |
|---|---|---|---|
| it1000 | `Mean reward` **z−30.3**（85.97→79.74） | z−16.5 | z−23.8 |
| it1250 | `peak_height_mean` z−27.8（0.046→0.044） | `angular_momentum_mean` z−25.6 | `angular_momentum_mean` z−25.1 |
| it1500 | `Mean reward` **z−38.9**（93.18→86.13） | `Mean reward` **z−88.2**（89.60→77.87） | `Mean reward` **z−95.4**（92.53→80.02） |
| it2000 | `Mean reward` **z−49.7**（90.11→82.95） | **z−79.5**（82.24→71.11） | **z−65.7**（83.85→72.55） |

**是否触发 AGENTS.md 的"课程节奏错了"？——没有，但要记一笔。** 判据是"掉下去之后回不来"。
实测（W=201 平滑，`/tmp/edyn/corr.py`）：

| it | official | full | fwd |
|---:|---:|---:|---:|
| 1000 | 82.84 | 79.16 | 83.19 |
| 1500 | 89.62 | 83.70 | 86.23 |
| **2000** | 86.50 | **76.62** | **78.15** |
| **2500** | 84.04 | **68.39（谷底）** | **70.42（谷底）** |
| 3000 | 86.52 | 73.59 | 71.49 |
| 4000 | 91.17 | 75.32 | 75.98 |
| 8000 | 98.57 | 83.41 | 82.85 |
| 8999 | **102.85** | **84.24** | 85.42 |

⇒ 两条带头部三项的臂在 **it2000–2500** 掉出一个深谷（−15 / −13），到 it4000 才回到边界前水平，
**恢复用了 ~1500 迭代**。这是**"tax 引入得略早/略陡"的弱证据**（AGENTS.md：「任何 attempt-tax
在硬技能探索期间生效会让'什么都不做'胜出」），但**不是 "wandb 指标在边界处永久下降" 那条禁令**：
终值高于边界前（official 102.85 / full 84.24 / fwd 85.42）。

**更重要的边界事实**：**it2000 之前所有臂的 P64 走直都同样差，臂间不可分**：

| 臂 | eff ck≤2000 均值 (n=2) | eff ck3000–8000 均值 (n=6) | \|yaw\| ck≤2000 | \|yaw\| ck3000–8000 |
|---|---:|---:|---:|---:|
| align_official | 0.035 | 0.085 | 91.31 | 24.95 |
| align_full | 0.104 | **0.255** | 8.92 | 7.06 |
| align_fwd | 0.044 | **0.190** | 44.81 | 12.08 |

（源：`logs/queue_align/sweep/straightness.tsv:2,3,7,10,18,19,27` 等 26 行。）

**走直最好的检查点全部在 it≥3000，且不对应任何边界**：

```
BEST  align_fwd_ck08000    eff 0.451  yawDC  -1.455   speed_along 0.081   (straightness.tsv:25)
BEST  align_full_ck03000   eff 0.411  yawDC  +0.931   speed_along 0.085   (straightness.tsv:7)
BEST  align_full_ck05000   eff 0.313  yawDC  -4.792                      (straightness.tsv:11)
BEST  align_full_ck07000   eff 0.266  yawDC  -7.147
BEST  align_fwd_ck06000    eff 0.249  yawDC  -6.637
WORST align_official_ck01000 eff 0.035 yawDC -137.132                     (straightness.tsv:3)
```

### 解耦区间：**奖励在涨、走直在变差**（`/tmp/edyn/decision.py` C 节）

| 臂 | 区间 | ΔReward | eff | Δ\|yaw\| | 判定 |
|---|---|---:|---|---:|---|
| align_official | ck5000→ck6000 | +1.89 | 0.105→0.074 | +5.50 | 解耦 |
| align_official | **ck8000→ck8999** | **+4.28** | **0.125→0.076** | **+7.35** | **解耦（末段）** |
| align_full | ck3000→ck4000 | +1.73 | 0.411→0.195 | +6.30 | 解耦 |
| align_full | ck5000→ck6000 | +2.28 | 0.313→0.118 | +10.05 | 解耦 |
| align_full | ck7000→ck8000 | +1.31 | 0.266→0.226 | +0.27 | 解耦 |
| align_full | **ck8000→ck8999** | **+0.82** | **0.226→0.128** | **+4.09** | **解耦（末段）** |
| align_fwd | ck6000→ck7000 | +0.28 | 0.249→0.158 | +5.32 | 解耦 |
| align_full | ck2000→ck3000 | **−3.03** | 0.101→**0.411** | −14.62 | **反向：奖励跌、走直变好** |
| align_fwd | ck1000→ck2000 | **−5.04** | 0.032→0.056 | −41.25 | **反向** |
| align_fwd | ck2000→ck3000 | **−6.66** | 0.056→0.061 | −3.69 | **反向** |

**两条硬结论**：
1. **在 3 条臂里有 2 条，最后 1000 迭代让走直变差**（official eff 0.125→0.076、full 0.226→0.128），
   而奖励是**涨**的（+4.28 / +0.82）。⇒ **交付件不该默认取终件，应取 it8000 并扫检查点。**
2. **走直改善最大的那几步，奖励往往是跌的**（full ck2000→3000 奖励 −3.03 而 eff +0.310；
   fwd ck1000→3000 奖励 −11.7 而 \|yaw\| 65.4→20.5）⇒ 奖励与目标在局部**反相关**，
   用"奖励还在涨"当"方向对"的判据会得到相反结论。

---

## night_queue 四臂对比

### 配方确认（读 `<run>/params/env.yaml`；`kp_fw` 见 `params/env.yaml`）

| 臂 | 奖励项数 | 配方 | `kp_fw` | 与用户描述一致？ |
|---|---:|---|---:|---|
| A | 22 | 我们配方（`stance_hip_yaw`/`action_over_limit`/`neck_motion`/`heading_hold`/`stillness_zero_cmd`/`no_stepping` 均在，权重由 curriculum 拉起） | 32 | ✓ |
| B | 22 | = A，`kp_fw=64` | **64** | ✓ |
| C | 19 | 官方 16 项 + `heading_hold`（其余 `stillness_zero_cmd=0`、`no_stepping=0` 为恒 0 槽位） | 32 | ✓ |
| D | 16 | **逐字官方** | 32 | ✓ |

A/B 的 `heading_hold=0.0`、`stance_hip_yaw=0.0` 是**初始权重**，由 curriculum 在 it300/611/950/1300
拉到 2.5（`logs/night_queue/C_s42.log`；A/B 同理）——**读时要按权重课程解释**。

### 结果（N32 × 20 s × vx=0.2 × seed 0 × DR 开；源 `logs/night_queue/*.json`）

| 臂 | yaw DC (deg/s) | net yaw (deg) | eff_med | vx | tilt | falls |
|---|---:|---:|---:|---:|---:|---:|
| A 我们配方 | **−1.747 ± 18.941** | −34.97 ± 378.48 | 0.564 ± 0.086 | 0.216 ± 0.003 | 3.193 ± 0.244 | 0/32 ×3 |
| B = A+kp64 | +6.806 ± 9.377 | +135.47 ± 187.44 | **0.691 ± 0.015** | 0.168 ± 0.026 | 3.306 ± 0.218 | 0/32 ×3 |
| C 官方+heading_hold | +2.746 ± 13.138 | +55.71 ± 262.41 | 0.633 ± 0.119 | 0.143 ± 0.003 | 3.584 ± 0.132 | 0/32 ×3 |
| D 逐字官方 | **+19.023 ± 10.667** | +379.59 ± 212.90 | 0.575 ± 0.102 | 0.117 ± 0.021 | 3.615 ± 0.461 | 0/32 ×3 |

（每臂 n=3 seed，`±` 是**跨 seed 的 SD**；`falls` 三 seed 全为 0。）

**同 seed 配对差**（12 条 run 用的是同一组 seed s42/s43/s44，可以配对）：

| 对比 | yaw DC 配对差 | 符号 | eff_med 配对差 | 符号 |
|---|---:|---:|---:|---:|
| D − A | **+20.770** | **3/3 正**（+45.2, +3.4, +13.7） | +0.011 | 1/3 |
| C − D（heading_hold 的作用） | **−16.277** | 1/3 正 | +0.058 | 1/3 |
| B − A（kp64 的作用） | +8.553 | **3/3 正** | **+0.127** | **3/3 正** |
| C − A | +4.493 | 2/3 | +0.069 | 2/3 |

### 能支持 / 不能支持什么

**能支持的（在"2026-09-14/15 的旧物理"这台机器上、N32 协议下）：**
1. **逐字官方臂 D 的偏航 DC 显著为正，且是四臂里最大的**：三 seed 全为正（+22.4 / +7.1 / +27.6），
   配对 D−A 三 seed 全正，均值 +20.8 deg/s。**这是符号一致的 3/3，但 n=3 的符号检验
   p=0.125（单侧），达不到显著。**
2. **B（+kp64）的 `path_eff_med` 三 seed 全部高于 A**（0.709/0.681/0.684 vs 0.473/0.646/0.574，
   配对差 +0.235/+0.036/+0.110），但代价是 vx 从 0.216 掉到 0.168（−22%）。
3. **`heading_hold`（C vs D）把偏航 DC 均值从 +19.0 拉到 +2.7**，但 3 seed 里有 1 个变差
   （s43 +7.1→+16.5）⇒ 方向对，未达显著。

**不能支持的（重要）：**
1. **不能推断今天的机器。** 12 条 run 全部训练于 `friction_base=0.0136 / friction_viscous=0.0081`
   （`grep -m1 friction_base logs/night_queue/{A,B,C,D}_s4{2,3,4}.log` → 12/12 相同），
   而三臂 align 与三条 r3 臂是 `0.0067 / 0.0055`。**这与 AGENTS.md 记录的
   "`-OfficialRecipe` A/B 训练于 2026-09-16 HD-1910 重新辨识之前 ⇒ 关于另一台机器"是同一条。**
   ⇒ **D 臂 +22.4/+7.1/+27.6 的结论不能用来裁决今天"官方 vs 我们"的问题。**
2. **臂间排序不可分离。** yaw DC 的**臂内 SD 均值 13.03 deg/s**，臂间极差仅 20.77 deg/s；
   `path_eff_med` 臂内 SD 均值 0.080，臂间极差 0.127。这与 AGENTS.md「4 arms × 3 seeds × 3000 iters
   分不开任何臂（path efficiency F=0.19, p=0.90；~52 seeds/arm）」完全一致——**night_queue 就是那批实验。**
3. **协议也不同**：night_queue 是 N32，align/r3 是 P64；`straightness.tsv` 用的是
   `path_efficiency__median`，而 `align_*_walk.log` 打印的是 aggregate 的 **mean**
   （例如 `align_full_ck01000`：tsv 0.107 vs walk.log 0.193）——**报数时必须写清是 mean 还是 median。**

---

## 训练时要盯的 3 个指标与提前停的判据

### 先说明：现在日志里**没有**能直接读出"DC 偏航"的量

`eval_straightness.py` 的 `yaw_dc_deg_s` 是 `mean(wz_body)`（`scripts/eval_straightness.py:554,677`），
**训练日志里没有任何对应的量**。最接近的三个候选都被振荡污染：
- `Metrics/twist/error_vel_yaw`：实测 3.3–4.0 rad/s vs 命令上界 1.0 rad/s（见"表 1/3"）；
- `Episode_Reward/track_angular_velocity`：终值 0.0835/2.0 = **0.042**（`align_fwd_train.log:522294`）
  即满权的 4%——和 FACTS.md:88 的 0.119/1.0 一个量级，
  **官方臂拿最高（0.1021）却走得最歪**；
- `Episode_Reward/heading_hold`（只在 A/B/C 与 r3 臂存在）：我实测它与 P64 的
  `Spearman = +0.13 (n=9, night_queue A/B/C)` / `+0.50 (n=3, r3)`，**方向都对不上**（应为负），
  因为它是 `exp(−wrap(yaw−yaw_spawn)²/std²)` 的逐步均值（`src/mjlab_microduck/tasks/mdp.py:5348-5361`），
  同样被步内摆动拉低。

**⇒ 结论：要"在训练中及早判断一条臂是否走对方向"，必须新增一个 DC 量的日志**
（例如每 episode 的 `|EMA_1s(wz)|` 或门控下的 `heading_hold` 的对数误差），
否则只能靠**定期 P64 抽检**。我在这份报告里能给的，只是"筛选 + 何时抽检"。

### 要盯的 3 个（按可用性排序，全部注明"能做什么/不能做什么"）

**① `Metrics/angular_momentum_mean` — 最强的早期筛子（但只是筛子）**
- it2000–8000 分离度 SNR **43–54**（it2000 44.3 / it3000 54.4 / it5000 42.7 / it8000 43.1），
  align official ≈ **0.011–0.012** vs full/fwd ≈ **0.0065–0.0075**，倍数 ~1.7
  （`/tmp/edyn/decision.py` 表 4；源 `align_official_train.log:468323` = 0.0102）。
- 臂内去均值后与 `path_eff_med` 的 Spearman = **−0.711，三条 align 臂符号全同**。
- **但不能当预言机**：round3 三条臂方向**翻正**（+0.60），night_queue 只有 +0.42。
- **用法**：it2000 时若仍 ≥0.010 ⇒ 该臂落在"官方式绕圈平滑步"盆地，预期 P64 \|yaw DC\| >20 deg/s。

**② `Metrics/air_time_mean` + `Metrics/peak_height_mean`（成对看）— "头部三项有没有咬住"的指纹**
- official：air ~0.126–0.131、peak ~0.030–0.038（慢节奏高抬腿）；
  两条带头部项的臂：air ~0.110–0.113、peak ~0.023–0.024（快节奏低抬腿）。
  分离从 it2000 起稳定（SNR 15–24 / 10–24）。
- **意义是"配方差异生效了"，不是"走直了"**（round3 里方向反转，见"表 2"）。
- **用法**：若新臂 it2000 的 air/peak 还在 official 那一档，说明头部项没起作用——
  先查权重课程，不要急着下"走不直"的结论。

**③ `Metrics/twist/error_vel_yaw` — 只当"步态成熟度"表，绝不当"走直"表**
- 健康形状：it700–900 冲顶到 **6.6–9.3** → 单调下降到 **≤4.0**，it2000 之后基本平台
  （official 7.65→3.75、full 5.06→3.34、fwd 5.87→4.00；`/tmp/edyn/corr.py` 对齐表）。
- **若 it2000 仍 >4.5 ⇒ 步态还没稳**（还在高振荡阶段），此时任何 P64 抽检都是噪声。
- **禁止**用它给臂排序：it8000 时 align 三臂 SNR 塌到 0.9；night_queue 里终值与 \|yaw DC\| 反相关
  （−0.32），最低的 D 臂最歪。

**以及第 4 个"不是指标但是纪律"的：`Mean reward` 在 D1/D2 里与目标反相关（−0.65 / −0.38），
在 D3 里近似为零（+0.05）——它从来不提供正向信息。**
注意 D1 三臂的奖励栈**项数不同**（official **16** 项 vs full **18** 项 vs fwd **19** 项，实测
`logs/rsl_rl/velocity/align_{official,full,fwd}/params/env.yaml` 的 `rewards:` 节），
所以 `Mean reward` **跨臂本来就不可比**；
即使是同权重的 `track_linear_velocity`，也是"越高越歪"（表 4 末行）。

### "第几迭代判死"（`/tmp/edyn/decision.py`、表 1/4 的证据）

| 迭代 | 能说什么 | 判据 |
|---|---|---|
| it0–2000 | **什么都不能说。** 课程还在跑；三臂 P64 eff 全在 0.032–0.107、\|yaw\| 2.3–137 | **不要在这个窗口杀任何臂**，也不要在这个窗口给臂排序 |
| it2000–2500 | 奖励深谷（full 68.4 / fwd 70.4），但 align_full 在 it2000 的 eff=0.101 到 it3000 变 0.411 | **不要杀**——这是"tax 刚满权"的恢复期，历史上 1500 迭代就回来了 |
| **it3000** | **最早的可靠分叉点。** 若 P64 抽检 \|yaw DC\| 仍 **>30 deg/s** 且 eff **<0.06** ⇒ 落在 official 式盆地。 | 参考：official 从 ck3000 的 −45.8 训到 ck8999 的 −20.7，**6000 个迭代只买到 25 deg/s** |
| it5000–8000 | 唯一值得在这区间反复抽检的地方；最好的检查点全部落在这里 | 单检查点不可信：align_fwd ck7000 \|yaw\|=11.96 而 ck8000=1.46 |
| it8000 | **性价比拐点。** 3 臂里 2 臂在 it8000→8999 走直变差而奖励还涨 | 建议 **it8000 停 + 扫 ck {3000,5000,6000,7000,8000}**，而不是取终件 |

### "如果再来一条臂，什么信号出现就提前停掉它省算力"

**信号 A（最早，it2000，零额外评测成本）——三条件合取：**

```
Metrics/angular_momentum_mean ≥ 0.010
且 Metrics/air_time_mean      ≥ 0.125
且 Metrics/peak_height_mean   ≥ 0.033
```

三条都是 align_official 臂在 it2000 的取值（0.0121 / 0.1277 / 0.0379），而两条"能走一点直"的臂
都在（0.0072 / 0.1091 / 0.0255）和（0.0075 / 0.1124 / 0.0269）。
**这是一条"配方没生效 ⇒ 落在官方盆地"的筛选器，不是"走直预言机"**：
它在 round3 那批臂上会给出错误建议（方向反转），所以**只用于"头部/姿态项是否生效"的同类配方**。
命中即停 ⇒ 省下 7000 迭代（≈ 9000×0.98 s ≈ 2.5 GPU-h，按 `align_fwd_train.log:522334-522335` 的
`Iteration time: 0.98s` 与 `Time elapsed: 02:34:56`）。

**信号 B（it3000，一次 P64 抽检 ≈1 GPU-min）——直接测量，唯一有物理意义的判据：**

```
64 env × 20 s × seed 0 × DR 开 × --freeze-pose-cmd, vx=0.2
|yaw_dc_deg_s| > 30  且  path_eff_med < 0.06   ⇒ 停
```

统计依据：26 个 align 检查点里，eff<0.06 的 **6** 个全部 \|yaw\| ≥ **24.2**，eff≥0.19 的 **7** 个全部 \|yaw\| ≤ **7.4**
（`logs/queue_align/sweep/straightness.tsv`）。且 **必须连续两个相隔 1000 的检查点都满足**才停
（align_fwd ck7000 单独看会被误杀）。

**信号 C（贯穿全程）——`Metrics/twist/error_vel_yaw` 在 it2000 后仍 >4.5** ⇒ 步态未成熟，
此时**不要**因为 P64 难看就停（P64 本身在噪声里）；先等 1000 迭代再看。

---

## 不确定项

1. **"没有可靠预言量"这个否定结论的样本量很小。** 三个 A/B 集分别只有 3/12/3 条臂；
   D1/D3 的跨臂秩相关基于 n=3，`rho=±0.87` 只等价于"3 条臂里 2 条同序"。
   表 2 的 `SAME SIGN` / `FLIPS` 是**方向一致性**，不是显著性。**没有任何一处到 p<0.05。**
2. **D2（night_queue）与 D1/D3 不是同一台机器**（`0.0136` vs `0.0067`），
   所以我用 D2 只做"同协议内 12 条 run 的相对排序"，没有把 D2 的绝对值和 D1/D3 混着算。
   **D2 与 D1/D3 之间任何"同一指标方向不一致"都可能来自物理差异，我不能排除。**
3. **`Metrics/peak_height_mean` / `angular_momentum_mean` / `air_time_mean` 的因果地位未知。**
   三臂 align 的差异是"有没有头部三项"这一个开关；它同时改变了抬脚高度、air_time、
   angular momentum 和走直程度。**我无法区分"头部件 → 低抬脚 → 走直"和
   "头部件 → 低抬脚 + 走直（两条独立通路）"。** 表 4 的 SNR 是**共标记**，不是机制。
4. **`heading_hold` 与 DC 偏航的关系没测准**（n=9 时 rho=+0.13，n=3 时 +0.50，方向都不对）。
   我没有把它的**门控**（`vel_gate_ref`）剥开单独算，所以不能说它无用——
   只能说**直接读 `Episode_Reward/heading_hold` 的日志值不能当 DC 判据**。
   它是否是一个好指标需要在有 P64 扫描的臂上重测。**UNCERTAIN。**
5. **`Total steps` / 奖励项数不同的臂之间 `Mean reward` 不可比**，我在报告里凡是用到
   `Mean reward` 跨臂比较的地方都标注了；如果读者要复现，请只用**同权重同参数**的
   `Episode_Reward/<term>`。
6. **it1000–2000 的 `|yaw_dc|` 数值本身可疑。** 例：`align_official_ck01000` 的
   `speed_along_mps = 0.000`（`straightness.tsv:3`）而 `achieved vx = +0.123 m/s`
   （`logs/queue_align/sweep/align_official_ck01000_walk.log`）——
   `speed_along` 是"沿初始航向的净推进/时间"（`scripts/eval_straightness.py:664`），
   对绕圈策略必然接近 0。所以「\||yaw\| 137 deg/s 而速度 0」不代表它站着不动，
   而是**以 0.123 m/s 边走边以 137 deg/s 打转**。这个区分不改变排序，但改变叙事。
7. **`logs/w5_train.log` 只到 it187**（被中断），**`logs/abd_ab_train.log` 只有 it7000–8999**
   （resume 段），两者**都没有对应的走直评测**，所以没有进入任何相关计算。它们不能用于本问题。
8. **`straightness.tsv` 缺 `align_fwd_ck08999`**（只有 ck01000–ck08000）。
   FACTS.md:40 的 `align_fwd` 终件 +0.40 来自 `logs/queue_align/align_fwd_walk.log:246`（`+0.395 ± 7.931`，
   同文件 `:255` `path efficiency 0.373`、`:257` `achieved vx +0.166`），
   **不在 tsv 里**。我在 D1 里对 `align_fwd` 用的是 **ck8000**，与终件同向但**不是同一个检查点**。
