# 走路 + 坐下起立训练队列（2026-09-15 夜）

> 需求（用户原话）：所有测试和评估做完、数据没问题之后，**重新安排队列训练走路和坐下起立模型**；
> **走路模型不要在静止状态下原地踏步**；把相关参数同步到 rl 里；最后开始训练。

## 1. 队列排法

| run | 任务 ID | 迭代 | envs | 冒烟实测 | 预计 | 预算(预计+2h) |
|---|---|---:|---:|---:|---:|---:|
| walk | `Mjlab-Velocity-Flat-MicroDuck` | 12000 | 2048 | 1.40 s/iter | 5.4 h | 7.4 h |
| sitstand | `Mjlab-SitStand-Flat-MicroDuck` | 8000 | 1024 | 2.42 s/iter | 6.2 h | 8.2 h |

- **两项并发**（彼此独立、不共享 checkpoint）。**2026-09-15 17:49 重排过一次**，原因与结论：
  - 首次启动（17:41）用"冒烟独占耗时"算预算 ⇒ walk 预算 7.4 h。但并发实测 walk **1.40 → 2.75 s/iter**
    （sitstand 几乎不受影响 2.42 → 2.45 ⇒ 它是 CPU 密集、walk 是 GPU 密集）⇒ walk 需 ~7.5 h，
    **会在完成前 6 分钟被预算杀掉**（步态+站定都还没训完）。⇒ 已改为
    `预算 = 冒烟耗时 × iters × 1.15 × 并发惩罚(×2) + 2h`，重排后 walk 预算 12.7 h / sitstand 14.6 h，无截断风险。
  - 顺带修掉 GPU 等待守卫的正则：`pgrep -f "train_cli Mjlab\|train Mjlab"` 里的 `\|` 在 ERE 中不是交替符
    （原写法可能永不匹配、守卫空转），已改为 `|`。
  - 被杀的首轮留下 `_r2` 半成品目录已清理（保留 11999 迭代的基线 run）。
- 预计完成：sitstand 约 5.5 h、walk 约 7.5 h（sitstand 先结束后 walk 独占 GPU 会加速）。
- 迭代数依据：走路按 `AGENTS.md` 步态预算上限（6000@4096 → 12000@2048）；坐起立按 episodic 口径
  8000（cfg 上限 15000 是天花板不是目标），最重的 env 用 1024 envs。
- 脚本：`scripts/train_walk_sitstand_queue.sh`
  ```bash
  setsid nohup bash scripts/train_walk_sitstand_queue.sh \
      > logs/queue_walk_sitstand.out 2>&1 < /dev/null &
  cat logs/queue_walk_sitstand_status.txt      # 进度
  tail -f logs/queue_walk_sitstand.out
  ```

## 2. 参数冻结守卫（09-10 事故的补丁）

`docs/policy_failure_analysis.md` 的头号问题是：**执行器参数在一次训练途中被静默替换，而 run 不记录用的是哪套参数**
（kt/kd 在 09-10 18:51 被换成正弦拟合值 `kd=192.68`，导致两条策略活在两套物理里，事后无人知晓）。本次双层防护：

1. **每次 run 前**把 `vendor/bam/bam/params/hd1910/` 全量快照到
   `logs/queue_walk_sitstand/params_snapshot_<时间戳>/`，并在 results.tsv 里记 `fp_before` / `fp_after`；
   训练结束复验，不一致即标记 **CONTAMINATED**（该 run 作废）。
2. **任何进程**（train/play/eval）导入任务插件时都会记录参数指纹：
   `src/mjlab_microduck/train_hook.py::record_actuator_params()` →
   打印 `[bam] actuator params fingerprint <sha16>` 并追加到 `logs/actuator_params_seen.tsv`
   （含 kt/kd/error_gain/friction_base/command_delay + pid + argv；kd 异常偏大时会告警）。

本次指纹：**`7ec8bdc54393e1b1`**（m5: kt=0.692, kd=0.5, error_gain=0.163, friction_base=0.00675, delay=0.00115）。

> `kd=0.5` 的来由：摆锤数据**对 D 不敏感**（200× 扫频只改 0.0014 rad），所以 D 由 **T5 阶跃扫频**反解：
> 过冲匹配给出 `D_eff=0.0205@kd_fw=40 ⇒ kd≈0.50`（R²=0.935）；代价是摆锤验证 MAE 0.00477 → 0.00504
> （+0.00027 rad）。详见 `bam/docs/identification/hd1910_7v5_reid_plan.md` §9.4。

## 3. 走路"静止不原地踏步"的机制（本次沿用，已在线）

`microduck_velocity_env_cfg.py` 里三项**按命令门控**（`cmd_speed < 0.01` 才生效 ⇒ 对行走步态零影响）：

| 项 | 权重 | 作用 |
|---|---:|---|
| `stillness_zero_cmd` | +0.8 | 零命令时正奖励随躯干速度递减（无阈值可钻） |
| `no_stepping` | −0.4 | 零命令时罚"脚离地"这个具体动作 |
| `standing_leg_vel` | −0.02 | 零命令时罚腿关节速度平方和（**稠密**梯度，实测把踏步策略 74.7 与站定策略 0.008 分得很开） |

权重由 `stance_tax_weight` 课程在**前 300 迭代**从 0 爬到目标；零命令环境占比由 `standing_envs` 课程
从 2% 爬到 25%（≈2000 迭代）。`air_time` 还带了一条"抬脚但身体不动不给分"的进度门。

## 4. 判据与基线（`scripts/queue_walk_sitstand_report.py`）

| 判据 | 阈值 | 基线实测 |
|---|---|---|
| 走路·零命令站定 | `resets==0` 且 **离地 < 8 mm** 且 **动作率 < 0.06** 且 `tilt_max < 15°` | 11999 策略 4 次重复全过；踏步样本全部不过（见下） |
| 走路·0.2 m/s 步态 | `resets==0` 且 步频 ∈ [0.5, 3.0] Hz | 待新 run |
| 坐下起立 | 末段 STAND 的 z ≥ 0.10 m 且 \|tilt\| < 45° | 09-11 策略：STAND 0.114 m / 坐 0.061 m / tilt 34° ⇒ 起立成功 |
| 参数指纹 | `fp_before == fp_after` | — |

> **判据为什么不是"步频"（2026-09-15 实测修正，两次）**：
> 1. 步频在站立时会被足底接触抖动混叠：同一 11999 检查点重复 4 次得 **0.083 / 0.167 / 8.333 / 8.333 Hz**；
> 2. 反过来，真踏步也可能给出中等步频。真正干净的判据是**离地高度**（foot p95−p5）加**动作变化率**：

| 样本（零命令 12 s，同一评测口径） | 离地 mm | 动作率 | 步频 Hz | 判定 |
|---|---:|---:|---:|---|
| 11999 站定（4 次重复） | 0.6 / 1.6 / 2.3 / 4.2 | **0.014–0.052** | 0.08–8.33 | 站定 ✅ |
| 旧 run 8000（踏步） | 19.4 | 0.064 | 3.58 | 踏步 ❌ |
| 新 run 1000（踏步） | 33.8 | 0.095 | 3.08 | 踏步 ❌ |
| **新 run 2000** | **4.9** | **0.019** | 2.83 | **站定 ✅** |

> 站定带（≤0.052）与踏步带（≥0.064）之间的分界取 **0.06**，两个判据同时成立才算站定（保守）。
> 新 run 在 **2000 迭代就已站定**，而旧 run 到 8000 还在踏步（19.4 mm / 0.064）
> ⇒ 新配方（零命令三件套 + kd=0.5）把站定学得早得多，12000 迭代预算有充足余量。

```bash
.venv/bin/python scripts/queue_walk_sitstand_report.py                 # 读队列结果直接判定
.venv/bin/python scripts/queue_walk_sitstand_report.py --repeats 5     # 验收用：自己重跑 5 次取中位数
```

**逐检查点时间线**（`scripts/eval_checkpoint_timeline.py`，回答"行为什么时候成形"）：

| 检查点（走路，零命令） | 步频 | 离地 | tilt_max | 判定 |
|---|---:|---:|---:|---|
| model_8000 | 3.58 Hz | **19.4 mm** | 8.1° | **原地踏步** ❌ |
| model_11999 | 0.083 Hz | 0.5 mm | 1.2° | 站定 ✅ |

⇒ **站定是"后期才成形"的行为**：8000 迭代时还在踏步，12000 才干净。这直接说明
(a) 本次 walk 的 12000 迭代预算不能再砍；(b) 上一轮若按 7.4 h 预算被杀在 ~9300 迭代，
拿到的很可能仍是**会踏步**的模型 —— 那个预算修复是必要的。

## 5. 本轮发现（先记下，不在本次判据内）

1. **站立时缓慢偏航漂移**（数值已按重复测量修正）：11999 策略零命令 12 s 的偏航漂移在
   6 次重复里为 **−17.4° / +0.7° / −18.1° / −101.4° / −158.3° / −2.4°**（中位 ≈ −17°，
   即 ~1.5°/s，但尾部到 −158°）。**单样本不可下结论**（我最初"12°/s"的说法来自一次抽样，偏大）；
   但"站立时会被扰动慢慢转走、且没有任何奖励项拉回来"这个现象在 4/6 次里出现。
   **机制已定位**（`mdp.py: heading_hold_reward` + cfg 常量）：该项有一个**乘性速度门**
   `HEADING_HOLD_VEL_GATE_REF = 0.15`，`|v_base| < 0.15 m/s` 时整项按 `|v|/0.15` 缩放到 **0**
   —— 这是**刻意**的（注释写明：站着不动本来就能完美保持航向，不设门就等于给"干脆别走"发钱，
   会挖出"停下"的吸引盆）。代价就是**静止时没有任何项约束偏航**。
   ⇒ 已按此修（见 §6），并已实测该项在零命令下生效、在 0.2 m/s 下恒为 0。
2. **坐起立的头下垂**：09-11 策略四段里 `neck_rel ≈ −1.93 rad`（远低于 HOME +0.35），
   即文档记录的"把头当第三条腿"局部最优（`policy_failure_analysis.md` §0）。报告工具会把它作为**警示**而非失败项。

## 6. 新增：零命令定点航向项（`yaw_hold_zero_cmd`）
**动因**（两条独立证据）：
1. 11999 迭代走路策略零命令 12 s 偏航漂 **−144°（≈12°/s）**，而其它静止指标正常
   ⇒ 现有奖励栈在静止时对偏航**无约束**（`heading_hold` 被速度门归零）；
2. `logs/night_queue/results.csv` 12 个 run 的统计：**|偏航偏置| ↔ 20 s 净前进 r = −0.900（r²=0.81）**、
   ↔ 路径效率 r = −0.781 ⇒ 航向是走路质量的主杠杆（完整分析见 `docs/night_queue_analysis.md`）。

**实现**（`mdp.py::yaw_hold_at_zero_command` + 走路 cfg 接线）：
- 门控：**整条 twist 指令**（|vx|,|vy|,|wz| ≤ 0.01）⇒ 命令一给立即返回 0；
- 参考角：只在"非零命令 + 刚 reset"时更新 ⇒ 站立时保持"停下那一刻"的朝向，可回纠；
- 形状：`exp(−err²/std²)`（角项，可把漂移转回来）`− rate_weight·|wz|`（角速度项，误差小时也有稠密梯度），
  `std=0.6 rad`、`rate_weight=1.0`、权重 0.8（与 `stillness_zero_cmd` 同量级）；
- 权重与零命令三件套共用 `stance_tax_weight` 课程，前 300 迭代从 0 爬到位。

**已验证**（`scripts/probe_yaw_hold_term.py`，自建 env、指令范围直接设零、不依赖评测脚本的冻结约定）：
零命令站姿下该项非零（首帧 +0.455，与 `hold − |wz|` 的数学一致）、**0.2 m/s 下严格为 0**（门控正确）。
**尚未验证**：训练层面能否真的把 12°/s 慢漂压掉 —— 正在跑的 walk run 不含此项（作"修前"基线），
下一次 walk 训练才是它的验证；届时用 `queue_walk_sitstand_report.py` 的零命令判据 + 偏航漂移数值对比。

> **变体可追溯性**：rsl_rl 会在每个 run 目录里存一份启动时的完整 git diff
> （`<run_dir>/git/microduck_rl.diff`，本 run 141 KB）。已核对：**在跑的 walk run 的 diff 里
> `yaw_hold` 出现 0 次** ⇒ 它确实是"无该项"的修前基线；将来带该项的 run 只需
> `grep -c yaw_hold <run>/git/*.diff` 即可自证 —— 不必再加环境变量开关。
2. **坐起立的头下垂**：09-11 策略四段里 `neck_rel ≈ −1.93 rad`（远低于 HOME +0.35），
   即文档记录的"把头当第三条腿"局部最优（`policy_failure_analysis.md` §0）。报告工具会把它作为**警示**而非失败项。
3. `train_walk_sitstand_queue.sh` 开头的 GPU 等待用了 `pgrep -f "train_cli Mjlab\|train Mjlab"`——
   `\|` 在 ERE 里不是交替符，该守卫可能空转失效。**本次队列运行中不改脚本**（bash 按偏移读取运行中的脚本，
   边跑边改会让它读到错位的行），等队列结束后修成 `"train_cli Mjlab|train Mjlab"`。

## 7. 训练完的交付路径（2026-09-15 确认）

| 环节 | 约定 |
|---|---|
| 本仓库导出件 | 仓库根：`walk_hd1910_<iters>.onnx` / `sitstand_hd1910_<iters>.onnx`（与既有 `walk_hd1910_15000.onnx`、`sitstand_hd1910_15000.onnx` 同风格） |
| 机器人运行时 | `/home/joyandai/microduck/policies/`（内含 `alpha_walking.onnx`、`alpha_sitstand.onnx` 等，属**另一个仓库**） |
| 运行时如何选中 | `robotd-params` 解析 `policies/<name>.onnx` 路径，配置里的 `walk = .../policies/alpha_walking.onnx` 决定用哪一个（见该仓库 `docs/onboarding.md:310`、`source-lessons/lessons/07-intents-config.md:365`） |

⚠️ **不擅自改动机器人仓库**：本会话只产出 ONNX + 判定报告；真正替换 `alpha_walking.onnx` / 改 `walk =` 指向
是用户在机器人侧的决定（涉及发布/release 流程）。交付时给出可复制的命令与校验哈希即可。

## 8. 早期预警的标定（"哪个迭代该出现什么行为"）

用**已有 run 的检查点**把两个技能的成形时间量出来，避免把正常进度误判成失败：

| 技能 | 证据 | 结论 |
|---|---|---|
| 走路·零命令站定 | 新 run **2000** 迭代即站定（离地 4.9 mm / 动作率 0.019）；旧 run 到 8000 仍踏步（19.4 mm / 0.064） | 站定在**本配方**下早期就成形；12000 余量充足 |
| 坐起立·能否起身 | 旧 8000 迭代 run：**model_4000 已能起身**（末段 STAND z=0.11 / tilt 31.6° / 坐姿 0.062），model_7999 同样；新 run model_2000 还不能（z=0.06），**model_3000 已能起身**（z=0.107 / **tilt 27.9°** / 坐姿 0.061） | 起身出现在 **2000–4000**；新 run 到 3000 就成形且**姿态更稳**（tilt 小 4–6°），8000 预算足够 |

> 判据口径：`eval_sitstand_head.py`（60 s episode、无推力、冻结姿势时序）的**稳态段** z 与 tilt；
> 起立成功 = 末段 STAND z ≥ 0.10 m 且 |tilt| < 45°。头下垂（`neck_rel` 远低于 HOME）作为**警示**单列，
> 不算失败（它是文档记录的既有局部最优）。

## 9. 最终结果（2026-09-16 09:58，队列 rc=0 全部完成）

| run | 迭代 | 用时 | 参数指纹 | 判定 |
|---|---:|---:|---|---|
| walk | 12000 | 6h48m | `7ec8bdc54393e1b1` 前后一致 ✓ | **PASS** |
| sitstand | 8000 | 5h12m | 同上 ✓ | **PASS** |

**走路·零命令站定（`--repeats 5`，18 s/次）**：5/5 全过 —— 离地中位 **0.4 mm**（最大 0.8 mm）、
`tilt_max` 中位 **2.3°**、步频中位 0.11 Hz、动作率 0.009 ⇒ **不在静止状态原地踏步** ✅
（对照：旧 run 8000 迭代时离地 19.4 mm / 动作率 0.064）

**走路·0.2 m/s 步态**：步频 2.3 Hz、离地 28.4 mm、`tilt_mean` 2.6° ⇒ 正常迈步 ✅
（`resets=1` 但 `tilt_max=7.2°` ⇒ 是 20 s episode 的**超时重置**，不是摔倒）

**坐起立（60 s 协议）**：末段 STAND z=**0.115 m**、坐姿 z=**0.055 m**、tilt 35.4° ⇒ 起立成功 ✅
（`neck_rel ≈ −1.93 rad` 头下垂，属既有"头三脚架"局部最优，非本次判据，已单列警示）

**交付件**（仓库根）：`walk_hd1910_12000.onnx`（sha256 `bebfc1bb…`）、
`sitstand_hd1910_8000.onnx`（sha256 `9ef1053f…`）；两者 obs[1,61]→actions[1,14]，
与运行时共享观测合同；`run_path` 元数据指向本次 run（可追溯）。最终报告：`docs/walk_sitstand_final_report.md`。

**遗留（不在本次判据内，供下一步决策）**：
1. **站立时缓慢偏航**：5 次重复中位 −82°（范围 −284…+26°/18 s）。机制已知（`heading_hold` 速度门在站立时归零），
   修法（`yaw_hold_zero_cmd`）**已实现并在配置里**，但本次 run 的 git diff 里没有它（配置修改晚于启动）
   ⇒ 需要一次**带该项的对照训练**验证（脚本与判据都已就绪）。
2. **坐起立头下垂**：见上，需奖励结构重设计（`policy_failure_analysis.md` 已取证）。

### 教训：`resets` 不等于摔倒（2026-09-16）

速度任务 `episode_length_s = 20 s`。用 `--seconds 20` 评测时**必然**产生一次超时重置，
最初判据把它当成失败 ⇒ 新策略被误判为 FAIL（实际 tilt_max 仅 10.4°、站得很好）。
判据已改为 **摔倒 = `resets>0` 且 `tilt_max > 45°`**，评测时长默认取 18 s 规避；
回归样本验证：真摔倒的 `model_0`（tilt 67°）仍判 FAIL ✓，踏步样本仍判 FAIL ✓。

## 10. 遗留问题复核（2026-09-16，用户追问"这两个问题还在吗"）

### 10.1 站立时偏航漂移：**仍在，且新模型比旧模型更明显**（n=6 各）

| 指标（零命令 18–20 s） | 新 run 12000 | 旧 run 11999 |
|---|---|---|
| 偏航漂移 6 次 | +26, **−284**, +15, −82, **−259**, −134 | −17, +1, −18, −101, −144, −0 |
| 中位 / \|中位\| | **−108° / 108°**（≈ −5.6°/s） | −18° / 18°（≈ −1.5°/s） |
| 离地中位 | **0.6 mm**（更好） | 1.9 mm |
| 动作率中位 | **0.014**（更好） | 0.024 |

- 两步都"站着不动"，**新模型腿更静（不踏步），但转向漂移更大**；n=6 时 Mann-Whitney U=9（p≈0.1），
  **差异是提示性、未达统计显著**，但中位数差 6 倍。
- **时间形状（新模型一次 18 s，逐步 yaw 序列）**：分段均值 −118.4 → −121.6 → −123.1 → −125.8 → −126.9 → −129.0°，
  即**近似线性匀速旋转 ≈ −2.1°/s**；逐步增量均值 −0.043°/步、标准差 0.924°/步 ⇒ **系统性偏置**（drift/jitter=0.05），
  不是随机游走。这正是 `heading_hold` 本来要治的 DC 偏置，而它的速度门在站立时把该项归零。
- 一个可解释的机制：旧模型站立时有轻微踏步（离地 1.9 mm），步态本身会**部分抵消**偏航偏置；
  新模型把腿完全钉住后，这个偏置**单调累积**，没有东西把它拉回来。
- **修法状态**：`yaw_hold_zero_cmd`（角项可回纠 + 角速度项稠密梯度）已实现、接线、探针验证（站立非零 / 0.2 m/s 恒为 0），
  但**本次 run 的 git diff 里没有它** ⇒ 需要一次对照训练才能确认有效（约 6.8 h，GPU 空闲即可跑）。

### 10.2 坐起立头下垂：**仍在，且与旧模型数值上完全一致**

| 相位 | 新 run（8000） | 旧 run（8000） |
|---|---|---|
| STAND | z=0.115, tilt 35.4°, **neck_rel −1.932** | z=0.114, tilt 34.0°, **neck_rel −1.933** |
| SIT | z=0.055, tilt 0.7°, neck_rel −1.932 | z=0.083, tilt 15.7°, neck_rel −1.931 |

- `head_rel ≈ 0`（头**跟踪被命令的姿态**）而 `neck_rel ≈ −1.93 rad`（≈ −90° 相对 HOME）⇒ 策略**主动**把脖子压低，
  即文档记录的"头当第三条腿"（头-地接触力 3.16 N ≈ 体重的 37%）。
- 躯干姿态反而改善了（新 35.4° vs 旧 34.0°，且中段 tilt 27.7° vs 33.8°），但**头下垂是同一个局部最优**。
- **为什么现有惩罚治不了**（`policy_failure_analysis.md` 取证）：垂头姿态下 `head_pose_bias` 只值 −0.0008/step，
  而用头撑地值 +0.87/step ⇒ 差 3 个数量级；把权重按比例调大（~1000×）会压垮其它项。
  可行的方向是**换目标**：直接惩罚"头/喙-地接触"（sensor 已有）或把站立姿态的高度/支撑面定义成不依赖头。

### 10.3 想自己跑对照时怎么跑（已备好，不必改代码）

```bash
cd /home/joyandai/microduck_rl

# A) 带 yaw_hold_zero_cmd（默认开）—— 验证"站着不转"
ONLY=walk CONCURRENCY=1 setsid nohup bash scripts/train_walk_sitstand_queue.sh \
    > logs/queue_walk_yawhold.out 2>&1 < /dev/null &

# B) 对照：显式关掉该项（跑出来的 run 目录 git diff 里会没有它，可自证）
ONLY=walk CONCURRENCY=1 MICRODUCK_YAW_HOLD_ZERO_CMD=0 setsid nohup bash scripts/... &

# 跑完判定（零命令重复 5 次、18 s/次；判据：离地<8mm + 动作率<0.06 + tilt<15° + 摔倒=resets>0且tilt>45°）
.venv/bin/python scripts/queue_walk_sitstand_report.py --out logs/queue_walk_sitstand --repeats 5

# 想看偏航漂移的**时间形状**（匀速偏置 vs 随机游走）
.venv/bin/python scripts/gait_forensics.py --onnx <onnx> --task Mjlab-Velocity-Flat-MicroDuck \
    --lin-vel-x 0 --lin-vel-y 0 --ang-vel-z 0 --seconds 18 --csv /tmp/yaw.csv \
    --params-json vendor/bam/bam/params/hd1910/m5.json
# 再看 /tmp/yaw.csv 的 trunk_yaw 列（分段均值判匀速、逐步增量均值/标准差判 drift/jitter）
```

> `MICRODUCK_YAW_HOLD_ZERO_CMD` 是本次新增的开关（默认 `1`=开；`0`=关），沿用仓库
> `MICRODUCK_STANCE_HYGIENE` 等既有 env 开关范式；两种状态都已实测（终点权重 0.8 / 0.0）。
