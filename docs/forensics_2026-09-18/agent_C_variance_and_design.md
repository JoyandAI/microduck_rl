# Agent C：方差分解与实验设计（对齐臂）

> 视角：**方差与实验设计**。只回答"现在的数据能支持什么结论 / 需要多大实验 / 有没有更便宜的判别方式"。
> 全部数字来自本仓库已有产物；脚本只写在 `/tmp/`（`/tmp/var_analysis.py`、`/tmp/infer_analysis.py`、`/tmp/power_scipy.py`），未改动仓库任何文件、未启动训练。

---

## 结论(3 行内)

1. **不能**说"`align_fwd` 比 `align_official` 好"——那是**臂（配方）级**结论，而每臂只有 **1 个训练种子**（三臂都是 seed 42，`logs/rsl_rl/velocity/align_*/params/agent.yaml:1`，来自 `mjlab/rl/config.py:86` 的默认值 42，即 `.venv/lib/python3.12/site-packages/mjlab/rl/config.py:86`），同配方跨种子的实测 SD 是 **8.60–13.54 deg/s**（`docs/night_queue_analysis.md:15,55-62`），与本次臂间差 **21.1 deg/s** 同量级 ⇒ 1 个种子分不开。
2. **能**说的只有两件"具体交付件"级别的事：`align_fwd` 终件与 `align_official` 终件在 P64 同 seed 上相差 **+21.07 deg/s（配对 SE 1.32，t=+15.9）**；且 `align_fwd` 的**未经挑选的终件**（|yaw DC| 0.395）优于 official 臂 9 个检查点里**最好的那个**（13.326）。但该排序**对指标极敏感**：换成"20 s 净偏航 <30° 的环境比例"，fwd 终件 **0.28** vs 官方参照 **0.72**；再换成中位净偏航，fwd 终件是 **78.2°** vs 官方 **2.8°**（FACTS.md:15 的"+8.5° vs +2.8°"是**均值 vs 中位数**的口径混用，见 §1.2）。
3. 下一步：先做**零成本的同检查点确定性消融**（见《推荐的确定性消融清单》，A 组 0 GPU·h + B 组约 0.5 GPU·h）回答"偏航从哪来 / 能不能转"；真正要判臂就上 **2 臂 × 8 seed × 9000 iter @2048**（MDD 20.4 deg/s，≈46 GPU·h），**不要**再跑 3 seed 级别的臂间对比。

---

## 方差分解表

统一协议 **P64** = 64 env × 20 s × seed 0 × DR 开 × 冻结指令（`scripts/queue_align_sweep.sh:80-82`）。
"换算到单次 P64 读数"一列 = 该方差源对**一次读数（一个标量）**的贡献。

| # | 方差来源 | 抽样单位 | 实测 SD | 出处（file:line / 命令） | 单次 P64 读数的 SE |
|---|---|---|---|---|---|
| 1 | **同检查点、跨环境（DR 抽样）** | 1 env | yaw_dc per-env SD 中位 **8.14**（27 次评测，范围 0.012–17.47） | `logs/queue_align/sweep/align_*_walk.json` → `aggregate.yaw_rate_bias_deg_s.std`（命令见 `/tmp/var_analysis.py` §2） | **1.02 deg/s** |
| 1b | 同上（路径效率） | 1 env | per-env SD 中位 **0.214** | 同上 `path_efficiency.std` | **0.027** |
| 2 | **重复评测噪声**（**bit-identical ONNX**、同 seed 0、同 flag，跑两次） | 1 次 P64 | official 终件 Δmean **+0.441**（per-env SD(diff) 2.363）；full 终件 Δmean **−0.134**（2.425）；per-env r = 0.924 / 0.967 | `logs/queue_align/align_{official,full}_walk.json` vs `logs/queue_align/sweep/align_{official,full}_ck08999_walk.json`（md5 相同：`ac994d365572` / `f1bebe83afa7`） | **0.295–0.303 deg/s**（eff: 0.014–0.019） |
| 3 | **同检查点、跨评测 seed**（固定策略，5 seed × P64） | 1 个 eval seed | `U_hd` yaw_dc **1.041**（均值 −12.409）；`O_xl` **0.330**（−0.597） | `logs/q1_matrix/{U_hd,O_xl}_vx0.2_s{0..4}.json`；矩阵生成器 `scripts/q1_straightness_matrix.sh:37,39`（NUM_ENVS=64, SEEDS="0 1 2 3 4"） | **≈1.0 deg/s**（与 #1 自洽：8.61/√64=1.08） |
| 3b | 同上（路径效率） | 1 个 eval seed | `U_hd` **0.0228**；`O_xl` **0.0314** | 同上 `path_efficiency` | 0.023–0.031 |
| 4 | **同臂、跨检查点**（sweep，每 1000 iter） | 1 个 checkpoint | yaw_dc: official **−39.22 ± 38.59**（min −137.13, max −13.33, n=9）；full **−7.25 ± 6.21**（n=9）；fwd **−17.97 ± 19.79**（n=9，含终件）。eff: 0.073±0.033 / 0.207±0.107 / 0.178±0.150 | `logs/queue_align/sweep/straightness.tsv`（26 行）+ `logs/queue_align/align_fwd_walk.log:246,255`（fwd 终件不在 TSV 里） | —（这是**被测量本身**的散布，不是噪声） |
| 4b | 同上，**去趋势**后（一阶差分/√2） | 相邻 checkpoint | official **23.6**、full **8.2**、fwd **10.8** deg/s | 同上 | — |
| 5 | **同臂、跨训练种子** | 1 个训练 run | yaw bias（signed）pooled within-arm SD **13.538**（每臂 9.58/5.83/7.52/10.67）；\|yaw bias\| **8.602**；path_eff **0.1911**；净前进 **0.5357** | `docs/night_queue_analysis.md:15-17,55-80`（数据 `logs/night_queue/results.csv`，4 臂 × 3 seed） | **13.5 deg/s**（eff 0.191） |
| 6 | **臂间**（终件，P64） | 1 臂（1 seed） | official −21.114 / full −11.373 / fwd +0.395 deg/s；eff −0.023 / 0.089 / 0.373 | `logs/queue_align/align_{official,full,fwd}_walk.log:246,255`；= FACTS.md:38-40 | 与 #5 同量级 ⇒ 不可分 |
| 6b | **臂间**（全部检查点集合，非参数） | checkpoint（非独立） | yaw_dc: full vs official **U=79/81, p=0.0002**；fwd vs official **U=61/81, p=0.077**；full vs fwd **U=56/81, p=0.190**（精确置换，命令见下） | 同上 TSV + fwd 终件 | — |

复现命令（只读）：

```bash
python3 /tmp/var_analysis.py        # 表 #1/#4
python3 /tmp/infer_analysis.py      # 表 #2/#3/#4/#5 + 精确 Mann-Whitney
python3 /tmp/power_scipy.py         # §3 的全部功效/样本量数字
```

### 1.1 三个必须写在结论旁边的限制

* **检查点不是独立样本**：同一 run 的 9 个 checkpoint 是时间序列（有强趋势，official 从 ck01000 的 −137.1 大致爬到 ck08000 的 −13.3，中间在 ck06000 有反复），所以 #6b 的 p 值是**乐观下界**，只能当"分布重叠程度"的参考。
* **重复评测不可复现**：#2 显示**同一个 ONNX、同一个 seed、同一组 flag** 两次跑出来的 per-env 偏航率最大差 6.47 deg/s、净偏航最大差 129°（`align_official` 两文件对比）。所以"同 seed 就配对"只是**近似**成立（r=0.92–0.97），引用配对 SE 时必须加 #2 的 0.30 deg/s 地板。
* **`--drop-dr` 不等于零噪声**：`logs/q1_matrix/O_hd_vx0.2_s0_nodr.json` 与 `..._s1_nodr.json` 的 yaw DC 是 **−7.84 vs −16.98**（DR 已关、只差 seed）。`--drop-dr` 只固定物理参数，**spawn/reset 仍随机**，因此 DR-off 也只能"比同一 draw 的排序"，不能报绝对值。

### 1.2 官方参照的真身

判据里"官方偏航 DC 0.2 / 路径效率 0.43"来自 `logs/axis/j_official_0.2.json` 的 `aggregate`：yaw DC **−0.188**、`path_efficiency__median` **0.430**、`|net_yaw|__median` **2.83**、64 env、seed 0、`freeze_pose_cmd: true`。
⚠️ 它是 **official policy + xl330 m6 参数 + kp_fw 200**（该文件 `overrides`），**不是**我们的 HD-1910/m5 机器；而且它 20 s 只走了 **0.038 m/s**（`along_end_m.mean/20`），我们的臂走 0.081–0.095 m/s。所以"偏航 DC 越接近 0.2 越好"这个判据**跨硬件、跨速度**，只能当"大概方向"，不能当等值比较。
⚠️ **口径混用的地方有两处，都会改变结论**：

| 指标 | FACTS.md 出处 | 官方那一格的定义 | 我们那一格的定义 | 同一口径重算（official REF → align_fwd 终件） |
|---|---|---|---|---|
| 路径效率 | FACTS.md:13-14 vs :38-40 | `path_efficiency__median` = **0.430** | mean = **0.373** | median: 0.430 → **0.463**（fwd 略好）；mean: 0.304 → **0.373**（fwd 略好）——**这个指标两种口径都不改变排序** |
| 20 s 净偏航 | FACTS.md:15 | `net_yaw_deg__median` = **2.8°** | `net_yaw_deg.mean` = **+8.5°** | median\|net\|: 2.83° → **78.16°**（fwd 差 **28×**）；mean signed: −3.72° → +8.46°（fwd 差 2.3×） |

⇒ "净偏航 +8.5° vs 官方 +2.8°"这个对比**不成立**：官方用的是中位数、我们用的是均值。按同一口径，fwd 终件的中位净偏航是 **78.2°**，不是 8.5°。（FACTS.md:38-40 表里的 −422°/+8.5°/−228° 都是 `net_yaw_deg.mean`，与官方参照不同源。）

**⇒ 排期判据建议**：主判据用 **`yaw_dc`（mean，与官方 −0.188 同源）+ `frac(|net yaw|<30°)`**，两者都必须是 P64 聚合值；不要用 `net_yaw` 的"中位 vs 均值"跨行比较。
（两个定义在本仓库里都合法：`scripts/eval_straightness.py:640` 是 along/total，`:736-740` 另加 `__median`；本文两套都报。）

### 1.3 `yaw_dc` 这个指标本身是**别名化的**（对训练排期影响最大的一条）

| 检查点 | yaw DC | 20 s 净偏航中位 | \|net\|<30° 环境比例 | 沿向速度 |
|---|---:|---:|---:|---:|
| `align_full_ck03000` | **+0.93**（过 <1.0 的线） | **99.5°** | 0.25 | 0.085 m/s |
| `align_fwd_ck08000` | **−1.46** | **93.3°** | 0.22 | 0.081 m/s |
| `j_ck7000_0.2` | **−0.70** | **118.5°** | 0.08 | 0.091 m/s |
| `align_fwd` 终件 | +0.40 | 78.2° | **0.28** | 0.095 m/s |
| `align_full_ck00000`（未训练！） | +0.02 | **0.3°** | **1.00** | **0.000 m/s** |
| **官方参照** | −0.19 | **2.8°** | **0.72** | 0.038 m/s |

出处：逐检查点 `logs/queue_align/sweep/*_walk.json` + `logs/queue_align/align_fwd_walk.log:256`（"within 30 deg yaw: 28%"）+ `logs/axis/j_official_0.2.json`。

**⇒ 单看 yaw DC 会把"转了一大圈但两个方向的偏航互相抵消"判成好**（`align_full_ck03000`：DC +0.93 但中位净偏航 99.5°）；反过来未训练的 ck00000 在 DC/净偏航上"完美"因为**它根本不动**。
**⇒ 任何验收/训练判据必须同时报**：① yaw DC、② 20 s 净偏航中位、③ `|net|<30°` 比例、④ 沿向速度（做下限门，例如 ≥0.08 m/s）。
三个 align 臂在 ③ 上的逐检查点最大值：official **0.02**（n=9）、full **0.28**（n=10，含未训练 ck00000 的 1.00——它不动，是退化值）、fwd **0.28**（n=9，终件）；把未训练的 ck00000 排除后**没有任何一个训练过的检查点达到 0.5**，而官方参照是 **0.72**。⇒ 按这条更硬的判据，三条臂离官方都还很远，且**互相之间分不开**。

按仓库自己的两条线（`scripts/queue_align_sweep.sh:208-210`：`|yaw DC|<1.0` 且 `path_eff≥0.30`）逐检查点命中率：official **0/9**、full **1/9**（ck03000）、fwd **1/9**（终件 8999，唯一一个"终件即最好"的臂）。

---

## 需要多少个 seed/迭代（算式）

### 2.1 目标量

* 主判据 D：**离官方参照有多远** —— `|yaw_dc − (−0.188)|`（越接近 0 越好，**不设 0**）。
* 副判据 E：`path_eff`（中位定义 `path_eff_med`，线 ≥0.30）。
* 决策相关的最小对比：**fwd 终件 vs official 终件**，实测 Δ = **21.51 deg/s**（`−21.114 → +0.395`，`align_official_walk.log:246` / `align_fwd_walk.log:246`），ΔE = **0.396**（−0.023 → 0.373，`:255`）。

### 2.2 σ 的来源与适用性

| σ 候选 | 值 | 出处 | 适用条件 / 是否可用于当前指标 |
|---|---:|---|---|
| 跨**训练种子**（signed yaw bias） | **13.538 deg/s** | `docs/night_queue_analysis.md:15,61`（= `scripts/night_queue_analysis.py` 输出） | ✅ 是"run 级 KPI 跨 seed"的**唯一**实测；⚠️ 旧物理轨（`logs/night_queue/A_s42.eval.log:46`：vin 4.75–5.25 V、force_limit ±0.97 Nm、friction_base 0.0136），32 env、`freeze_pose_cmd=False`（`logs/night_queue/A_s42.json`），3000 iter；今天 m5 是 7.1–7.9 V / ±1.46 Nm / 0.0067（`logs/queue_align/align_official_walk.log:49`） |
| 跨**训练种子**（\|yaw bias\|） | **8.602 deg/s** | 同上（脚本对 signed 值做 ANOVA，\|\|·\|\| 只用于展示 ⇒ 这是重算值） | 同上 |
| 跨**训练种子**（path_eff） | **0.1911** | 同上 `:16,71` | 同上 |
| 跨**检查点**（当前轨，同 run） | 6.21 / 19.79 / 38.59 | `straightness.tsv` 三臂 | ⚠️ 混了趋势 + 训练随机游走，**不是** seed SD；只用于说明"检查点选择"本身有多贵 |
| 评测噪声 | 1.0 deg/s（yaw）、0.023（eff） | 表 #3 `logs/q1_matrix/*_vx0.2_s*.json` | ✅ 当前轨；相对 seed SD 可忽略 |

**⇒ 关键判断：可用 σ 只有旧轨的 13.54 / 8.60 / 0.191，必须标注为"跨轨外推"。** 方向上它偏保守：旧轨舵机力限低 33%（±0.97 vs ±1.46 Nm），越接近力矩饱和，种子间分散通常越大。

### 2.3 公式

两臂比较（双侧 α=0.05、power=0.80，正态近似）：

```
n/arm = (z_{0.975} + z_{0.80})² · 2σ² / Δ²  =  15.698 · (σ/Δ)²
MDD(n) = (z_{0.975} + z_{0.80}) · σ · √(2/n)  =  2.8016 · σ · √(2/n)
```

上表数字用 **scipy 精确非中心 t / 非中心 F** 复核（`/tmp/power_scipy.py`；两臂情形 `T² ~ 非中心 F(1, 2n−2, δ²)`）。正态近似在 n≥20 时几乎一致（Δ=5 → 116 vs 精确 117），在 n≈8 时**低估 1 个**（Δ=20 → 近似 8 vs 精确 9），所以下文一律引用精确值。

### 2.4 数字

yaw DC，σ = 13.538（旧轨 signed）：

| Δ (deg/s) | 2 | 5 | 10 | **20** | 40 |
|---|---:|---:|---:|---:|---:|
| **n/arm（精确）** | 721 | 117 | 30 | **9** | 4 |

| n/arm | 2 | 3 | 5 | **8** | 12 | 20 |
|---|---:|---:|---:|---:|---:|---:|
| **MDD (deg/s)** | 76.5 | 41.6 | 27.4 | **20.4** | 16.2 | 12.3 |

σ = 8.602（旧轨 \|yaw bias\|）：Δ=20 → n=5；Δ=10 → n=13；MDD(n=8)=13.0。
path_eff，σ = 0.1911：Δ=0.30 → **n=8**；Δ=0.20 → n=16；Δ=0.10 → n=59；Δ=0.05 → n=231；MDD(n=8)=**0.288**。
path_eff，σ = 0.1047（当前轨 pooled 检查点 SD，**不是 seed SD**，仅作乐观界）：Δ=0.20 → n=6。

**⇒ 对当前效应量的答案：yaw DC 需要 9 个 seed/臂（σ=13.54）或 5 个（σ=8.60）；path_eff 需要 5–8 个 seed/臂。取两指标的交集 = 8 个 seed/臂。**

### 2.5 核实 AGENTS.md 的两个数字（**这是本次任务明确要求核实的**）

出处已定位：`AGENTS.md:261-264` ← `docs/night_queue_analysis.md:15-17,55-80` ← `scripts/night_queue_analysis.py`（数据 `logs/night_queue/results.csv`，2026-09-14/15，A/B/C/D × 3 seed × 3000 iter @2048）。

* 52 与 8 都是 **k=4 单因素 ANOVA** 的样本量，用脚本自己的近似式 `n/arm = (2.802+0.84)²/(k·f²)`（`scripts/night_queue_analysis.py:158-166`），其中 f = 臂间 SD / 臂内合并 SD。
* 我用**精确非中心 F** 重算（`/tmp/power_scipy.py` §1）：

| 指标 | σ_臂内 | σ_臂间 | 仓库近似 | **精确** |
|---|---:|---:|---:|---:|
| yaw bias（signed deg/s） | 13.538 | 8.923 | **8** | **10** |
| \|yaw bias\|（deg/s） | 8.602 | 4.686 | 12 | **14** |
| path efficiency | 0.1911 | 0.0485 | **52** | **58** |
| 20 s 净前进（m） | 0.5357 | 0.1519 | 42 | **47** |

**结论（核实结果）**：

1. **数字可复现**，但**语义是"分辨当时观测到的那个臂间差（0.0485 path_eff / 8.92 deg/s）"所需的量**，不是"分辨任意有用差值"的通用常数。脚本的近似把所需 n 低估约 10–20%（52 vs 58、8 vs 10）。
2. **52 不适用于当前问题**：它对应 Δ_path_eff = 0.0485 这种极小臂间差；当前要分辨的是 0.396，只需 **5–8 seed/臂**。"52 seed/臂"若被当成通用门槛会**把可行的实验判死**。
3. **8（精确 10）在形状上适用于当前 yaw 指标**，但它的 σ 是旧轨/32 env/`freeze_pose_cmd=False`/3000 iter 的，不是当前 m5 的。当前判据下 MDD(n=8) = 20.4 deg/s，与实测臂间差 21.5 deg/s 刚好同量级 ⇒ **8 seed/臂是"刚好能看见当前这个效应量"的下限**，不是余量充足的设计。
4. 该分析的另一个适用性缺口：它用**一个 onnx/run 一个读数**（32 env、seed 0），没有"最佳检查点"这一步；当前臂的 KPI 是 9 个检查点里的最优，**选择偏差**使真实可分辨性比这里算的更差。

### 2.6 每个 seed 多少迭代 / 评几个检查点

* **迭代数必须是臂定义的一部分，且预先钉死。** 证据：同配置同 seed，3500 vs 12000 iter 的 path_eff 是 **0.57 vs 0.12**（`AGENTS.md:264-265`）；本次 `logs/ckpt_yaw_sweep/Mjlab*r2_*.json` 单个 run 的 13 个检查点 yaw DC 从 **−12.52 到 +39.12**（SD 14.17）；三个 align 臂的检查点 SD 6.2/19.8/38.6。**"迭代数"这个自由度带来的散布 ≈ 种子散布**，不钉死就等于多了一个隐藏的臂间因子。
* 建议：**9000 iter @2048**（与三条已有臂逐字可比：`logs/queue_align/status.txt:11,12,59`），并且**主读数用预先注册的终件 `model_8999`**（不是"最好的那个"）。理由：只有 `align_fwd` 的终件恰好是最优（|0.395| < |−1.455|），另两条臂若按"最优"读会白拿 7–11 deg/s 的选择红利。
* **评几个检查点**：每 run 评 **3 个**（`{5000, 7000, 8999}` 的粗网格 + 终件）。用途不是降方差（同 run 检查点是随机游走，平均降不了多少），而是**（a）事后审计"终件是不是代表性读数"，（b）让"最佳检查点"这个次要读数带上下文**。单次 P64（20 s 窗口）实测墙钟 **26 s**（`logs/queue_align/status.txt:53 → :57`，与 smoke 并发）到 **41 s**（`:63 → :64`，独占），⇒ 每 run 3 次 ≈ 1.5–2 min，16 个 run 共 48 次 ≈ **20–35 min**，成本可忽略。
* **评测 seed 数**：主读数用 **seed 0**（与全部历史可比）；确认时另跑 **seed 1、2** 各一次（每次 P64 的 seed 噪声 ≈1.0 deg/s，远小于 seed SD 13.5，所以 1 个就够；3 个是为了防"seed 0 特别幸运"）。

### 2.7 推荐排期（数字）

| 方案 | 规模 | 判据可达 | 实测成本 |
|---|---|---|---|
| **A（推荐）** | **2 臂（align_fwd vs align_official）× 8 seed × 9000 iter @2048** = 16 run（复用 seed 42 ⇒ 新跑 14 个）；每 run 3 次 P64 | yaw DC **MDD 20.4 deg/s**；path_eff **MDD 0.288** | 2 并发 = **5.78 h/批 2 个 run**（`status.txt:11-12 → 50-51`，=2.31 s/iter；单跑 1.06 s/iter，`:59 → 70`）⇒ 8 批 ≈ **46 h ≈ 1.9 天**；+48 次 P64（26–41 s 级 ⇒ 约 30 min） |
| B（若只要方向） | 2 臂 × 5 seed | MDD 27.4 deg/s / 0.387 | ≈ 29 h |
| C（给 3 臂） | 3 臂 × 8 seed = 24 run | 同上 | ≈ 69 h（但 `align_full` 在**终件**层面被 fwd 支配：yaw DC −11.37 vs +0.395、eff 0.089 vs 0.373；检查点集合层面 full vs fwd 的 yaw 也分不开，p=0.190） |

**复用**：seed 42 的三条已有 run 就是第 1 个种子（`align_*/params/agent.yaml:1`），所以方案 A 只需新跑 **7 seed × 2 臂 = 14 run**（≈40 h），直接省掉 1/8。⚠️ 前提是**臂定义不变**（仍是"逐字官方"vs"官方+头部三项"）；一旦改配方，seed 42 那一份作废，要跑满 16 个。
**不要**把 3 个臂一起上：AGENTS.md:258-261 的并发实测（velocity@2048 = 2880 MiB，2 个并发 1.40→2.75 s/iter），3 并发会把单批时间再拉长，臂数越多单位算力信息量越低。

---

## 推荐的确定性消融清单

**总原则**（`AGENTS.md:261-266` + `docs/night_queue_analysis.md:38-41`）：同一 checkpoint、冻结指令、无推力、单因素改动；**单环境绝对值不可引用**（`AGENTS.md:11` 段：可差 20–44×且翻符号），所以下面每一条都写清抽样数，并把"可引用的量"限定为 **P64 的聚合值** 或 **同 draw 内的排序**。

所有命令都用 `scripts/eval_straightness.py`（已支持 `--seed/--num-envs/--drop-dr/--head-pose-cmd/--ablate-head/--ablate-joints/--ang-vel-z`，`:79-116`），对象固定为 **`logs/queue_align/align_fwd.onnx`（终件）**，并**同时跑 `align_official.onnx`（终件）做配对 A/B**。

### A 组：零成本（不跑任何 rollout，只重读现成产物）

| 编号 | 做法 | 能排除什么 / 得到什么 | 抽样数 |
|---|---|---|---|
| **A0** | 从 27 个 `logs/queue_align/sweep/*_walk.json` 里取 `per_env`，算 `hip_yaw_L/R_rel_deg`、`foot_pair_torsion_deg` 与 `yaw_rate_bias_deg_s` 的逐环境相关与斜率 | **偏航是否来自"脚对扭转/髋 yaw 偏置"**。若 r 高且斜率 ≈1 deg/s per deg torsion，则偏航是几何（站姿）问题而非控制问题 | n=64 env × 27 检查点（现成） |
| **A1** | 同上，同时报 ① yaw DC ② `median|net_yaw|` ③ `frac(|net|<30°)` ④ `speed_along_mps` | 排除"用 DC 一个数就把臂排序"的别名化（§1.3） | 同上 |
| **A2** | 重读 `logs/q1_matrix/{U_hd,O_xl}_vx0.2_s{0..4}.json` 与 `O_hd_vx0.2_s{0,1}_nodr.json` | 得到**评测噪声地板**（1.0 deg/s）与"DR-off 仍随 seed 变"（−7.84 vs −16.98）⇒ 排除"把单次读数当结论" | 5 seed + 2 nodr（现成） |
| **A3** | `python3 /tmp/infer_analysis.py`（本文脚本） | 复现 §2 的全部功效数字；给出"现在的 n=1/臂 能不能说话"的明确答案：**不能** | — |

### B 组：需新跑的确定性消融（每条都给抽样数与判据）

约定：`S` = `--num-envs 64 --seed 0 --seconds 20 --lin-vel-x 0.2 --freeze-pose-cmd`（= P64，`scripts/queue_align_sweep.sh:80-82`）。单次 P64 实测墙钟 **26–41 s**（`logs/queue_align/status.txt:53→57` / `:63→64`）；下面用 12 s 窗口的条目会更短。

| 编号 | 命令（在 P64 基础上加参数） | 能排除什么 | 抽样数 | 判据 |
|---|---|---|---|---|
| **B1 先证明配对可复现** | 对同一 onnx **跑两次 S** | 排除"用同 seed 就以为逐环境配对成立"。基线：Δmean 0.13–0.44 deg/s，per-env r 0.924/0.967 | 2 次 × 2 策略 = 4 次 | 两次 Δmean(yaw DC) ≤1 deg/s 才允许引用配对 SE；否则按独立样本（SE 1.0）算 |
| **B2 头耦合**（回答"偏航从哪来"） | 加 `--ablate-head`（`eval_straightness.py:106`：把头 4 关节动作清零 = 头锁在 HOME） | 排除/确认"**289 g = 37% 体重的头**的摆动是躯干偏航来源"（FACTS.md:106）。若清零后 yaw DC 显著变小，则头是主通道 | 2 策略 × 2 模式 × 3 seed = 12 次 | Δyaw DC 与 `√(SE_base²+SE_abl²)` 比较；同时报 eff/F30 |
| **B3 髋 yaw 通道** | 加 `--ablate-joints left_hip_yaw,right_hip_yaw`（`:111`） | 排除"偏航由被指令驱动的髋 yaw 自由度产生"。与官方验收里的 `hip_yaw 顶限位 99.5%`（FACTS.md:69）互为证据 | 同上 12 次 | 同上；另报 `hip_yaw_*_at_limit_frac` |
| **B4 物理随机化** | 加 `--drop-dr`（`:88`） | 排除"偏航 DC 是 DR 抽样产物"。若 DC 在 DR-off 下仍同号同量级 ⇒ 是**策略的系统性偏置**（`drift_jitter_ratio` 对 official 是 0.615，`logs/queue_align/align_official_walk.log:269`） | 2 策略 × 2 模式 × 2 seed = 8 次 | 报聚合 yaw DC ± per-env SD；**不得**引用单环境值 |
| **B5 转向指令活性**（回答"能不能转"） | `--ang-vel-z w`，w ∈ {−0.6,−0.3,0,+0.3,+0.6}（≈±34 deg/s），12 s 窗口 | 排除"wz 指令没进网络 / 被冻结覆盖"。现成形态：`scripts/queue_round2_report.py:407-453`（`--steer`，64 env、seed 0、12 s、两档 wz） | 2 策略 × 5 档 × 3 seed = 30 次 | 斜率（deg/s per deg/s 指令）与**符号**；参照官方 +18.5 deg/s（`docs/experiment_report_2026-09-17.md:117`）与现状"<2%、原地符号不对"（FACTS.md:89） |
| **B6 头作为备用转向通道** | `--head-pose-cmd 0,0,y,0`，y ∈ {−1.4,−0.7,0,+0.7,+1.4}（`:96`） | 排除"完全不存在 yaw 通道"；量化"head_yaw 指令 → 身体偏航率"增益 | 2 策略 × 5 档 × 2 seed = 20 次（12 s） | 单调性 + 斜率。FACTS.md:93 报 ck7000 在 −1.40 处 −22.93 deg/s（**原始产物未定位，见文末不确定项 #3**） |
| **B7 奖励梯度是否还活着** | `scripts/eval_reward_terms.py --onnx <ck> --ang-vel-z 0.6`（`:43`；样例用法见 `docs/q1_q2_root_cause_2026-09-16.md:765-771`） | 排除"`track_angular_velocity`（权重 2.0）在当前工作点已经死了"（高斯尾部梯度 0）。现状只拿 0.119/1.0（FACTS.md:88） | 2 策略 × 3 档 wz（不需 seed） | 报该项**原始值 / 加权值 / 占总奖励比例**；比例落在"已被证明能改变均衡"的 5–10% 区间才叫"定价了" |
| **B8 零命令静止** | 复用现成 `*_stand_forensics.json`（`gait_forensics.py`，**硬编码 1 env**，`scripts/gait_forensics.py:211`）+ 需要数值时改读 `straightness.tsv` 的 `tilt_deg` / `marching.tsv` 的 `zero_cmd_clearance_mm` | 排除"偏航是用踏步/失稳换来的"。基线：official 终件离地 0.7 mm、full 终件 20.5 mm、fwd **ck8000** 18.1 mm（FACTS.md:79-84，fwd 终件未测）⇒ **两条臂都在踏步** | 1 env × 27（现成）；要引用数值 → P64 的 `tilt_deg` 列 | 离地 <8 mm（`scripts/queue_align_sweep.sh:203`）+ tilt |

**执行顺序建议**：A0–A3（0 GPU·h）→ B1（校准地板）→ B5/B7（回答"能不能转"，最便宜且直接对应用户问题③）→ B2/B3/B4（归因偏航）→ B6 → B8。
B 组总计 **92 次** rollout（B1 4 + B2 12 + B3 12 + B4 8 + B5 30 + B6 20 + B7 6），其中大部分是 12 s 窗口（比 P64 的 26–41 s 更短）⇒ ≈ **0.4–0.7 GPU·h**，**远低于方案 A 的 46 h**，而且**结论是确定性的**（可复现、可归因），不受种子散布影响。

---

## 不确定项

1. **当前 m5 轨的跨训练种子 SD 不存在**：搜遍 `logs/rsl_rl` 下 97 个 run 目录（`logs/rsl_rl/*/*/params/env.yaml`，把 `seed:` 行归一化后哈希；命令见本文脚本运行记录），只有 4 组"同配方 ≥2 个不同种子"的 run，**全部是 `night_queue`（09-14/15，旧轨）**。§2 的 σ=13.538 / 0.1911 是跨轨外推，方向未知（判断为偏保守，但**未实测**）。
2. **配对设计的收益未知**：三臂共用 seed 42 ⇒ 若按同 seed 配对做多种子实验，方差是 `2σ²(1−ρ)/n`，而 ρ（seed 主效应占比）**无法从 1 个种子估计**。§2.7 的 n 全部按独立样本算，是上界。
3. **FACTS.md:93 的原始产物未定位**："ck7000 head_yaw −1.40 → 身体偏航 −22.93 deg/s"在任何 log/json 里都 grep 不到（`grep -rn "22\.93" logs/ --include=*.log --include=*.json` 只命中 mean reward / episode length）。B6 就是为了把它变成可引用的实测。
4. **重复评测差异的根因未定位**：#2 里同一 ONNX 同 seed 的两次跑的差异来源（全局 RNG 消耗顺序？GPU 非确定性？）没有查明；在查明前，一切"配对"统计都要带 0.30 deg/s 的地板。
5. **`logs/q1_matrix` 的策略不是 align 臂**：`U_hd` 是 `walk_hd1910_12000.onnx`（`scripts/q1_straightness_matrix.sh:60-61`），只用作"固定策略的评测噪声"参照；它的 `O_hd`/`U_xl` cell 有 47–55/64 个环境摔倒（`reset_reasons.fell_over=710`），其 path_eff 不可引用。
6. **sweep TSV 缺 `align_fwd_ck08999`**（只有 8 个 fwd 检查点），终件从 `logs/queue_align/align_fwd_walk.log:246,255` 另取；两处 yaw DC 差 0.44 deg/s（−20.673 vs −21.114 对 official 也是 0.44），量级同 #2 的重复评测噪声。
7. **检查点集合的 p 值不是干净的**：同一 run 的检查点有单调趋势（official 从 −137 爬到 −13），§1 表 #6b 的 Mann-Whitney 只能说明"分布重叠程度"，不能当独立样本检验。
8. **官方参照跨硬件/跨速度**：`logs/axis/j_official_0.2.json` 的 `overrides` 是 xl330 m6 + kp_fw 200，地面速度 0.038 m/s（我们 0.095 m/s）。"偏航 DC 越接近 0.2 越好"没有做速度归一化；仓库里**没有**现成的"每米偏航"聚合量（`lateral_per_metre` 分母是前进距离，接近 0 时爆掉）。
9. **`path_efficiency` 两个定义在用**（along/total vs along/net displacement，`AGENTS.md:11` 段明确警告过），FACTS.md:13 用中位数、:38-40 用均值；本文两套都列，但**排期判据必须二选一**（建议中位定义，与 `straightness.tsv` 的 `path_eff_med` 一致）。
10. **"旧轨更差"是假设不是实测**：13.54 是否高估当前 σ，取决于旧轨 33% 更低的力限是否真的放大了种子散布——没有对照实验。
