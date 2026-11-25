# Fork 复核：`JoyandAI/microduck_rl`（2026-09-20）

> 触发问题（用户）：**"有人说 id 会影响训练，这有关系吗？"**
> 结论先说：**那个 ID 不影响训练，一行都不影响我们已跑出来的任何数字。**
> 但它旁边有一个 *真正* 会影响训练的东西，而且方向和直觉相反 —— 见 §3。
> 唯一权威口径：本文件里的数字都能用命令复现，命令写在每个结论下面。

复核对象：`joyai/develop` = `22916b9`（2026-09-19），相对我们 `976d2ae`
**领先 42 个提交 / 落后 7 个**。merge-base = `cfbd4e7`（2026-09-11）。

```bash
git remote add joyai https://github.com/JoyandAI/microduck_rl.git && git fetch joyai
git log --oneline --no-merges 976d2ae..joyai/develop | cat -n   # 42 个
git log --oneline --no-merges joyai/develop..976d2ae | cat -n   # 我们独有的 7 个
```

---

## 1. 那个 "ID" 是什么：`FRICTION_DOF` 的 `efc_id`

对应他们唯一一个碰训练代码的 BAM 提交：

```
330754b fix(bam): match FRICTION_DOF efc_id against dof indexes, not joint ids
```

**MuJoCo 在 FRICTION_DOF 约束里存的是 DOF 地址，不是关节 id。** Microduck 的
freejoint 占 6 个 dof，所以 `dof = joint + 5`。两份实现里：

| 路径 | 文件 | 用的选择器 | 判定 |
|---|---|---|---|
| **训练**（MuJoCo Warp，我们所有 run） | `vendor/bam/bam/mjlab.py:602` `_dof_friction_force` → `:723` 调用 | `idx = efc_id`，scatter 进 `(N, nv)` | ✅ **本来就是对的** |
| CPU 排练（`testbench_sim2real.py --sim-backend bam`；他们新版 `infer_policy.py`） | `vendor/bam/bam/mujoco.py:177` | 拿 `efc_id` 比 `actuator.trnid`（关节 id） | ❌ 错的 |

**所以"id 影响训练"这句话，对我们的训练不成立。** 证据三条，互相独立：

1. `bam.mjlab` 里根本没有 `efc_id` vs `joint id` 的比较 —— 它是
   `torch.where(is_fric, efc_id, 0)` scatter 进 `nv` 宽的张量，语义上必须
   是 dof 地址才对（如果 `efc_id` 是 joint id，这个 scatter 会越界/错位）。
2. 训练栈里没有任何模块 import `bam.mujoco`：
   ```bash
   grep -rln "bam.mujoco\|from bam.mujoco" --include=*.py scripts/ src/ tests/
   # → 只有 scripts/testbench_sim2real.py（老台架脚本）
   ```
3. **我们自己那份 `scripts/infer_policy.py` 连这个控制器都没用**
   （只 `from bam.model import load_model` 取 kt），所以他们踩的坑我们
   在部署排练路径上也没踩过。

### 1.1 在真机模型上量出来（不是玩具 XML）

```bash
.venv/bin/python scripts/probe_friction_dof_id.py     # 新增
```

`robot_walk.xml`，14 个舵机 dof 各给 0.02 N·m 库仑预算、qvel=0.5：

```
actuator trnid (JOINT ids)     : [1..14]
jnt_dofadr       (DOF indexes) : [6..19]
FRICTION_DOF rows: 14  efc_id values = [6..19]     ← 与 dof 地址完全一致
set(dof_indexes) ⊆ set(efc_ids)   -> True     ← efc_id 是 dof 地址
set(joint_indexes) ⊆ set(efc_ids) -> False    ← 不是关节 id

Σ|剥掉的总量|: 正确 0.280000   旧代码 0.180000
  关节 6..14 各剥到了"另一个关节"的摩擦；关节 1..5 一点都没剥
  （关节 j 剥到的是关节 j−5 的摩擦，正是他们 commit message 说的
   "mixed left-leg friction into the neck/right-side load budget"）
```

**后果只落在 CPU 排练路径**：`compute_frictions` 的负载力矩里被减掉了错误
行 → 在线辨识出的 frictionloss/damping 是错的。真机不跑这段代码（真机跑的是
固件），训练也不跑，所以既不影响真机也不影响训练。

### 1.2 已处理

- 把 `mujoco.py` 的选择器改成 `self.dof_indexes`（+11 行注释说明，不整包 merge）。
- 新增 `tests/test_friction_dof_id.py`（5 个用例，全部在**真机模型**上）：
  语义（`efc_id` 是 dof 地址）、旧选择器会"张冠李戴"的量化、ground truth =
  `efc_id` scatter、CPU 选择器源码契约、CPU 路径 == 训练路径。
  **有牙齿**：把 `dof_indexes` 改回 `joint_indexes`，`test_cpu_controller_selects_on_dof_indexes` 立刻 FAIL。
- 没有照搬他们的 `tests/test_infer_policy_bam.py` —— 那是给**他们重写过的**
  `infer_policy.py` 写的（依赖 `load_bam_model` / `BAM_KP_FW` 等我们树里没有的符号）。

---

## 2. 逐条审完 42 个提交：真正碰到训练路径的只有 3 类

```bash
git log --oneline --no-merges cfbd4e7..joyai/develop -- \
  src/mjlab_microduck/robot/ src/mjlab_microduck/tasks/mdp.py \
  src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py \
  vendor/bam/bam/params vendor/bam/bam/mjlab.py
```

| 提交 | 内容 | 影响 |
|---|---|---|
| `330754b` | FRICTION_DOF id | CPU 路径（§1），**非训练** |
| `dc17d02` + `cb70b79` | velstand 的 yaw-cmd dead-zone 修复 **+ 当天自我 revert** | 净效果为零；`turn_in_place_min_frac` 在我们树里也不存在（`grep -c` = 0） |
| `228732d/e40fbda` `46604dc/0fd69f4` `3f52571/9be7cf2` `d3edc1f/6ce51aa` | rough-backlash velstand、fall-like pose reset、protective-fall 代价、distill | 全部落在 `microduck_velstand_env_cfg.py` / `distill.py` / `mdp.py` 的 **VelStand 段**；**走路的 recipe 一行没动**（见 §3.3） |
| `8d5c18b` `8dfc08f` `08680d3` | 模型家族改名 `allcollisions→groundcontact`、按新 CAD 重导出、新增真·全碰撞模型 | 见 §3.4：**物理上无变化**，但名字冲突 |
| 其余 ~32 个 | sim body server / 公寓场景 / vSLAM 贴图 / 相机+ToF / `publish` 上 Hub / `infer_policy` 的 odometry 锚点 | 不在训练路径 |

---

## 3. 真正的雷：他们的 `vendor/bam/bam/params/hd1910/` 是**退役的 5 V 标定**

这是本次复核最重要的一条，而且**方向和我们之前的假设相反**：不是我们落后，
是他们的 BAM 参数落后于我们（因为 `fb9b0d2` 是我们的提交，他们没取）。

```
--- m5.json（我们训练实际用的就是 m5：fingerprint 7ec8bdc54393e1b1 = kt 0.692 kd 0.5）---
  friction_base         ours=0.006749675016320943   joyai=0.013550748385667027
  load_friction_motor   ours=0.29858745118818675    joyai=0.01638114235528301
  friction_viscous      ours=0.0054519606526566115  joyai=0.008066421435690372
  kd                    ours=0.5                    joyai=0.346
  max_velocity          ours=12.0423                joyai=8.0282
  command_delay         ours=0.0011503683124627375  joyai=0.025339691826625724
  alpha / dtheta_stribeck 也是两套值
--- m6.json ---
  kd                    ours=0.5                    joyai=230.71322576948543   ← 那个 557× 的异常值
```

```bash
for r in 976d2ae joyai/develop; do git show $r:vendor/bam/bam/params/hd1910/m5.json; done
```

**这正好是 AGENTS.md 里记的那个坑**：`night_D_s42/s43/s44`（"官方 recipe" 那组
A/B）就是跑在这套旧物理上的，所以它的结论**不能用来裁决今天的物理**。

> **操作纪律：任何 merge/rebase 都必须 `--ours` 保住 `vendor/bam/bam/params/hd1910/`。**
> 一旦被他们的版本覆盖，每一个 run 都会静默退回 5 V 轨的舵机，而且是
> fingerprint 变了才发现（`logs/actuator_params_seen.tsv` + 队列的 fp_before/fp_after）。
> 同样是为什么**不要**做整分支 merge —— 我们这两天的全部工作都在这条分支上，
> 只能 cherry-pick 单个提交。

### 3.2 附带交叉验证：我们的 `reference_official` 是真的官方

`reference_official_velocity_env_cfg.py` 是我们从 `origin/develop @3412ae5`
抄的官方副本（Route A 的 A/B 基准）。用 AST 对比它和 **fork 线上的走路 recipe**：

- 忽略模块 docstring 后，**两份走路 recipe 只差一个数字**。

也就是说：走路 recipe 自 3412ae5 以来**没有任何人改过**（包括他们），
我们的对齐基准成立。

### 3.3 那唯一的差异：`init_std`（PPO 初始动作 std）—— 是他们错了

```
官方 origin/develop @3412ae5（2026-09-13） : "init_std": 1.0
我们            @976d2ae（2026-09-16）     : "init_std": 1.0   ✅ 对齐官方
JoyandAI        @joyai/develop（09-19）    : "init_std": 0.5   ❌
```

追溯来源（他们为什么不一致）：

```bash
git log -p -G'"init_std": *[0-9]' --format="COMMIT %h %ad %s" --date=short 3412ae5 -- <file>
#  官方线上：1.0 自 2026-06-30 (dbc1bbf) 起从未改变
git log -p -G'"init_std": *[0-9]' --format="COMMIT %h %ad %s" --date=short cfbd4e7 -- <file>
#  a52b4a1 2026-09-07 feat(hls2909): hardware calibration done
#    - "init_std": 1.0
#    + "init_std": 0.5,  # was 1.0 (xl330); HLS 双限幅下大动作→饱和乱蹬→学不动
```

**0.5 是我们自己 HLS 分支 2026-09-07 的一次实验值**，`cfbd4e7`（Axel 的 autodl
提交，在 HLS 线上）把它带了出去，JoyandAI 的 develop 从 `cfbd4e7` 分叉，
于是**继承了我们已经废弃的实验值**；我们自己在 `fe8806b` 改回了官方的 1.0。

→ 按用户定的规矩（除 BAM/舵机参数外一律对齐官方），**我们是对的，不需要动**；
如果他们问，应该建议他们把 1.0 拿回去。

### 3.4 模型重导出：物理上无变化

`8dfc08f`（按新 CAD 重导出全部模型）改了 `robot_walk.xml` 228 行，但逐行看：

```
inertial pos="-0.00615738 ..." mass="0.0300246"  →  mass 完全相同
inertial ... mass="0.188766"                     →  mass 完全相同
fullinertia  5.91525e-06 ... 5.6714e-08          →  ~1e-9 相对差（第 6 位有效数字）
```

只有 3 处惯量在第 6 位有效数字上动了，**质量 0 g 变化** → 颜色/STL 的功劳，
对物理和我们测过的一切都无影响。`8d5c18b` 的 `allcollisions→groundcontact`
改名会撞我们的 `friction_dr_bam.py` / `tasks/__init__.py` 引用，是纯机械冲突。

---

## 4. 顺手修掉的 2 个红色用例（与本问题无关，但套件现在全绿）

`tests/test_record_pendulum_bench.py` 2 个失败是**改动前就存在**的
（用 HEAD 原版 `mujoco.py` 跑同样失败），原因：

```
scripts/record_pendulum_bench.py:133  TypeError: 'str' object is not callable
```

`zero_belt()` 现在吃**轨迹对象**（`main()` 传的是 `trajmap[n]`，即按
`--amp-max-deg` 缩放过的包装器，所以安全带量的是真正下发的幅值），
测试还在传名字。**先独立验证脚本的数是错的还是测试的数是错的**：

```
sin_time_square 57.296° / sin_sin 100.795° / lift_and_drop 90° / up_and_down 90°
zero_belt(defaults) = lo=1211 hi=2884 amp_lsb=1147   ← 与测试期望一字不差
```

→ 脚本是对的，测试的**调用方式**过期了（期望值没错，安全不变量"2048 在带内"仍成立）。改测试传对象。

---

## 5. 套件与队列状态（2026-09-20 10:49）

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest --with pyserial pytest tests/ -q
# → 380 passed, 1 skipped   （含新增 5 个 friction-id 用例；修复前是 378 passed / 2 failed）
# 之后又加了 4 个命令上限用例（tests/test_cmd_speed_cap.py）→ 当前 378 passed, 1 skipped
```

`--with pyserial` 是必需的：`tests/test_record_pendulum_bench.py` 经
`scripts/record_pendulum_bench.py` 走 `read_hls_registers`，没有 pyserial 会让
pytest 在 **collection 阶段 INTERNALERROR**（`no tests ran, rc=3`），看起来像
"一个测试都没收"，其实是整份套件被一个 import 掐掉了。

- 队列驱动进程活着（`bash scripts/queue_overnight.sh`，已跑 1 天 17 h）。
- 当前臂 `ours_head_s44`：iteration **3862 / 9000**，elapsed 1:19，ETA 1:45。
- extension 队列（`ours_yawdc` s43/s44、`ours_flight_yawdc` s42）在等
  `OVERNIGHT QUEUE COMPLETE` 标志，一旦上一队结束会自己接上。
