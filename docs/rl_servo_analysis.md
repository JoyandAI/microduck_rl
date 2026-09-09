# Microduck RL 代码解析 —— 训练管线与舵机(BAM)相关代码

> 本仓库: 874g / 25cm 双足机器人, 14 个舵机(现役 Dynamixel XL330 → 计划换
> Feetech HL-2909-C001), mjlab(MuJoCo Warp) + rsl_rl(PPO) 训练, 50 Hz,
> 导出 ONNX 后由 `pollen-robotics/microduck` 运行时部署到真机。
> 配套可运行注释 demo: `scripts/demo_servo_action_flow.py`
> (`.venv/bin/python scripts/demo_servo_action_flow.py`)。

---

## 0. 一分钟地图

```
策略网络(ONNX/rsl_rl, 50 Hz)
   │  输入 obs: 61D = 48 本体感知 + 13D 指令块 [twist(3) | head_pose(4) | body_pose(6)]
   ▼
14D 位置目标 action ──► mjlab 动作管理器(JointPositionAction, scale=1.0)
   ▼                                          │
BamActuator.compute ──► 电压控制律 ──► DC 电机扭矩 ──► MuJoCo Warp 求解
   │  (固件 P 环 + 每 env 电压/增益/摩擦 DR)           │ (重力/接触/关节限位)
   ▼                                                  ▼
dof_frictionloss/dof_damping 写入(摩擦预算) ──► 下一拍关节位置/速度 q, dq
                                                      │
观测: joint_pos(+encoder bias) / joint_vel(1 拍延迟) / IMU / 足端接触 …
```

训练代码全部在 `src/mjlab_microduck/`,舵机的物理模型在 **vendor/bam/**(git
子包, `better-actuator-models`),两者通过 `microduck_constants.py` 的
`_BAM_ACTUATOR_KWARGS` 对接。

### 关键文件

| 文件 | 作用 |
|---|---|
| `src/mjlab_microduck/tasks/mdp.py` (7188 行) | 所有自定义 MDP: 奖励/事件/观测/指令/课程。舵机相关: `randomize_bam_friction`、`expand_bam_friction_fields`、`_servo_joint_ids` |
| `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py` | 主行走配方 + 全家族共享底座(机器人、DR、obs、指令) |
| `src/mjlab_microduck/robot/microduck_constants.py` | 机器人 cfg、HOME 姿态、**BAM 舵机执行器配置** |
| `src/mjlab_microduck/actuator/friction_dr_bam.py` | 本仓库对 bam 的扩展: 每 env 摩擦 DR + 间隙编码器 |
| `vendor/bam/bam/mjlab.py` | BAM × mjlab 桥接: `BamActuator.compute`(全部向量化, (N,J)) |
| `vendor/bam/bam/actuator.py` | 舵机控制律本体: `VoltageControlledActuator`(duty → 电压 → 扭矩) |
| `vendor/bam/bam/params/{xl330,hls2909}/` | 标定出的电机参数 (kt, R, 摩擦预算) |
| `scripts/export.py` / `scripts/infer_policy.py` | ONNX 导出(烘焙归一化) / 真机部署排练 |
| `tests/` | CPU 回归测试, 锁定不变量(如 `test_hls2909_cfg.py`) |

---

## 1. 训练管线总览

```bash
uv run list-envs                                 # 任务注册表
uv run train <TASK_ID> --env.scene.num-envs 64 --agent.max_iterations 5   # 冒烟(必做)
uv run train <TASK_ID> --env.scene.num-envs 4096                          # 正式训练
uv run play <TASK_ID> --wandb-run-path <...>                              # 回放
uv run scripts/export.py <TASK_ID> --wandb-run-path <...>  → ONNX
uv run scripts/infer_policy.py --walking out.onnx        # CPU MuJoCo 部署排练
```

- **环境**: mjlab `ManagerBasedRlEnv`,GPU 上并行 4096 个世界;`cfg.events`
  (reset/startup/interval)、`cfg.rewards`、`cfg.observations`、`cfg.terminations`
  全部由 manager 按名注册,同一份 cfg 描述符被 deepcopy 到每个世界。
- **训练**: rsl_rl PPO,每迭代 24 步/环境(`NUM_STEPS_PER_ENV = 24`)。
- **任务家族**: velocity(行走) / velstand / standup(站起) / sitstand /
  roulade(侧滚翻) / ground_pick / ball_kick / spin / roller_* / swizzle,
  以及每个的 `-Backlash-` 间隙变体(`tasks/backlash.py` 包装)。
  共享底座是 `make_microduck_velocity*_env_cfg` —— 继承它就自动继承
  DR / 观测噪声 / 延迟 / NaN 防护整套栈。

---

## 2. 舵机相关代码深度解析

### 2.1 舵机在 RL 里扮演什么角色

策略输出的 **action 就是 14 个舵机的"位置目标"**(不是扭矩,不是 PWM):
`JointPositionActionCfg(scale=1.0)`(`microduck_velocity_env_cfg.py:271-273`)。
舵机固件自己闭环(位置模式),模拟侧用 **BAM 电压控制执行器**复现
"固件 P 位置环 + DC 电机 + 齿轮摩擦 + 电压饱和"的完整动力学。

**为什么不用 MuJoCo 自带位置执行器?** 真机舵机是"电压→PWM"的电压有源
系统,反电动势、电流限制、负载相关摩擦都真实存在;用理想 PD + 定摩擦建模,
训练出的策略在硬件上会抖振/滞后。BAM 就是为此做 sim2real 的。

### 2.2 舵机配置入口: `microduck_constants.py`

```python
# 见 microduck_constants.py:125-138(当前值, hls2909 参数为占位)
_BAM_ACTUATOR_KWARGS = dict(
    motor_name="hls2909",            # 舵机型号(was "xl330")
    model="m1",                      # BAM 摩擦模型档位(was "m6"; 先低阶, 标定后升级)
    target_names_expr=(r"^(?!passive_).*",),  # 只驱动"主动"关节(见 2.6)
    kp_fw=200.0,                     # 固件位置环 P 增益
    vin_range=(10.5, 12.6),          # 每 env 上电时采样的"电池电压" → 电压 DR
    vin_drop_gain_range=(0.0, 0.15), # 负载压降: V_drop = gain * Σ|τ_上一拍|
    vin_min=10.0,                    # 压降后的电压下限(欠压)
    delay_min_lag=3, delay_max_lag=6, # 总线延迟 DR(60-120 ms @50Hz)
)
actuators = FrictionDRBamActuatorCfg(**_BAM_ACTUATOR_KWARGS)
```

要点:

- **执行器类型是 `FrictionDRBamActuatorCfg`** —— 本仓库的扩展(见 2.4),
  在标准 BAM 上加了每环境 `friction_scale` 与(间隙模型用)编码器反馈。
- **`target_names_expr=(r"^(?!passive_).*",)`**: 被动关节(滚轮、间隙铰、
  颌部连杆)全部命名为 `passive_*`;正则排除它们,动作空间保持 14 维。
- **电压 DR 模拟电池老化/批次差异**: `vin` 在 `initialize()` 时每 env 采样
  一次、**不随 reset 重采样**(mjlab.py:336-343, 注释明确写了这一点);
  电压还会按上一拍电机扭矩之和"塌陷"(模拟电源内阻), 有 `vin_min` 下限。
- `kp_fw=200` 目前是**占位值**(代码注释: 示波器/出厂寄存器 21 标定后定,
  换算 200×0.166/8 ≈ 4.15 duty/rad);HLS 换装后大量数值待标定,
  **标定前禁止长训**(见 `docs/actuator_physics_swap_plan.md` 的 G1–G4 门)。

### 2.3 桥接层: `vendor/bam/bam/mjlab.py` — `BamActuator.compute`

每个控制周期 mjlab 把 `(N 环境, J 关节)` 张量喂给 `compute(cmd)`,
**五步流水线**(mjlab.py:580-675):

```
① 每环境电压   vin = vin_tensor(上电采样) − vin_drop_gain·Σ|τ_prev|, 钳到 ≥vin_min
② 固件控制律   duty = clip((q_target − q)·kp·kp_scale·error_gain, 限流窗口, ±max_pwm)
               电压 V = vin·duty                     ← bam/actuator.py compute_control
③ 电机扭矩     τ = kt·V/R − kt²·dq/R(反电动势, 即"电气阻尼")
                                                  ← bam/actuator.py compute_torque
④ 摩擦预算     frictionloss = f(基础摩擦, Stribeck, 负载相关项)(N,J)
               viscous 项 → dof_damping; 预算 → dof_frictionloss
⑤ 返回 τ → MuJoCo 求解器做静摩擦削顶(摩擦不是加在扭矩上, 而是求解器原生的
              dof_frictionloss 约束, 与真机"静摩擦锁死"行为一致)
```

关键设计点:

- **摩擦写在 `dof_frictionloss/dof_damping` 里,而不是被动力矩**:
  求解器会正确地处理"|净力矩| ≤ 摩擦预算 → 关节钉住"的静摩擦切换
  (BAM Algorithm 1),这正是 2.4 说的 DR 陷阱的根源。
- `edit_spec`(mjlab.py:234-293)把 XML 里的 position 执行器统一改成
  **motor 模式**,置 `forcerange=±vin_max·kt/R`(扭矩上限由电压上限推算),
  `frictionloss=0, damping=0` —— BAM 每拍自己重写这两个场。
- 所有参数 `(N,1)` 广播、`TorchBackend` 向量化 `(N,J)`,4096 env 全并行。

### 2.4 本项目扩展: `src/mjlab_microduck/actuator/friction_dr_bam.py`

```python
class FrictionDRBamActuator(BamActuator):        # friction_dr_bam.py:28
    def initialize(...):
        # 仿照 kp_scale: 每环境一个摩擦乘数 (N,1)
        self.friction_scale = torch.ones_like(self.kp_scale)
        self.default_friction_scale = self.friction_scale.clone()

    def _compute_friction_budget(...):           # 覆写摩擦预算
        base = super()._compute_friction_budget(...)
        return base if fs is None else base * fs   # (N,J) * (N,1)

    def set_friction_scale(env_ids, s): ...      # reset 事件用
    def reset_friction_scale(env_ids): ...       # 恢复 1.0, 防累积
```

**为什么必须有这个扩展?** BAM 在 `edit_spec` 里把 `dof_frictionloss` 清零,
自己每拍从参数算摩擦 —— 所以 mjlab 标准的 `dr.dof_frictionloss` 是**静默
no-op**。给真机做"每台舵机摩擦不同"的域随机化,只能走 BAM 内部的摩擦预算。

调用方是 `mdp.py:3212 randomize_bam_friction`(reset 事件):

```python
# mdp.py:3236-3240 —— 先恢复标称再采样, 保证"不累积"
actuator.reset_friction_scale(env_ids)                        # → 1.0
samples = torch.rand(len(env_ids), 1) * (hi - lo) + lo
actuator.set_friction_scale(env_ids, samples)
```

配置在 `microduck_velocity_env_cfg.py:471-483`:
`JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)`(±10%)。
同类还有**增益 DR** `randomize_delayed_actuator_gains`(mdp.py:3180,
kp ±15% / kd ±10%,同样先 `reset_gains` 再 `set_gains`)。

另一个扩展 `BacklashEncoderBamActuator`(friction_dr_bam.py:64):对带
`passive_<joint>_backlash` 铰(±1° 间隙)的模型,**固件 PD 读取的是
"主轴 + 间隙铰"的合计角度**,复现"舵机在空行程里扫过时编码器读数不变、
PD 误差不发散"的物理。`cmd.pos + qpos[backlash]·mask`,`cmd.vel` 故意留在
电机侧(反电动势/摩擦属于转子物理,不是固件编码器信号)。

### 2.5 控制律本体: `vendor/bam/bam/actuator.py` — VoltageControlledActuator

```python
# actuator.py:251-267
duty_cycle = (q_target - q) * self.kp * self.error_gain     # P 位置环
if self.max_current is not None:                             # 固件限流 0.6A
    back_emf  = self.model.kt.value * dq
    span      = self.model.R.value * self.max_current / vin
    center    = back_emf / vin                               # 反电动势抬升窗口中心
    duty_cycle = clamp(duty_cycle, center-span, center+span) # 限流"尝试"
duty_cycle = clamp(duty_cycle, -max_pwm, max_pwm)            # 物理 PWM 上限(最后)
return vin * duty_cycle                                       # 电压[V]
```

```python
# actuator.py:289-292
torque = kt·V/R − kt²·dq/R        # DC 电机方程, 反电动势项=电气阻尼
```

- `error_gain`: 把 "kp × 误差" 换算成占空比(依赖编码器分辨率与增益缩放),
  HLS 占位 0.166;注释 `200×0.166/8 ≈ 4.15 duty/rad` 说明固件内部还有 1/8 缩放。
- **限流窗口的物理**: 固件只能钳 PWM,不能合成任意电压;高速时反电动势把
  窗口整体推到 `max_pwm` 之外,电流限制"够不着" —— 和真机行为一致。
- 注意 HLS 固件还有 **P-D-I 位置环**(寄存器 21/22/23 → 50/51/52,
  Kp 1/8、Kd 1/4 缩放)和**目标速度/加速度双限幅**(寄存器 41/46-47)。
  当前 BAM 电压类只建了 P + 电气阻尼(等效 D) + 我们的 demo 里近似了双限幅;
  **这是换装后最大的 sim2real 风险**,见 `docs/actuator_physics_swap_plan.md`。

### 2.6 关节布局与"servo 视角"助手: `mdp.py`

14 个舵机: **0–4 左腿**(hip_yaw, hip_roll, hip_pitch, knee, ankle)/
**5–8 头颈**(neck_pitch, head_pitch, head_yaw, head_roll)/
**9–13 右腿**。凡是有被动关节的模型(滚轮、间隙、颌部)关节数组会**交错**,
所以奖励/观测**一律不写死索引**,必须走助手:

```python
# mdp.py:126-155
def _servo_joint_ids(env, asset) -> list:
    """^(?!passive_).* 正则选出"舵机"关节, 缓存; 纯模型上就是恒等映射。"""
    ids, _ = asset.find_joints(r"^(?!passive_).*")
def _servo_joint_pos(env, asset):   # (N, 14) 的舵机视角位置
```

同模块还有 `_servo_joint_vel` / `_servo_default_joint_pos`,以及
`expand_bam_friction_fields`(mdp.py:3198)这个"无操作"启动事件 ——
**它的存在只是为了触发 `@requires_model_fields("dof_frictionloss","dof_damping")`
装饰器**,让 mjlab 把这两个场按世界数展开(BAM 每拍写每 env 的值,不展开就
所有 world 共享一个 buffer,写穿)。任何 standalone env 手工搭起来时漏掉它,
BAM 摩擦 DR 会静默失效 —— AGENTS.md 把这条列在不变量里。

### 2.7 观测侧的舵机细节: `microduck_velocity_env_cfg.py`

- **`joint_pos`/`joint_vel` 用 `^(?!passive_).*` 过滤**(cfg:620-624),
  使观测维 = 动作维 = 14(否则咬合连杆等会撑到 16D, 破坏 61D 契约)。
- **编码器偏差 DR**(cfg:630-633):每 env 一个恒定角度偏差(±0.86°),
  **actor 看到"有偏"的位置, critic 看真值** —— 模拟批量舵机零点不一致,
  同时给 critic 特权信息(AGENTS.md: "奖励必须和观测同视角")。
- **`joint_vel` 延迟 1 个控制周期**(cfg:604-613):Dynamixel 固件的
  present_velocity 是前一个位置采样窗内的滑动平均,策略读到的速度天生滞后
  一拍 —— 不建模这个,策略会依赖瞬时 qdot 反馈,而真机没有。
- **观测噪声**(cfg:587-590)压得很小(joint_pos ±0.001 rad / joint_vel
  ±0.25 rad/s),因为真机码盘相当干净;噪声过大只会浪费策略容量。
- **50 Hz 控制周期**(`dt=0.02`)与运行时一致。

### 2.8 部署链路(为什么 sim 里能跑不代表真机能跑)

```bash
uv run scripts/export.py <TASK_ID> --wandb-run-path <...>   # ONNX + 烘焙归一化
uv run scripts/infer_policy.py --walking out.onnx           # CPU MuJoCo 排练
```

- **归一化必须烘焙进 ONNX**(`export.py` 做);自己手转 checkpoint 会在真机
  上静默失败(sim 内 play 会掩盖,因为 play 也应用归一化)。
- 运行时 50 Hz 把 `[twist(3) | head_pose(4) | body_pose(6)]` 13D 指令块
  写进 obs;`twist.vx` 槽里还编码"站/走"姿态旗标 —— 喂全零 = "站立",
  看起来像"策略无视按钮"。
- 部署前用 `infer_policy.py --walking` 排练(AGENTS.md 铁律)。

---

## 3. 注释实例 demo: `scripts/demo_servo_action_flow.py`

运行: `.venv/bin/python scripts/demo_servo_action_flow.py`
(输出首行的 `[WARN] Failed to load task package ...` 是 mjlab 入口点扫描的
既有噪音,与本 demo 无关,见脚本 docstring。)

三个部分:

1. **Part 1 单舵机 60° 阶跃** —— 用**真实的** `VoltageControlledActuator`
   (hls2909 m1 参数, kp_fw=200, 0.6A 限流, 12V)逐拍打印
   duty/电压/电流/扭矩, 实现"限流起步 → 满速加减速 → 电气阻尼减速 →
   静摩擦死区"全过程。**能直接看到**: 起步电流压在 0.6A、
   误差大时 duty 饱和 0.97、收敛后的残余极限环(P-only + 50Hz 离散化的
   欠阻尼体现 —— 真机靠固件 D 项抑制,这正是 sim 需要建模它们的原因)。
2. **Part 2 并行训练视角** —— N=4 env × J=14 关节形状下复现
   `BamActuator.compute` 的每环境 DR:(a) 上电电压采样 (b) kp_scale
   (c) **调用真实的 `FrictionDRBamActuator._compute_friction_budget`**
   展示 `friction_scale` 乘在摩擦预算上 (d) 对照旧 XL330 m6 模型:
   预算里 Stribeck + 负载相关项随速度/载荷变化(低速摩擦更高)。
3. **Part 3 数据流图** —— action → 舵机 → 状态 → obs 的文字版链路,
   每行标注真实文件:行号。

典型输出节选(Part 1):

```
 t(ms)   q(deg)  dq(rad/s) err(mrad)   duty voltage current  torque  备注
     0    0.885      0.772    1031.8  0.970   11.64   0.582   0.846  起步: 限流 0.6A 生效
   320   66.062      3.767     105.8 -0.450   -5.40  -0.600  -0.872  限流: I 被压在 0.6A 附近
   960   60.877     -0.433      15.3 -0.796   -9.55  -0.502  -0.730  运动
  1180   59.934      0.051       1.1  0.072    0.87   0.049   0.071  运动
✗ 有残留极限环(欠阻尼 P 控制律 + 50Hz 离散化)
```

---

## 4. 常见坑(与舵机直接相关)

| 坑 | 原因 | 正确做法 |
|---|---|---|
| `dr.dof_frictionloss` 无效 | BAM 在 `edit_spec` 清零该场, 每拍自算 | 用 `randomize_bam_friction` → `friction_scale` |
| 漏注册 `expand_bam_friction_fields` | 摩擦场未按世界展开, 写穿共享 buffer | 所有 standalone env 注册该 startup 事件 |
| DR 累积 | 每 reset 只 `set_*` 不 `reset_*` | 先恢复标称(1.0/默认增益)再采样 |
| 关节索引错位 | 间隙/滚轮模型关节交错 | 只用 `_servo_joint_ids` / `^(?!passive_).*` |
| 直接用 mjlab 模板建 env | 缺 encoder-bias/IMU 失准/延迟/NaN 防护 | 建立在 `make_microduck_velocity*_env_cfg` 上 |
| 手转 ONNX checkpoint | 归一化没烘焙 | 只走 `scripts/export.py` |
| HLS 参数未标定就长训 | kt/R/kp_fw/电压全都是占位 | 等 testbench 标定(gate 文档 G1-G4) |

---

## 5. 下一步建议

1. **标定 HLS-2909**: 按 `docs/actuator_physics_swap_plan.md` G1–G4 门。
   注意两处现状更新(2026-09):
   - **双限幅已在 bam 实现**: `vendor/bam/bam/feetech/actuator.py` 的
     `HLS2909Actuator`(目标二阶爬坡 + P(1/8) + D(1/4) + 0.6A 限流 + PWM 钳位)
     已落地并注册(`actuators.py:32`),不需要再写代码,只需把
     寄存器 84/85/86、21/22/23 的**读了的值**填进 JSON/`kp_fw`。
   - **G2 可以软件化**: 固件反馈寄存器 `Present_Load(60)` = 当前占空比(0.1%),
     `Present_Current(69)` = 当前电流(6.5mA)—— error_gain/max_pwm 可**纯软件
     标定**(给已知误差读占空比),kt 可用砝码力矩 ÷ 电流反解,示波器不再是
     硬门槛(寄存器表交叉验证: `lerobot_xlerobot/src/lerobot/motors/feetech/tables.py`
     的 `HLS_SERIES_CONTROL_TABLE`,比仓库 memtable 更全,含 25/34/35/37/39/50-52)。
2. **摆锤标定不可省**: R 真值 / armature / 摩擦全家桶(见 §6.3)没有寄存器可读,
   仍需 `bam.fit`(单舵机阶跃 MAE <10% 峰值时间误差验收)。
3. 标定后把 `_BAM_ACTUATOR_KWARGS` 的占位值替换为实测值, 跑冒烟测试
   (64 env × 5 iter) 再开长训。

---

## 6. BAM 公式详解与参数敏感性 —— 为什么每个参数都要标定

### 6.1 三层方程链(每条都能对到代码)

BAM 把真舵机拆成三层,每层一组参数;策略在 sim 里"感受"到的是三层方程的
复合映射 `action(位置目标) → 下一拍 (q, dq)`。

**(①) 固件位置环(命令 → 电压)** — `bam/feetech/actuator.py:159-211`(HLS)
与 `bam/actuator.py:251-269`(基类):

```
目标二阶爬坡(HLS 固件特性, XL330 没有):
    v_t = clamp((q_cmd − q_smooth)/dt, ±v_max)              ← max_velocity
    v_t = clamp(v_t,  v_s−a_max·dt,  v_s+a_max·dt)          ← max_acceleration
    v_t = clamp(v_t, ±|q_cmd − q_smooth|/dt)                ← 末段"缓停"约束
    q_smooth += v_t·dt

固件 P-D 位置环(官方内存表: Kp 1/8 缩放, Kd 1/4, Ki 位置模式无效):
    duty_raw = (q_smooth − q)·k_p·error_gain/8 + (−dq)·k_d·error_gain/4
                  └─ kp_fw ──┘└─ error_gain ─┘       └─ kd ─┘

固件限流(只能钳 duty, 由 I=(duty·vin − kt·dq)/R 反解):
    duty ∈ [ kt·dq/vin − R·I_max/vin ,  kt·dq/vin + R·I_max/vin ]   ← max_current

物理 PWM 钳位:  duty ∈ [−max_pwm, +max_pwm]                        ← max_pwm
输出电压:      V = vin_eff·duty                                    ← vin
```

**(②) DC 电机方程(电压 → 扭矩)** — `bam/actuator.py:289-292`:

```
     V − kt·dq            kt·V        kt²·dq
I = ──────────    τ = kt·I = ─────── − ───────
        R                        R         R
                        └ 驱动项 ─┘ └─ 反电动势=电气阻尼 ─┘
```

**(③) 摩擦预算(扭矩 → 净运动)** — `bam/mjlab.py:425-479` + `:661-669`:

```
τ_fric,budget = friction_base                                  ← m1(全部)
              + S(dq)·friction_stribeck                        ← m2
              + |τ_ext·L_ext − τ_mot·L_mot|                    ← m3/m5 负载相关
              + S(dq)·|τ_ext·L_ext,st − τ_mot·L_mot,st|        ← m4/m6
              + (m6: 按"驱动/背驱"分段的二次项 τ²)
写回: dof_frictionloss = budget × friction_scale ; dof_damping = friction_viscous
求解器: |所有其他力矩| ≤ budget → 关节锁死; 否则按 budget 削顶(静摩擦)

Stribeck 衰减系数:   S(dq) = exp( −(|dq|/θ_stribeck)^α )       ← 低速≈1, 高速≈0
```

**(④) 电气环境** — `bam/mjlab.py:601-607` + `microduck_constants.py:135-136`:

```
vin_eff = vin − gain_drop·Σ|τ_prev| , 钳到 ≥ vin_min        ← vin_range/vin_drop_gain_range/vin_min
指令延迟 = delay_min_lag ~ delay_max_lag 拍(@50 Hz = 60~120 ms)
```

**(⑤) 训练时的 DR 乘数层** — `friction_dr_bam.py:43-47` / `mjlab.py:400-419`:

```
budget × friction_scale (0.9–1.1)      kp_eff = kp_fw × kp_scale (0.85–1.15)
```

### 6.2 派生物理量 —— RL 真正"感觉"到的量

| 派生量 | 公式 | 占位 R=20Ω | 参考拟合 R=2.68Ω | 影响 |
|---|---|---|---|---|
| 位置刚度 K | `kt·vin·kp·error_gain/(8R)` | **3.62** Nm/rad | **27.0** Nm/rad(7.5×) | 同样误差 → 多大反向扭矩 → 闭环"软/硬" |
| 电气阻尼 c | `kt²/R` | 0.106 Nm·s/rad | 0.789(7.5×) | 高速自减速能力、振铃程度 |
| 限流扭矩 | `kt·I_max` | 0.873 N·m | 同(受 kt 影响) | 饱和扭矩(9 kg·cm) |
| 力矩上限 | `vin_max·kt/R`(`edit_spec`:249-254) | 0.92 N·m | — | MuJoCo forcerange |
| 关节惯量 | 连杆 + `armature`(`edit_spec`:269) | +1e-4 | +0.0284(≈284×) | 带宽 `ω≈√(K/I)`、加速度上限 |

**数值算例(demo Part 1 的起步拍, 全部照公式算)**:
```
Δq = 1.047 rad(60°), dq = 0:
duty_raw = 1.047·200·0.166/8 = 4.35 → 限流窗口 [−1.0, 1.0] → 1.0
        → PWM 钳位 0.97 → V = 12·0.97 = 11.64 V
I = (11.64 − 1.454·0)/20 = 0.582 A          (注意: 0.582 < 0.6 ——
        PWM 上限 0.97 使电流够不到 0.6A, 0.582 = 0.6×0.97 恰好是"限流×钳位"乘积)
τ = kt·I = 1.454·0.582 = 0.846 N·m           (vs 摩擦预算 0.05 → 必动, 起步加速)
```
**振铃解释**: 占位 R=20Ω 下, 50 Hz 离散 P-only 回路 ζ≈0.31(欠阻尼);
换成拟合 R=2.68Ω 后 ζ≈0.58(合理阻尼)—— 这正是 demo 里"残留极限环"的根源,
也是为什么"占位参数训练出的策略"会为不存在的振荡付出代价。

### 6.3 参数 → 方程 → 后果 全表

| 参数 | 代码位置 | 出现在 | 占位值 | 参考标定值 | 错了会怎样 |
|---|---|---|---|---|---|
| `kt` | feetech/actuator.py:138 / m1.json | ②驱动项、②阻尼、限流窗口 | 1.454 | 1.454(锚点: 0.873Nm/0.6A) | 扭矩包络整体缩放 |
| `R` | :140 / m1.json | ②全程、限流窗口 | 20(限流等效!) | 2.68(STS3215 拟合) | K 与阻尼 **7.5×**; 占位下 sim 明显欠阻尼 |
| `armature` | :142 / m1.json | 关节惯量 | 1e-4 | 0.0284 | 带宽差 ~300×; 策略依赖不存在的瞬时加速度 |
| `q_offset` | :143 / m1.json | 零点 | 0 | −0.051(STS3215) | 零点偏置 → 常值力矩对抗 |
| `friction_base` | m1.json | ③ | 0.05 | 0.052(STS3215); 旧XL330=0.0048 | 死区大小; 换舵机后**量级×10** |
| `friction_viscous` | m1.json | ③粘性 | 0.06 | 0.059 | 运动阻尼, 自由滑/迟缓 |
| `friction_stribeck` 等 | m6.json(待) | ③ | — | 见 xl330/m6 | 低速咬死/脱阻行为; BAM 论文 m6 比 m1 误差降 >50% |
| `error_gain` | :145 + JSON | ① | 0.166 | G2 标定 | K 线性失真 ← 可软件标定(见 §5) |
| `kp_fw` | constants.py:129 | ①P 项 | 200 | 寄存器 21 | K 线性失真 |
| `kd` | :146 + JSON | ①D 项 | 0(无 D 项!) | 寄存器 22/51 | 真机有 P-D-I; sim 振铃而真机稳定 |
| `max_velocity` | :147 + JSON | ①爬坡 | 8.06 | 寄存器 84 | **头号 sim2real 风险**: 命令相位滞后 |
| `max_acceleration` | :150 + JSON | ①爬坡 | 38.6 | 寄存器 85×86 | 同上(缓起缓停) |
| `max_pwm` | :122(类内) | ①钳位 | 0.97 | G2 示波器/软件 | 速度扭矩上限 |
| `max_current` | :123(类内) | ①限流 | 0.6 | 官网锚点 | 饱和扭矩; 已是最优锚点 |
| `vin_range` | constants.py:131 | ④ | (10.5,12.6) | 整机实测 | τ ∝ vin; 两端差 20% |
| `vin_drop_gain_range` | :132 | ④ | (0,0.15) | 线阻实测 | 负载压降 → 大扭矩时欠压 |
| `vin_min` | :133 | ④ | 10.0 | 系统欠压 | 压降下限 |
| `delay_*` | :135-136 | ④ | 3-6 拍 | G4 总线实测 | 策略命令到达时刻 |

### 6.4 为什么不能手填, 必须标定拟合

1. **参数强耦合**: `kp·error_gain/8` 只以乘积出现、`kt²/R` 同时出现在驱动项与
   阻尼项、`friction_scale` 乘在整个预算上 —— 手改一个必然破坏其他,
   且无法"按规格逐个对齐"(规格不提供这些系数)。
2. **规格值可能是"限流假象"**: R=20Ω 由 "12V/0.6A" 推出, 但 0.6A 是**固件
   限流值**, 不是绕组特性 —— 真实 R 只能实测(STS3215 拟合结果 2.68Ω 印证)。
3. **无寄存器可读的四类**: 绕组 R / 转子惯量 armature / 齿轮摩擦 / 固件 duty
   斜率 error_gain —— 前三个是纯物理, 第四个是固件黑盒,
   只能靠"摆锤阶跃数据 + 电流/占空比反馈"反演。
4. **验收标准**: `scripts/testbench_sim2real.py` + `validate_bam_testbench.py`
   单舵机 60°/120° 阶跃, sim-vs-真机 **MAE < 10% 峰值时间误差**; 达到后
   `bam.fit` 产出的全套参数才能写进 `m1.json`(再升 m6)。

> 一句话总结: 每条公式里的每个系数 = 标定清单; PPO 学的是"动作→状态"的
> 动力学映射, 公式里有系数错了, 策略在真机上就是错的。
