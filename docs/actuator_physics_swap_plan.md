# 执行器物理修改计划 —— HL-2909-C001（bam 层）

> 分支：`feat/feetech-hls2909`。本文档是"③ 执行器物理"的落地计划（代码级）。
> 依据：官方内存表 `docs/feetech_hls_memtable.md` + 协议文档 `docs/ftscs_protocol.md` +
> 规格页（`docs/feetech_hls2915_servo_swap.md`）。
> 结论先行：**策略/观测/动作接口零改动；换的是 sim 的电机+固件动力学模型，必须在
> sim 重训之前落地并标定。**

## 0. 目标与验收

- 目标：sim 中 14 个关节的动力学 = HL-2909-C001 真机（电压控制律 + 固件
  "加速度→速度→位置"双限幅 + P-D 位置环 + 0.6A 限流 + 12V 供电）。
- 验收：
  1. `pytest tests/` 全绿（新增 `test_hls2909_cfg.py`）；
  2. 单舵机 testbench 阶跃（60°/120°，ACC=0，SPEED=max）sim-vs-真机时间曲线
     MAE 达标的判定值在标定时定（参考 STS3215 标定精度，目标 <10% 峰值时间误差）；
  3. 冒烟（64 envs × 5 iters）无 NaN、61D obs、forcerange=±kin/R 与 0.873 N·m
     同量级；
  4. 训练评估：新指令范围下可达速度达成（§doc: 达速扫描）。

## 1. 依赖决策：bam 改在哪（必须先定）

bam 是 git 源依赖（`better-actuator-models = {git=Rhoban/bam, branch=mjlab_frictionloss}`）。
新执行器类必须改 bam 源码，三选一：

| 方案 | 做法 | 优缺点 |
|---|---|---|
| **A. 本地 vendor（推荐）** | `git clone` bam 到 `vendor/bam/`，pyproject 改为 `{path="vendor/bam", editable=true}` | diff 全在本仓库、可审可回滚；uv sync 会重建 venv（几分钟） |
| B. fork GitHub | fork 后改 `[tool.uv.sources]` 指向 fork | 依赖用户 GitHub 组织和网络 |
| C. 上游 PR 先行 | 代码写好提 PR 等合入 | 阻塞、不推荐先做 |

> 本计划按 **A** 编写；B/C 仅需替换 pyproject 一行。

## 2. bam 包改动（文件级）

### 2.1 `bam/feetech/actuator.py`：新增 `HLS2909Actuator(VoltageControlledActuator)`

对照现有 `STS3215Actuator`（同文件）来实现：

```python
class HLS2909Actuator(VoltageControlledActuator):
    """Feetech HL-2909-C001 (12V 恒力舵机, 位置伺服模式)。"""

    def __init__(self, testbench_class):
        super().__init__(
            testbench_class,
            vin=12.0,              # 官网 9-14V 标称 12V
            kp=None,               # ← 标定: 出厂寄存器 21(EPROM P) 读取值, 占位 40
            error_gain=None,       # ← 标定: 示波器 duty/error 斜率, 占位 0.16
            max_pwm=None,          # ← 标定: 示波器, 占位 0.97
        )
        self.max_current = 0.6     # 官网: 堵转电流 0.6A@12V (= 内存表 28 保护电流)
        # 固件运动学限幅: 出厂参数区 84/85/86 (只读)
        #   84 最大速度限制     → max_velocity [rad/s]  (0.732 RPM/LSB 换算)
        #   85×86 加速度限制×倍数 → max_acceleration [rad/s²] (8.7°/s²/LSB 换算)
        self.default_max_velocity = None   # ← 标定后填
        self.default_max_acceleration = None  # ← 标定后填

    def get_extra_inertia(self): return self.model.armature.value

    def initialize(self):
        # 电机参数 (出厂参数区 77-86 锚点 + testbench 拟合)
        self.model.kt = Parameter(1.454, 0.5, 3.0)   # 官网 KT=14.83 kg·cm/A
        self.model.R = Parameter(20.0, 5.0, 40.0)    # 12V/0.6A 等效, 待拟合
        self.model.armature = Parameter(1e-4, ...)   # ← 标定 (STS3215 拟合 0.0284, 跨度大)
        self.model.q_offset = Parameter(0.0, ...)
        # 摩擦预算参数随 bam.fit 产出 (先 m1: friction_base/friction_viscous)
        # 新增: 双限幅参数
        self.model.max_velocity = Parameter(self.default_max_velocity, 0.1x, 10x)
        self.model.max_acceleration = Parameter(self.default_max_acceleration, 0.1x, 10x)

    def compute_control(self, q_target, q, dq, dt):
        """重写: 固件位置伺服 = 二阶目标爬坡 + P(1/8)·D(1/4) + 限流 + PWM 钳位。"""
        # 1) 目标二阶爬坡 (加速度限幅 → 速度限幅), 状态 q_target_smooth & v_target_smooth
        #    参照 STS3215 的 q_target_smooth, 但二阶: 先爬速度再爬位置
        # 2) duty = ((q_target_smooth - q) * kp/8 + (dq_target - dq) * kd/4) * error_gain
        #    (官方: Kp 1/8 缩放, Kd 1/4 缩放, Ki 位置模式无效)
        #    kd 为新的类属性 → 从 model 读 (位置环 D 系数, 出厂 22/51)
        # 3) 电流限幅: 沿用基类 max_current 的 duty_span 钳位逻辑 (I=(duty·vin-kt·dq)/R)
        # 4) duty = clamp(duty, -max_pwm, +max_pwm)
        # 5) 返回 vin * duty
```

**关键实现细节**：
- `q_target_smooth` 状态用 torch tensor（backend 是 `TorchBackend`，形状
  `(num_envs, num_joints)`）：`STS3215Actuator` 用 numpy，需换成按 backend
  clamp 的 tensor；在 `compute_control` 里做 `self.backend.clamp(...)` 逐拍积分。
- D 项用 `dq`（当前速度）即可（目标速度爬坡状态可作为可选项，先做
  `d = -dq * kd/4 * error_gain` 的简化版，阶跃对比后决定是否升级）；
- `model.max_velocity / max_acceleration` 用 `Parameter` 以便与 DP 一致；
- **标定值全部以 `None`/注释占位**，未标定前禁止跑训练（沿用 §4 的 gate）。

### 2.2 `bam/actuators.py`：注册

```python
from .feetech.actuator import HLS2909Actuator  # 同文件已有 STS3215Actuator
actuators["hls2909"] = lambda: HLS2909Actuator(Pendulum)
```

### 2.3 `bam/params/hls2909/`：参数 JSON（新建目录）

- 先做 **m1**（只需 `kt/R/armature/q_offset/friction_base/friction_viscous`，
  数据量小、参数少），模板 = `feetech_sts3215_7_4V/m1.json` +
  `"actuator": "hls2909"`；
- 跑通后再按标定数据升 m6（需要摆锤+阶跃数据集，BAM `bam.fit` 标准流程）；
- 未标定前的占位文件内容用**官方锚点值并标注 `# TODO: fitted by testbench`**：
  `kt=1.454, R=20.0, friction_base≈0.05, friction_viscous≈0.06`（参照 STS3215 m1
  的量级, 金属齿 320:1 摩擦显著大于 XL330 的 0.0048）。

## 3. 本仓库接入改动（`microduck_constants.py` 第 120-133 行）

```python
_BAM_ACTUATOR_KWARGS = dict(
    motor_name="hls2909",          # ← 从 "xl330" 改
    model="m1",                    # ← 从 "m6" 改 (先低阶, 标定后再升)
    kp_fw=None_probe,              # ← 标定后填 (出厂 21 读取, 1/8 缩放反解)
    vin_range=(10.5, 12.6),        # ← 待整机供电实测 (规格 9-14V)
    vin_drop_gain_range=(0.0, 0.15), # ← 待实测, 初值按 12V 系统比例
    vin_min=10.0,                  # ← 12V 系统欠压下限
    delay_min_lag=3, delay_max_lag=6,  # ← 待总线实测 (初值沿用)
)
```

## 4. 实测数据获取（计划的前置 GATE，缺一不可）

| GATE | 数据 | 获取方式 | 阻塞哪一步 |
|---|---|---|---|
| G1 | 出厂寄存器：0-4 版本、21/22/23 PID 默认、16/28、**77-86**（vFk/vKgI/pFk/DTs/eFk/Vk/速度限制/加速度限制/倍数） | `scripts/read_hls_registers.py`（待写，USB-TTL 接 1 个舵机） | kp/kd/max_velocity/max_acceleration 初值 |
| G2 | 示波器：duty vs error 斜率、最大占空比 | 示波器（同 STS3215 标定法） | error_gain、max_pwm |
| G3 | testbench 摆锤 + 阶跃（60°/120°） | `scripts/testbench_sim2real.py`（需参数化 motor）+ `validate_bam_testbench.py` | kt/R/armature/摩擦/阶跃验收 |
| G4 | 总线延迟 | 真机 14 舵机回读计时 | delay_*_lag |

G1 不依赖示波器/摆锤，可最先做；**G2/G3 是质量保证关键，未完成前只冒烟不训练。**

## 5. 标定管线改造（仓库内脚本）

| 文件 | 改动 |
|---|---|
| `scripts/testbench_sim2real.py`（120/132/314/433/461/472/567 行硬编码 xl330） | 加 `--motor hls2909`（默认保持 xl330 兼容），kp_fw 从参数文件读 |
| `scripts/validate_bam_testbench.py`（24/30/89/151） | 同参数化 + 阶跃判据对 HLS 用 77-86 的速度/加速度限制值 |
| `src/mjlab_microduck/robot/testbench_constants.py` | 新增 `hls2909_test_bench`：摆锤几何/质量按 27.8g 舵机 + 120g 臂重；`motor_name="hls2909"` |
| `scripts/infer_policy.py:892`（`load_model(xl330,m6)`）+ `--current-limit` | kt 来源改 `hls2909`；默认 `--current-limit 0.6`（0.6A×1.454≈0.87 N·m） |

## 6. 分阶段实施

| 阶段 | 内容 | 依赖 |
|---|---|---|
| P1 骨架 | vendor bam（或 fork）→ 2.1/2.2/2.3 代码+占位参数 → 单元测试（参数加载、双限幅数值、forcerange） | 无 |
| P2 读数 | `scripts/read_hls_registers.py` 写+跑（需 1 个舵机+USB-TTL）→ 填 G1 初值 | 硬件 |
| P3 标定 | testbench 参数化 + 示波器 + 摆锤 → `bam.fit` 产出真实 params → 填 G2/G3 | 硬件 |
| P4 接入 | `microduck_constants.py` 接入 + 冒烟 + settle 测试 + 新增 `tests/test_hls2909_cfg.py` | P2/P3 |
| P5 验证 | 达速扫描定指令范围 → 重训 → 导出 → infer 演练 → 真机对照 | P4 |

> 质量/重心（②）与运动学（① 无需改）见 `mechanical_swap_requirement.md`，
> 属于独立并行任务（MJCF inertial 批量脚本），不阻塞本计划 P1/P2。

## 7. 风险与待确认

1. **Kp 1/8、Kd 1/4 缩放的语义**：官方内存表标注"位置环比例系数(1/8)…微分系数(1/4)"，
   实现里按 `(kp/8, kd/4)` 处理；最终以 G2 示波器实测（error→duty 斜率）为准，
   若不符则反解出一个 `error_gain_kp / error_gain_kd` 组合。
2. **ACC=0 语义**：内存表"0 表示最大加速度"，但最大值的数值 = 出厂寄存器 85×86？
   或 254×8.7°/s²？由 G1 读完 85/86 后确定（暂按 85×86）。
3. **I 项**：官方表说 Ki 位置模式无效 → 不建模（挂空）。
4. **12V 下 kt/R 温度漂移**：先按 25°C 标定；DR 范围覆盖（±10%）。
5. **venv 重建**：vendor/fork 后 `uv sync` 需重跑（几分钟），训练前确认
   `import bam.actuators` 里有 `hls2909`。
