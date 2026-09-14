#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Microduck RL —— 舵机(BAM)动作→扭矩链路 注释实例 demo
======================================================

本脚本可独立运行(.venv/bin/python scripts/demo_servo_action_flow.py),
不复现全套 mjlab 环境(那需要 GPU + MuJoCo Warp),而是用【真实的 bam 库】
(vendor/bam, 与训练时完全相同的代码路径)演示 RL 训练中
"策略动作 → 舵机位置目标 → 固件 P 控制律 → 电压 → 电机扭矩 → 摩擦预算"的
每一步,并打印数值。

对应关系(真实代码位置):
  策略输出 action (14D)               → mjlab JointPositionActionCfg(scale=1.0)
                                          microduck_velocity_env_cfg.py:271-273
  位置目标 → 舵机固件 P 控制律(duty)   → bam/actuator.py VoltageControlledActuator.compute_control
  电压 → 电机扭矩(反电动势方程)       → bam/actuator.py VoltageControlledActuator.compute_torque
  每 env 电压/增益/摩擦 DR            → vendor/bam/bam/mjlab.py BamActuator.compute
  摩擦预算 → MuJoCo dof_frictionloss  → vendor/bam/bam/mjlab.py _compute_friction_budget
  摩擦 × friction_scale(本仓库扩展)   → src/mjlab_microduck/actuator/friction_dr_bam.py

运行:  .venv/bin/python scripts/demo_servo_action_flow.py

说明: 输出最前面的一行 "[WARN] Failed to load task package ..." 是 mjlab 扫描
      mjlab.tasks 入口点时, 在 bam.mjlab 尚未导入完成时先碰到 mjlab_microduck
      产生的既有打印(仓库里任何外部导入 bam.mjlab 都会触发), 与本 demo 无关。
"""

import math
import os

import torch

# ── 真实 bam 库(与训练同一份 vendored 源码) ─────────────────────────────────
from bam.actuator import VoltageControlledActuator, TorchBackend
from bam.model import Model, load_model

# 本仓库对 bam 的扩展:每个环境独立的摩擦缩放(RL 域随机化专用)
from mjlab_microduck.actuator.friction_dr_bam import FrictionDRBamActuator

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 与 microduck_constants.py 中 _BAM_ACTUATOR_KWARGS 一致的关键值 ──────────
# 注意: 目前 hls2909 各项是【占位值】(见 docs/actuator_physics_swap_plan.md G1-G4),
#       用于演示链路;标定前不要用它们启动长训练。
KP_FW = 200.0            # 固件位置环 P 增益 (kp_fw=200, microduck_constants.py:129)
ERROR_GAIN = 0.166       # 占位: error_gain (duty/rad 换算, 来自示例参数 m1.json)
MAX_PWM = 0.97           # 占位: 最大占空比
MAX_CURRENT = 0.6        # HLS 固件限流 0.6A@12V (占位, 注释见 constants.py:134)
DT = 0.02                # 50 Hz 控制周期 (策略 50 Hz)
VIN_NOM = 12.0           # 标称 12V (DR 范围 (10.5, 12.6) 见 constants.py:131)

# HLS 固件"目标位置按速度+加速度限幅追踪"(寄存器 41/46-47, 只读出厂值 84/85/86)——
# 这是 XL330 没有、HLS 有的特性, 也是该文档里最大的 sim2real 风险之一。
# sim 侧 BAM 实现尚未落地(见 docs/actuator_physics_swap_plan.md 2.1), 此处用
# 占位值(m1.json 里的 max_velocity/max_acceleration)近似该双限幅, 仅作演示。
MAX_VEL = 8.06            # 77 rpm 空载 → 8.06 rad/s
MAX_ACC = 38.6            # 加速度限幅占位值 [rad/s²]


def make_hls2909_actuator():
    """构造一个"固件位置伺服"电机模型(hls2909 m1, 与 vendor 参数 JSON 同值).

    这就是 BamActuator.__init__ 里 load_model(cfg._resolved_json_path) 得到的东西:
    kt/R/friction_base/friction_viscous 全部来自 bam/params/hls2909/m1.json。
    """
    model = Model(name="m1")                       # m1 = 只有 Coulomb 摩擦(见 model.py)
    act = VoltageControlledActuator(
        None, vin=VIN_NOM, kp=KP_FW,
        error_gain=ERROR_GAIN, max_pwm=MAX_PWM, max_current=MAX_CURRENT,
    )
    model.set_actuator(act)                        # 在 model 上创建 kt/R/摩擦参数
    model.kt.value = 1.454                         # 扭矩常数 [Nm/A] (官网 KT=14.83 kg·cm/A)
    model.R.value = 20.0                           # 绕组电阻 [Ω] (12V/0.6A 等效)
    model.friction_base.value = 0.05               # Coulomb 摩擦预算 [Nm] (占位)
    model.friction_viscous.value = 0.06            # 粘性摩擦系数 [Nm·s/rad] (占位)
    act.backend = TorchBackend()                   # mjlab 下就是 Torch backend, 向量化 (N,J)
    return model, act


def step_servo(q, dq, q_target, act, model, inertia, dt=DT):
    """单关节 50 Hz 位置伺服: action→电压→扭矩→(摩擦)→积分 一小步.

    对照 mjlab.py BamActuator.compute 的 5 步流程:
      1. 电压: 固件 P 控制律 (compute_control)
      2. 扭矩: 直流电机方程 + 反电动势 (compute_torque)
      3. 摩擦预算: _compute_friction_budget → dof_frictionloss/dof_damping
      4. MuJoCo 求解器对摩擦做"静摩擦削顶" (此处用 1-DOF 解析等效)
      5. 积分前进 (mujoco 的 euler 积分)

    Returns: (q_new, dq_new, info) — info 是给打印用的中间量字典。
    """
    # 编码器视角的当前状态(模拟中直接取真值; 训练里还叠加 encoder-bias DR/1拍延迟)
    q_t = torch.tensor([q_target], dtype=torch.float32)   # 策略下发的目标位置
    qm = torch.tensor([q], dtype=torch.float32)           # 位置测量
    dq_m = torch.tensor([dq], dtype=torch.float32)        # 速度测量

    # (1) 固件控制律: duty = clip((q_target - q) * kp * error_gain, ...)
    #     先按"限流窗口"夹取(I = (duty·vin − kt·dq)/R 落在 ±0.6A 内),
    #     再按物理 PWM 上限夹取。见 bam/actuator.py:251-268 (compute_control)
    control = act.compute_control(q_t, qm, dq_m, dt)
    voltage = control.item()

    # (2) 直流电机扭矩: τ = kt·V/R − kt²·dq/R  (反电动势项 = 电气阻尼)
    #     见 bam/actuator.py:289-292 (compute_torque)
    torque = act.compute_torque(control, True, qm, dq_m).item()

    # (3) 摩擦预算(速度无关部分): m1 模型 = friction_base (Coulomb)
    #     mjlab 里这一步会写成每 env 的 dof_frictionloss / dof_damping,
    #     由 MuJoCo 求解器做静摩擦削顶 — 见 mjlab.py:661-669。
    friction_loss = model.friction_base.value          # [Nm] 干摩擦预算
    viscous = model.friction_viscous.value             # [Nm·s/rad] 粘性摩擦

    # (4) 1-DOF 解析等效"静摩擦削顶"(MuJoCo 约束求解的简化):
    #     净力矩 = 电机扭矩 − 粘性; 干摩擦最多抵消 |τ_net|。
    tau_net = torque - viscous * dq
    if abs(dq) < 1e-9:                                  # 静止: 静摩擦可以锁死
        if abs(tau_net) <= friction_loss:
            return q, 0.0, _info(q_target, q, dq, voltage, torque, tau_net, friction_loss, "静摩擦锁死")
        tau_drive = tau_net - math.copysign(friction_loss, tau_net)   # 克服静摩擦起转
    else:                                               # 运动: 干摩擦始终反向
        tau_drive = tau_net - math.copysign(friction_loss, dq)

    # (5) 固件运动学限幅(近似) + 欧拉积分 (I = 关节等效惯量, 含转子 armature)
    dq_new = dq + tau_drive / inertia * dt
    dq_new = max(-MAX_VEL, min(MAX_VEL, dq_new))        # 目标速度限幅
    dq_new = max(dq - MAX_ACC * dt, min(dq + MAX_ACC * dt, dq_new))  # 目标加速度限幅
    q_new = q + dq_new * dt
    info = _info(q_target, q, dq, voltage, torque, tau_net, friction_loss, "运动")
    info["q_new"], info["dq_new"] = q_new, dq_new
    return q_new, dq_new, info


def _info(q_target, q, dq, voltage, torque, tau_net, friction_loss, state):
    # 电流: I = (duty·vin − kt·dq)/R — 看看 0.6A 限流有没有生效
    duty = voltage / VIN_NOM
    current = (duty * VIN_NOM - 1.454 * dq) / 20.0
    return {
        "target": q_target, "q": q, "dq": dq, "voltage": voltage,
        "torque": torque, "tau_net": tau_net,
        "friction": friction_loss, "current": current, "state": state,
    }


def demo_single_servo():
    """Part 1: 单关节 60° 阶跃响应 —— 打印逐拍表。"""
    print("=" * 100)
    print("Part 1  单舵机 60° 阶跃响应 (hls2909 m1, kp_fw=200, 50Hz, 无重力)")
    print("对照: bam/actuator.py compute_control + compute_torque")
    print("=" * 100)

    model, act = make_hls2909_actuator()
    inertia = 0.02  # [kg·m²] 演示用关节等效惯量(含转子 armature), 无重力
    q, dq = 0.0, 0.0
    q_target = math.radians(60.0)  # 60° = 1.047 rad
    settled = False

    print(f"{'t(ms)':>6} {'q(deg)':>8} {'dq(rad/s)':>10} {'err(mrad)':>9} "
          f"{'duty':>6} {'voltage':>7} {'current':>7} {'torque':>7}  {'备注'}")
    for step in range(60):  # 1.2 s
        q, dq, info = step_servo(q, dq, q_target, act, model, inertia)
        if step % 4 == 0 or step in (1, 59):
            note = info["state"]
            if step == 0:
                note = "起步: 限流 0.6A 生效"
            elif abs(dq) > 7.9:
                note = "限速: |dq|≈8.06 rad/s (77 rpm 空载上限)"
            elif abs(info["current"]) > 0.599:
                note = "限流: I 被压在 0.6A 附近"
            elif info["state"] == "静摩擦锁死":
                note = "静摩擦锁死: |τ_net| ≤ 0.05 Nm"
            print(f"{step*20:>6} {math.degrees(q):>8.3f} {dq:>10.3f} "
                  f"{abs(q_target-q)*1e3:>9.1f} {info['voltage']/VIN_NOM:>6.3f} "
                  f"{info['voltage']:>7.2f} {info['current']:>7.3f} "
                  f"{info['torque']:>7.3f}  {note}")
        if abs(q - q_target) < 1e-3 and abs(dq) < 1e-3 and step > 10:
            settled = True
            print(f"  → 阶跃在 t={step*20}ms 收敛(±1 mrad / ±1e-3 rad/s)")
            break

    print(f"\n{'✓ 收敛' if settled else '✗ 有残留极限环(欠阻尼 P 控制律 + 50Hz 离散化)'}")
    print("说明1: 起步阶段电流被 0.6A 限流器压在极限 → 扭矩恒定 0.87 N·m 的加减速段;")
    print("      高速段反电动势(kt²·dq/R)是主要阻尼, 即 BAM 的\"电气阻尼\"等效 D 项。")
    print("说明2: 收敛死角 |τ_net| ≤ friction_base(0.05 Nm) 由静摩擦锁死 → 真机上的死区;")
    print("      mjlab 里摩擦预算写入 dof_frictionloss, MuJoCo 求解器执行同款削顶。")
    print("说明3: 真机 HLS 固件还有 P-D 位置环(1/8 与 1/4 缩放) + 目标双限幅,")
    print("      本 demo 只近似后者 —— 详见 docs/actuator_physics_swap_plan.md G1-G4 门。")


def demo_parallel_dr():
    """Part 2: 并行环境视角 —— 每 env 电压 DR / 增益 DR / 摩擦 DR (BAM 特有)。"""
    print()
    print("=" * 100)
    print("Part 2  并行训练视角 (N=4 envs × J=14 关节): BAM 每环境 DR")
    print("对照: vendor/bam/bam/mjlab.py BamActuator  +  src/.../friction_dr_bam.py")
    print("       +  src/mjlab_microduck/tasks/mdp.py randomize_bam_friction")
    print("=" * 100)

    torch.manual_seed(0)
    N, J = 4, 14
    device = "cpu"  # 训练时是 cuda, 数学完全一致

    # (a) 每 env 电压: 上电时从 vin_range 采样一次, 之后【不变】(mjlab.py:336-343)
    vin = torch.empty(N, 1).uniform_(10.5, 12.6)
    print(f"\n(a) 每环境电池电压 vin (vin_range=(10.5,12.6), constants.py:131)")
    print("    " + "  ".join(f"env{i}={v:.2f}V" for i, v in enumerate(vin.squeeze(1))))

    # (b) 每 env 固件增益: kp_scale (N,1), DR 在 reset 时 randomize_delayed_actuator_gains
    #     (mdp.py:3180-3194; 先 reset_gains 再 set_gains → 不累积)
    kp_scale = torch.ones(N, 1)
    kp_dr = torch.rand(N, 1) * (1.25 - 0.75) + 0.75  # 假设 kp DR 范围 (0.75, 1.25)
    kp_scale = kp_scale * kp_dr
    print(f"\n(b) 每环境固件 P 增益缩放 kp_scale (kp=200×scale):")
    print("    " + "  ".join(f"env{i}={200.0*s:.1f}" for i, s in enumerate(kp_scale.squeeze(1).tolist())))

    # (c) BAM 摩擦预算的 DR: 这是 stock dr.dof_frictionloss 在 BAM 下行不通的原因
    #     (edit_spec 把 frictionloss 清零, BAM 在 compute 里自己写预算), 所以本仓库
    #     用 FrictionDRBamActuator.friction_scale 乘在预算上 (friction_dr_bam.py:37-47)。
    #     调用序列 = mdp.py randomize_bam_friction (mdp.py:3212-3240):
    #       reset_friction_scale(env_ids) → 采样 → set_friction_scale(env_ids, samples)
    scale_range = (0.5, 1.5)  # JOINT_FRICTION_RANDOMIZATION_RANGE 演示值
    friction_scale = torch.rand(N, 1) * (scale_range[1] - scale_range[0]) + scale_range[0]

    # 用真实的 FrictionDRBamActuator._compute_friction_budget 算预算:
    # (避开 mjwarp 依赖 —— 只验证"预算×friction_scale"这一步)
    obj = FrictionDRBamActuator.__new__(FrictionDRBamActuator)  # 不调用 __init__
    model, act = make_hls2909_actuator()
    obj._bam_model = load_model(os.path.join(REPO, "vendor/bam/bam/params/hls2909/m1.json"))
    obj.friction_scale = friction_scale

    motor_torque = torch.rand(N, J) * 0.8 - 0.4     # 上一步电机扭矩 (N,J)
    external_torque = torch.rand(N, J) * 0.6 - 0.3  # 重力/约束载荷 (N,J)
    stribeck = torch.zeros(N, J)                    # m1 无 Stribeck 项
    obj.friction_scale = torch.ones(N, J)           # 先取标称值 → 基准预算
    budget_nominal = obj._compute_friction_budget(motor_torque, external_torque, stribeck)
    obj.friction_scale = friction_scale             # 再乘上每 env 的 DR 采样
    budget_scaled = obj._compute_friction_budget(motor_torque, external_torque, stribeck)

    print(f"\n(c) 摩擦预算 × friction_scale (randomize_bam_friction):")
    print(f"    m1 模型: budget = friction_base({obj._bam_model.friction_base.value:.3f} Nm) + 负载项(此时为0)")
    for i in range(N):
        print(f"    env{i}: friction_scale={friction_scale[i,0].item():.3f} → "
              f"budget={budget_scaled[i,0].item():.4f} Nm (基准 {budget_nominal[i,0].item():.4f})")

    # (d) 对照: 旧 XL330 m6 模型的预算里还有什么 (Stribeck + 负载相关项)
    print("\n(d) 对照 XL330 m6 (旧舵机, 标定过的模型): 预算里多了 Stribeck 与负载项")
    m6 = load_model(os.path.join(REPO, "vendor/bam/bam/params/xl330/m6.json"))
    obj6 = FrictionDRBamActuator.__new__(FrictionDRBamActuator)
    obj6._bam_model = m6
    obj6.friction_scale = torch.ones(N, J)
    vels = torch.tensor([[0.0], [0.1], [1.0], [5.0]])  # (N,1) 广播到 (N,J)
    dtheta = m6.dtheta_stribeck.value
    alpha = m6.alpha.value
    stribeck6 = torch.exp(-torch.pow((vels.abs() / dtheta), alpha))
    bud6 = obj6._compute_friction_budget(motor_torque, external_torque, stribeck6)
    for i in range(N):
        v = vels[i, 0].item()
        note = "Stribeck 抬升(静摩擦更强)" if v < 1.0 else "Stribeck 消失, 以负载项为主"
        print(f"    |dq|={v:.1f} rad/s → budget≈{bud6[i,0].item():.4f} Nm ({note})")


def demo_pipeline_map():
    """Part 3: RL 数据流全链路图示(文字版) —— action → 舵机 → 状态。"""
    print()
    print("=" * 100)
    print("Part 3  训练/部署数据流 (舵机视角)")
    print("=" * 100)
    print(r"""
  策略网络 (PPO, rsl_rl, 50 Hz)
        │  obs: 61D = 48 本体 + [twist(3), head_pose(4), body_pose(6)]
        ▼
  action: 14D 关节位置目标 [rad]        ← scale=1.0, 直接映射
        │  (JointPositionActionCfg, velocity_env_cfg.py:271-273)
        ▼
  mjlab 动作管理器 → actuator 转矩接口
        │  cmd.position = q_target; cmd.vel = q_dot  (BamActuator.command_field="position")
        ▼
  BamActuator.compute (mjlab.py:580)
        ├─ 每 env 电压 vin (上电采样, (10.5,12.6)V) + 压降
        ├─ 固件 P 控制律  duty = clip(kp·Δq·error_gain)   ← VoltageControlledActuator
        ├─ 电机扭矩        τ = kt·V/R − kt²·dq/R          ← 反电动势=电气阻尼
        ├─ 摩擦预算        → dof_frictionloss/dof_damping ← 由 MuJoCo 求解器做静摩擦削顶
        │     (× friction_scale: FrictionDRBamActuator, 每 env 随机化, 不累积)
        ▼
  MuJoCo Warp 并行求解 → 下一拍状态 q, dq
        ▼
  观测: joint_pos(+encoder_bias)/joint_vel(1拍延迟) + 回环到策略
        (velocity_env_cfg.py:604-633)

  部署 (scripts/export.py → ONNX 含归一化 → 运行时 50 Hz 写位置目标到
  UART 总线上的 14 个舵机; 舵机固件内部 P-D 位置环闭环, 与 sim 一致。)
""")


if __name__ == "__main__":
    demo_single_servo()
    demo_parallel_dr()
    demo_pipeline_map()
