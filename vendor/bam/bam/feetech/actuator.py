# Copyright 2025 Marc Duclusaud & Grégoire Passault

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Union
from bam.message import yellow, print_parameter, bright
from bam.actuator import VoltageControlledActuator
from bam.parameter import Parameter
from bam.testbench import Testbench, Pendulum

if TYPE_CHECKING:
    from bam.actuator import ArrayLike


class STS3215Actuator(VoltageControlledActuator):
    """
    Feetech STS3215 7.4v

    The firmware rate-limits the target position it feeds to its P controller,
    which makes :meth:`compute_control` stateful (see :attr:`q_target_smooth`).
    """

    stateful = True

    def __init__(self, testbench_class: Testbench):
        super().__init__(
            testbench_class,
            vin=7.4,
            kp=32,
            # This gain, if multiplied by a position error and firmware KP, gives duty cycle
            # It was determined using an oscilloscope and STS3215 actuators
            # here, firmware_kp = kp
            error_gain=0.166,
            # self.error_gain = 0.001 * np.rad2deg(1.0)
            # Maximum allowable duty cycle, also determined with oscilloscope
            max_pwm=0.97,  # TODO, but can we assume 1.0 ?
        )

        self.default_max_velocity = (3400 * 2 * np.pi) / 4096

        # Firmware's internal target position, rate-limited to max_velocity.
        # It is lazily initialized on the first compute_control() call so that it
        # adopts the type and shape of the state it is fed (float, numpy array or
        # torch tensor), and does not require a log to be loaded.
        self.q_target_smooth: ArrayLike | None = None

        # Environments whose internal target still has to be re-seeded, see reset()
        self._to_reset_env_ids: list = []

    def get_extra_inertia(self) -> float:
        return self.model.armature.value

    def reset(self, env_ids=...) -> None:
        """Reset the firmware's internal (rate-limited) target position.

        That target is a joint position, which is only known when the controller
        runs: the reset is therefore deferred to the next :meth:`compute_control`,
        which re-seeds the selected environments from the position it is given.

        :param env_ids: Environments to reset, ``...`` (default) for all of them.
        """
        if env_ids is ...:
            self.q_target_smooth = None
            self._to_reset_env_ids.clear()
        else:
            self._to_reset_env_ids.append(env_ids)

    def get_state(self):
        # Internal target position, plus the resets that are still pending
        return self.q_target_smooth, list(self._to_reset_env_ids)

    def set_state(self, state) -> None:
        q_target_smooth, to_reset_env_ids = (None, []) if state is None else state
        self.q_target_smooth = q_target_smooth
        self._to_reset_env_ids = list(to_reset_env_ids)

    def initialize(self):
        # Torque constant [Nm/A] or [V/(rad/s)]
        self.model.kt = Parameter(0.784532, 0.05, 2.5)  # docs says 8 kg.cm / A

        self.model.error_gain_ratio = Parameter(1.0, 0.1, 10.0)

        # Motor resistance [Ohm]
        self.model.R = Parameter(2.0, 0.1, 10.0)

        # Motor armature / apparent inertia [kg m^2]
        self.model.armature = Parameter(0.0001, 0.00001, 0.04)

        self.model.q_offset = Parameter(0, -0.2, 0.2)

        self.model.max_velocity = Parameter(
            self.default_max_velocity,
            0.1 * self.default_max_velocity,
            10.0 * self.default_max_velocity,
        )

    def compute_control(
        self, q_target: ArrayLike, q: ArrayLike, dq: ArrayLike, dt: float
    ) -> ArrayLike | None:
        """Compute the voltage command from the rate-limited position error.

        The firmware does not track ``q_target`` directly: it moves an internal
        target towards it at most at ``max_velocity``, and runs its P controller
        on that internal target. This is stateful, so :meth:`compute_control`
        must be called once per timestep, in order.

        The state is kept in ``self.q_target_smooth`` and is (re)initialized to
        the current position ``q`` on the first call after a :meth:`reset` — the
        way the servo behaves when it powers on. It therefore inherits the type
        and shape of ``q`` and works on any backend (float, numpy array or torch
        tensor, in the latter case vectorized over all environments).

        :param q_target: Target joint angle(s) [rad].
        :param q: Current joint angle(s) [rad].
        :param dq: Current joint velocity(ies) [rad/s] (unused here).
        :param dt: Timestep [s], used by the internal target rate limiter.
        :returns: Voltage [V] sent to the motor.
        """
        q_target_smooth = self.q_target_smooth
        if q_target_smooth is None:
            # First call, or first call after a full reset
            q_target_smooth = q
            self._to_reset_env_ids.clear()
        elif self._to_reset_env_ids:
            # Environments reset since the last call: their internal target has to
            # follow their (teleported) position
            for env_ids in self._to_reset_env_ids:
                q_target_smooth[env_ids] = q[env_ids]
            self._to_reset_env_ids.clear()

        # Internal target position is clipped using maximum velocity
        max_step = self.model.max_velocity.value * dt
        self.q_target_smooth = self.backend.clamp(
            q_target,
            q_target_smooth - max_step,
            q_target_smooth + max_step,
        )

        duty_cycle = (
            (self.q_target_smooth - q)
            * self.kp
            * self.error_gain
            * self.model.error_gain_ratio.value
        )
        duty_cycle = self.backend.clamp(duty_cycle, -self.max_pwm, self.max_pwm)
        self.duty_cycle = duty_cycle  # for logging (and the battery drop model)

        return self.vin * duty_cycle


class HLS2909Actuator(VoltageControlledActuator):
    """Feetech HL-2909-C001 12 V TTL bus servo (position servo mode).

    Reference numbers (identification in ``docs/feetech_hls_memtable.md`` of the
    Microduck repository, firmware 3.45, all 15 servos identical):

    * ``kt`` = 1.454 N·m/A on the *output* side (datasheet KT = 14.83 kg·cm/A),
      gear ratio 320:1;
    * ``R`` = 20 Ω (12 V / 0.6 A stall-current equivalent — the datasheet group
      ``kt``/stall-torque/R is internally consistent, a probe regression gave
      6.8 Ω and the stall point 3.5 Ω, both flagged unreliable);
    * firmware current limit = 1.95 A (register 28/44, 6.5 mA/LSB × 300 LSB);
    * maximum running speed ≈ 19.2 rad/s (register 84 = 250 LSB = 183 RPM — the
      physical no-load speed of 8.06 rad/s *emerges* from ``kt``/``R``, the
      firmware ramp is not the limiting element);
    * position loop gains: Kp = 32 (registers 21/50, firmware scale 1/8),
      Kd = 32 (registers 22/51, firmware scale 1/4), Ki = 0 (inactive in
      position mode);
    * armature ≈ 1e-3 kg·m² and the ``m1`` friction budget were calibrated
      against a known-good policy standing up on the real robot.

    Compared to :class:`STS3215Actuator`, the firmware rate-limits its internal
    target position with **both** a velocity and an acceleration cap
    (registers 46/47 "running speed" and 41 "acceleration", where 0 = firmware
    maximum — a second-order ramp), so :meth:`compute_control` carries two
    internal states (:attr:`q_target_smooth` and :attr:`v_target_smooth`) and
    is :attr:`stateful`. The position loop is P-D (P scaled 1/8, D scaled 1/4),
    and the motor current is capped by a duty-cycle window derived from the
    firmware current limit (register 28), exactly like the base class.

    The motor/friction values above are the *current* best estimates; the
    friction budget and armature still await a proper pendulum identification.
    They are all model parameters, so they can be fitted directly
    (see :class:`~bam.model.Model`) and are stored in
    ``bam/params/hls2909/mX.json``.
    """

    stateful = True

    def __init__(self, testbench_class: Testbench = None):
        super().__init__(
            testbench_class,
            vin=12.0,
            # Firmware position-loop P gain (register 21/50). mjlab can override
            # it per-environment via BamActuatorCfg.kp_fw.
            kp=32,
            # Converts kp * Δq to a duty cycle; determined on the bench for the
            # STS3215 and confirmed by behavior probe for the HLS firmware
            # (duty = Δq · kp · error_gain / 8, see compute_control).
            error_gain=0.166,
            # Maximum physical duty cycle, oscilloscope-measured for feetech units.
            max_pwm=0.97,
            # Firmware current limit (register 28/44 = 300 × 6.5 mA). Also
            # mirrored as a model parameter so it can be set per-JSON.
            max_current=1.95,
        )

        # Firmware "running speed" hard cap [rad/s]: register 84 = 250 LSB =
        # 183 RPM. The physical no-load speed (8.06 rad/s) is what kt/R produce,
        # it does not appear here.
        self.default_max_velocity = 19.1637
        # Firmware "acceleration" cap [rad/s²]: register 85×86 = 0 (= firmware
        # maximum, register 41 = 0). A large but finite value keeps the ramp
        # numerically tractable; the microduck robot is trained with 500.0.
        self.default_max_acceleration = 500.0

        # 便捷可见性属性（纯加法，不改动现有默认值/公式；供日志与自测直接读取）。
        # 这些初值与 initialize() 中创建的 model 参数初值保持一致。
        self.kt = 1.454
        self.R = 20.0
        self.max_velocity = self.default_max_velocity
        self.max_acceleration = self.default_max_acceleration
        self.kd = 32.0

        # Firmware's internal target position and velocity, rate-limited to
        # max_velocity / max_acceleration. Lazily initialized so they adopt the
        # type and shape of the state they are fed (float, numpy array or torch
        # tensor), and do not require a log to be loaded.
        self.q_target_smooth: ArrayLike | None = None
        self.v_target_smooth: ArrayLike | None = None

        # Environments whose internal target still has to be re-seeded, see reset()
        self._to_reset_env_ids: list = []

    def get_extra_inertia(self) -> float:
        return self.model.armature.value

    def reset(self, env_ids=...) -> None:
        """Reset the firmware's internal (rate-limited) target position/velocity.

        That target is a joint position, which is only known when the controller
        runs: the reset is therefore deferred to the next :meth:`compute_control`,
        which re-seeds the selected environments from the position it is given
        (with a zero target velocity, i.e. servo power-on behaviour).

        :param env_ids: Environments to reset, ``...`` (default) for all of them.
        """
        if env_ids is ...:
            self.q_target_smooth = None
            self.v_target_smooth = None
            self._to_reset_env_ids.clear()
        else:
            self._to_reset_env_ids.append(env_ids)

    def get_state(self):
        # Internal target position/velocity, plus the resets that are still pending
        return self.q_target_smooth, self.v_target_smooth, list(self._to_reset_env_ids)

    def set_state(self, state) -> None:
        q_target_smooth, v_target_smooth, to_reset_env_ids = (
            (None, None, []) if state is None else state
        )
        self.q_target_smooth = q_target_smooth
        self.v_target_smooth = v_target_smooth
        self._to_reset_env_ids = list(to_reset_env_ids)

    def initialize(self):
        # Torque constant [Nm/A] or [V/(rad/s)], output side (datasheet KT)
        self.model.kt = Parameter(1.454, 0.2, 3.0)

        # Motor resistance [Ohm]
        self.model.R = Parameter(20.0, 2.0, 50.0)

        # Reflected armature / apparent inertia [kg m^2]
        self.model.armature = Parameter(1e-3, 1e-6, 0.05)

        # Firmware position-loop gains. The duty cycle acts on the *error seen
        # by the P controller*: duty = Δq · kp · error_gain / 8 (P scale 1/8)
        # + dq · kd · error_gain / 4 (D scale 1/4).
        self.model.error_gain = Parameter(self.error_gain, 0.01, 2.0)
        self.model.kd = Parameter(32.0, 0.0, 254.0)

        # Firmware target ramp limits [rad/s], [rad/s²]
        self.model.max_velocity = Parameter(
            self.default_max_velocity,
            0.1 * self.default_max_velocity,
            10.0 * self.default_max_velocity,
        )
        self.model.max_acceleration = Parameter(
            self.default_max_acceleration,
            0.1 * self.default_max_acceleration,
            10.0 * self.default_max_acceleration,
        )

        # Firmware current limit [A] (register 28/44, 6.5 mA/LSB). Like the base
        # class, it is modelled as a duty-cycle window; 0 disables it.
        self.model.max_current = Parameter(1.95, 0.1, 5.0)

    def compute_control(
        self, q_target: ArrayLike, q: ArrayLike, dq: ArrayLike, dt: float
    ) -> ArrayLike | None:
        """Compute the voltage command from the second-order rate-limited error.

        The firmware does not track ``q_target`` directly: it moves an internal
        target towards it with a velocity **and** acceleration limit, and runs
        its P-D controller on that internal target. This is stateful, so
        :meth:`compute_control` must be called once per timestep, in order.

        The internal target (position/velocity) is (re)initialized to the
        current position / zero on the first call after a :meth:`reset` — the
        way the servo behaves when it powers on. It therefore inherits the type
        and shape of ``q`` and works on any backend (float, numpy array or torch
        tensor, in the latter case vectorized over all environments).

        :param q_target: Target joint angle(s) [rad].
        :param q: Current joint angle(s) [rad].
        :param dq: Current joint velocity(ies) [rad/s].
        :param dt: Timestep [s], used by the internal target rate limiter.
        :returns: Voltage [V] sent to the motor.
        """
        q_target_smooth = self.q_target_smooth
        if q_target_smooth is None:
            # First call, or first call after a full reset: the internal target
            # starts at the current position, at rest (servo power-on behaviour)
            q_target_smooth = q
            v_target_smooth = q * 0
            self._to_reset_env_ids.clear()
        else:
            v_target_smooth = self.v_target_smooth
            if self._to_reset_env_ids:
                # Environments reset since the last call: their internal target
                # has to follow their (teleported) position, at rest
                for env_ids in self._to_reset_env_ids:
                    q_target_smooth[env_ids] = q[env_ids]
                    v_target_smooth[env_ids] = q[env_ids] * 0
                self._to_reset_env_ids.clear()

        # Firmware target profile: velocity limit, then acceleration limit, then
        # a soft stop so the profile parks exactly on the target when the
        # remaining distance is short (no overshoot of the second-order ramp).
        # A zero timestep (e.g. a controller's first update at simulation time
        # 0) cannot advance the ramp — hold the internal target as-is, which
        # also keeps the divisions below NaN-free.
        v_max = self.model.max_velocity.value
        a_max = self.model.max_acceleration.value
        # dt is a scalar (single rollout, mjlab) or the per-log vector of a
        # batched fit (bam.fit rolls out all logs at once). All logs of a batch
        # share the same timestep grid, so a scalar truth-value check suffices;
        # np.any avoids the "ambiguous truth value" error on the vector case.
        if np.any(np.asarray(dt) > 0):
            v_target = (q_target - q_target_smooth) / dt
            v_target = self.backend.clamp(v_target, -v_max, v_max)
            v_target = self.backend.clamp(
                v_target,
                v_target_smooth - a_max * dt,
                v_target_smooth + a_max * dt,
            )
            remaining = abs(q_target - q_target_smooth) / dt
            v_target = self.backend.clamp(v_target, -remaining, remaining)
        else:
            v_target = v_target_smooth

        q_target_smooth = q_target_smooth + v_target * dt
        self.q_target_smooth = q_target_smooth
        self.v_target_smooth = v_target

        # Firmware position loop: P (1/8 scale) + D (1/4 scale); Ki is inactive
        # in position mode. The D term runs on the actual joint velocity.
        kp = self.kp
        error_gain = self.model.error_gain.value
        kd = self.model.kd.value
        duty_cycle = (
            (q_target_smooth - q) * kp * error_gain / 8.0
            + (-dq) * kd * error_gain / 4.0
        )

        # Firmware current limiter: bound the duty cycle so the motor current
        # I = (duty * vin - kt * dq) / R stays within [-max_current, max_current]
        # (same attempt semantics as the base class).
        if self.model.max_current.value > 0:
            back_emf = self.model.kt.value * dq
            duty_span = self.model.R.value * self.model.max_current.value / self.vin
            duty_center = back_emf / self.vin
            duty_cycle = self.backend.clamp(
                duty_cycle, duty_center - duty_span, duty_center + duty_span
            )

        # Physical PWM limit (voltage bounded by the battery) — applied last.
        duty_cycle = self.backend.clamp(duty_cycle, -self.max_pwm, self.max_pwm)
        self.duty_cycle = duty_cycle  # for logging (and the battery drop model)

        return self.vin * duty_cycle


class HD1910Actuator(VoltageControlledActuator):
    """Feetech HD-1910-C001 6 V 级 TTL 总线舵机（工作电压 4–8.4 V，典型 6 V；
    运行模式 4 = 纯位置 PD，Sim2Real，出厂默认）。

    规格书 2026-09-07 A/0 版给出的关键参数：

    * ``kt`` = 0.7358 N·m/A（输出轴，规格书直接给出的 7.5 kg·cm/A），减速比 320:1；
    * ``R`` ≈ 3.75 Ω（6 V / 1.6 A 堵转），交叉验证 4.8 V/1.2 A → 4.0 Ω、7.4 V/2.0 A → 3.7 Ω；
    * 空载速度 @6 V 92 RPM = 9.63 rad/s（@4.8 V 73 RPM = 7.64，@7.4 V 113 RPM = 11.83）；
    * 工作电压 4–8.4 V（典型 6 V）；空载电流 ≤ 180 mA；额定 3 kg·cm / 690 mA；
      堵转 12 kg·cm / 1.6 A（均为 @6 V）；
    * 12-bit 磁编码器，0.088°/LSB = 2π/4096 rad/LSB，360°(0–4095)，中位 2048，
      顺时针 0→4095；回差 ≤ 0.5°（≈0.0087 rad）；位置更新率 1 kHz；
    * 固件运行模式：0=角度伺服(限力)、1=恒速、2=恒流、3=PWM 开环、4=纯位置 PD
      （Sim2Real，出厂默认）。

    真机回读（USB-TTL 直连，ID=14，固件 3.46，FT-SCS 1Mbps）确认/更新了下列关键值：
    reg33=4（纯位置 PD，实测确认）；出厂/生效增益 reg21/22/23=reg50/51/52=32/40/0 →
    **Kp=32、Kd=40、Ki=0**（注意 Kd=40 不是 32）；reg28/44=500 LSB = **3.25 A**；
    reg84/85/86=0（HLS 语义的出厂限速/加速度）→ 支持"模式 4 无目标速度/加速度限幅"；
    reg81 DTs=10ms（100Hz 控制周期）、reg82 eFk(*10)=60、reg77 vFk=60、reg79 pFk=60、
    reg78 vKgI=2、reg83 Vk=20ms；供电 reg62=5.1V（当前电源 5.1V，规格典型 6V）；
    温度 33°C；空载电流 ≈ 20 mA；状态位 0；reg19/20（卸载/LED 报警条件）= 4（bit2）。

    .. note::
        Kp/Kd/Ki 已真机回读确认 = 32/40/0（2026-09，ID14 实测，无需再用 HLS 惯例初值）。
        模式 4 控制律经 2026-09-09 小阶跃回归自测：**无 1/8 P 缩放**（``p_scale=1.0``、
        ``error_gain≈1.0`` 已同步为默认值）；``d_scale=4.0``（D 缩放≈1/4，沿用 HLS 惯例）
        与 error_gain 的精确语义仍待官方内存表/精细扫频确认；电流限幅语义见
        :meth:`__init__` 的 ``max_current`` 说明。

    .. warning::
        实测 mode 4 的**位置误差不做 ±2048 回绕**：目标从 4094 写到 338 时，舵机走
        -3755 LSB 的长路径（原始差直驱）。若命令目标靠近编码器 0/4095 边界
        （即 q_zero≈0 或 ≈2π），真实舵机不会按最短路径回绕，而是沿长路径大幅转动；
        本模型用连续弧度（无回绕）建模，仿真/拟合应避开编码器回绕区或显式处理
        该长路径，否则会与真机出现明显偏差。
        已实测可动的写序：A) 46=0→46=32767→42=目标→40=1；B) 40=1→46=32767→42=目标；
        实测不动的写序（HLS record 风格）：46=0→42=目标→40=1→46=32767。
        摆锤实验须保证 q_zero 远离编码器 0/4095，或采集脚本做环绕感知。

    与 :class:`HLS2909Actuator` 相比，本模型在默认配置（``use_rate_limiting=False``）
    下假设模式 4 的固件 **不做** 目标速度/加速度二阶爬坡（纯 PD，直接跟踪
    ``q_target``），因此是否做二阶限幅、电流限幅是否启用、位置环 P/D 缩放、
    ``error_gain`` 语义都做成了**可配置项**（见 :meth:`__init__` 与
    :meth:`initialize`）。默认值已按 2026-09-09 自测同步为
    ``error_gain=1.0 / max_pwm=1.0 / p_scale=1.0 / d_scale=4.0``（其中 p_scale 与
    error_gain 为实测结论，d_scale 沿用 HLS 惯例、精确值待精细扫频确认；
    见各 ``TODO(核对)`` 注释）。

    .. note::
        电机电流由 :meth:`~bam.actuator.VoltageControlledActuator.compute_torque`
        按直流电机方程给出（``kt``/``R``/反电动势），本类只负责 :meth:`compute_control` 的
        控制律（占空比 → 电压）。armature 为占位值，与 ``m1`` 摩擦预算一起待摆识别。
    """

    stateful = True

    def __init__(
        self,
        testbench_class: Testbench = None,
        *,
        vin: float = 5.0,
        kp: float = 32,
        error_gain: float = 1.0,
        max_pwm: float = 1.0,
        max_current: float = 3.25,
        use_rate_limiting: bool = False,
        p_scale: float = 1.0,
        d_scale: float = 4.0,
    ):
        """构造 HD-1910-C001 执行器模型。

        除 ``testbench_class``（工厂/管道会传入 :class:`~bam.testbench.Pendulum`，
        默认为 ``None``，仅供裸实例化与自测读取默认值）外，其余参数为带默认值的
        可配置项；其中带 ``TODO(真机核对)`` 的项均为规格书未覆盖的模型语义假设：

        :param testbench_class: 测试台类（见 :class:`~bam.actuator.Actuator`）。
        :param vin: 供电电压 [V]，默认 6.0（典型工作电压）。
        :param kp: 固件位置环 P 增益，默认 32（真机回读 reg21/50 = 32，已确认）。
        :param error_gain: 把 ``kp * Δq`` 换算成占空比的系数，默认 1.0。
            自测(2026-09-09): 100 LSB 小阶跃回归 duty≈0.45–0.61%/LSB @kp=32，且大误差
            (>≈400 LSB)实测**饱和 100%**；据此估计的有效 P 斜率 ≈ kp×0.93/rad
            （≈4.6%/LSB），与上述每 LSB 斜率相差约 10×，疑为 reg60 占空比反馈的
            单位口径（0.1% vs %）差异所致。模型采用后一口径：**模式 4 基本无 1/8 缩放、
            error_gain≈1**（HLS 的 1/8 缩放+0.166 增益不适用）；error_gain 的精确
            语义仍待官方内存表/精细扫频确认，拟合时在 (0.01, 10) 内自由吸收。
        :param max_pwm: 最大物理占空比，默认 1.0（2026-09-09 实测模式下饱和点为 100%，
            非 0.97 — 示波器 0.97 为其他模式/电压下的值）。
            TODO(核对): 0.97 vs 1.0 对拟合影响极小（摆锤 duty 峰值 ≈35%），如需精确可再测。
        :param max_current: 固件电流限幅 [A]，默认 3.25（真机回读 reg28/44 = 500 LSB
            = 3.25 A），``0`` 表示关闭。
            TODO(真机核对): 模式 4 出厂默认关闭过流保护；且 2026-09-09 实测 reg44 在
            mode 4 **不钳流**（reg44=0.195A 时电流峰值仍 ≈2A 且照常走完大行程）——
            本模型默认"保留限幅 + 3.25 A"仅为保守设定，可改为 0（关闭）模拟真机。
        :param use_rate_limiting: 是否启用目标速度/加速度二阶限幅，默认 ``False``。
            真机回读 reg84/85/86 = 0，支持"模式 4 无目标限幅（纯 PD）"假设，故默认关。
        :param p_scale: 位置环 P 缩放分母，默认 1.0（2026-09-09 小阶跃回归实测
            斜率 ≈ kp×0.93/rad → 模式 4 **无 1/8 缩放**；与 HLS 的差异由
            ``error_gain=1.0`` + 拟合吸收）。
        :param d_scale: 位置环 D 缩放分母，默认 4.0（沿用 HLS 惯例）。
            TODO(核对): 2026-09-09 回归中 D 项系数≈0（采样 4 ms 太粗、行程太短，
            瞬态被饱和淹没）；直接过冲实验（kd=40 vs 16）显示 kd 有**弱阻尼作用**
            （峰值 80 vs 82 LSB）→ D 项存在但弱，缩放待精细采样/扫频确认。
        """
        super().__init__(
            testbench_class,
            vin,
            kp,
            error_gain,
            max_pwm,
            max_current,
        )

        # 可配置的模型语义开关与缩放系数，供 initialize()/compute_control() 使用。
        self.use_rate_limiting = use_rate_limiting
        self.p_scale = p_scale
        self.d_scale = d_scale

        # 固件"运行速度"上限 [rad/s] = 规格书空载转速 @6 V 92 RPM 按 5.0 V 折算 = 8.03 rad/s；
        # （2026-09-10 起台架与整机统一按 5.0 V 供电，故默认值同步为 5 V 折算值）
        # 真机回读 reg84/85/86 = 0（模式 4 无目标速度/加速度限幅），故默认不做爬坡，
        # 该值仅在 use_rate_limiting=True 时才参与目标限幅。
        # TODO(标定): 直流电机空载速度（kt/R 推出 6/0.7358 ≈ 8.15 rad/s）与规格书
        # 9.63 rad/s 存在出入，需扫频/实测标定。
        self.default_max_velocity = 8.0282   # 5.0 V 供电(台架/整机统一 5 V; @6V 规格 9.6338)
        # 固件"加速度"上限 [rad/s²]：规格书未给，暂用 HLS 惯例（500.0）。真机回读
        # reg85/86=0，默认（纯 PD）下该值不生效。
        # TODO(标定): 加速度/爬坡在模式 4 经回读确认不存在，此值仅保留给 use_rate_limiting。
        self.default_max_acceleration = 500.0

        # 便捷可见性属性（与 initialize() 中 model 参数初值保持一致，供日志/自测读取）。
        self.kt = 0.7358
        self.R = 3.75
        self.max_velocity = self.default_max_velocity
        self.max_acceleration = self.default_max_acceleration
        self.kd = 40.0  # 真机回读 reg22/51 = 40（注意不是 32）。

        # 内部目标（位置/速度）。与 HLS 一致延迟初始化以继承输入的类型/形状；即使默认
        # 不做爬坡也保留这两个字段，以维持 get_state()/set_state() 与拟合/回放管线兼容。
        self.q_target_smooth: ArrayLike | None = None
        self.v_target_smooth: ArrayLike | None = None

        # 仍需重新播种内部目标的环境列表，见 reset()。
        self._to_reset_env_ids: list = []

    def get_extra_inertia(self) -> float:
        return self.model.armature.value

    def reset(self, env_ids=...) -> None:
        """重置固件内部（限幅）目标位置/速度。

        该目标是关节位置，只有控制器运行时才知道，因此把重置推迟到下一次
        :meth:`compute_control`，届时从给定位置重新播种所选环境（目标速度置零，
        即舵机上电行为）。

        :param env_ids: 要重置的环境索引，``...``（默认）表示全部。
        """
        if env_ids is ...:
            self.q_target_smooth = None
            self.v_target_smooth = None
            self._to_reset_env_ids.clear()
        else:
            self._to_reset_env_ids.append(env_ids)

    def get_state(self):
        # 内部目标位置/速度，以及尚未执行的重置
        return self.q_target_smooth, self.v_target_smooth, list(self._to_reset_env_ids)

    def set_state(self, state) -> None:
        q_target_smooth, v_target_smooth, to_reset_env_ids = (
            (None, None, []) if state is None else state
        )
        self.q_target_smooth = q_target_smooth
        self.v_target_smooth = v_target_smooth
        self._to_reset_env_ids = list(to_reset_env_ids)

    def initialize(self):
        # 力矩常数 [Nm/A] 或 [V/(rad/s)]，输出轴（规格书 7.5 kg·cm/A = 0.7358 N·m/A）。
        self.model.kt = Parameter(0.7358, 0.1, 2.0)

        # 电机电阻 [Ohm]（6 V / 1.6 A 堵转 ≈ 3.75 Ω，交叉验证 4.0/3.7 Ω）。
        self.model.R = Parameter(3.75, 0.5, 10.0)

        # 折算到输出轴的电机电枢/表观惯量 [kg·m²]。减速比 320:1；数值为占位，
        # 同 HLS，待摆识别。
        # TODO(真机核对): armature 与 m1 摩擦预算的占位，待摆识别校准。
        self.model.armature = Parameter(1e-3, 1e-6, 0.05)

        # 注意：q_offset 由 Model.set_actuator() 统一创建（界 ±0.1），无需在此重复
        # 定义（旧实现在此处定义的 ±0.2 会被 set_actuator 覆盖，属死代码）。

        # 固件位置环增益。duty = Δq · kp · error_gain / p_scale (P) + (-dq) · kd ·
        # error_gain / d_scale (D)；Ki 在位置模式下不生效（真机回读 Ki=0）。
        # kp/kd 取自真机回读（reg21/50=32，reg22/51=40）。
        # error_gain 界放宽到 (0.01, 10)：2026-09-09 自测模式 4 无 1/8 缩放（P≈kp×0.93/rad），
        # 有效 error_gain≈1（HLS 的 0.166 会落在旧界外无法覆盖真值）。
        self.model.error_gain = Parameter(self.error_gain, 0.01, 10.0)
        self.model.kd = Parameter(40.0, 0.0, 254.0)  # 真机回读 reg22/51 = 40。

        # 固件目标爬坡限 [rad/s]、[rad/s²]。默认（use_rate_limiting=False，纯 PD）不参与
        # 限幅；真机回读 reg84/85/86=0，确认模式 4 无目标速度/加速度限幅。
        self.model.max_velocity = Parameter(
            self.default_max_velocity,
            0.1 * self.default_max_velocity,
            10.0 * self.default_max_velocity,
        )
        self.model.max_acceleration = Parameter(
            self.default_max_acceleration,
            0.1 * self.default_max_acceleration,
            10.0 * self.default_max_acceleration,
        )

        # 固件电流限幅 [A]：真机回读 reg28/44 = 500 LSB = 3.25 A。本模型默认保留该限幅，
        # 与基类一致建模为占空比窗口；0 表示关闭。
        # TODO(真机核对): 模式 4 出厂默认关闭过流保护，且实测 reg44 不钳流；此处默认
        # "保留限幅 + 初值 3.25 A"仅为保守设定。该窗口在摆锤辨识域几乎不生效
        # （duty_span = R·I/vin ≈ 2.03 > 物理占空比 1.0），仅在高反电动势/大负占空比
        # 时略紧于物理钳位；如需完全镜像真机"不钳流"，把默认与 m1.json 的 max_current 设为 0。
        self.model.max_current = Parameter(3.25, 0.1, 5.0)

    def compute_control(
        self, q_target: ArrayLike, q: ArrayLike, dq: ArrayLike, dt: float
    ) -> ArrayLike | None:
        """根据位置误差计算电压指令（可配置是否做二阶目标限幅）。

        当 ``use_rate_limiting=True`` 时，固件并不直接跟踪 ``q_target``，而是以一个内部目标
        以速度 + 加速度限幅朝它移动，并在该内部目标上运行 P-D 控制器（与
        :class:`HLS2909Actuator` 相同）。默认 ``use_rate_limiting=False``（模式 4 纯位置 PD，
        假设固件不爬坡），因此内部目标就是 ``q_target``。两种情形都是有状态的，
        :meth:`compute_control` 需要按时间顺序每个仿真步调用一次。

        内部目标（位置/速度）在 :meth:`reset` 后的首次调用时被重播种到当前
        位置 / 静止（舵机上电行为），并继承 ``q`` 的类型与形状：既支持 Python
        float，也支持 numpy 数组 / torch tensor（后者对所有环境向量化）。

        :param q_target: 目标关节角 [rad]。
        :param q: 当前关节角 [rad]。
        :param dq: 当前关节角速度 [rad/s]。
        :param dt: 时间步 [s]，仅在 ``use_rate_limiting`` 爬坡时用于目标限幅。
        :returns: 发送给电机的电压 [V]。
        """
        q_target_smooth = self.q_target_smooth
        if q_target_smooth is None:
            # 首次调用，或全量重置后的首次调用：内部目标从当前位置、静止开始
            q_target_smooth = q
            v_target_smooth = q * 0
            self._to_reset_env_ids.clear()
        else:
            v_target_smooth = self.v_target_smooth
            if self._to_reset_env_ids:
                # 被重置的环境：其内部目标须跟随（瞬移后的）位置，保持静止
                for env_ids in self._to_reset_env_ids:
                    q_target_smooth[env_ids] = q[env_ids]
                    v_target_smooth[env_ids] = q[env_ids] * 0
                self._to_reset_env_ids.clear()

        if self.use_rate_limiting:
            # 与 HLS 相同的二阶爬坡：速度限幅 → 加速度限幅 → 短距离软停车。
            v_max = self.model.max_velocity.value
            a_max = self.model.max_acceleration.value
            # dt 是标量（单条 rollout，mjlab）或批量拟合的各日志向量（bam.fit 一次 rollout
            # 所有日志）。同一批日志共享相同的时间步网格，故用标量真值判断即可；np.any
            # 避免向量情况下 "ambiguous truth value" 报错。
            if np.any(np.asarray(dt) > 0):
                v_target = (q_target - q_target_smooth) / dt
                v_target = self.backend.clamp(v_target, -v_max, v_max)
                v_target = self.backend.clamp(
                    v_target,
                    v_target_smooth - a_max * dt,
                    v_target_smooth + a_max * dt,
                )
                remaining = abs(q_target - q_target_smooth) / dt
                v_target = self.backend.clamp(v_target, -remaining, remaining)
            else:
                v_target = v_target_smooth

            q_target_smooth = q_target_smooth + v_target * dt
            self.v_target_smooth = v_target
        else:
            # TODO(真机核对): 模式 4 纯位置 PD 的假设——固件直接跟踪 q_target，不做目标
            # 限幅，因此内部目标在数值上等于 q_target。
            q_target_smooth = q_target
            self.v_target_smooth = q_target * 0

        self.q_target_smooth = q_target_smooth

        # 固件位置环：P（1/p_scale）+ D（1/d_scale）；Ki 在位置模式下不生效。D 项作用于
        # 实际关节角速度。缩放分母可通过 p_scale/d_scale 配置（默认 p_scale=1.0——模式 4
        # 实测无 1/8 缩放；d_scale=4.0——D 缩放≈1/4，沿用 HLS 惯例，精确值待精细确认）。
        kp = self.kp
        error_gain = self.model.error_gain.value
        kd = self.model.kd.value
        duty_cycle = (
            (q_target_smooth - q) * kp * error_gain / self.p_scale
            + (-dq) * kd * error_gain / self.d_scale
        )

        # 固件电流限幅：把占空比约束到使电机电流 I = (duty · vin - kt · dq) / R 维持在
        # [-max_current, max_current] 的窗口（与基类相同的"尝试"语义）。
        if self.model.max_current.value > 0:
            back_emf = self.model.kt.value * dq
            duty_span = self.model.R.value * self.model.max_current.value / self.vin
            duty_center = back_emf / self.vin
            duty_cycle = self.backend.clamp(
                duty_cycle, duty_center - duty_span, duty_center + duty_span
            )

        # 物理 PWM 限制（电压受电池限制）——最后应用。
        duty_cycle = self.backend.clamp(duty_cycle, -self.max_pwm, self.max_pwm)
        self.duty_cycle = duty_cycle  # 供日志（以及电池压降模型）使用

        return self.vin * duty_cycle
