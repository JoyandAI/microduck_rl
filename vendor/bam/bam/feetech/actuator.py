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
    """

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

    def get_extra_inertia(self) -> float:
        return self.model.armature.value

    def load_log(self, log: dict):
        super().load_log(log)

        self.q_target_smooth = np.zeros_like(self.kp)

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
        """
        Assumes the motor is using a kp controller
        This can be overloaded if more custom behaviour is used
        """

        # Internal target position is clipped using maximum velocity
        self.q_target_smooth = self.backend.clamp(
            q_target,
            self.q_target_smooth - self.model.max_velocity.value * dt,
            self.q_target_smooth + self.model.max_velocity.value * dt,
        )

        duty_cycle = (
            (self.q_target_smooth - q)
            * self.kp
            * self.error_gain
            * self.model.error_gain_ratio.value
        )
        duty_cycle = self.backend.clamp(duty_cycle, -self.max_pwm, self.max_pwm)

        return self.vin * duty_cycle


class HLS2909Actuator(VoltageControlledActuator):
    """
    Feetech HL-2909-C001 (12V 9kg.cm 恒力 TTL 舵机, 位置伺服模式 mode 0).

    与 STS3215 的差别（官方内存表 docs/feetech_hls_memtable.md）:
      * 固件目标是"加速度限幅 → 速度限幅 → 位置"二阶爬坡（STS3215 只有速度限幅）;
      * 位置环是 P-D-I（Kp 1/8 缩放, Kd 1/4 缩放, Ki 位置模式无效）;
      * 堵转电流 0.6A@12V  → 固件限流建模为 duty 约束（max_current）;
      * 电压 12V (9-14V), 减速比 320:1 金属齿, 摩擦预算显著大于 XL330。

    未标定参数的取值规则: 占位值仅在冒烟/联调用; 训练前必须由
    scripts/read_hls_registers.py (出厂参数区 77-86) + testbench/示波器填入,
    并把数值写入 bam/params/hls2909/mX.json (error_gain/kd/max_velocity/
    max_acceleration 均为 model 参数, 可直接配置在 JSON 里)。
    """

    def __init__(self, testbench_class: Testbench):
        super().__init__(
            testbench_class,
            vin=12.0,
            # 占位: 出厂寄存器 21 (位置环P) 读取/示波器反解后填入; mjlab 侧由
            # BamActuatorCfg.kp_fw 覆盖, 此处仅为独立使用时的默认值。
            kp=200.0,
            # 占位: 示波器标定 (duty/error 斜率 × 1/8 缩放组合后的等效值)
            error_gain=0.166,
            max_pwm=0.97,
            max_current=1.95,  # 出厂真值: reg28/44 = 300×6.5mA (官网 0.6A 是等效测试电流)
        )
        # 占位 (出厂参数 84/85/86 或阶跃实测后填入):
        #   84 最大速度限制 → max_velocity [rad/s] (空载 77rpm = 8.06 rad/s 为上限)
        #   85×86 加速度限制×倍数 → max_acceleration [rad/s²] (254×8.7°/s² = 38.6 为上限)
        self.default_max_velocity = 8.06
        self.default_max_acceleration = 38.6
        # 位置环 D 系数 (出厂 22/51, 1/4 缩放); 0 = 无 D 项（占位, 标定后填）
        self._kd_placeholder = 0.0

    def get_extra_inertia(self) -> float:
        return self.model.armature.value

    def initialize(self):
        # 官网 KT=14.83 kg.cm/A = 1.454 Nm/A (输出端)
        self.model.kt = Parameter(1.454, 0.2, 3.0)
        # 12V / 0.6A 堵转电流的等效电阻 (待测试台拟合修正)
        self.model.R = Parameter(20.0, 2.0, 50.0)
        # 反射转子惯量 (占位: STS3215 默认同值; 标定后填)
        self.model.armature = Parameter(1e-4, 1e-6, 0.05)
        self.model.q_offset = Parameter(0.0, -0.2, 0.2)
        # 固件运动学限幅 (可经 JSON 配置: "error_gain"/"kd"/"max_velocity"/...)
        self.model.error_gain = Parameter(self.error_gain, 0.01, 2.0)
        self.model.kd = Parameter(self._kd_placeholder, 0.0, 254.0)
        self.model.max_velocity = Parameter(
            self.default_max_velocity, 0.1 * self.default_max_velocity, 10 * self.default_max_velocity
        )
        self.model.max_acceleration = Parameter(
            self.default_max_acceleration,
            0.1 * self.default_max_acceleration,
            10 * self.default_max_acceleration,
        )
        # 固件电流上限 (reg28 保护电流, 出厂 = 44 目标电流): BAM 限流约束用
        self.model.max_current = Parameter(1.95, 0.1, 5.0)
        # 二阶爬坡状态 (lazy-init in compute_control: mjlab 路径无 load_log)
        self._q_target_smooth = None
        self._v_target_smooth = None

    def compute_control(
        self, q_target, q, dq, dt
    ):
        """固件位置伺服控制律: 加速度+速度双限幅目标爬坡 + P(1/8) + D(1/4) + 限流。

        返回电压 [V]。
        """
        v_max = self.model.max_velocity.value
        a_max = self.model.max_acceleration.value
        kp = self.kp
        error_gain = self.model.error_gain.value
        kd = self.model.kd.value

        # -- state (lazy init: 与 q 同形状 (N, J)) -----------------------------
        if self._q_target_smooth is None:
            # q*0 → 与 q 同形状的零 (numpy/torch 通用, torch 下保留 autograd 语义)
            self._q_target_smooth = q * 0
            self._v_target_smooth = q * 0

        # -- 1. 目标位置二阶爬坡: accel -> velocity -> position ----------------
        # 期望速度 = (目标 - 平滑目标)/dt, 先限速度, 再按加速度限幅爬坡,
        # 最后加"缓停"末段约束: 剩余距离不足时按剩余距离减速 (防超调)。
        v_target = (q_target - self._q_target_smooth) / dt
        v_target = self.backend.clamp(v_target, -v_max, v_max)
        v_target = self.backend.clamp(
            v_target,
            self._v_target_smooth - a_max * dt,
            self._v_target_smooth + a_max * dt,
        )
        rem_abs = abs(q_target - self._q_target_smooth) / dt
        v_target = self.backend.clamp(v_target, -rem_abs, rem_abs)
        self._q_target_smooth = self._q_target_smooth + v_target * dt
        self._v_target_smooth = v_target

        # -- 2. 位置环 P(1/8) + D(1/4) (官方内存表: Kp 1/8 缩放, Kd 1/4, Ki 位置模式无效)
        duty_cycle = (
            (self._q_target_smooth - q) * kp * error_gain / 8.0
            + (-dq) * kd * error_gain / 4.0
        )

        # -- 3. 固件电流限幅 (仅能约束 duty, 与基类 VoltageControlledActuator 同式)
        if self.model.max_current.value > 0:
            back_emf = self.model.kt.value * dq
            duty_span = self.model.R.value * self.model.max_current.value / self.vin
            duty_center = back_emf / self.vin
            duty_cycle = self.backend.clamp(
                duty_cycle, duty_center - duty_span, duty_center + duty_span
            )

        # -- 4. 物理 PWM 钳位
        duty_cycle = self.backend.clamp(duty_cycle, -self.max_pwm, self.max_pwm)

        return self.vin * duty_cycle
