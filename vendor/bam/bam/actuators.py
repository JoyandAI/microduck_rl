# Copyright 2025 Marc Duclusaud & Grégoire Passault

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

from .testbench import Pendulum
from .erob.actuator import ErobActuator
from .dynamixel.actuator import (
    MXActuator,
    XL320Actuator,
    XL330Actuator,
    XL330CurrentActuator,
)
from .feetech.actuator import HD1910Actuator, HLS2909Actuator, STS3215Actuator
from .unitree.actuator import UnitreeGo1Actuator
from .waveshare.actuator import ST3025Actuator

actuators = {
    # Dynamixel MX series
    "mx64": lambda: MXActuator(Pendulum),
    "mx106": lambda: MXActuator(Pendulum),
    # Dynamixel XL series
    "xl320": lambda: XL320Actuator(Pendulum),
    "xl330": lambda: XL330Actuator(Pendulum),
    "xl330i": lambda: XL330CurrentActuator(Pendulum),
    # eRob actuators with custom PD controller
    "erob80_100": lambda: ErobActuator(Pendulum, damping=2.0),
    "erob80_50": lambda: ErobActuator(Pendulum, damping=1.0),
    # Feetech STS3215
    "sts3215": lambda: STS3215Actuator(Pendulum),
    # Feetech HL-2909-C001 (12V, position servo mode with a firmware
    # velocity+acceleration rate-limited target and P(1/8)+D(1/4) gains)
    "hls2909": lambda: HLS2909Actuator(Pendulum),
    # Feetech HD-1910-C001 (6V, position servo mode; brother of HLS2909Actuator,
    # same interface — 6V/1.6A stall, kt = 7.5 kg·cm/A output side).
    # This is the servo the Microduck runs; it is the DEFAULT for all training
    # (see microduck_constants._BAM_ACTUATOR_KWARGS).
    "hd1910": lambda: HD1910Actuator(Pendulum),
    # "hd1909" is an alias — the same HD-1910-C001 servo as referred to in the
    # lab's shorthand. Registered so `--motor hd1909` / `motor_name="hd1909"`
    # (and the matching params dir) cannot silently fall through; the canonical
    # name remains "hd1910" (the spec sheet's model number).
    "hd1909": lambda: HD1910Actuator(Pendulum),
    # Waveshare ST3025
    "waveshare_st3025": lambda: ST3025Actuator(Pendulum),
    # Unitree Go1
    "unitree_go1": lambda: UnitreeGo1Actuator(Pendulum),
}
