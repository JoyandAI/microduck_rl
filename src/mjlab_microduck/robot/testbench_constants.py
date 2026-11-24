"""Single-servo testbench entity configuration for sim2real validation.

The bench uses a Feetech HD-1910-C001 servo (21 g, 5 V rail).
The XML file/dir keep their historical `xl330_test_bench` name (renaming them
would churn the bench tooling); only the *actuator model* is updated to the servo
actually mounted, so bench sims match the identified parameters.
"""

import os
from pathlib import Path

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from bam.mjlab import BamActuatorCfg


_TESTBENCH_DIR: Path = Path(os.path.dirname(__file__)) / "xl330_test_bench"
# Use the robot-only XML (no floor / no lights): mjlab's TerrainImporterCfg
# adds its own ground plane, so scene.xml would give a duplicated floor.
TESTBENCH_XML: Path = _TESTBENCH_DIR / "xl330_test_bench.xml"

assert TESTBENCH_XML.exists(), f"XML not found: {TESTBENCH_XML}"


# Real-device payload mass (120 g)
TESTBENCH_ARM_MASS: float = 0.12


def _set_arm_mass(spec: mujoco.MjSpec, mass: float) -> None:
    for body in spec.bodies:
        if body.name == "arm":
            original = body.mass
            if original > 0:
                scale = mass / original
                body.mass = mass
                body.fullinertia = [x * scale for x in body.fullinertia]
            break


def get_testbench_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(TESTBENCH_XML))
    _set_arm_mass(spec, TESTBENCH_ARM_MASS)
    return spec


HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={"1": 0.0},
    joint_vel={".*": 0.0},
)


# The 5 V pendulum fit uses the HD-1910 m5 control law at kp_fw=32.
# Robot training uses the 7.5 V m6 fit at kp_fw=5.
testbench_actuators = BamActuatorCfg(
    motor_name="hd1910",
    model="m5",
    target_names_expr=(r"1",),
    kp_fw=32.0,
    vin_range=(4.75, 5.25),   # bench supply is a regulated 5 V rail (5.0-5.2 V measured)
    vin_min=4.0,              # servo reg15 under-voltage alarm
    delay_min_lag=0,
    delay_max_lag=3,
)


XL330_TESTBENCH_ROBOT_CFG = EntityCfg(
    spec_fn=get_testbench_spec,
    init_state=HOME_FRAME,
    collisions=(),
    articulation=EntityArticulationInfoCfg(
        actuators=(testbench_actuators,),
        soft_joint_pos_limit_factor=1.0,
    ),
)
