"""OpenMD robot cfgs.

Geometry and the STAND / SIT / FOLD angles come from the exported MJCF
(openmicroduck.xml + openmicroduck_flat_scene.xml), rebuilt by
``robot/openmd/build_models.py``. Actuator physics is the same HD-1910 m6
identification the MicroDuck tasks train with (``vendor/bam/bam/params/hd1910/m6.json``
via ``_BAM_ACTUATOR_KWARGS``). Joint names are the training layout so the
existing velocity / standup / velstand recipes resolve.
"""

from pathlib import Path

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from mjlab_microduck.robot.microduck_constants import (
    FULL_COLLISION,
    actuators,
    backlash_actuators,
)

_ROBOT_DIR: Path = Path(__file__).resolve().parent / "openmd"

OPENMD_WALK_XML: Path = _ROBOT_DIR / "robot_walk.xml"
OPENMD_STAND_XML: Path = _ROBOT_DIR / "robot_stand.xml"
OPENMD_ALLCOLLISIONS_XML: Path = _ROBOT_DIR / "robot_allcollisions.xml"
OPENMD_WALK_BACKLASH_XML: Path = _ROBOT_DIR / "robot_walk_backlash.xml"
OPENMD_ALLCOLLISIONS_BACKLASH_XML: Path = _ROBOT_DIR / "robot_allcollisions_backlash.xml"

assert OPENMD_WALK_XML.exists(), f"XML not found: {OPENMD_WALK_XML}"
assert OPENMD_STAND_XML.exists(), f"XML not found: {OPENMD_STAND_XML}"
assert OPENMD_ALLCOLLISIONS_XML.exists(), f"XML not found: {OPENMD_ALLCOLLISIONS_XML}"
assert OPENMD_WALK_BACKLASH_XML.exists(), f"XML not found: {OPENMD_WALK_BACKLASH_XML}"
assert OPENMD_ALLCOLLISIONS_BACKLASH_XML.exists(), f"XML not found: {OPENMD_ALLCOLLISIONS_BACKLASH_XML}"

# STAND keyframe in openmicroduck_flat_scene.xml. Knee is 0 (the source scene),
# not MicroDuck's ±0.28 deg. Trunk height in that keyframe is 0.12 m.
_L_HIP_ROLL = -0.08726646259971647
_L_HIP_PITCH = -0.457924
_L_ANKLE = 0.452984
_R_HIP_ROLL = 0.08726646259971647
_R_HIP_PITCH = 0.457924
_R_ANKLE = -0.452984
_NECK = 0.3490658503988659
_HEAD_PITCH = 0.3490658503988659

OPENMD_HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={
        r".*hip_yaw.*": 0.0,
        r".*left_hip_roll.*": _L_HIP_ROLL,
        r".*right_hip_roll.*": _R_HIP_ROLL,
        r".*left_hip_pitch.*": _L_HIP_PITCH,
        r".*right_hip_pitch.*": _R_HIP_PITCH,
        r".*left_knee.*": 0.0,
        r".*right_knee.*": 0.0,
        r".*left_ankle.*": _L_ANKLE,
        r".*right_ankle.*": _R_ANKLE,
        r".*neck_pitch.*": _NECK,
        r".*head_pitch.*": _HEAD_PITCH,
        r".*head_yaw.*": 0.0,
        r".*head_roll.*": 0.0,
    },
    joint_vel={".*": 0.0},
)

# Backlash hinges must match first: `.*left_hip_roll.*` would also hit
# passive_left_hip_roll_backlash and put it outside its ±1° range.
OPENMD_BACKLASH_HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={r".*_backlash$": 0.0, **OPENMD_HOME_FRAME.joint_pos},
    joint_vel={".*": 0.0},
)


def get_openmd_walk_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(OPENMD_WALK_XML))


def get_openmd_stand_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(OPENMD_STAND_XML))


def _name_body_collision_geoms(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Map CAD collision twins to the names selected by FULL_COLLISION."""
    counts: dict[str, int] = {}
    for geom in spec.geoms:
        name = geom.name or ""
        if geom.contype == 0 or not name.endswith(".001"):
            continue
        base = name[: -len(".001")]
        new = f"{base}_collision"
        n = counts.get(new, 0)
        counts[new] = n + 1
        geom.name = new if n == 0 else f"{new}_{n}"
    return spec


def get_openmd_allcollisions_spec() -> mujoco.MjSpec:
    return _name_body_collision_geoms(
        mujoco.MjSpec.from_file(str(OPENMD_ALLCOLLISIONS_XML)))


def get_openmd_walk_backlash_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(OPENMD_WALK_BACKLASH_XML))


def get_openmd_allcollisions_backlash_spec() -> mujoco.MjSpec:
    return _name_body_collision_geoms(
        mujoco.MjSpec.from_file(str(OPENMD_ALLCOLLISIONS_BACKLASH_XML)))


# Jaw centerline tip at the STAND keyframe, expressed in bottom_head_shell.
# Site +X is the beak direction (mostly world +X, slightly down). The ground-pick
# perpendicular reward treats that axis pointing at the floor as a peck.
_MOUTH_TIP_POS = (0.00699, -0.01311, 0.07709)
_MOUTH_TIP_QUAT = (0.73574, 0.05650, -0.67206, -0.06185)


def get_openmd_ground_pick_spec() -> mujoco.MjSpec:
    """Stand model plus a mouth_tip site. OpenMD has no such frame in the CAD export."""
    spec = mujoco.MjSpec.from_file(str(OPENMD_STAND_XML))
    site = spec.body("bottom_head_shell").add_site()
    site.name = "mouth_tip"
    site.pos = list(_MOUTH_TIP_POS)
    site.quat = list(_MOUTH_TIP_QUAT)
    site.group = 3
    return spec


def _robot(spec_fn, init_state, acts, *, collisions=(FULL_COLLISION,)) -> EntityCfg:
    return EntityCfg(
        spec_fn=spec_fn,
        init_state=init_state,
        collisions=collisions,
        articulation=EntityArticulationInfoCfg(
            actuators=(acts,),
            soft_joint_pos_limit_factor=0.9,
        ),
    )


OPENMD_WALK_ROBOT_CFG = _robot(get_openmd_walk_spec, OPENMD_HOME_FRAME, actuators)
# robot_stand.xml owns the contact settings for both training and native replay.
OPENMD_STAND_ROBOT_CFG = _robot(
    get_openmd_stand_spec, OPENMD_HOME_FRAME, actuators, collisions=()
)
OPENMD_ALLCOLLISIONS_ROBOT_CFG = _robot(
    get_openmd_allcollisions_spec, OPENMD_HOME_FRAME, actuators
)
OPENMD_WALK_BACKLASH_ROBOT_CFG = _robot(
    get_openmd_walk_backlash_spec, OPENMD_BACKLASH_HOME_FRAME, backlash_actuators
)
OPENMD_ALLCOLLISIONS_BACKLASH_ROBOT_CFG = _robot(
    get_openmd_allcollisions_backlash_spec,
    OPENMD_BACKLASH_HOME_FRAME,
    backlash_actuators,
)
OPENMD_GROUND_PICK_ROBOT_CFG = _robot(
    get_openmd_ground_pick_spec, OPENMD_HOME_FRAME, actuators, collisions=()
)
