"""OpenMD models load, and the registered tasks point at them."""

from __future__ import annotations

import mujoco
import pytest

from mjlab_microduck.robot.openmd_constants import (
    OPENMD_ALLCOLLISIONS_XML,
    OPENMD_ALLCOLLISIONS_BACKLASH_XML,
    OPENMD_HOME_FRAME,
    OPENMD_STAND_XML,
    OPENMD_WALK_BACKLASH_XML,
    OPENMD_WALK_XML,
    get_openmd_allcollisions_backlash_spec,
    get_openmd_allcollisions_spec,
    get_openmd_ground_pick_spec,
    get_openmd_stand_spec,
    get_openmd_walk_backlash_spec,
    get_openmd_walk_spec,
)

SERVO_JOINTS = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)

# STAND qpos joint slice from openmicroduck_flat_scene.xml.
STAND_JOINTS = (
    0.0,
    -0.08726646259971647,
    -0.457924,
    0.0,
    0.452984,
    0.3490658503988659,
    0.3490658503988659,
    0.0,
    0.0,
    0.0,
    0.08726646259971647,
    0.457924,
    0.0,
    -0.452984,
)


def _hinge_names(xml) -> list[str]:
    model = mujoco.MjModel.from_xml_path(str(xml))
    names = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))
    return names


def test_walk_joint_layout_and_stand_angles() -> None:
    model = mujoco.MjModel.from_xml_path(str(OPENMD_WALK_XML))
    assert _hinge_names(OPENMD_WALK_XML) == list(SERVO_JOINTS)
    assert model.nu == 14
    assert model.nq == 21
    scene = OPENMD_WALK_XML.with_name("scene_walk.xml")
    scene_model = mujoco.MjModel.from_xml_path(str(scene))
    key = mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    qpos = scene_model.key_qpos[key]
    assert qpos[7:] == pytest.approx(STAND_JOINTS, abs=1e-9)


def test_home_frame_matches_the_source_stand_keyframe() -> None:
    """The numbers HOME applies are the source scene's STAND joint angles."""
    assert OPENMD_HOME_FRAME.joint_pos[r".*left_hip_pitch.*"] == -0.457924
    assert OPENMD_HOME_FRAME.joint_pos[r".*left_knee.*"] == 0.0
    assert OPENMD_HOME_FRAME.joint_pos[r".*right_knee.*"] == 0.0
    assert OPENMD_HOME_FRAME.joint_pos[r".*left_ankle.*"] == 0.452984
    assert OPENMD_HOME_FRAME.joint_pos[r".*neck_pitch.*"] == 0.3490658503988659


def test_collision_variants_and_backlash() -> None:
    walk = mujoco.MjModel.from_xml_path(str(OPENMD_WALK_XML))
    stand = mujoco.MjModel.from_xml_path(str(OPENMD_STAND_XML))
    full = mujoco.MjModel.from_xml_path(str(OPENMD_ALLCOLLISIONS_XML))
    back = mujoco.MjModel.from_xml_path(str(OPENMD_WALK_BACKLASH_XML))
    full_back = mujoco.MjModel.from_xml_path(str(OPENMD_ALLCOLLISIONS_BACKLASH_XML))

    def colliding(model: mujoco.MjModel) -> int:
        return sum(1 for g in range(model.ngeom) if model.geom_contype[g] != 0)

    assert colliding(walk) < colliding(stand) < colliding(full)
    assert back.nq == 35
    assert back.nu == 14
    assert full_back.nq == 35
    assert full_back.nu == 14
    passive = [n for n in _hinge_names(OPENMD_WALK_BACKLASH_XML) if n.startswith("passive_")]
    assert len(passive) == 14
    assert all(n.endswith("_backlash") for n in passive)
    full_passive = [
        n for n in _hinge_names(OPENMD_ALLCOLLISIONS_BACKLASH_XML) if n.startswith("passive_")
    ]
    assert full_passive == [
        f"passive_{name}_backlash" for name in SERVO_JOINTS
    ]
    servo_geoms = [
        mujoco.mj_id2name(full, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        for g in range(full.ngeom)
    ]
    assert any(name.endswith("_servo_collision") for name in servo_geoms)


def test_openmd_tasks_use_the_hd1910_m6_actuator() -> None:
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    expected = {
        "Mjlab-StandUp-Flat-OpenMD": get_openmd_stand_spec,
        "Mjlab-Velocity-Flat-OpenMD": get_openmd_walk_spec,
        "Mjlab-Velocity-Flat-Backlash-OpenMD": get_openmd_walk_backlash_spec,
        "Mjlab-VelStand-Flat-OpenMD": get_openmd_allcollisions_spec,
        "Mjlab-VelStand-Rough-OpenMD": get_openmd_allcollisions_spec,
        "Mjlab-VelStand-Flat-Backlash-OpenMD": get_openmd_allcollisions_backlash_spec,
        "Mjlab-VelStand-Rough-Backlash-OpenMD": get_openmd_allcollisions_backlash_spec,
        "Mjlab-GroundPick-Flat-OpenMD": get_openmd_ground_pick_spec,
        "Mjlab-GroundPick-Rough-OpenMD": get_openmd_ground_pick_spec,
        "Mjlab-BallKick-Flat-OpenMD-Right": get_openmd_stand_spec,
        "Mjlab-BallKick-Flat-OpenMD-Left": get_openmd_stand_spec,
        "Mjlab-SitStand-Flat-OpenMD": get_openmd_stand_spec,
        "Mjlab-SitStand-Rough-OpenMD": get_openmd_stand_spec,
        "Mjlab-Roulade-Flat-OpenMD": get_openmd_stand_spec,
    }
    roulade = load_env_cfg("Mjlab-Roulade-Flat-OpenMD", play=True)
    head = next(s for s in roulade.scene.sensors if s.name == "head_ground_contact")
    assert head.primary.pattern == "bottom_head_shell"
    assert roulade.sim.nconmax >= 200
    pick = load_env_cfg("Mjlab-GroundPick-Flat-OpenMD", play=True)
    assert pick.rewards["mouth_payload_force"].params["asset_cfg"].body_names == (
        "bottom_head_shell",
    )
    assert pick.sim.nconmax >= 200
    spec = get_openmd_ground_pick_spec()
    assert "mouth_tip" in [s.name for s in spec.sites]
    kick = load_env_cfg("Mjlab-BallKick-Flat-OpenMD-Right", play=True)
    assert "ball" in kick.scene.entities
    support = next(s for s in kick.scene.sensors if s.name == "support_foot_ground_contact")
    assert support.primary.pattern == "^left_foot_collision$"
    assert kick.events["reset_ball"].params["offset"][1] < 0
    assert kick.sim.nconmax >= 200
    assert load_rl_cfg("Mjlab-BallKick-Flat-OpenMD-Right").experiment_name == "openmd_ball_kick_right"
    kick_left = load_env_cfg("Mjlab-BallKick-Flat-OpenMD-Left", play=True)
    support_left = next(
        s for s in kick_left.scene.sensors if s.name == "support_foot_ground_contact"
    )
    assert support_left.primary.pattern == "^right_foot_collision$"
    assert kick_left.events["reset_ball"].params["offset"][1] > 0
    assert load_rl_cfg("Mjlab-BallKick-Flat-OpenMD-Left").experiment_name == (
        "openmd_ball_kick_left"
    )
    sitstand = load_env_cfg("Mjlab-SitStand-Flat-OpenMD", play=True)
    head = next(s for s in sitstand.scene.sensors if s.name == "head_ground_contact")
    assert head.primary.pattern == "bottom_head_shell"
    assert sitstand.sim.nconmax >= 200
    for task_id, spec_fn in expected.items():
        cfg = load_env_cfg(task_id, play=True)
        robot = cfg.scene.entities["robot"]
        act = robot.articulation.actuators[0]
        assert act.motor_name == "hd1910", task_id
        assert act.model == "m6", task_id
        assert act.kp_fw == 5.0, task_id
        assert robot.spec_fn is spec_fn, task_id
        assert spec_fn().modelname == "openmd", task_id
        if "Backlash" in task_id:
            term = cfg.observations["actor"].terms["joint_pos"]
            assert term.func.__name__ == "joint_pos_rel_backlash", task_id
