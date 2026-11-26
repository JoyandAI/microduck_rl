import mujoco
import numpy as np
import pytest
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg

from mjlab_microduck.robot.microduck_constants import FULL_COLLISION
from mjlab_microduck.robot.openmd_constants import OPENMD_STAND_XML, get_openmd_stand_spec


@pytest.fixture(scope="module")
def models():
    cfg = load_env_cfg("Mjlab-StandUp-Flat-OpenMD", play=False)
    cfg.scene.num_envs = 1
    trained = Scene(cfg.scene, device="cpu").compile()
    native = mujoco.MjModel.from_xml_path(str(OPENMD_STAND_XML.with_name("scene_stand.xml")))
    return trained, native


def test_stand_scene_matches_trained_collision_properties(models):
    trained, native = models
    names = {native.geom(i).name for i in range(native.ngeom)} - {"floor"}
    assert names == {trained.geom(i).name.removeprefix("robot/") for i in range(trained.ngeom)} - {"terrain"}
    # Check the contact recipe that was in use when existing checkpoints were trained.
    spec = get_openmd_stand_spec()
    FULL_COLLISION.edit_spec(spec)
    for name in sorted(names | {"floor"}):
        actual = native.geom(name)
        expected = trained.geom("terrain" if name == "floor" else "robot/" + name)
        for field in ("contype", "conaffinity", "condim", "priority", "friction", "solref", "solimp"):
            np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field),
                                          err_msg=f"{name}.{field}")
            if name != "floor":
                np.testing.assert_allclose(getattr(actual, field), getattr(spec.geom(name), field),
                                           err_msg=f"training recipe: {name}.{field}")
    assert trained.nexclude == native.nexclude == 0


@pytest.mark.parametrize("probe", ["suspended", "feet", "head"])
def test_stand_scene_contact_probes(models, probe):
    trained, native = models
    for model, prefix, floor_name in ((trained, "robot/", "terrain"), (native, "", "floor")):
        data = mujoco.MjData(model)
        data.qpos[:] = native.key("STAND").qpos
        data.qpos[2] = 0.5
        if probe == "head":
            data.qpos[3:7] = (0, 0, 1, 0)
        mujoco.mj_forward(model, data)
        if probe == "suspended":
            pairs = {
                frozenset(model.geom(i).name.removeprefix(prefix) for i in contact.geom)
                for contact in data.contact
            }
            assert pairs == {
                frozenset(("neck_pitch_collision", "jaw_collision")),
                frozenset(("neck_pitch_collision", "bottom_head_shell_collision")),
            }
            continue
        floor = model.geom(floor_name).id
        targets = ("left_foot_collision", "right_foot_collision") if probe == "feet" else ("top_head_shell_collision",)
        ids = {model.geom(prefix + name).id for name in targets}
        distance = min(mujoco.mj_geomDistance(model, data, floor, i, 2.0, None) for i in ids)
        data.qpos[2] -= distance + 0.001
        mujoco.mj_forward(model, data)
        loaded = set()
        for k, contact in enumerate(data.contact):
            if floor in contact.geom:
                wrench = np.zeros(6)
                mujoco.mj_contactForce(model, data, k, wrench)
                if wrench[0] > 0:
                    loaded.update(i for i in contact.geom if i != floor)
        assert ids <= loaded
