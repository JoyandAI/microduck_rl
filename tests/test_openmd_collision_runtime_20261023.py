"""Check compiled physics, not merely collision names in an unprocessed spec."""
import pytest

from scripts.audit_openmd_collision_runtime_20261023 import inspect
from mjlab_microduck.robot.openmd_constants import (
    OPENMD_ALLCOLLISIONS_ROBOT_CFG,
    OPENMD_ALLCOLLISIONS_BACKLASH_ROBOT_CFG,
    OPENMD_STAND_ROBOT_CFG,
)


@pytest.fixture(scope="module", params=[OPENMD_ALLCOLLISIONS_ROBOT_CFG,
                                       OPENMD_ALLCOLLISIONS_BACKLASH_ROBOT_CFG])
def comparison(request):
    return inspect(request.param, raw=True), inspect(request.param)


def test_compiled_shell_masks_recover_collision(comparison):
    before, after = comparison
    for part in ("top_head_shell", "bottom_head_shell", "jaw", "neck_pitch", "trunk_base"):
        assert before["masks"][part]
        assert after["masks"][part]
        assert all(g["contype"] and g["conaffinity"] for g in after["masks"][part])
    assert after["lost_on_entity_build"] == []
    assert after["active_count"] >= before["active_count"]


def test_fix_changes_contact_but_preserves_inertial_properties(comparison):
    before, after = comparison
    assert before["body_mass"] == after["body_mass"]
    assert before["body_inertia"] == after["body_inertia"]
    # A regenerated XML can already have corrected names. Do not require the
    # raw source to remain broken just to keep this regression test passing.
    if "top_head_shell.001" in before["lost_on_entity_build"]:
        assert not before["floor_probe"]
    assert any(c["geom"] == "top_head_shell_collision" and c["normal_force_n"] > 0
               for c in after["floor_probe"])


def test_stand_family_already_has_head_collision():
    before, after = inspect(OPENMD_STAND_ROBOT_CFG, raw=True), inspect(OPENMD_STAND_ROBOT_CFG)
    assert before["active_geoms"] == after["active_geoms"]
    assert before["floor_probe"] == after["floor_probe"]
    assert any(c["geom"] == "top_head_shell_collision" and c["normal_force_n"] > 0
               for c in after["floor_probe"])
