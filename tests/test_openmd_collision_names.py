"""OpenMD collision geom names must match the entity collision selectors."""
from mjlab_microduck.robot.openmd_constants import (
    get_openmd_allcollisions_backlash_spec,
    get_openmd_allcollisions_spec,
)


def _bad_collision_geoms(spec):
    return [
        (g.name or "<unnamed>")
        for g in spec.geoms
        if g.contype and not (g.name or "").endswith("_collision")
    ]


def test_allcollisions_every_collision_geom_is_selectable():
    assert _bad_collision_geoms(get_openmd_allcollisions_spec()) == []


def test_allcollisions_backlash_every_collision_geom_is_selectable():
    assert _bad_collision_geoms(get_openmd_allcollisions_backlash_spec()) == []


def test_head_shells_and_jaw_are_collidable():
    for get in (get_openmd_allcollisions_spec,
                get_openmd_allcollisions_backlash_spec):
        spec = get()  # keep alive: MjSpec geoms are owned by the spec
        names = {g.name for g in spec.geoms if g.contype}
        for part in ("bottom_head_shell", "top_head_shell", "jaw", "neck_pitch"):
            assert f"{part}_collision" in names, (part, sorted(names)[:5])
