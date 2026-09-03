#!/usr/bin/env python3
"""Feetech HL-2909-C001 swap: update servo masses in the MJCF models.

Structural geometry and joint axes are UNCHANGED (confirmed mechanically),
so this only updates the mass properties: each driven servo joint adds
+delta_m to the body that carries the servo CASE (the joint's parent body —
the servo housing is fixed to the parent link, the output shaft drives the
child). Inertia is scaled proportionally (mass*ratio); CoM is kept.

Model mass summary before/after is printed; the check `tests/` + smoke test
are the acceptance gate. Models updated: robot_walk, robot_allcollisions,
robot_allcollisions_rollers and their backlash twins (same bodies).

Usage:
    python scripts/update_servo_mass.py                     # apply to all models
    python scripts/update_servo_mass.py --delta-m 4.5 --dry-run
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import mujoco
from mujoco import mjtJoint

ROBOT_DIR = Path("src/mjlab_microduck/robot/microduck")

MODELS = [
    "robot_walk.xml",
    "robot_allcollisions.xml",
    "robot_allcollisions_rollers.xml",
    "robot_walk_backlash.xml",
    "robot_allcollisions_backlash.xml",
    "robot_allcollisions_rollers_backlash.xml",
]


def servo_parents(xml: Path) -> dict[str, str]:
    """joint name -> parent body NAME that carries the servo case."""
    m = mujoco.MjModel.from_xml_path(str(xml))
    out = {}
    for j in range(m.njnt):
        if m.jnt_type[j] != mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if not name or name.startswith("passive_"):
            continue  # backlash hinges / wheels: no servo
        b = m.jnt_bodyid[j]
        parent = m.body_parentid[b]
        pname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, parent)
        out[name] = pname
    return out


_INERTIAL_RE = re.compile(
    r'(<inertial pos="[^"]*" mass=")([0-9.eE+-]+)(" fullinertia=")([^"]*)("/>)'
)


def body_inertial(xml_text: str, body: str) -> tuple[float, list[float]]:
    """Parse (mass, fullinertia[6]) of a body directly from the XML text.

    Reading MjModel here is a trap: body_mass/body_inertia round-trip can
    reorder or drop elements vs the XML attribute; parsing the text is the
    single source of truth for both read and write.
    """
    block_pat = re.compile(
        rf'<body name="{re.escape(body)}"[^>]*>.*?(<inertial[^>]*/>)', re.S
    )
    m = block_pat.search(xml_text)
    assert m, f"body {body}: no <inertial> found"
    im = _INERTIAL_RE.match(m.group(1))
    assert im, f"body {body}: unexpected inertial line: {m.group(1)[:80]}"
    inertia = [float(x) for x in im.group(4).split()]
    assert len(inertia) == 6, f"body {body}: fullinertia has {len(inertia)} values"
    return float(im.group(2)), inertia


def apply_mass(xml: Path, delta_m: float, dry_run: bool) -> tuple[int, float, float]:
    """Add delta_m [kg] to each servo-case body; proportional inertia. Returns
    (n_bodies_updated, mass_before_total, mass_after_total)."""
    text = xml.read_text(encoding="utf-8")
    parents = servo_parents(xml)

    # group joint deltas per body (trunk_base carries 3 servos)
    per_body: dict[str, float] = {}
    for _jn, body in parents.items():
        per_body[body] = per_body.get(body, 0.0) + delta_m

    # capture OLD masses once, before any replacement (after-mass must not
    # re-read the edited text, or the delta is counted twice)
    old_masses = {b: body_inertial(text, b)[0] for b in per_body}
    total_before = sum(old_masses.values())
    n = 0
    for body, dm in sorted(per_body.items()):
        mass, inertia = old_masses[body], body_inertial(text, body)[1]
        ratio = (mass + dm) / mass
        new_inertia = [v * ratio for v in inertia]
        block_pat = re.compile(
            rf'(<body name="{re.escape(body)}"[^>]*>.*?)(<inertial[^>]*/>)(.*?</body>)',
            re.S,
        )
        found = False

        def block_repl(mo: re.Match) -> str:
            nonlocal found
            im = _INERTIAL_RE.match(mo.group(2))
            assert im, f"unexpected inertial line in {body}: {mo.group(2)[:80]}"
            found = True
            fv = " ".join(f"{v:.9g}" for v in new_inertia)
            return (
                f"{mo.group(1)}{im.group(1)}{mass + dm:.9g}"
                f"{im.group(3)}{fv}{im.group(5)}{mo.group(3)}"
            )

        text = block_pat.sub(block_repl, text, count=1)
        if not found:
            raise RuntimeError(f"body {body}: no <inertial> found in {xml}")
        n += 1

    total_after = sum(old_masses[b] + per_body[b] for b in per_body)

    if not dry_run:
        bak = xml.with_suffix(xml.suffix + ".bak_servoswap")
        if not bak.exists():
            shutil.copy2(xml, bak)
        xml.write_text(text, encoding="utf-8")
    return n, total_before, total_after


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--delta-m", type=float, default=0.0045,
                    help="mass increase per servo [kg] (XL330 18g -> HL-2909 22.5g)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--models", nargs="*", default=MODELS,
                    help="XML filenames under ROBOT_DIR")
    args = ap.parse_args()

    for fname in args.models:
        xml = ROBOT_DIR / fname
        if not xml.exists():
            print(f"SKIP {fname} (missing)")
            continue
        n, before, after = apply_mass(xml, args.delta_m, args.dry_run)
        tag = "DRY-RUN " if args.dry_run else "UPDATED "
        print(f"{tag}{fname:<44} {n:>2} bodies  {before*1000:7.1f}g -> {after*1000:7.1f}g"
              f"  (+{(after-before)*1000:6.1f}g = {n} servos x {args.delta_m*1000:g}g)")


if __name__ == "__main__":
    main()
