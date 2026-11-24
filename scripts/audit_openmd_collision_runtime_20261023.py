"""Compare raw-export and current Entity collision masks, including a floor probe.

No training, servo changes, or edits to the source robot. The contact probe is
static MuJoCo forward dynamics, not a BAM policy rollout or a stability test.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import mjlab_microduck.tasks  # register before importing robot constants
from mjlab.entity import Entity
from mjlab_microduck.robot import openmd_constants as c


def inspect(cfg, raw=False):
    cfg = deepcopy(cfg)
    xml = {c.get_openmd_stand_spec: c.OPENMD_STAND_XML,
           c.get_openmd_allcollisions_spec: c.OPENMD_ALLCOLLISIONS_XML,
           c.get_openmd_allcollisions_backlash_spec: c.OPENMD_ALLCOLLISIONS_BACKLASH_XML}[cfg.spec_fn]
    if raw:
        cfg.spec_fn = lambda: mujoco.MjSpec.from_file(str(xml))
    source = cfg.spec_fn()
    source_on = {g.name for g in source.geoms if g.contype or g.conaffinity}
    entity = Entity(cfg)  # applies CollisionCfg and actuator spec edits
    entity.spec.worldbody.add_geom(name="audit_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                  size=[2, 2, .1], contype=1, conaffinity=1)
    m = entity.compile()
    active = [m.geom(i).name for i in range(m.ngeom)
              if m.geom(i).name != "audit_floor" and (m.geom_contype[i] or m.geom_conaffinity[i])]
    names = ("bottom_head_shell", "top_head_shell", "jaw", "neck_pitch", "trunk_base", "left_shell", "leg_left")
    masks = {}
    for part in names:
        matches = [i for i in range(m.ngeom)
                   if m.geom(i).name in (part+".001", part+"_collision")]
        masks[part] = [dict(name=m.geom(i).name, contype=int(m.geom_contype[i]),
                           conaffinity=int(m.geom_conaffinity[i])) for i in matches]
    # Put the HOME robot upside-down, then move it until its lowest head-shell
    # collision surface is 1 mm below the floor. mj_geomDistance ignores masks,
    # which is exactly what we need to compare enabled vs disabled surfaces.
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    free = next(i for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE)
    adr = m.jnt_qposadr[free]
    d.qpos[adr:adr+3] = [0, 0, .5]
    mujoco.mj_forward(m, d)
    home_self_contacts = [dict(geom1=m.geom(d.contact[k].geom1).name,
                               geom2=m.geom(d.contact[k].geom2).name,
                               distance_m=float(d.contact[k].dist)) for k in range(d.ncon)
                          if d.contact[k].dist < -.0001]
    d.qpos[adr+3:adr+7] = [0, 0, 1, 0]
    mujoco.mj_forward(m, d)
    floor = m.geom("audit_floor").id
    head = [i for i in range(m.ngeom) if m.geom(i).name in
            ("bottom_head_shell.001", "top_head_shell.001", "jaw.001",
             "bottom_head_shell_collision", "top_head_shell_collision", "jaw_collision")]
    minimum = min(mujoco.mj_geomDistance(m, d, floor, g, 2., None) for g in head)
    d.qpos[adr+2] -= minimum + .001
    mujoco.mj_forward(m, d)
    contacts = []
    for k in range(d.ncon):
        contact = d.contact[k]
        pair = [int(contact.geom1), int(contact.geom2)]
        if floor not in pair:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(m, d, k, force)
        other = pair[1] if pair[0] == floor else pair[0]
        contacts.append(dict(geom=m.geom(other).name, distance_m=float(contact.dist),
                             normal_force_n=float(force[0])))
    result = dict(xml=str(xml), xml_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),
                  raw_export_loader=raw, active_geoms=active, active_count=len(active),
                  lost_on_entity_build=sorted(source_on-set(active)), masks=masks,
                  mass_kg=float(m.body_mass.sum()),
                  head_subtree_mass_kg=float(m.body_subtreemass[m.body("neck").id]),
                  body_mass=m.body_mass.tolist(), body_inertia=m.body_inertia.tolist(),
                  home_self_contacts=home_self_contacts,
                  floor_probe=contacts,
                  floor_probe_scope="Static forward at upside-down HOME, shell penetration 1 mm; not policy dynamics")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", type=Path, required=True)
    a = p.parse_args()
    report = {}
    for label, cfg in (("stand", c.OPENMD_STAND_ROBOT_CFG),
                       ("allcollisions", c.OPENMD_ALLCOLLISIONS_ROBOT_CFG),
                       ("allcollisions_backlash", c.OPENMD_ALLCOLLISIONS_BACKLASH_ROBOT_CFG)):
        before, after = inspect(cfg, raw=True), inspect(cfg)
        report[label] = dict(raw_loader=before, current_loader=after,
                             mass_identical=before["body_mass"] == after["body_mass"],
                             inertia_identical=before["body_inertia"] == after["body_inertia"])
        print(label, "active", before["active_count"], "->", after["active_count"],
              "mass", after["mass_kg"], "head", after["head_subtree_mass_kg"], flush=True)

    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
