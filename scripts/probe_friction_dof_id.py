#!/usr/bin/env python
"""Is a FRICTION_DOF constraint's `efc_id` a DOF index or a JOINT index?

Upstream `JoyandAI/microduck_rl` commit 330754b ("fix(bam): match FRICTION_DOF
efc_id against dof indexes, not joint ids") claims MuJoCo stores a **dof
address** in FRICTION_DOF constraints, so `bam.mujoco.MujocoController` matched
the wrong rows: actuator `trnid` is a *joint* id, and on Microduck the freejoint
shifts every hinge joint by 5.

This probe answers the question on the REAL robot model — not a toy XML — and
quantifies what each selector actually picks up:

  * `set(dof_indexes) <= set(efc_ids)`  → efc_id is a dof index
    ⇒ the WARP/TRAINING path (`bam.mjlab.BamActuator._friction_dof_force`,
    which scatters with `idx = efc_id` into an `nv`-wide tensor) is CORRECT.
  * `set(joint_indexes) <= set(efc_ids)` → efc_id is a joint id
    ⇒ the CPU path (`bam.mujoco.MujocoController`) was correct all along.

Run:  uv run python scripts/probe_friction_dof_id.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_XML  # noqa: E402

FRICTION_DOF = int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)


def build_model(xml: Path) -> mujoco.MjModel:
    """Walk model with a live Coulomb budget on the 14 servo dofs.

    Mirrors what BAM writes each step: XML friction is zeroed and the budget is
    rewritten into `dof_frictionloss` (see friction_dr_bam / bam.mjlab).
    """
    model = mujoco.MjModel.from_xml_path(str(xml))
    # `actuator_trnid` holds a JOINT id; `jnt_dofadr` maps it to the DOF address
    # — the exact pair of quantities the two implementations disagree about.
    joint_indexes = np.asarray(model.actuator_trnid[: model.nu, 0], dtype=int)
    dof_indexes = np.asarray(model.jnt_dofadr[joint_indexes], dtype=int)

    model.dof_frictionloss[:] = 0.0
    model.dof_frictionloss[dof_indexes] = 0.02  # N·m, order of the m5 budget
    return model, joint_indexes, dof_indexes


def friction_rows(model: mujoco.MjModel, data: mujoco.MjData):
    nefc = int(data.nefc)
    types = np.asarray(data.efc_type[:nefc])
    ids = np.asarray(data.efc_id[:nefc])
    force = np.asarray(data.efc_force[:nefc])
    m = types == FRICTION_DOF
    return ids[m], force[m]


def selector_sum(efc_ids, efc_force, target_id):
    """Exactly `bam.mujoco.MujocoController`'s batched element-wise test."""
    return float(np.sum(efc_force * (efc_ids == target_id)))


def main() -> int:
    xml = Path(MICRODUCK_WALK_XML)
    print(f"model: {xml.relative_to(REPO)}")
    model, joint_indexes, dof_indexes = build_model(xml)
    data = mujoco.MjData(model)

    print(f"nv={model.nv}  nu={model.nu}  nq={model.nq}")
    print(f"actuator trnid (JOINT ids)     : {joint_indexes.tolist()}")
    print(f"jnt_dofadr       (DOF indexes) : {dof_indexes.tolist()}")
    overlap = sorted(set(joint_indexes.tolist()) & set(dof_indexes.tolist()))
    print(
        f"joint-id ∩ dof-index = {overlap}  "
        f"({len(overlap)}/{model.nu} ids alias to a DIFFERENT joint)"
    )

    # Give every servo dof a velocity so the Coulomb rows carry force.
    data.qvel[:] = 0.0
    data.qvel[dof_indexes] = 0.5
    data.qpos[2] = 0.25  # free joint height, keeps it off the floor
    mujoco.mj_forward(model, data)

    efc_ids, efc_force = friction_rows(model, data)
    uniq = sorted(set(efc_ids.tolist()))
    print(f"\nFRICTION_DOF rows: {len(efc_ids)}  efc_id values = {uniq}")
    print(f"efc_id range: [{min(uniq)}, {max(uniq)}]   nv-1 = {model.nv - 1}")

    dof_ok = set(dof_indexes.tolist()).issubset(set(efc_ids.tolist()))
    jnt_ok = set(joint_indexes.tolist()).issubset(set(efc_ids.tolist()))
    print(f"\nset(dof_indexes) ⊆ set(efc_ids)  -> {dof_ok}   (efc_id IS a dof index)")
    print(f"set(joint_indexes) ⊆ set(efc_ids) -> {jnt_ok}   (efc_id IS a joint id)")

    # Per-joint friction force each selector collects.
    print("\n per-joint friction force collected by each selector (N·m)")
    print(f"{'jnt':>4} {'dof':>4} {'CORRECT (dof match)':>20} {'CPU BUG (joint match)':>22}")
    tot_right = tot_wrong = 0.0
    for j, d in zip(joint_indexes, dof_indexes):
        right = selector_sum(efc_ids, efc_force, d)
        wrong = selector_sum(efc_ids, efc_force, j)
        tot_right += abs(right)
        tot_wrong += abs(wrong)
        flag = "" if abs(right - wrong) < 1e-12 else "  <-- MISATTRIBUTED"
        print(f"{j:>4} {d:>4} {right:>20.6f} {wrong:>22.6f}{flag}")
    print(f"{'Σ|·|':>9} {tot_right:>20.6f} {tot_wrong:>22.6f}")

    # Who receives the mis-stripped force?
    print("\n which joint's friction the CPU selector actually picks up per actuator:")
    for i, j in enumerate(joint_indexes):
        wrong = selector_sum(efc_ids, efc_force, j)
        if abs(wrong) < 1e-12:
            continue
        donor = [jj for jj, dd in zip(joint_indexes, dof_indexes) if dd == j]
        print(f"  actuator[joint {j:>2}] strips the friction of joint {donor}")

    print("\nverdict:", end=" ")
    if dof_ok and not jnt_ok:
        print("efc_id is a DOF index.")
        print("  TRAINING path (bam/mjlab.py, scatter idx=efc_id into nv)  = CORRECT")
        print("  CPU path (bam/mujoco.py, compare to actuator trnid)       = BUGGY")
    elif jnt_ok and not dof_ok:
        print("efc_id is a JOINT id — the upstream fix would be wrong.")
    else:
        print("ambiguous on this model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
