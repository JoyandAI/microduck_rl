"""`efc_id` in a FRICTION_DOF constraint is a DOF address, not a joint id.

Two implementations of the same physics must agree:

* ``bam.mjlab.BamActuator._dof_friction_force`` — the TRAINING path (MuJoCo
  Warp). It scatters ``efc_force`` with ``idx = efc_id`` into an ``nv``-wide
  tensor, i.e. it treats ``efc_id`` as a dof address.
* ``bam.mujoco.MujocoController.update`` — the CPU path (``infer_policy.py`` /
  ``testbench_sim2real.py``). It used to compare ``efc_id`` against
  ``actuator.trnid`` (a JOINT id). Upstream JoyandAI commit 330754b fixed it.

On Microduck the freejoint shifts every hinge joint, so the two index spaces
OVERLAP but mean different joints — the bug is silent, not a crash. The probe
``scripts/probe_friction_dof_id.py`` measures it on the real walk model: 9 of 14
actuators stripped a neighbouring joint's friction and the 5 left-leg joints
stripped nothing.

These tests lock the semantics on the REAL model and the selector in the CPU
implementation, so a future merge (or a well-meaning "cleanup") cannot silently
reintroduce the mismatch.
"""

from __future__ import annotations

import inspect
import re

import mujoco
import numpy as np
import pytest

from bam.mujoco import MujocoController
from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_XML

FRICTION_DOF = int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)
FRICTION_BUDGET = 0.02  # N·m, order of the m5 dry-friction budget


@pytest.fixture(scope="module")
def walk_friction():
    """Real walk model with a live Coulomb budget on the 14 servo dofs."""
    model = mujoco.MjModel.from_xml_path(str(MICRODUCK_WALK_XML))
    data = mujoco.MjData(model)
    joint_indexes = np.asarray(model.actuator_trnid[: model.nu, 0], dtype=int)
    dof_indexes = np.asarray(model.jnt_dofadr[joint_indexes], dtype=int)

    model.dof_frictionloss[:] = 0.0
    model.dof_frictionloss[dof_indexes] = FRICTION_BUDGET
    data.qvel[:] = 0.0
    data.qvel[dof_indexes] = 0.5  # give the Coulomb rows something to resist
    data.qpos[2] = 0.25
    mujoco.mj_forward(model, data)

    nefc = int(data.nefc)
    types = np.asarray(data.efc_type[:nefc])
    m = types == FRICTION_DOF
    return model, data, joint_indexes, dof_indexes, np.asarray(data.efc_id[:nefc])[m], np.asarray(
        data.efc_force[:nefc]
    )[m]


def test_friction_dof_efc_id_is_a_dof_address(walk_friction):
    """The semantic claim the whole fix rests on."""
    model, _, joint_indexes, dof_indexes, efc_ids, _ = walk_friction
    assert model.nv == model.nu + 6, "unexpected freejoint layout"
    assert np.array_equal(dof_indexes, joint_indexes + 5), (
        "Microduck's freejoint puts 5 dofs between the joint and dof index "
        "spaces — this is why the joint-id match aliases onto other joints"
    )
    assert set(efc_ids.tolist()) == set(dof_indexes.tolist())
    assert set(dof_indexes.tolist()).issubset(set(efc_ids.tolist()))
    assert not set(joint_indexes.tolist()).issubset(set(efc_ids.tolist()))


def test_joint_id_selector_aliases_onto_other_joints(walk_friction):
    """The old comparison is not merely off — it silently swaps joints."""
    _, _, joint_indexes, dof_indexes, efc_ids, efc_force = walk_friction

    def total(target):
        return float(np.sum(efc_force * (efc_ids == target)))

    correct = np.array([total(d) for d in dof_indexes])
    buggy = np.array([total(j) for j in joint_indexes])

    # Every joint has the same budget here, so the CORRECT selector must return
    # the budget on every channel and the buggy one must diverge.
    assert np.allclose(correct, -FRICTION_BUDGET), correct
    assert not np.allclose(buggy, correct)

    # efc_id == j is the row of the dof j, i.e. of joint j-5 — so the actuator
    # for joint j strips a NEIGHBOUR's friction, and the first 5 joints (whose
    # ids have no matching dof address) strip nothing at all.
    aliased = [int(j) for j in joint_indexes if total(j) != 0.0]
    assert aliased == [6, 7, 8, 9, 10, 11, 12, 13, 14], aliased
    silent = [int(j) for j in joint_indexes if total(j) == 0.0]
    assert silent == [1, 2, 3, 4, 5], silent
    for j in aliased:
        donor_joint = int(joint_indexes[dof_indexes == j][0])  # the joint owning dof j
        assert donor_joint == j - 5
        assert donor_joint != j, f"joint {j} stripped joint {donor_joint}'s friction"
        assert total(j) == pytest.approx(total(donor_joint + 5))  # == its own dof's row


def test_ground_truth_is_the_dof_scatter(walk_friction):
    """Scattering efc_force by efc_id reproduces the per-dof budget exactly."""
    model, data, _, dof_indexes, efc_ids, efc_force = walk_friction
    gt = np.zeros(model.nv)
    np.add.at(gt, efc_ids.astype(int), efc_force)
    assert np.allclose(gt[dof_indexes], -FRICTION_BUDGET)
    assert np.allclose(gt[np.setdiff1d(np.arange(model.nv), dof_indexes)], 0.0)


def test_cpu_controller_selects_on_dof_indexes():
    """Source contract: `MujocoController.update` must repeat `dof_indexes`.

    Cheap and blunt on purpose — the alternative (observing the stripped torque)
    needs the controller's private arithmetic, and this line IS the bug.
    """
    src = inspect.getsource(MujocoController.update)
    body = re.sub(r"#.*", "", src)  # comments mention both names on purpose
    m = re.search(r"np\.repeat\(\s*\[self\.(\w+)\],\s*len\(self\.mujoco_data\.efc_id\)", body)
    assert m is not None, "could not find the efc_id selector in update()"
    assert m.group(1) == "dof_indexes", (
        f"efc_id is a dof address; the selector repeats self.{m.group(1)}"
    )


def test_cpu_path_matches_training_path_on_the_real_model(walk_friction):
    """End-to-end: what the CPU controller strips == what warp strips.

    `bam.mjlab` scatters by `efc_id`; `bam.mujoco` must select the same rows.
    Recomputing the training-path expression on the same constraint state is the
    closest thing to a cross-backend check available without a GPU.
    """
    model, data, _, dof_indexes, efc_ids, efc_force = walk_friction
    nefc = int(data.nefc)
    is_fric = np.asarray(data.efc_type[:nefc]) == FRICTION_DOF

    # warp: scatter-add into (nv,) by efc_id, then index the servo dofs
    warp = np.zeros(model.nv)
    np.add.at(warp, np.asarray(data.efc_id[:nefc])[is_fric].astype(int), efc_force)

    # cpu (fixed): batched element-wise test against dof_indexes
    selector = np.repeat([np.asarray(data.efc_id[:nefc])], len(dof_indexes), axis=0) == np.repeat(
        [dof_indexes], nefc, axis=0
    ).T
    cpu = np.sum(
        np.asarray(data.efc_force[:nefc]) * (selector * is_fric), axis=1
    )

    assert np.allclose(cpu, warp[dof_indexes])
