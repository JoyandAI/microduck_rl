#!/usr/bin/env python3
"""Left/right mirror-symmetry audit of the Microduck MJCFs (CPU-only, no GPU).

Answers the half of the "is the veer in the MODEL?" question that does NOT need a
running sim: if the robot model is NOT mirror-symmetric, a symmetric policy can
still walk in a circle. If it IS symmetric, then a per-seed-varying veer cannot
come from the model geometry itself (only from per-seed randomization or from
the actuator/controller layer).

Method (no sign-convention guessing): work in the WORLD frame at the HOME joint
configuration via `mj_forward`, then reflect through the plane y = 0:
  * body CoM    : `data.xipos`      -> reflected L must equal R
  * body inertia: `data.cinert`     -> reflected as M I M with M = diag(1,-1,1)
  * joint range : `jnt_range` of the left joint vs the NEGATED-AND-SWAPPED right
                  joint range (a hinge around a mirrored axis flips sign)
  * foot meshes : world vertex clouds of the sole collision geoms, reflected.

Usage:
  .venv/bin/python scripts/mirror_symmetry_audit.py
"""
from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

REPO = Path("/home/joyandai/microduck_rl")
MDIR = REPO / "src/mjlab_microduck/robot/microduck"

MODELS = {
    "HD-1910 (current)": MDIR / "robot_walk.xml",
    "XL330 (origin/develop)": MDIR / "robot_walk_preswap.xml",
}
BODY_PAIRS = [
    ("yaw2roll", "bearing_roll"),
    ("hip_l", "hip_l_2"),
    ("upper_leg_left", "upper_leg_right"),
    ("leg", "leg_2"),
    ("ankle_left", "ankle_right"),
]
JOINT_PAIRS = [
    ("left_hip_yaw", "right_hip_yaw"),
    ("left_hip_roll", "right_hip_roll"),
    ("left_hip_pitch", "right_hip_pitch"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
]
HOME = {
    "left_hip_yaw": 0.0, "left_hip_roll": -0.0873, "left_hip_pitch": -0.4579,
    "left_knee": 0.0, "left_ankle": 0.4530,
    "neck_pitch": 0.3491, "head_pitch": 0.3491, "head_yaw": 0.0, "head_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0873, "right_hip_pitch": 0.4579,
    "right_knee": 0.0, "right_ankle": -0.4530,
}


def audit(path: Path):
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    for name, val in HOME.items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            d.qpos[m.jnt_qposadr[jid]] = val
    mujoco.mj_forward(m, d)

    M = np.diag([1.0, -1.0, 1.0])
    out = {"total_mass_g": float(m.body_mass.sum() * 1000.0), "bodies": {}, "joints": {}}

    for lb, rb in BODY_PAIRS:
        li = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, lb)
        ri = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, rb)
        if li < 0 or ri < 0:
            out["bodies"][f"{lb}|{rb}"] = "MISSING"
            continue
        pL = np.array(d.xipos[li], dtype=float)
        pR = np.array(d.xipos[ri], dtype=float)
        # data.cinert is a 10-vector: [mass, com(3), xx, yy, zz, xy, xz, yz]
        def iner(i):
            c = np.array(d.cinert[i], dtype=float)
            xx, yy, zz, xy, xz, yz = c[4:10]
            return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
        iL, iR = iner(li), iner(ri)
        out["bodies"][f"{lb}|{rb}"] = {
            "mass_L": float(m.body_mass[li]), "mass_R": float(m.body_mass[ri]),
            "com_L_reflected_mm": (M @ pL * 1000.0).round(3).tolist(),
            "com_R_mm": (pR * 1000.0).round(3).tolist(),
            "com_mismatch_mm": float(np.linalg.norm(M @ pL - pR) * 1000.0),
            "inertia_mismatch_fro": float(np.linalg.norm(M @ iL @ M - iR)),
            "inertia_scale": float(np.linalg.norm(iR)),
        }

    for lj, rj in JOINT_PAIRS:
        lid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, lj)
        rid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, rj)
        lr = m.jnt_range[lid]
        rr = m.jnt_range[rid]
        # mirrored hinge: range flips sign AND swaps ends
        out["joints"][f"{lj}|{rj}"] = {
            "L_deg": np.degrees(lr).round(3).tolist(),
            "R_deg": np.degrees(rr).round(3).tolist(),
            "R_mirrored_deg": (-np.degrees(rr)[::-1]).round(3).tolist(),
            "mismatch_deg": float(np.max(np.abs(np.degrees(lr) - (-np.degrees(rr)[::-1])))),
        }

    # whole-body CoM at HOME: symmetric model => y ~ 0
    com = np.array(d.subtree_com[0], dtype=float)
    out["home_com_mm"] = (com * 1000.0).round(3).tolist()
    out["home_com_y_abs_mm"] = float(abs(com[1]) * 1000.0)

    # foot sole meshes: reflect the left cloud through y=0 and compare to the right
    def cloud(prefix):
        pts = []
        for g in range(m.ngeom):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if nm.startswith(prefix) and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
                v = m.mesh_vert[va:va + vn].astype(float)
                R = d.geom_xmat[g].reshape(3, 3)
                pts.append(v @ R.T + d.geom_xpos[g])
        return np.concatenate(pts) if pts else np.zeros((0, 3))

    cl, cr = cloud("left_foot_collision"), cloud("right_foot_collision")
    if len(cl) and len(cr):
        rl = cl.copy()
        rl[:, 1] *= -1.0
        # nearest-neighbour Hausdorff-style distance (sorted cloud comparison)
        A = rl[np.argsort(rl[:, 1])]
        B = cr[np.argsort(cr[:, 1])]
        n = min(len(A), len(B))
        dmax = float(np.max(np.linalg.norm(A[:n] - B[:n], axis=1))) * 1000.0
        out["foot_cloud"] = {
            "n_left": int(len(cl)), "n_right": int(len(cr)),
            "sorted_reflected_max_dev_mm": dmax,
            "centroid_L_reflected_mm": (rl.mean(axis=0) * 1000).round(3).tolist(),
            "centroid_R_mm": (cr.mean(axis=0) * 1000).round(3).tolist(),
        }
    # ---- decisive, frame-independent mirror test -------------------------
    # For a mirror-symmetric mechanism, a MIRRORED state must have the same
    # kinetic energy (this exercises the COMPLETE mass matrix: every body's
    # mass, CoM and full inertia, plus all the kinematics). It needs no
    # assumption about body-frame conventions, which a naive world-frame
    # inertia comparison does.
    #
    # Mirror map (measured, not assumed): swap left<->right leg joints and
    # negate; negate head_yaw/head_roll (roll and yaw are pseudoscalars);
    # keep neck_pitch/head_pitch. Free joint: v -> M v, omega -> -M omega.
    def mirror_qpos(d):
        for lj, rj in JOINT_PAIRS:
            li = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, lj)
            ri = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, rj)
            ql, qr = d.qpos[m.jnt_qposadr[li]], d.qpos[m.jnt_qposadr[ri]]
            d.qpos[m.jnt_qposadr[li]], d.qpos[m.jnt_qposadr[ri]] = -qr, -ql
        for hj in ("head_yaw", "head_roll"):
            hi = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, hj)
            if hi >= 0:
                d.qpos[m.jnt_qposadr[hi]] *= -1.0
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        j = m.body_jntadr[b]                       # the free joint
        a = m.jnt_qposadr[j]
        d.qpos[a + 1] *= -1.0                      # free-joint y  -> -y
        # Quaternion of M R M with M = diag(1,-1,1): R(axis a, theta) maps to
        # R(a, -theta) exactly when a is parallel to y and to R(M a, -theta)
        # otherwise, which in (w,x,y,z) form is (w, -x, +y, -z).
        # VERIFIED numerically below against the matrix identity.
        d.qpos[a + 4] *= -1.0                      # x -> -x
        d.qpos[a + 6] *= -1.0                      # z -> -z

    def mirror_qvel(d):
        for lj, rj in JOINT_PAIRS:
            li = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, lj)
            ri = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, rj)
            if m.jnt_type[li] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            vl, vr = d.qvel[m.jnt_dofadr[li]], d.qvel[m.jnt_dofadr[ri]]
            d.qvel[m.jnt_dofadr[li]], d.qvel[m.jnt_dofadr[ri]] = -vr, -vl
        for hj in ("head_yaw", "head_roll"):
            hi = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, hj)
            if hi >= 0:
                d.qvel[m.jnt_dofadr[hi]] *= -1.0
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        j = m.body_jntadr[b]
        a = m.jnt_dofadr[j]
        d.qvel[a + 1] *= -1.0                      # linear v_y -> -v_y
        d.qvel[a + 3] *= -1.0                      # angular wx -> -wx (pseudovector)
        d.qvel[a + 5] *= -1.0                      # angular wz -> -wz

    # sanity: verify the quaternion mirror map against the matrix identity
    _qv = rng0 = np.random.default_rng(7).normal(size=4)
    _qv /= np.linalg.norm(_qv)
    _w, _x, _y, _z = _qv
    _R = np.array([
        [1 - 2 * (_y * _y + _z * _z), 2 * (_x * _y - _w * _z), 2 * (_x * _z + _w * _y)],
        [2 * (_x * _y + _w * _z), 1 - 2 * (_x * _x + _z * _z), 2 * (_y * _z - _w * _x)],
        [2 * (_x * _z - _w * _y), 2 * (_y * _z + _w * _x), 1 - 2 * (_x * _x + _y * _y)]])
    _wm, _xm, _ym, _zm = _w, -_x, _y, -_z
    _Rm = np.array([
        [1 - 2 * (_ym * _ym + _zm * _zm), 2 * (_xm * _ym - _wm * _zm), 2 * (_xm * _zm + _wm * _ym)],
        [2 * (_xm * _ym + _wm * _zm), 1 - 2 * (_xm * _xm + _zm * _zm), 2 * (_ym * _zm - _wm * _xm)],
        [2 * (_xm * _zm - _wm * _ym), 2 * (_ym * _zm + _wm * _xm), 1 - 2 * (_xm * _xm + _ym * _ym)]])
    out["quat_mirror_map_error"] = float(np.max(np.abs(_Rm - M @ _R @ M)))

    rng = np.random.default_rng(0)
    ke_err, com_err = [], []
    for _ in range(64):
        d2 = mujoco.MjData(m)
        mujoco.mj_resetData(m, d2)
        for name, _v in HOME.items():
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                d2.qpos[m.jnt_qposadr[jid]] = HOME[name]
        d2.qpos[0:3] = rng.uniform(-0.1, 0.1, 3)
        qr = rng.normal(size=4)
        qr /= np.linalg.norm(qr)
        d2.qpos[3:7] = qr
        for name in HOME:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                d2.qpos[m.jnt_qposadr[jid]] += rng.uniform(-0.3, 0.3)
        d2.qvel[:] = rng.uniform(-1.0, 1.0, m.nv)
        mujoco.mj_forward(m, d2)
        ke1 = float(d2.energy[1])
        com1 = np.array(d2.subtree_com[0], dtype=float)
        d3 = mujoco.MjData(m)
        d3.qpos[:] = d2.qpos
        d3.qvel[:] = d2.qvel
        mirror_qpos(d3)
        mirror_qvel(d3)
        mujoco.mj_forward(m, d3)
        ke2 = float(d3.energy[1])
        com2 = np.array(d3.subtree_com[0], dtype=float)
        ke_err.append(abs(ke1 - ke2) / max(abs(ke1), 1e-12))
        com_err.append(float(np.linalg.norm(M @ com1 - com2) * 1000.0))
    out["mirror_ke_rel_err_max"] = float(np.max(ke_err))
    out["mirror_com_dev_max_mm"] = float(np.max(com_err))

    return out


def main() -> int:
    import json
    for tag, p in MODELS.items():
        a = audit(p)
        print(f"\n{'='*88}\n{tag}   {p.name}   total mass {a['total_mass_g']:.1f} g\n{'='*88}")
        print(f"  whole-body CoM at HOME: {a['home_com_mm']} mm   |y| = {a['home_com_y_abs_mm']:.3f} mm")
        for k, v in a["bodies"].items():
            if v == "MISSING":
                print(f"  body {k:<34} MISSING")
                continue
            print(f"  body {k:<34} mass L/R = {v['mass_L']:.6f}/{v['mass_R']:.6f} kg"
                  f"  CoM-mirror dev = {v['com_mismatch_mm']:.4f} mm"
                  f"  inertia-mirror dev = {v['inertia_mismatch_fro']:.3e}"
                  f" (|I|={v['inertia_scale']:.3e})")
        for k, v in a["joints"].items():
            print(f"  joint {k:<34} L={v['L_deg']}  R={v['R_deg']}"
                  f"  R-mirrored={v['R_mirrored_deg']}  mismatch={v['mismatch_deg']:.4f} deg")
        print(f"  quaternion mirror map ||R' - M R M||_inf = {a['quat_mirror_map_error']:.3e}")
        print(f"  MIRROR TEST (64 random states): max |KE-mirror KE|/KE = {a['mirror_ke_rel_err_max']:.3e}"
              f"   max CoM mirror dev = {a['mirror_com_dev_max_mm']:.4f} mm")
        if "foot_cloud" in a:
            f = a["foot_cloud"]
            print(f"  sole cloud: nL={f['n_left']} nR={f['n_right']}"
                  f"  centroidL_mirrored={f['centroid_L_reflected_mm']}"
                  f"  centroidR={f['centroid_R_mm']}"
                  f"  sorted max dev={f['sorted_reflected_max_dev_mm']:.4f} mm")
        Path(f"/tmp/mirror_audit_{p.stem}.json").write_text(json.dumps(a, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
