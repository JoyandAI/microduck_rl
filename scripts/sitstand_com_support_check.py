#!/usr/bin/env python3
"""Is the sit-stand head droop a REWARD problem or a CENTRE-OF-MASS problem?

CPU-only, `mj_forward`-only (the simulator is never stepped), so it is safe to run
next to a training/eval job and is fully deterministic.

The question the user asked ("是奖励出了问题，还是重心出了问题") has a specific,
decidable form, and this repo has already answered it once by accident:

  `microduck_constants.HOME_FRAME`'s docstring says the OLD home pose put the CoM
  "~5mm behind [the ankle axis], which biased the robot backward and made the
  standup policy droop its head forward as a counterweight".  That is a CoM
  diagnosis, and it was fixed by leaning the leg chain forward (STAND2).

But STAND2 predates the HD-1910 swap: the robot gained +42 g, +12 g of it in the
head assembly, and the head sits at a 75.6 mm lever arm. So the question is
whether the CURRENT model still satisfies the repo's own standing criterion:

  * is the whole-body CoM inside the FOOT SUPPORT POLYGON, and
  * is it over the ANKLE AXIS (the criterion STAND2 was built for), and
  * what does folding the neck to its hard stop do to both?

What this script measures, per pose:
  * whole-body CoM (mujoco `subtree_com[0]`, mass-weighted, exact),
  * the contact support polygon from the actual foot collision mesh vertices,
  * the fore-aft / lateral margin of the CoM inside that polygon,
  * the CoM offset from the ankle axis (the repo's own STAND2 criterion),
  * the neck gravitational torque (is the servo even near its limit?).

Poses are swept over trunk pitch because the observed sit-stand stand pose is
tilted 35.4 deg, so "the CoM at the commanded pose" must be evaluated at that
tilt, not only upright.

Usage:
    uv run python scripts/sitstand_com_support_check.py
    uv run python scripts/sitstand_com_support_check.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

ROBOT_XML = Path("src/mjlab_microduck/robot/microduck/robot_allcollisions.xml")
FOOT_GEOMS = ("left_foot_collision", "right_foot_collision")
NECK_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
ANKLE_JOINTS = ("left_ankle", "right_ankle")

HOME = {
    "left_hip_yaw": 0.0, "left_hip_roll": -0.0873, "left_hip_pitch": -0.4579,
    "left_knee": 0.0, "left_ankle": 0.4530,
    "neck_pitch": 0.3491, "head_pitch": 0.3491, "head_yaw": 0.0, "head_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0873, "right_hip_pitch": 0.4579,
    "right_knee": 0.0, "right_ankle": -0.4530,
}
# The observed sit-stand STAND steady state (eval_sitstand_head.py, 60 s protocol,
# delivered sitstand_hd1910_8000.onnx): trunk z 0.115 m, tilt 35.4 deg.
# That checkpoint's HOME was knees ±0.0049 and neck/head +0.3491 (20 deg).
# neck_pitch -1.583 rad absolute (= -1.932 relative to that HOME).
OBSERVED = dict(HOME)
OBSERVED["left_knee"] = 0
OBSERVED["right_knee"] = 0
OBSERVED["head_pitch"] = 0.3491
OBSERVED["neck_pitch"] = -1.583


def _hull(points: np.ndarray) -> np.ndarray:
    """Convex hull (monotone chain) of an (M,2) array -> hull vertices CCW."""
    pts = np.unique(points, axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2:
                a, b = out[-2], out[-1]
                if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) <= 0:
                    out.pop()
                else:
                    break
            out.append(p)
        return out

    lower, upper = half(pts), half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def _edge_margins(hull: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """(min inward distance to any edge, signed area sign) — robust to winding."""
    if len(hull) < 3:
        return float("-inf"), 0.0
    # ensure CCW
    area = 0.5 * float(np.sum(hull[:, 0] * np.roll(hull[:, 1], -1)
                              - np.roll(hull[:, 0], -1) * hull[:, 1]))
    h = hull if area > 0 else hull[::-1]
    best = float("inf")
    for i in range(len(h)):
        a, b = h[i], h[(i + 1) % len(h)]
        e = b - a
        n = np.array([-e[1], e[0]])
        n /= max(np.linalg.norm(n), 1e-12)
        best = min(best, float(np.dot(p - a, n)))
    return best, float(np.sign(area))


def _foot_vertices(m: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray:
    """World-frame vertices of the foot collision meshes, as (K,3)."""
    out = []
    for gname in FOOT_GEOMS:
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gname)
        if gid < 0:
            raise SystemExit(f"geom {gname} not found in {ROBOT_XML}")
        did = int(m.geom_dataid[gid])
        if did < 0:
            # primitive geom: fall back to its centre
            out.append(d.geom_xpos[gid][None, :])
            continue
        adr, num = int(m.mesh_vertadr[did]), int(m.mesh_vertnum[did])
        v = m.mesh_vert[adr:adr + num].astype(np.float64)
        R = d.geom_xmat[gid].reshape(3, 3)
        out.append(v @ R.T + d.geom_xpos[gid])
    return np.concatenate(out, axis=0)


def _place(m: mujoco.MjModel, d: mujoco.MjData, joint_pos: dict, pitch_deg: float,
           trunk_z: float) -> None:
    """Set the pose: freejoint (x,y,z,quat) + named joints."""
    d.qpos[:] = m.qpos0
    # freejoint of trunk_base is the first 7 qpos entries
    d.qpos[0:3] = (0.0, 0.0, trunk_z)
    half = math.radians(pitch_deg) / 2.0
    # pitch about world +y (nose-down positive per the model's x-forward frame)
    d.qpos[3:7] = (math.cos(half), 0.0, math.sin(half), 0.0)
    for jname, val in joint_pos.items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            continue
        d.qpos[int(m.jnt_qposadr[jid])] = val
    mujoco.mj_forward(m, d)


FEET_FLAT_PATCH_MM = 40.0


def analyse(m: mujoco.MjModel, joint_pos: dict, pitch_deg: float, trunk_z: float,
            label: str) -> dict:
    # ⚠️ Pitching the trunk pitches the whole serial chain, so the SOLES tilt with
    # it and the contact patch collapses to a narrow strip. A leaning row is
    # therefore NOT "a robot standing with a leaning trunk" — it is "standing on
    # an edge". We do not pretend to compensate that (the chain's hip/knee/ankle
    # sign conventions make a naive ankle offset wrong, and a wrong fix is worse
    # than a flagged caveat); instead every row reports its patch extent and any
    # row below FEET_FLAT_PATCH_MM is marked, so its margin is read as
    # indicative-only. The DECISIVE rows are the feet-flat ones.
    d = mujoco.MjData(m)
    _place(m, d, joint_pos, pitch_deg, trunk_z)
    v = _foot_vertices(m, d)
    dz = float(v[:, 2].min())
    _place(m, d, joint_pos, pitch_deg, trunk_z - dz)
    v = _foot_vertices(m, d)

    com = np.array(d.subtree_com[0], dtype=np.float64)
    total_mass = float(m.body_subtreemass[0])
    # contact patch = vertices within 1.5 mm of the lowest point
    zmin = float(v[:, 2].min())
    patch = v[v[:, 2] < zmin + 1.5e-3][:, :2]
    hull = _hull(patch)
    margin, _ = _edge_margins(hull, com[:2])

    ankle_x = []
    for jn in ANKLE_JOINTS:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        ankle_x.append(float(d.xanchor[jid][0]))
    ankle_x_mean = float(np.mean(ankle_x))

    # Neck gravity torque about the neck_pitch axis (the servo must hold it).
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "neck_pitch")
    body = int(m.jnt_bodyid[jid])
    head_mass = 0.0
    head_com = np.zeros(3)
    for b in range(m.nbody):
        if body == 0:
            continue
        # bodies in the neck_pitch subtree
        anc = b
        while anc > 0 and anc != body:
            anc = int(m.body_parentid[anc])
        if anc == body:
            mi = float(m.body_mass[b])
            head_mass += mi
            head_com += mi * np.array(d.xipos[b], dtype=np.float64)
    if head_mass > 0:
        head_com /= head_mass
        axis = np.array(d.xaxis[jid], dtype=np.float64)
        anchor = np.array(d.xanchor[jid], dtype=np.float64)
        r = head_com - anchor
        gravity_force = np.array([0.0, 0.0, -9.81 * head_mass])
        tau = float(np.linalg.norm(np.cross(r, gravity_force)))

    return {
        "label": label,
        "trunk_pitch_deg": pitch_deg,
        "contact_patch_x_extent_mm": float(hull[:, 0].max() - hull[:, 0].min()) * 1e3,
        "feet_flat": bool((hull[:, 0].max() - hull[:, 0].min()) * 1e3
                          >= FEET_FLAT_PATCH_MM),
        "trunk_z_m": trunk_z - dz,
        "com_x_m": float(com[0]), "com_y_m": float(com[1]), "com_z_m": float(com[2]),
        "total_mass_g": total_mass * 1000.0,
        "support_poly_x_range_mm": [float(hull[:, 0].min() * 1e3),
                                    float(hull[:, 0].max() * 1e3)],
        "support_poly_y_range_mm": [float(hull[:, 1].min() * 1e3),
                                    float(hull[:, 1].max() * 1e3)],
        "support_poly_n_verts": int(len(hull)),
        "com_inside_margin_mm": float(margin * 1e3),
        "com_over_ankle_offset_mm": float((com[0] - ankle_x_mean) * 1e3),
        "head_mass_g": head_mass * 1000.0,
        "neck_gravity_torque_Nm": tau,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=str(ROBOT_XML))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(args.xml)
    print(f"=== CoM / support-polygon audit: {args.xml} ===")
    print(f"  total mass {float(m.body_subtreemass[0])*1000:.2f} g\n")

    rows = []
    for pitch in (0.0, 10.0, 20.0, 30.0, 35.4, 40.0):
        rows.append(analyse(m, HOME, pitch, 0.115, f"HOME legs, head at HOME, pitch {pitch:g}"))
    for pitch in (0.0, 20.0, 35.4):
        rows.append(analyse(m, OBSERVED, pitch, 0.115,
                            f"HOME legs, neck at HARD STOP, pitch {pitch:g}"))

    hdr = (f"{'pose':<44}{'z_mm':>7}{'CoM x':>8}{'margin':>9}{'vs ankle':>10}"
           f"{'τ_neck':>9}{'patch':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        mark = "" if r["feet_flat"] else " *"
        print(f"{r['label']:<44}{r['trunk_z_m']*1e3:>7.1f}"
              f"{r['com_x_m']*1e3:>8.2f}{r['com_inside_margin_mm']:>9.2f}"
              f"{r['com_over_ankle_offset_mm']:>10.2f}"
              f"{r['neck_gravity_torque_Nm']:>9.4f}"
              f"{r['contact_patch_x_extent_mm']:>8.1f}{mark}")
    print(f"\n  patch: fore-aft extent of the ground contact patch (mm). A feet-flat")
    print(f"         stand reads >= {FEET_FLAT_PATCH_MM:.0f} mm. Rows marked * are standing on an")
    print( "         EDGE (the whole leg chain pitched with the trunk), so their margin")
    print( "         is indicative only — read the feet-flat rows for the verdict.")

    print("\n  'margin' > 0  => the CoM is INSIDE the foot support polygon (statically")
    print("                   standable); < 0 => it cannot stand at that pose at all.")
    print("  'vs ankle'    => CoM x minus the ankle-axis x. HOME_FRAME's docstring")
    print("                   says the OLD pose sat ~-5 mm (CoM BEHIND the ankles) and")
    print("                   that THIS is what made the standup policy droop its head.")
    print("  'τ_neck'      => gravitational torque the neck servo must hold.")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
        print(f"\n[i] -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
