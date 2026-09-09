#!/usr/bin/env python3
"""Render the duck at its ZERO pose vs its HOME pose — for servo zero calibration.

The "zero" (机械零位) of every joint is its angle 0 in the model frame. The
original XL330 duck was zero-calibrated at exactly this pose; the servo-swap
requirement keeps the joint axes in place (<=0.5 mm), so the same physical
pose is the zero for the new HL-2909 servos.

These images let you POSE the duck back to that zero by eye:

  * zero_pose_*.png  — every servo joint at angle 0 (the calibration pose);
  * home_pose_*.png  — the standing HOME pose (same camera) for comparison.

Both are rendered with the trunk at its default height. Joint-by-joint, you
can also use the printed table: starting from the standing pose, rotate each
joint by `-home_angle` (its own positive direction as in the model) to reach
the zero pose; the image shows the resulting geometry.

Usage:
    uv run python scripts/render_zero_pose.py [--out zeronotes]
    # saves zeronotes_zero_side.png / _front / _quarter + _home_* and prints the table
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import mujoco

import sys

sys.path.insert(0, str(Path(__file__).parent))
from infer_policy import DEFAULT_POSE  # noqa: E402  (home pose, 14 joints)

SCENE_XML = "src/mjlab_microduck/robot/microduck/scene.xml"


def servo_qpos_indices(model: mujoco.MjModel) -> list[int]:
    """qpos addresses of the 14 actuated joints (via actuator transmission)."""
    return [int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)]


def pose_robot(model: mujoco.MjModel, data: mujoco.MjData, joint_q: np.ndarray,
               trunk_z: float = 0.125) -> None:
    fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qadr = int(model.jnt_qposadr[fid])
    data.qpos[:] = 0.0
    data.qpos[qadr + 2] = trunk_z
    data.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]
    for val, qidx in zip(joint_q, servo_qpos_indices(model)):
        data.qpos[qidx] = val
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def render(model: mujoco.MjModel, data: mujoco.MjData, path: Path,
           azimuth: float, elevation: float, lookat=None, distance: float | None = None) -> None:
    cam = mujoco.MjvCamera()
    cam.azimuth = azimuth
    cam.elevation = elevation
    cam.distance = distance if distance else 1.6 * float(model.stat.extent)
    if lookat is None:
        lookat = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")]
    cam.lookat = np.asarray(lookat).tolist()
    with mujoco.Renderer(model, height=480, width=640) as ren:
        ren.update_scene(data, camera=cam)
        img = ren.render()
    from PIL import Image
    Image.fromarray(img).save(path)
    print(f"[+] {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="zeronotes", help="output prefix (no extension)")
    ap.add_argument("--trunk-z", type=float, default=0.125,
                    help="trunk height for both poses (m); default = sim stand height")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)

    # The compiled model default qpos is all-zero (that IS the ZERO pose); the
    # HOME (standing) pose comes from DEFAULT_POSE and must be set explicitly.
    mujoco.mj_forward(model, data)
    qidx = servo_qpos_indices(model)
    home_in_model = np.array([float(data.qpos[i]) for i in qidx])
    if np.allclose(home_in_model, 0.0):
        print("[i] model compiled default qpos == 0 (this is the ZERO pose — "
              "helpful to know: the CAD zero IS the model's rest pose)")
    else:
        print(f"[i] model default qpos nonzero: {np.round(home_in_model, 4)} "
              "(not used — poses are set explicitly below)")
    home_pose = DEFAULT_POSE[: model.nu]
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,
                                     int(model.actuator_trnid[j, 0]))
                   for j in range(model.nu)]

    print("\n=== per-joint table (zero ← from HOME) ===")
    print(f"{'joint':<15} {'home(rad)':>10} {'zero':>6} {'Δ(rad)':>8} {'Δ(deg)':>8}")
    for j in range(model.nu):
        print(f"{joint_names[j]:<15} {home_pose[j]:+10.4f} {'0.0':>6} "
              f"{-home_pose[j]:+8.4f} {np.degrees(-home_pose[j]):+8.2f}")

    out = Path(args.out)

    def leg_views(tag: str) -> None:
        """Closeup of the left leg: the clearest per-joint reference."""
        hip = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hip_l")]
        ankle = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ankle_left")]
        lookat = (hip + ankle) / 2.0
        render(model, data, out.with_name(f"{out.name}_{tag}_leg_side.png"),
               90, -10, lookat=lookat, distance=0.55 * float(model.stat.extent))

    # HOME pose (stand) images
    pose_robot(model, data, home_pose, trunk_z=args.trunk_z)
    lookat = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")]
    for az, view in ((90, "side"), (0, "front"), (135, "quarter")):
        render(model, data, out.with_name(f"{out.name}_home_{view}.png"),
               az, -15, lookat=lookat)
    leg_views("home")

    # ZERO pose images
    zero = np.zeros(model.nu)
    pose_robot(model, data, zero, trunk_z=args.trunk_z)
    for az, view in ((90, "side"), (0, "front"), (135, "quarter")):
        render(model, data, out.with_name(f"{out.name}_zero_{view}.png"),
               az, -15, lookat=lookat)
    leg_views("zero")

    print("\n[i] 用法: 让每个关节手工对齐 zero_* 图中的几何 → 逐台 0x0B 校到中位"
          " (provision-hls.py --home) → 站直后用 scripts/check_hls_zero.py 复核")


if __name__ == "__main__":
    main()
