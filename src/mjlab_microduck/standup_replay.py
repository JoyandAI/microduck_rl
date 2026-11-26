"""Nominal CPU MuJoCo replay contract for the OpenMD StandUp actor."""

from pathlib import Path

import mujoco
import numpy as np
from onnxruntime import InferenceSession

TASK_ID = "Mjlab-StandUp-Flat-OpenMD"
SCENE = Path(__file__).resolve().parent / "robot/openmd/scene_stand.xml"
OBSERVATION_NAMES = (
    "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions",
    "command", "head_command", "body_command",
)
START_POSES = ("stand", "sit", "prone", "supine")


def select_task(requested: str | None, exported: str | None, standing_only: bool) -> bool:
    if requested and not standing_only:
        raise ValueError(f"--task {requested} requires a single --standing policy")
    if requested and exported and exported != requested:
        raise ValueError(f"ONNX task_id {exported!r} disagrees with --task {requested!r}")
    return standing_only and (requested or exported) == TASK_ID


def policy_settings(
    model: mujoco.MjModel, session: InferenceSession, action_scale: float | None,
) -> tuple[np.ndarray, float | np.ndarray]:
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (
        len(inputs) != 1 or len(outputs) != 1 or model.nu != 14
        or len(inputs[0].shape) != 2 or inputs[0].shape[1] != 61
        or len(outputs[0].shape) != 2 or outputs[0].shape[1] != model.nu
    ):
        raise ValueError(f"{TASK_ID} expects one [batch,61] input and one [batch,14] output")
    meta = session.get_modelmeta().custom_metadata_map
    joints = [model.joint(j).name for j in model.actuator_trnid[:, 0]]
    if "joint_names" in meta and meta["joint_names"].split(",") != joints:
        raise ValueError("ONNX joint_names do not match the OpenMD StandUp actuators")
    if "observation_names" in meta and tuple(meta["observation_names"].split(",")) != OBSERVATION_NAMES:
        raise ValueError("ONNX observation_names do not match the OpenMD StandUp actor")
    indices = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    home = model.key("STAND").qpos[indices].astype(np.float32)
    if "default_joint_pos" in meta:
        declared = np.asarray(meta["default_joint_pos"].split(","), dtype=np.float32)
        # mjlab writes joint metadata to three decimal places.
        if declared.shape != home.shape or not np.allclose(declared, home, atol=.00051, rtol=0):
            raise ValueError("ONNX default_joint_pos does not match the OpenMD STAND keyframe")
    if "action_scale" in meta:
        scales = np.asarray(meta["action_scale"].split(","), dtype=np.float32)
        if scales.size not in (1, model.nu) or not np.isfinite(scales).all():
            raise ValueError("Invalid ONNX action_scale")
        if action_scale is not None and not np.allclose(scales, action_scale):
            raise ValueError("--action-scale disagrees with ONNX action_scale")
        action_scale = float(scales[0]) if np.all(scales == scales[0]) else scales
    return home, 1.0 if action_scale is None else action_scale


def set_start_pose(model: mujoco.MjModel, data: mujoco.MjData, pose: str) -> None:
    """Place reproducible unperturbed poses in the independent flat scene."""
    if pose not in START_POSES:
        raise ValueError(f"Unknown StandUp pose {pose!r}")
    mujoco.mj_resetDataKeyframe(model, data, model.key("SIT" if pose == "sit" else "STAND").id)
    if pose in ("prone", "supine"):
        root = int(model.joint("trunk_base_freejoint").qposadr[0])
        half = np.sqrt(0.5)
        data.qpos[root + 3:root + 7] = (half, 0, half if pose == "prone" else -half, 0)
        mujoco.mj_forward(model, data)
        floor = model.geom("floor").id
        active = np.flatnonzero((model.geom_bodyid != 0) & (model.geom_contype != 0))
        gap = min(mujoco.mj_geomDistance(model, data, floor, g, 1., None) for g in active)
        data.qpos[root + 2] += 0.001 - gap
    mujoco.mj_forward(model, data)
