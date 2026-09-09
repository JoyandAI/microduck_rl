#!/usr/bin/env python3
"""Headless deployment rehearsal: fixed vel command, N seconds, stats out.

Same scene/policy path as scripts/infer_policy.py (keyboard), but no GUI:
answers "does this ONNX actually stand/walk in the deployment scene?"

Usage:
    uv run python scripts/headless_eval.py --onnx hls2909_walk.onnx
    uv run python scripts/headless_eval.py --onnx walk.onnx --lin-vel-x 0.2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).parent))
from infer_policy import PolicyInference  # noqa: E402

SCENE_XML = "src/mjlab_microduck/robot/microduck/scene.xml"


def run_eval(onnx_path: str, lin_vel_x: float = 0.0, seconds: float = 5.0,
             current_limit: float = 1.75, motor: str = "xl330",
             model: str = "m6", use_projected_gravity: bool = True,
             delay_min_lag: int = 0, delay_max_lag: int = 0) -> dict:
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)

    # same fource-limit trick as infer_policy: model is a plain PD position
    # actuator scene; clip its output with the motor's kt * current limit.
    if current_limit > 0:
        from bam.model import load_model
        model_variant = "m6" if motor != "hls2909" else "m1"  # hls2909 目前只有 m1 档
        kt = load_model(motor_name=motor, model=model_variant).kt.value
        tl = kt * current_limit
        model.actuator_forcerange[:, 0] = -tl
        model.actuator_forcerange[:, 1] = tl
        model.actuator_forcelimited[:] = 1
        print(f"force limit: +/-{tl:.4f} Nm (motor={motor}, I={current_limit}A)")

    policy = PolicyInference(model, data, walking_onnx_path=onnx_path,
                             new_cmd_obs=True,
                             use_projected_gravity=use_projected_gravity,
                             delay_min_lag=delay_min_lag, delay_max_lag=delay_max_lag)
    policy.set_vel_cmd(lin_vel_x, 0.0, 0.0)

    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    freejoint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qadr = int(model.jnt_qposadr[freejoint])
    vadr = int(model.jnt_dofadr[freejoint])

    zs, tilts, qdots, ctrls = [], [], [], []
    ctrl_period = 0.02  # 50 Hz
    steps = int(seconds / model.opt.timestep)
    for i in range(steps):
        mujoco.mj_step(model, data)
        if i % int(ctrl_period / model.opt.timestep) == 0:
            action = policy.infer()
            policy.apply_action(action)
        if i % (int(0.05 / model.opt.timestep)) == 0:  # 20 Hz metrics
            quat = data.xquat[trunk]  # [w, x, y, z]
            tilt_rad = float(2 * np.arctan2(np.hypot(quat[1], quat[2]), abs(quat[3])))
            zs.append(float(data.xpos[trunk][2]))
            tilts.append(tilt_rad)
            qdots.append(float(np.abs(data.qvel[vadr + 3:]).mean() * 0))
            ctrl = data.ctrl[:]
            ctrls.append(float(np.abs(ctrl[5:]).mean()))  # legs only

    z_arr = np.array(zs)
    tilt_arr = np.array(tilts)
    # torso body spin (from freejoint angular qvel)
    spin = float(np.abs(data.qvel[vadr + 3:vadr + 6]).mean())
    fell = bool(z_arr.mean() < 0.10 or tilt_arr.mean() > np.deg2rad(50))
    return {
        "trunk_z_mean_m": float(z_arr.mean()),
        "trunk_z_min_m": float(z_arr.min()),
        "tilt_mean_deg": float(np.rad2deg(tilt_arr.mean())),
        "torso_spin_rad_s": spin,
        "leg_ctrl_mean_rad": float(np.mean(ctrls)),
        "STANDED_OK (z>0.10 & tilt<50°)": "NO" if fell else "YES",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--lin-vel-x", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--current-limit", type=float, default=1.75, help="A; <=0 disables")
    ap.add_argument("--motor", default="xl330")
    ap.add_argument("--model", default=None, help="bam model tier (hls2909 -> m1)")
    ap.add_argument("--delay-min", type=int, default=0)
    ap.add_argument("--delay-max", type=int, default=0)
    args = ap.parse_args()
    print(f"=== headless eval: {args.onnx}  cmd=({args.lin_vel_x}, 0, 0) ===")
    model_tier = args.model or ("m1" if args.motor == "hls2909" else "m6")
    res = run_eval(args.onnx, args.lin_vel_x, args.seconds,
                   args.current_limit, args.motor, model=model_tier,
                   delay_min_lag=args.delay_min, delay_max_lag=args.delay_max)
    for k, v in res.items():
        print(f"  {k:<26} = {v}")


if __name__ == "__main__":
    main()
