#!/usr/bin/env python3
"""Straight-line quality + per-leg asymmetry, FAITHFUL BAM env, 2x2 factorial.

Answers the 2026-09-14 question: does the walking policy VEER, and if so is the
cause the POLICY, the ROBOT-MODEL GEOMETRY, or the ACTUATOR PARAMETERS?

The 2x2 factorial (one policy held fixed, four environments):

                     HD-1910 actuator      XL330 actuator
    HD-1910 geom          (a) home              (b)
    XL330 geom            (c)                   (d)

`--override-model {hd1910,xl330}` and `--override-actuator {hd1910,xl330}` are
independent flags, so all four cells are produced without editing any cfg file.

Faithfulness (AGENTS.md + scripts/eval_onnx_bam.py docstring):
  * REAL training env, i.e. the BAM voltage actuator model -- never scene.xml.
  * push_robot REMOVED (play=True turns it into a 0.5-1 s +-0.3 m/s shove).
  * twist / head_pose / body_pose commands FROZEN (command_manager.compute()
    otherwise resamples them every step and the policy is evaluated against
    random commands).
  * episode_length_s raised out of the way so a "reset" means a FALL, not the
    task's 20 s timeout.
  * trunk yaw is UNWRAPPED everywhere. The `yaw_drift_deg` field written by
    scripts/gait_forensics.py subtracts RAW wrapped Euler angles (a policy that
    drifted 5 deg in 20 s reads 327 deg there). Never trust that field.
  * N envs run at once => N independent randomization seeds; a single rollout
    with DR active is not evidence. `--no-dr --fix-spawn` removes all of it and
    gives the DETERMINISTIC component.

Usage:
  uv run python scripts/eval_straightness_factorial.py --onnx P.onnx \
      --override-model hd1910 --override-actuator hd1910 \
      --lin-vel-x 0.2 --seconds 20 --num-envs 8 \
      --csv out.csv --json-out out.json --tag "cell a"
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

CTRL_DT = 0.02  # 50 Hz

HOME = {
    "left_hip_yaw": 0.0, "left_hip_roll": -0.0873, "left_hip_pitch": -0.4579,
    "left_knee": 0.0, "left_ankle": 0.4530,
    "neck_pitch": 0.3491, "head_pitch": 0.3491, "head_yaw": 0.0, "head_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0873, "right_hip_pitch": 0.4579,
    "right_knee": 0.0, "right_ankle": -0.4530,
}
LEG_L = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle"]
LEG_R = ["right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]

REPO = Path("/home/joyandai/microduck_rl")
# Byte-identical (md5 fe7cac82f986c9bf4fa102525423a06d) to
# `origin/develop:src/mjlab_microduck/robot/microduck/robot_walk.xml` -- the true
# upstream XL330-era model (737.2 g). NOTE: `git show a91465c^:...` is NOT that
# model; it is the intermediate HLS2909 revision (742.9 g). Verified by md5.
PRESWAP_XML = REPO / "src/mjlab_microduck/robot/microduck/robot_walk_preswap.xml"
HD1910_XML = REPO / "src/mjlab_microduck/robot/microduck/robot_walk.xml"
XL330_JSON = REPO / "vendor/bam/bam/params/xl330/m6.json"
HD1910_JSON = REPO / "vendor/bam/bam/params/hd1910/m5.json"

# origin/develop:microduck_constants.py _BAM_ACTUATOR_KWARGS (XL330 era), verbatim
# from docs/forensics/artifacts/harness/ovr.py.
XL330_KW = dict(kp_fw=200.0, vin_range=(6.5, 8.2), vin_min=6.0,
                vin_drop=(0.0, 0.2), delay=(3, 6))

# ⚠️ "hd1910" is DERIVED from the live training constants. It used to be
# hardcoded `vin_range=(4.75, 5.25), vin_min=4.0` — the 5 V rail this repo
# retired on 2026-09-15 — so every hd1910-cell measurement made afterwards was
# taken on a 27%-weaker actuator than the policy trained with, silently.
# Deriving it makes that drift impossible. For the archived 5 V identification
# pass --params-json vendor/bam/bam/params/hd1910_5v/m5.json explicitly.
from mjlab_microduck.robot.microduck_constants import (  # noqa: E402
    _BAM_ACTUATOR_KWARGS as _BAMK,
)

HD1910_JSON = REPO / f"vendor/bam/bam/params/hd1910/{_BAMK['model']}.json"
HD1910_KW = dict(kp_fw=float(_BAMK["kp_fw"]), vin_range=tuple(_BAMK["vin_range"]),
                 vin_min=float(_BAMK["vin_min"]),
                 vin_drop=tuple(_BAMK["vin_drop_resistance_range"]),
                 delay=(int(_BAMK["delay_min_lag"]), int(_BAMK["delay_max_lag"])))

DR_EVENTS = (
    "base_com", "randomize_com", "randomize_head_com", "randomize_mass_inertia",
    "randomize_joint_friction", "randomize_joint_damping", "randomize_armature",
    "randomize_base_orientation", "foot_friction", "encoder_bias",
    "randomize_motor_gains",
)


# --------------------------------------------------------------------------- #
def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_yaw(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def deg(x):
    return math.degrees(float(x))


def fit_line_residual(p):
    c = p - p.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    d = vt[0]
    nrm = np.array([-d[1], d[0]])
    perp = c @ nrm
    return (float(np.sqrt(np.mean(perp ** 2))), float(np.max(np.abs(perp))),
            float(math.atan2(d[1], d[0])))


def polyfit_heading(t, yaw):
    out = {}
    if len(t) < 4:
        return {"slope_deg_s": float("nan"), "quad_deg_s2": float("nan")}
    lin = np.polyfit(t, yaw, 1)
    quad = np.polyfit(t, yaw, 2)
    out["slope_deg_s"] = deg(lin[0])
    out["quad_deg_s2"] = deg(2.0 * quad[0])
    res = yaw - np.polyval(lin, t)
    out["linfit_resid_rms_deg"] = deg(float(np.sqrt(np.mean(res ** 2))))
    return out


def window_slopes(t, yaw, n_win=4):
    n = len(t)
    out = []
    for k in range(n_win):
        a, b = int(k * n / n_win), int((k + 1) * n / n_win)
        out.append(deg(np.polyfit(t[a:b], yaw[a:b], 1)[0]) if b - a >= 4 else float("nan"))
    return out


def dominant_freq(x, dt):
    x = np.asarray(x, dtype=float)
    v = x - x.mean()
    spec = np.abs(np.fft.rfft(v))
    f = np.fft.rfftfreq(len(v), dt)
    if len(spec) < 3:
        return 0.0, 0.0
    k = int(np.argmax(spec[1:]) + 1)
    return float(f[k]), float(spec[k] * 2.0 / len(v))


def crossings(mask):
    m = np.asarray(mask, dtype=bool)
    rise = np.where((~m[:-1]) & m[1:])[0] + 1
    fall = np.where(m[:-1] & (~m[1:]))[0] + 1
    return rise, fall


# --------------------------------------------------------------------------- #
def build_cfg(args):
    import mujoco
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = args.num_envs
    cfg.episode_length_s = args.episode_s
    print(f"[i] push_robot removed: {cfg.events.pop('push_robot', None) is not None}")

    tcfg = cfg.commands["twist"]
    tcfg.resampling_time_range = (1e9, 1e9)
    for attr, val in (("rel_standing_envs", 0.0), ("rel_heading_envs", 0.0),
                      ("rel_turn_in_place_envs", 0.0)):
        if hasattr(tcfg, attr):
            setattr(tcfg, attr, val)
    if getattr(tcfg, "heading_command", False):
        tcfg.heading_command = False
        if hasattr(tcfg.ranges, "heading"):
            tcfg.ranges.heading = None
    for cname in ("head_pose", "body_pose"):
        pc = cfg.commands.get(cname)
        if pc is None:
            continue
        pc.resampling_time_range = (1e9, 1e9)
        if hasattr(pc, "zero_command_prob"):
            pc.zero_command_prob = 0.0
    print("[i] twist/head_pose/body_pose commands frozen (no resampling)")

    if args.fix_spawn:
        rb = cfg.events["reset_base"]
        rb.params["pose_range"] = {k: (0.0, 0.0) for k in rb.params["pose_range"]}
        rb.params["velocity_range"] = {}
        rj = cfg.events["reset_robot_joints"]
        # position_range is already (0,0); velocity_range must stay a TUPLE of
        # zeros -- `mdp.reset_joints_by_offset` does `sample_uniform(*velocity_range)`
        # and an empty dict raises TypeError.
        rj.params["position_range"] = (0.0, 0.0)
        rj.params["velocity_range"] = (0.0, 0.0)
        print("[i] spawn FIXED at nominal init_state")

    no_dr = args.no_dr or args.no_phenomena or args.no_physics_dr
    no_noise = args.no_dr or args.no_phenomena or args.no_obs_noise
    no_imu = args.no_dr or args.no_phenomena or args.no_imu_dr
    if no_dr:
        popped = [n for n in DR_EVENTS if cfg.events.pop(n, None) is not None]
        # `com_range` / `head_com_range` curricula call
        # event_manager.get_term_cfg("randomize_com") and CRASH if the event was
        # popped, so they must go too.
        for cname in ("com_range", "head_com_range"):
            if cfg.curriculum.pop(cname, None) is not None:
                popped.append(f"curriculum:{cname}")
        print(f"[i] DR OFF: popped={popped}")
    if no_noise:
        # NOTE: play=True does NOT disable observation noise -- the actor group's
        # `enable_corruption` is True, and ObservationManager sets
        # `term_cfg.noise = None` only when it is False. So every eval that does
        # not clear this flag feeds the policy +-0.03 rad/s on base_ang_vel,
        # +-0.25 rad/s on joint_vel, +-0.001 rad on joint_pos and +-0.01 on
        # projected_gravity. On a 25 cm biped (a chaotic limit cycle) that noise
        # alone seeds large run-to-run divergence in heading.
        cfg.observations["actor"].enable_corruption = False
        print("[i] OBS NOISE disabled (actor group enable_corruption=False)")
    if no_imu:
        from mjlab.envs import mdp as _bmdp
        for term, fn in (("base_ang_vel", _bmdp.base_ang_vel),
                         ("projected_gravity", _bmdp.projected_gravity)):
            cfg.observations["actor"].terms[term].func = fn
            cfg.observations["actor"].terms[term].params = {}
        print("[i] IMU misalignment DR reverted to plain obs funcs")

    robot = cfg.scene.entities["robot"]
    if args.override_model == "xl330":
        xml = str(args.model_xml or PRESWAP_XML)
        object.__setattr__(robot, "spec_fn", (lambda p=xml: mujoco.MjSpec.from_file(p)))
        print(f"[i] MODEL override -> {xml}")
    else:
        print(f"[i] MODEL -> {args.model_xml or HD1910_XML}")

    act = robot.articulation.actuators[0]
    kw, params = {}, None
    if args.override_actuator == "xl330":
        kw, params = dict(XL330_KW), str(args.params_json or XL330_JSON)
    elif args.override_actuator == "hd1910":
        kw, params = dict(HD1910_KW), str(args.params_json or HD1910_JSON)
    if params:
        object.__setattr__(act, "json_path", params)
        object.__setattr__(act, "motor_name", None)
        object.__setattr__(act, "model", None)
        object.__setattr__(act, "_resolved_json_path", params)
    for k, v in kw.items():
        if k == "delay":
            object.__setattr__(act, "delay_min_lag", int(v[0]))
            object.__setattr__(act, "delay_max_lag", int(v[1]))
        elif k == "vin_drop":
            object.__setattr__(act, "vin_drop_resistance_range", tuple(v))
        else:
            object.__setattr__(act, k, v if not isinstance(v, tuple) else tuple(v))
    rec = dict(
        xml=(str(args.model_xml or PRESWAP_XML) if args.override_model == "xl330"
             else str(args.model_xml or HD1910_XML)),
        params_json=getattr(act, "_resolved_json_path", None),
        kp_fw=getattr(act, "kp_fw", None),
        vin_range=getattr(act, "vin_range", None),
        vin_min=getattr(act, "vin_min", None),
        vin_drop=getattr(act, "vin_drop_resistance_range", None),
        delay=(getattr(act, "delay_min_lag", None), getattr(act, "delay_max_lag", None)),
    )
    print("[i] ACTUATOR " + json.dumps(rec, default=str))
    return cfg, rec


# --------------------------------------------------------------------------- #
def rollout(args):
    import onnxruntime as ort
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

    cfg, ovr_rec = build_cfg(args)
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    env = RslRlVecEnvWrapper(env, clip_actions=None)
    u = env.unwrapped
    robot = u.scene.entities["robot"]
    nenv = args.num_envs

    sess = ort.InferenceSession(args.onnx)
    iname, oname = sess.get_inputs()[0].name, sess.get_outputs()[0].name
    assert sess.get_inputs()[0].shape[1] == 61, sess.get_inputs()[0].shape

    jn = list(robot.joint_names)
    jidx = {n: i for i, n in enumerate(jn)}
    missing = [k for k in HOME if k not in jidx]
    if missing:
        raise SystemExit(f"joints missing: {missing}")
    trunk = list(robot.body_names).index("trunk_base")
    site = {}
    for s in ("left_foot", "right_foot"):
        ids, _ = robot.find_sites(s)
        site[s] = int(ids[0])
    try:
        csensor = u.scene.sensors["feet_ground_contact"]
    except Exception:
        csensor = None

    # NOTE: u.sim.model arrays are indexed by MUJOCO ids (world body = 0, free
    # joint = 0) while robot.joint_names / robot.body_names are the ENTITY's
    # actuated joints / bodies. Mixing the two silently shifts by one, which is
    # how a first draft of this script reported the left hip_yaw limit as 0 deg.
    # The MECHANICAL limits are therefore read from the effective XML by NAME.
    import mujoco as _mj
    _m = _mj.MjModel.from_xml_path(ovr_rec["xml"])
    hard = {}
    for _n in jn:
        _jid = _mj.mj_name2id(_m, _mj.mjtObj.mjOBJ_JOINT, _n)
        hard[_n] = (float(_m.jnt_range[_jid][0]), float(_m.jnt_range[_jid][1]))
    soft = robot.data.joint_pos_limits[0].cpu().numpy().astype(float)
    print("[i] hip_yaw limits: XML-hard L/R = "
          f"{np.degrees(hard['left_hip_yaw']).round(2)}/{np.degrees(hard['right_hip_yaw']).round(2)} deg"
          f" | mjlab-soft L/R = {np.degrees(soft[jidx['left_hip_yaw']]).round(2)}/"
          f"{np.degrees(soft[jidx['right_hip_yaw']]).round(2)} deg")
    print(f"[i] sim jnt_range (mujoco ids) env0[:3] = "
          f"{np.asarray(u.sim.model.jnt_range.cpu()).astype(float)[0, :3].round(4).tolist()}")

    term = u.command_manager.get_term("twist")
    hterm = u.command_manager.get_term("head_pose")
    bterm = None
    try:
        bterm = u.command_manager.get_term("body_pose")
    except Exception:
        pass

    obs_td = env.reset()[0]
    n_warm = int(round(args.warmup / CTRL_DT))
    n = int(round(args.seconds / CTRL_DT))

    # ---- document what the (D)omain (R)andomization actually did per env ----
    phys = {}
    try:
        phys["total_mass_g"] = (np.asarray(
            u.sim.model.body_mass.cpu()).astype(float).sum(axis=1) * 1000.0).tolist()
        gf = np.asarray(u.sim.model.geom_friction.cpu()).astype(float)
        names = list(getattr(u.sim.model, "geom_names", []) or [])
        for want in ("left_foot_collision", "right_foot_collision"):
            if want in names:
                phys[f"slide_friction_{want}"] = gf[:, names.index(want), 0].tolist()
        phys["dof_armature_max"] = np.asarray(
            u.sim.model.dof_armature.cpu()).astype(float).max(axis=1).tolist()
    except Exception as e:
        phys["error"] = f"{type(e).__name__}: {e}"
    try:
        fs = getattr(robot.articulation.actuators[0], "friction_scale", None)
        if fs is not None:
            phys["friction_scale"] = np.asarray(
                fs.detach().cpu()).reshape(nenv, -1)[:, 0].tolist()
    except Exception:
        pass

    diag = args.diag_command
    tcfg = cfg.commands["twist"]
    if diag:
        print("[DIAG] command path per step: obs[48:51] = what the POLICY consumed;"
              " vcb = term.vel_command_b (the tensor compute() uses)")
        print(f"[DIAG] cfg: heading_command={getattr(tcfg, 'heading_command', None)}"
              f" rel_standing={getattr(tcfg, 'rel_standing_envs', None)}"
              f" rel_heading={getattr(tcfg, 'rel_heading_envs', None)}"
              f" rel_world={getattr(tcfg, 'rel_world_envs', None)}"
              f" rel_forward={getattr(tcfg, 'rel_forward_envs', None)}"
              f" rel_turn_in_place={getattr(tcfg, 'rel_turn_in_place_envs', None)}")
        try:
            print("[DIAG] masks: is_standing_env="
                  f"{term.is_standing_env.cpu().numpy().astype(int).tolist()}"
                  " is_heading_env="
                  f"{term.is_heading_env.cpu().numpy().astype(int).tolist()}"
                  " is_world_env="
                  f"{term.is_world_env.cpu().numpy().astype(int).tolist()}"
                  " is_forward_env="
                  f"{term.is_forward_env.cpu().numpy().astype(int).tolist()}")
        except Exception as _e:
            print(f"[DIAG] masks n/a: {_e}")

    rows = []
    reset_kinds = Counter()
    reset_terms = Counter()
    first_reset = np.full(nenv, -1, dtype=int)
    batched = True

    for i in range(n_warm + n):
        term.command[:, 0] = args.lin_vel_x
        term.command[:, 1] = args.lin_vel_y
        term.command[:, 2] = args.ang_vel_z
        hterm.command[:] = 0.0
        if bterm is not None:
            bterm.command[:] = 0.0

        obs = obs_td["actor"].cpu().numpy()
        if batched:
            try:
                action = sess.run([oname], {iname: obs})[0].astype(np.float32)
            except Exception as e:
                print(f"[i] batched ONNX inference failed ({type(e).__name__}); "
                      "falling back to per-env inference")
                batched = False
        if not batched:
            action = np.concatenate(
                [sess.run([oname], {iname: obs[e:e + 1]})[0] for e in range(nenv)], axis=0
            ).astype(np.float32)
        if action.shape[0] != nenv:
            raise SystemExit(f"ONNX returned batch {action.shape[0]} != {nenv}")

        obs_td, _, dones, _ = env.step(torch.from_numpy(action).to(args.device))
        done = dones.cpu().numpy().astype(bool)
        if done.any():
            rterm = u.reset_terminated.cpu().numpy()
            rtout = u.reset_time_outs.cpu().numpy()
            for e in np.where(done)[0]:
                reset_kinds["terminated" if rterm[e] else
                            ("timeout" if rtout[e] else "other")] += 1
                if first_reset[e] < 0:
                    first_reset[e] = i - n_warm
            for name in list(getattr(u.termination_manager, "active_terms", [])):
                try:
                    t = u.termination_manager.get_term(name)
                    tt = t.terminated.cpu().numpy()
                    if tt[done].any():
                        reset_terms[name] += int(tt[done].sum())
                except Exception:
                    pass

        if diag and (i < 6 or i % 100 == 0):
            try:
                vcb = term.vel_command_b.cpu().numpy()
                vcw = term.vel_command_w.cpu().numpy()
                print(f"[DIAG] step {i:4d} obs[48:51] env0={obs[0,48:51].round(4).tolist()}"
                      f"  obs vx all envs={obs[:,48].round(3).tolist()}"
                      f"  obs wz all envs={obs[:,50].round(3).tolist()}"
                      f"  vel_command_b[0]={vcb[0].round(4).tolist()}"
                      f"  vel_command_w[0]={vcw[0].round(4).tolist()}"
                      f"  time_left[0]={float(term.time_left[0]):.3g}")
            except Exception as _e:
                print(f"[DIAG] step {i}: {_e}")

        if i < n_warm:
            continue

        q = robot.data.joint_pos.cpu().numpy().astype(float)
        a = np.asarray(action, dtype=float)
        p = robot.data.body_link_pos_w[:, trunk, :].cpu().numpy().astype(float)
        qt = robot.data.body_link_quat_w[:, trunk, :].cpu().numpy().astype(float)
        vw = robot.data.root_link_vel_w[:, 0:3].cpu().numpy().astype(float)
        aw = robot.data.root_link_ang_vel_w[:, 0:3].cpu().numpy().astype(float)
        if csensor is not None:
            found = csensor.data.found.cpu().numpy().astype(float)
            force = csensor.data.force.cpu().numpy().astype(float)
        else:
            found = np.zeros((nenv, 2))
            force = np.zeros((nenv, 2, 3))
        sp = {k: robot.data.site_pos_w[:, si, :].cpu().numpy().astype(float)
              for k, si in site.items()}
        try:
            comw = robot.data.body_com_pos_w[:, trunk, :].cpu().numpy().astype(float)
        except Exception:
            comw = p

        for e in range(nenv):
            R = quat_to_mat(qt[e])
            Rinv = R.T
            vb = Rinv @ vw[e]
            fl = Rinv @ (sp["left_foot"][e] - p[e])
            fr = Rinv @ (sp["right_foot"][e] - p[e])
            cb = Rinv @ (comw[e] - p[e])   # effective trunk CoM offset in body frame
            row = {
                "env": e, "t": (i - n_warm) * CTRL_DT,
                "trunk_x": p[e, 0], "trunk_y": p[e, 1], "trunk_z": p[e, 2],
                "yaw_raw": quat_yaw(qt[e]),
                "qw": qt[e, 0], "qx": qt[e, 1], "qy": qt[e, 2], "qz": qt[e, 3],
                "vx_w": vw[e, 0], "vy_w": vw[e, 1],
                "vx_b": float(vb[0]), "vy_b": float(vb[1]),
                "wz_w": aw[e, 2], "wz_b": float(Rinv[2] @ aw[e]),
                "fwd_x": R[0, 0], "fwd_y": R[1, 0],
                "foot_l_x_b": float(fl[0]), "foot_l_y_b": float(fl[1]), "foot_l_z_b": float(fl[2]),
                "foot_r_x_b": float(fr[0]), "foot_r_y_b": float(fr[1]), "foot_r_z_b": float(fr[2]),
                "contact_l": float(found[e, 0]), "contact_r": float(found[e, 1]),
                "force_l": float(np.linalg.norm(force[e, 0])),
                "force_r": float(np.linalg.norm(force[e, 1])),
                "force_l_z": float(force[e, 0, 2]), "force_r_z": float(force[e, 1, 2]),
                "cmd_vx": args.lin_vel_x, "cmd_vy": args.lin_vel_y, "cmd_wz": args.ang_vel_z,
                "com_off_x_b": float(cb[0]), "com_off_y_b": float(cb[1]),
                "com_off_z_b": float(cb[2]),
            }
            row["obs_cmd_vx"] = float(obs[e, 48])
            row["obs_cmd_vy"] = float(obs[e, 49])
            row["obs_cmd_wz"] = float(obs[e, 50])
            for name in HOME:
                row[f"q_{name}"] = float(q[e, jidx[name]])
            for name in jn:
                row[f"a_{name}"] = float(a[e, jidx[name]])
            rows.append(row)

    return dict(rows=rows, reset_kinds=dict(reset_kinds), reset_terms=dict(reset_terms),
                first_reset=first_reset, hard_limits=hard, phys=phys, ovr=ovr_rec,
                joint_names=jn, contact_sensor=csensor is not None)


# --------------------------------------------------------------------------- #
def metrics(res, args):
    rows = res["rows"]
    nenv = args.num_envs
    by_env = {e: [] for e in range(nenv)}
    for r in rows:
        by_env[int(r["env"])].append(r)

    per_env = {}
    for e, rr in by_env.items():
        fr = res["first_reset"][e]
        cut = len(rr) if fr < 0 else min(fr, len(rr))
        d = {k: np.array([x[k] for x in rr[:cut]], dtype=float) for k in rr[0]}
        m = {"fell": bool(fr >= 0),
             "fall_time_s": (float(fr) * CTRL_DT) if fr >= 0 else None,
             "n_steps": int(cut)}
        if cut < 25:
            m["error"] = "too few steps before fall"
            per_env[e] = m
            continue
        t = d["t"]
        T = float(t[-1] - t[0])

        # ---------------- 1. heading drift (UNWRAPPED) ----------------
        yaw = np.unwrap(d["yaw_raw"])
        yaw = yaw - yaw[0]
        m["heading_net_deg"] = deg(yaw[-1])
        m["heading_rate_deg_s"] = deg(yaw[-1]) / T
        m["heading_abs_max_deg"] = deg(float(np.max(np.abs(yaw))))
        m["heading_t_at_absmax_s"] = float(t[int(np.argmax(np.abs(yaw)))])
        m.update({f"heading_{k}": v for k, v in polyfit_heading(t, yaw).items()})
        m["heading_window_slopes_deg_s"] = window_slopes(t, yaw, 4)
        absy = np.abs(yaw)
        imax = int(np.argmax(absy))
        m["heading_final_over_max"] = float(absy[-1] / absy[imax]) if absy[imax] > 1e-9 else float("nan")
        m["heading_corrected_after_max_deg"] = deg(absy[imax] - absy[-1])
        lin = np.polyfit(t, yaw, 1)
        m["heading_osc_std_deg"] = deg(float(np.std(yaw - np.polyval(lin, t))))
        f, amp = dominant_freq(yaw, CTRL_DT)
        m["heading_osc_freq_hz"], m["heading_osc_amp_deg"] = f, deg(amp)
        m["wz_world_mean_deg_s"] = deg(float(np.mean(d["wz_w"])))
        m["wz_world_std_deg_s"] = deg(float(np.std(d["wz_w"])))
        m["wz_body_mean_deg_s"] = deg(float(np.mean(d["wz_b"])))
        m["wz_body_std_deg_s"] = deg(float(np.std(d["wz_b"])))

        # ---------------- 2. path geometry (initial-facing frame) ----------------
        yaw0 = float(d["yaw_raw"][0])
        u_ = np.array([math.cos(yaw0), math.sin(yaw0)])
        n_ = np.array([-math.sin(yaw0), math.cos(yaw0)])
        xy = np.stack([d["trunk_x"], d["trunk_y"]], axis=1)
        rel = xy - xy[0]
        fwd = rel @ u_
        lat = rel @ n_
        m["forward_m"] = float(fwd[-1])
        m["lateral_m"] = float(lat[-1])
        m["path_length_m"] = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
        m["lateral_per_m"] = float(lat[-1] / fwd[-1]) if abs(fwd[-1]) > 1e-6 else float("nan")
        m["path_heading_deg"] = deg(math.atan2(lat[-1], fwd[-1]))
        m["lateral_absmax_m"] = float(np.max(np.abs(lat)))
        rms, mx, _ = fit_line_residual(rel)
        m["path_line_rms_mm"], m["path_line_max_mm"] = rms * 1000.0, mx * 1000.0
        spd = np.linalg.norm(np.stack([d["vx_w"], d["vy_w"]], axis=1), axis=1)
        ok = spd > 0.02
        m["vel_dir_net_deg"] = (deg(float(np.unwrap(np.arctan2(
            d["vy_w"][ok], d["vx_w"][ok]))[-1] - np.unwrap(np.arctan2(
            d["vy_w"][ok], d["vx_w"][ok]))[0])) if ok.sum() > 5 else float("nan"))
        m["crab_angle_deg"] = deg(float(np.arctan2(np.mean(d["vy_b"]), np.mean(d["vx_b"]))))
        m["vy_body_mean"] = float(np.mean(d["vy_b"]))
        m["vy_body_std"] = float(np.std(d["vy_b"]))

        # ---------------- 6. sanity ----------------
        m["vx_body_mean"] = float(np.mean(d["vx_b"]))
        m["vx_body_std"] = float(np.std(d["vx_b"]))
        m["vx_track_err"] = float(np.mean(d["vx_b"]) - args.lin_vel_x)
        m["trunk_z_mean"] = float(np.mean(d["trunk_z"]))
        m["trunk_z_min"] = float(np.min(d["trunk_z"]))
        tilt = np.arccos(np.clip(1.0 - 2.0 * (d["qx"] ** 2 + d["qy"] ** 2), -1, 1)) \
            if "qx" in d else np.zeros_like(d["trunk_z"])
        m["trunk_tilt_mean_deg"] = deg(float(np.mean(tilt)))
        m["trunk_tilt_max_deg"] = deg(float(np.max(tilt)))
        if "com_off_y_b" in d:
            m["com_off_x_b_mm"] = float(np.median(d["com_off_x_b"])) * 1000.0
            m["com_off_y_b_mm"] = float(np.median(d["com_off_y_b"])) * 1000.0
            m["com_off_z_b_mm"] = float(np.median(d["com_off_z_b"])) * 1000.0

        # ---------------- 4. per-leg joints ----------------
        for j in LEG_L + LEG_R:
            q = d[f"q_{j}"]
            m[f"q_{j}_mean_deg"] = deg(float(np.mean(q)))
            m[f"q_{j}_median_deg"] = deg(float(np.median(q)))
            m[f"q_{j}_std_deg"] = deg(float(np.std(q)))
            m[f"q_{j}_dev_deg"] = deg(float(np.mean(q - HOME[j])))

        # ---------------- 4. feet: stance / swing / step ----------------
        for side, km in (("L", "l"), ("R", "r")):
            mask = d[f"contact_{km}"] > 0.5
            m[f"duty_{side}"] = float(np.mean(mask))
            rise, fall = crossings(mask)
            m[f"n_steps_{side}"] = int(len(rise))
            per = np.diff(rise) if len(rise) > 1 else np.array([])
            if len(per):
                m[f"step_period_{side}_s"] = float(np.mean(per) * CTRL_DT)
                m[f"gait_freq_{side}_hz"] = 1.0 / float(np.mean(per) * CTRL_DT)
            sw = []
            for a_ in fall:
                nxt = rise[rise > a_]
                if len(nxt):
                    sw.append((nxt[0] - a_) * CTRL_DT)
            m[f"swing_time_{side}_s"] = float(np.mean(sw)) if sw else float("nan")
            if sw and len(per):
                m[f"stance_time_{side}_s"] = float(np.mean(per) * CTRL_DT - np.mean(sw))
            if len(rise) > 1:
                wx = d["trunk_x"][rise]
                wy = d["trunk_y"][rise]
                dw = np.stack([np.diff(wx), np.diff(wy)], axis=1)
                sl = dw @ u_
                m[f"step_length_{side}_m"] = float(np.mean(sl))
                m[f"step_length_{side}_std_m"] = float(np.std(sl))
                yb = d[f"foot_{km}_y_b"][rise]
                m[f"foot_y_body_at_td_{side}_mm"] = float(np.mean(yb)) * 1000.0
                m[f"foot_y_body_at_td_{side}_std_mm"] = float(np.std(yb)) * 1000.0
                xb = d[f"foot_{km}_x_b"][rise]
                m[f"foot_x_body_at_td_{side}_mm"] = float(np.mean(xb)) * 1000.0
            zk = f"foot_{km}_z_b"
            m[f"foot_clearance_{side}_mm"] = float(
                np.percentile(d[zk], 95) - np.percentile(d[zk], 5)) * 1000.0
            m[f"force_{side}_mean_N"] = float(np.mean(d[f"force_{km}"]))
            m[f"force_{side}_peak_N"] = float(np.max(d[f"force_{km}"]))
            m[f"force_{side}_fz_mean_N"] = float(np.mean(d[f"force_{km}_z"]))
            m[f"force_{side}_fz_p95_N"] = float(np.percentile(d[f"force_{km}_z"], 95))
            m[f"force_{side}_fz_peak_N"] = float(np.max(d[f"force_{km}_z"]))
        mL, mR = m.get("force_L_fz_mean_N", np.nan), m.get("force_R_fz_mean_N", np.nan)
        if mL == mL and mR == mR and abs(mL) + abs(mR) > 1e-9:
            m["contact_force_share_L"] = float(mL / (mL + mR))
        m["foot_sep_body_y_mean_mm"] = float(np.mean(d["foot_l_y_b"] - d["foot_r_y_b"])) * 1000.0

        # ---------------- 5. hip_yaw mechanical limit ----------------
        for j in ("left_hip_yaw", "right_hip_yaw"):
            lo, hi = res["hard_limits"][j]
            q = d[f"q_{j}"]
            m[f"{j}_hi_deg"] = deg(hi)
            m[f"{j}_lo_deg"] = deg(lo)
            m[f"{j}_frac_at_hi_1deg"] = float(np.mean(q > hi - math.radians(1.0)))
            m[f"{j}_frac_at_lo_1deg"] = float(np.mean(q < lo + math.radians(1.0)))
            m[f"{j}_frac_at_limit_1deg"] = float(np.mean(
                (q > hi - math.radians(1.0)) | (q < lo + math.radians(1.0))))
            m[f"{j}_toeout_mean_deg"] = deg(float(np.mean(q)))
            m[f"{j}_absmean_deg"] = deg(float(np.mean(np.abs(q))))
            km = "l" if j.startswith("left") else "r"
            rise, _ = crossings(d[f"contact_{km}"] > 0.5)
            if len(rise):
                qs = q[rise]
                m[f"{j}_frac_steps_at_hi_1deg"] = float(np.mean(qs > hi - math.radians(1.0)))
                m[f"{j}_frac_steps_at_limit_1deg"] = float(np.mean(
                    (qs > hi - math.radians(1.0)) | (qs < lo + math.radians(1.0))))
                m[f"{j}_mean_over_steps_deg"] = deg(float(np.mean(qs)))
            if len(rise) > 1:
                peaks = [np.max(np.abs(q[rise[k]:rise[k + 1]])) for k in range(len(rise) - 1)]
                m[f"{j}_step_peak_absmean_deg"] = deg(float(np.mean(peaks)))
        per_env[e] = m
    return per_env


def agg(per_env):
    keys = set()
    for m in per_env.values():
        keys |= {k for k, v in m.items() if isinstance(v, (int, float))}
    out = {}
    for k in sorted(keys):
        vals = [m[k] for m in per_env.values()
                if isinstance(m.get(k), (int, float)) and m[k] == m[k]]
        if vals:
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                      "min": float(np.min(vals)), "max": float(np.max(vals)), "n": len(vals)}
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--task", default="Mjlab-Velocity-Flat-MicroDuck")
    ap.add_argument("--lin-vel-x", type=float, default=0.2)
    ap.add_argument("--lin-vel-y", type=float, default=0.0)
    ap.add_argument("--ang-vel-z", type=float, default=0.0)
    ap.add_argument("--warmup", type=float, default=2.0)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--episode-s", type=float, default=1.0e6)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--override-model", choices=("hd1910", "xl330", "asconfigured"),
                    default="hd1910")
    ap.add_argument("--override-actuator", choices=("hd1910", "xl330", "none"),
                    default="hd1910")
    ap.add_argument("--model-xml", default=None)
    ap.add_argument("--params-json", default=None)
    ap.add_argument("--no-dr", action="store_true",
                    help="all of: physics DR off + obs noise off + IMU-DR off")
    ap.add_argument("--no-phenomena", action="store_true",
                    help="alias of --no-dr (phenomenon-free env)")
    ap.add_argument("--no-physics-dr", action="store_true",
                    help="pop the physics DR events (+ their curricula) but KEEP "
                         "observation noise and IMU-misalignment DR")
    ap.add_argument("--no-obs-noise", action="store_true",
                    help="disable actor observation noise only (keeps physics DR)")
    ap.add_argument("--no-imu-dr", action="store_true",
                    help="revert the IMU-misalignment observation funcs only")
    ap.add_argument("--fix-spawn", action="store_true")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--diag-command", action="store_true",
                    help="print the REAL per-step command the policy receives")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    res = rollout(args)
    per_env = metrics(res, args)
    summary = agg(per_env)

    print(f"\n{'='*100}\nSTRAIGHTNESS  {args.tag}\n{'='*100}")
    print(f"  onnx      : {args.onnx}")
    print(f"  cell      : model={args.override_model} actuator={args.override_actuator}"
          f"  dr={'OFF' if args.no_dr else 'ON'} spawn={'fixed' if args.fix_spawn else 'random'}")
    print(f"  env       : {res['ovr']['xml']}\n              {res['ovr']['params_json']}"
          f"  kp_fw={res['ovr']['kp_fw']} vin={res['ovr']['vin_range']} delay={res['ovr']['delay']}")
    print(f"  cmd       : vx={args.lin_vel_x:+.2f} vy={args.lin_vel_y:+.2f} wz={args.ang_vel_z:+.2f}"
          f"   {args.seconds:.0f}s x {args.num_envs} envs (contact sensor={res['contact_sensor']})")
    nfell = sum(1 for m in per_env.values() if m.get("fell"))
    print(f"  resets    : {res['reset_kinds']}  terms={res['reset_terms']}")
    print(f"  envs fell : {nfell}/{len(per_env)}" + (
        "   times=" + str([round(m["fall_time_s"], 2) for m in per_env.values() if m.get("fell")])
        if nfell else ""))

    def row(label, key, fmt="{:+.3f}", scale=1.0, sfmt=None):
        a = summary.get(key)
        if not a:
            print(f"    {label:<36} (n/a)")
            return
        print(f"    {label:<36} {fmt.format(a['mean']*scale)} +- "
              f"{(sfmt or fmt).format(a['std']*scale)}")

    print("\n  [1. HEADING DRIFT (unwrapped)]")
    row("net heading change (deg)", "heading_net_deg", "{:+.2f}")
    row("net heading rate (deg/s)", "heading_rate_deg_s", "{:+.3f}")
    row("|heading| max (deg)", "heading_abs_max_deg", "{:.2f}")
    row("linear-fit yaw bias (deg/s)", "heading_slope_deg_s", "{:+.3f}")
    row("quadratic term (deg/s^2)", "heading_quad_deg_s2", "{:+.4f}")
    row("yaw-rate bias wz world (deg/s)", "wz_world_mean_deg_s", "{:+.3f}")
    row("yaw-rate std wz world (deg/s)", "wz_world_std_deg_s", "{:.3f}")
    row("yaw oscillation std (deg)", "heading_osc_std_deg", "{:.2f}")
    row("yaw osc freq (Hz)", "heading_osc_freq_hz", "{:.2f}")
    ws = np.array([m.get("heading_window_slopes_deg_s", [np.nan]*4) for m in per_env.values()],
                  dtype=float)
    print(f"    {'windowed yaw-rate deg/s (4 bins)':<36} "
          + "  ".join(f"{v:+.3f}" for v in np.nanmean(ws, axis=0)))
    row("final/max |heading| ratio", "heading_final_over_max", "{:.3f}")

    print("\n  [2. PATH (initial-facing frame)]")
    row("forward distance (m)", "forward_m", "{:.3f}")
    row("lateral displacement (m)", "lateral_m", "{:+.4f}")
    row("lateral per metre (m/m)", "lateral_per_m", "{:+.4f}")
    row("lateral |max| (m)", "lateral_absmax_m", "{:.4f}")
    row("path heading vs cmd (deg)", "path_heading_deg", "{:+.2f}")
    row("straight-line fit RMS resid (mm)", "path_line_rms_mm", "{:.1f}")
    row("straight-line fit max resid (mm)", "path_line_max_mm", "{:.1f}")
    row("vel-direction net change (deg)", "vel_dir_net_deg", "{:+.2f}")
    row("crab angle (deg)", "crab_angle_deg", "{:+.2f}")

    print("\n  [6. LOCOMOTION SANITY]")
    row("achieved vx body (m/s)", "vx_body_mean", "{:+.3f}")
    row("vx std (m/s)", "vx_body_std", "{:.3f}")
    row("achieved vy body (m/s)", "vy_body_mean", "{:+.4f}")
    row("trunk z (m)", "trunk_z_mean", "{:.4f}")
    row("trunk tilt mean (deg)", "trunk_tilt_mean_deg", "{:.2f}")

    print("\n  [5. hip_yaw vs MECHANICAL LIMIT]")
    for j in ("left_hip_yaw", "right_hip_yaw"):
        hi = summary.get(f"{j}_hi_deg", {}).get("mean", float("nan"))
        print(f"    {j:<16} limit_hi={hi:+.2f}deg  toe-out={summary[f'{j}_toeout_mean_deg']['mean']:+.2f}"
              f"+-{summary[f'{j}_toeout_mean_deg']['std']:.2f}deg"
              f"  frac_time@limit={summary[f'{j}_frac_at_limit_1deg']['mean']*100:.1f}%"
              f"  frac_STEPS@limit={summary.get(f'{j}_frac_steps_at_limit_1deg', {}).get('mean', float('nan'))*100:.1f}%")

    print("\n  [4. PER-LEG ASYMMETRY]")
    for j in LEG_L + LEG_R:
        print(f"    {j:<18} mean={summary[f'q_{j}_mean_deg']['mean']:+7.2f}deg"
              f"  dev_HOME={summary[f'q_{j}_dev_deg']['mean']:+7.2f}deg"
              f"  std={summary[f'q_{j}_std_deg']['mean']:6.2f}deg")
    for k in ("duty_L", "duty_R", "step_period_L_s", "step_period_R_s",
              "swing_time_L_s", "swing_time_R_s", "step_length_L_m", "step_length_R_m",
              "foot_y_body_at_td_L_mm", "foot_y_body_at_td_R_mm",
              "foot_sep_body_y_mean_mm", "force_L_fz_mean_N", "force_R_fz_mean_N",
              "force_L_fz_peak_N", "force_R_fz_peak_N", "contact_force_share_L"):
        if k in summary:
            print(f"    {k:<28} {summary[k]['mean']:+.4f} +- {summary[k]['std']:.4f}")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(
            {"tag": args.tag, "onnx": args.onnx, "args": vars(args), "override": res["ovr"],
             "reset_kinds": res["reset_kinds"], "reset_terms": res["reset_terms"],
             "physics": res["phys"], "hard_limits": res["hard_limits"],
             "summary": summary, "per_env": {str(k): v for k, v in per_env.items()}},
            indent=2, default=str))
        print(f"\n[i] summary -> {args.json_out}")
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(res["rows"][0].keys()))
            w.writeheader()
            w.writerows(res["rows"])
        print(f"[i] raw     -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
