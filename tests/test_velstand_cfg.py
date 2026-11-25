"""Cfg invariants for the protective-fall VelStand rebuild (2026-09).

CPU only: builds cfgs and compiles the MuJoCo models, never steps an env.
"""
import math

import mujoco
import pytest
import torch

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab_microduck.robot.microduck_constants import (
    SERVO_GEOM_SUFFIX,
    get_allcollisions_backlash_spec,
    get_allcollisions_spec,
    get_standup_spec,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_velstand_env_cfg as vs
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)

PROTECTION_COSTS = ("servo_impact", "head_impact", "trunk_impact", "servo_acc_spike", "servo_stall")


def _geom_names(model):
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)]


# ── Model ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("spec_fn", [get_allcollisions_spec, get_allcollisions_backlash_spec])
def test_allcollisions_servo_geoms_are_named(spec_fn):
    model = spec_fn().compile()
    servo = [n for n in _geom_names(model) if n and n.endswith(SERVO_GEOM_SUFFIX)]
    # 14 servos; one body carries a 15th xl330 housing mesh in the export.
    assert len(servo) >= 14
    for n in servo:
        g = model.geom(n)
        assert g.contype[0] or g.conaffinity[0], f"{n} is not a collision geom"
        assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, int(g.dataid[0])) == "xl330"


def test_allcollisions_matches_groundcontact_kinematics():
    a = get_allcollisions_spec().compile()
    g = get_standup_spec().compile()
    assert a.njnt == g.njnt
    assert [mujoco.mj_id2name(a, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(a.njnt)] == \
           [mujoco.mj_id2name(g, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(g.njnt)]
    assert (a.jnt_range == g.jnt_range).all()
    assert abs(a.body_subtreemass[0] - g.body_subtreemass[0]) < 1e-9
    assert a.ngeom > g.ngeom  # it really is the fuller collision set


def test_backlash_twin_has_14_backlash_joints():
    m = get_allcollisions_backlash_spec().compile()
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    assert sum(n.startswith("passive_") and n.endswith("_backlash") for n in names) == 14


# ── Cfg ──────────────────────────────────────────────────────────────────────

def test_env_builds_train_and_play():
    assert vs.make_microduck_velstand_env_cfg() is not None
    assert vs.make_microduck_velstand_env_cfg(play=True) is not None
    assert vs.make_microduck_velstand_env_cfg(rough=True) is not None


def test_uses_allcollisions_robot_and_servo_sensor():
    cfg = vs.make_microduck_velstand_env_cfg()
    assert cfg.scene.entities["robot"].spec_fn is get_allcollisions_spec
    names = {s.name for s in cfg.scene.sensors}
    assert {"servo_ground_contact", "head_ground_contact", "trunk_ground_contact"} <= names
    assert cfg.sim.nconmax >= 200
    servo_sensor = next(s for s in cfg.scene.sensors if s.name == "servo_ground_contact")
    model = get_allcollisions_spec().compile()
    import re
    pat = re.compile(servo_sensor.primary.pattern)
    assert sum(bool(n and pat.match(n)) for n in _geom_names(model)) >= 14


def test_backlash_variant_mirrors_base_model():
    from mjlab_microduck.tasks import _BACKLASH_TASKS
    from mjlab_microduck.robot.microduck_constants import MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG
    rows = [r for r in _BACKLASH_TASKS if r[0].startswith("Mjlab-VelStand-")]
    assert len(rows) == 2
    for row in rows:
        assert row[4] is MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG


def test_penalty_weight_signs():
    cfg = vs.make_microduck_velstand_env_cfg()
    # mjlab-style costs return >= 0 → negative weight
    for name in PROTECTION_COSTS:
        assert cfg.rewards[name].weight < 0, name
    # self-negating microduck penalty → POSITIVE weight
    assert cfg.rewards["gentle_rise"].func is microduck_mdp.trunk_vertical_accel_penalty
    assert cfg.rewards["gentle_rise"].weight > 0
    assert cfg.rewards["joint_torque_rate_l2"].weight < 0
    # protection ramp: stage 0 is the reduced weight, final is the full weight, same sign
    for name in PROTECTION_COSTS + ("gentle_rise",):
        stages = cfg.curriculum[f"{name}_weight"].params["weight_stages"]
        assert stages[0]["step"] == 0
        assert math.isclose(stages[0]["weight"], cfg.rewards[name].weight)
        assert abs(stages[-1]["weight"]) > abs(stages[0]["weight"])
        assert math.copysign(1, stages[-1]["weight"]) == math.copysign(1, stages[0]["weight"])


def test_impact_thresholds_spare_quasi_static_pushoff():
    # Body weight ≈ 0.74 kg × 9.81 ≈ 7.3 N: pushing off the head/trunk to get up
    # must stay free; only impact spikes are billed (2026-07 lesson).
    cfg = vs.make_microduck_velstand_env_cfg()
    assert cfg.rewards["head_impact"].params["threshold"] >= 2 * 7.3
    assert cfg.rewards["trunk_impact"].params["threshold"] >= 2 * 7.3


def test_warm_start_collapses_inherited_curricula_only():
    if not vs.WARM_START:
        pytest.skip("warm start disabled")
    base = make_microduck_velocity_env_cfg()
    cfg = vs.make_microduck_velstand_env_cfg()
    # body_pose_range is REPLACED by velstand's own body-control ramp (ENABLE_BODY_CONTROL)
    replaced = {"body_pose_range"} if vs.ENABLE_BODY_CONTROL else set()
    for name, term in base.curriculum.items():
        if name in replaced:
            continue
        for key, val in cfg.curriculum[name].params.items():
            if key.endswith("_stages"):
                assert len(val) == 1 and val[0]["step"] == 0, (name, key)
                final = {k: v for k, v in term.params[key][-1].items() if k != "step"}
                assert {k: v for k, v in val[0].items() if k != "step"} == final, (name, key)
    # the loaded walk's action_rate weight is the FINAL one from step 0
    assert cfg.rewards["action_rate_l2"].weight == base.curriculum["action_rate_weight"].params["weight_stages"][-1]["weight"]
    # velstand's own curricula still ramp
    for name in ("fell_over_disable", "prone_init_prob", "fallen_tax_weight", "recovery_success_weight", "topple_push_range"):
        stages = next(v for k, v in cfg.curriculum[name].params.items() if k.endswith("_stages"))
        assert len(stages) > 1, name
    # play cfg keeps the velocity curricula untouched (they don't run in play anyway)
    play = vs.make_microduck_velstand_env_cfg(play=True)
    assert len(play.curriculum["action_rate_weight"].params["weight_stages"]) > 1


def test_phase_order():
    # fell_over off → tax-free window → economics → protection full; prone spawns after econ
    assert vs.FELL_OVER_DISABLE_ITER < vs.RECOVERY_ECON_KICKIN_ITER < vs.PROTECT_FULL_ITER
    prone_start = next(s["step"] for s in vs.PRONE_RAMP_STAGES if s["params"]["prone_prob"] > 0)
    assert prone_start >= vs.RECOVERY_ECON_KICKIN_ITER * vs.NUM_STEPS_PER_ENV
    # topple ramp only grows once fell_over is off
    assert vs.TOPPLE_PUSH_STAGES[1]["step"] == vs.FELL_OVER_DISABLE_ITER * vs.NUM_STEPS_PER_ENV
    assert vs.TOPPLE_PUSH_STAGES[-1]["velocity_range"]["x"][1] >= 1.0  # ≥ the Δv that topples in the CPU baseline


def test_topple_push_event_and_play_range():
    cfg = vs.make_microduck_velstand_env_cfg()
    ev = cfg.events["topple_push"]
    assert ev.mode == "interval"
    assert ev.params["velocity_range"] == vs.TOPPLE_PUSH_STAGES[0]["velocity_range"]
    play = vs.make_microduck_velstand_env_cfg(play=True)
    assert play.events["topple_push"].params["velocity_range"] == vs.TOPPLE_PUSH_STAGES[-1]["velocity_range"]
    assert "topple_push_range" not in play.curriculum


# ── mdp functions (pure torch, fake env) ─────────────────────────────────────

class _Data:
    pass


class _Asset:
    def __init__(self, torque, vel):
        self.data = _Data()
        self.data.actuator_force = torque
        self.data.joint_vel = vel

    def find_joints(self, pattern):
        n = self.data.joint_vel.shape[1]
        return list(range(n)), [f"j{i}" for i in range(n)]


class _Env:
    def __init__(self, torque, vel):
        self.scene = {"robot": _Asset(torque, vel)}
        self.num_envs = torque.shape[0]
        self.device = "cpu"
        self.step_dt = 0.02
        self.episode_length_buf = torch.full((self.num_envs,), 10)


def test_servo_stall_penalty_counts_only_pinned_servos():
    torque = torch.zeros(2, 14)
    vel = torch.zeros(2, 14)
    torque[0, :3] = 0.9          # env 0: three servos at high torque, not moving → stalled
    torque[1, :3] = 0.9
    vel[1, :3] = 3.0             # env 1: same torque but moving → not stalled
    out = microduck_mdp.servo_stall_penalty(_Env(torque, vel))
    assert out.tolist() == [3.0, 0.0]
    assert (out >= 0).all()


def test_servo_acc_spike_penalty_bills_only_above_threshold_and_not_at_reset():
    vel0 = torch.zeros(2, 14)
    env = _Env(torch.zeros(2, 14), vel0)
    microduck_mdp.servo_acc_spike_penalty(env, acc_thresh=300.0)  # prime prev
    vel1 = torch.zeros(2, 14)
    vel1[0, 0] = 10.0            # 10 rad/s in 0.02 s → 500 rad/s² → 200 above threshold
    vel1[1, 0] = 10.0
    env.scene["robot"].data.joint_vel = vel1
    env.episode_length_buf[1] = 0  # env 1 just reset → no bill
    out = microduck_mdp.servo_acc_spike_penalty(env, acc_thresh=300.0)
    assert math.isclose(out[0].item(), 200.0, rel_tol=1e-5)
    assert out[1].item() == 0.0


# ── Patch 5: warm start ──────────────────────────────────────────────────────

def test_warm_start_patch_resets_counters(monkeypatch):
    """MICRODUCK_WARM_START=1 zeroes the env step counter and the iteration
    after a checkpoint load; unset, the restored values are kept (plain resume)."""
    from mjlab.rl.runner import MjlabOnPolicyRunner

    class _Unwrapped:
        common_step_counter = 90024

    class _Env:
        unwrapped = _Unwrapped()

    class _Runner:
        env = _Env()
        current_learning_iteration = 3750

    monkeypatch.setattr(microduck_mdp, "_orig_runner_load", lambda self, path, *a, **k: {"env_state": {"common_step_counter": 90024}})
    monkeypatch.setenv(microduck_mdp.WARM_START_ENV, "1")
    r = _Runner()
    MjlabOnPolicyRunner.load(r, "model_3750.pt")
    assert r.env.unwrapped.common_step_counter == 0
    assert r.current_learning_iteration == 0

    monkeypatch.delenv(microduck_mdp.WARM_START_ENV)
    r = _Runner(); r.env = _Env(); r.env.unwrapped = _Unwrapped()
    MjlabOnPolicyRunner.load(r, "model_3750.pt")
    assert r.env.unwrapped.common_step_counter == 90024
    assert r.current_learning_iteration == 3750


# ── Run-1 fixes: fallen-scaled smoothness + side spawns ──────────────────────

def test_smoothness_costs_are_fallen_scaled():
    cfg = vs.make_microduck_velstand_env_cfg()
    for name in ("action_rate_l2", "joint_torque_rate_l2"):
        term = cfg.rewards[name]
        assert term.func.__name__.endswith("_fallen_scaled"), name
        assert term.weight < 0
        assert 0.0 < term.params["fallen_scale"] < 1.0
        assert term.params["gate_tilt_above_deg"] == vs.REWARD_GATE_TILT_DEG
    # the walk's full action_rate weight is still what the loaded policy was trained with
    assert cfg.rewards["action_rate_l2"].weight == -1.0


def test_prone_init_has_side_spawns():
    cfg = vs.make_microduck_velstand_env_cfg()
    assert cfg.events["random_prone_init"].params["side_prob"] == vs.PRONE_SIDE_PROB > 0
    for st in vs.PRONE_RAMP_STAGES:
        assert st["params"]["side_prob"] == vs.PRONE_SIDE_PROB


class _SimData:
    def __init__(self, n):
        self.qpos = torch.zeros(n, 21); self.qpos[:, 3] = 1.0
        self.qvel = torch.zeros(n, 20)


class _Sim:
    def __init__(self, n): self.data = _SimData(n)


class _FlatScene(dict):
    def __init__(self, n):
        super().__init__(robot=object()); self.env_origins = torch.zeros(n, 3)


class _SpawnEnv:
    def __init__(self, n):
        self.num_envs = n; self.device = "cpu"; self.sim = _Sim(n); self.scene = _FlatScene(n)


def _tilt_and_axes(q):
    w, x, y, z = q.unbind(1)
    R22 = 1 - 2 * (x * x + y * y)                 # body z · world z
    xz = 2 * (x * z - w * y)                      # body x · world z (nose up/down)
    yz = 2 * (y * z + w * x)                      # body y · world z (side)
    return torch.rad2deg(torch.acos(R22.clamp(-1, 1))), xz, yz


def test_side_spawn_quaternions_lie_on_a_side():
    torch.manual_seed(0)
    env = _SpawnEnv(256)
    microduck_mdp.set_random_prone_orientation(env, torch.arange(256), face_down_prob=0.5, side_prob=1.0)
    q = env.sim.data.qpos[:, 3:7]
    assert torch.allclose(q.norm(dim=1), torch.ones(256), atol=1e-5)
    tilt, xz, yz = _tilt_and_axes(q)
    assert (tilt > 89).all() and (tilt < 91).all()
    assert (xz.abs() < 1e-4).all()                # nose horizontal → not face down/up
    assert ((yz - 1).abs() < 1e-4).sum() > 80 and ((yz + 1).abs() < 1e-4).sum() > 80  # both sides used


def test_side_prob_zero_keeps_face_spawns():
    torch.manual_seed(0)
    env = _SpawnEnv(128)
    microduck_mdp.set_random_prone_orientation(env, torch.arange(128), face_down_prob=1.0, side_prob=0.0)
    tilt, xz, yz = _tilt_and_axes(env.sim.data.qpos[:, 3:7])
    assert (tilt > 89).all() and (xz < -0.99).all()   # face-down: nose points down


def test_fallen_scaled_action_rate(monkeypatch):
    class _AM:
        action = torch.tensor([[1.0] * 14, [1.0] * 14]); prev_action = torch.zeros(2, 14)

    class _E:
        num_envs = 2; device = "cpu"; action_manager = _AM(); scene = {"robot": object()}

    calls = {}
    def fake_mask(env, asset, z_below, tilt_above):
        calls["tilt"] = tilt_above; return torch.tensor([True, False])
    monkeypatch.setattr(microduck_mdp, "_fallen_mask", fake_mask)
    out = microduck_mdp.action_rate_l2_fallen_scaled(_E(), fallen_scale=0.1, gate_tilt_above_deg=40.0)
    assert calls["tilt"] == 40.0
    assert torch.allclose(out, torch.tensor([1.4, 14.0]))


# ── Run-2 fix: fallen-gated expert BC ────────────────────────────────────────

def test_runner_cfg_uses_expert_bc():
    from rsl_rl.utils import resolve_callable
    from mjlab_microduck.tasks import distill
    alg = vs.MicroduckVelStandRlCfg.algorithm
    assert resolve_callable(alg.class_name) is distill.PpoWithExpertBc
    if vs.ENABLE_EXPERT_BC:
        assert alg.bc_cfg["coef"] > 0
        assert alg.bc_cfg["gate_tilt_deg"] <= vs.REWARD_GATE_TILT_DEG
        assert tuple(alg.bc_cfg["gravity_slice"]) == (3, 6)
        assert tuple(alg.bc_cfg["twist_slice"]) == (48, 51)
        assert alg.bc_cfg["wandb_run_path"].endswith("69u48n8l")  # alpha_stand.onnx's run


def test_fallen_mask_from_obs():
    from mjlab_microduck.tasks.distill import fallen_mask_from_obs, expert_input
    obs = torch.zeros(4, 61)
    obs[0, 3:6] = torch.tensor([0.0, 0.0, -1.0])                       # upright
    obs[1, 3:6] = torch.tensor([math.sin(math.radians(30)), 0, -math.cos(math.radians(30))])   # 30°
    obs[2, 3:6] = torch.tensor([math.sin(math.radians(60)), 0, -math.cos(math.radians(60))])   # 60°
    obs[3, 3:6] = torch.tensor([0.0, 1.0, 0.0])                        # on the side
    m = fallen_mask_from_obs(obs, (3, 6), 35.0)
    assert m.tolist() == [False, False, True, True]
    obs[:, 48:51] = 1.0
    e = expert_input(obs, (48, 51))
    assert (e[:, 48:51] == 0).all() and (obs[:, 48:51] == 1).all()   # copy, not in place
    assert torch.equal(e[:, :48], obs[:, :48])


def test_load_expert_is_frozen_copy():
    from mjlab_microduck.tasks.distill import load_expert_from
    actor = torch.nn.Sequential(torch.nn.Linear(3, 2))
    sd = {"0.weight": torch.ones(2, 3), "0.bias": torch.zeros(2)}
    exp = load_expert_from(actor, sd)
    assert exp is not actor and not exp.training
    assert all(not p.requires_grad for p in exp.parameters())
    assert torch.equal(exp[0].weight, torch.ones(2, 3)) and not torch.equal(actor[0].weight, torch.ones(2, 3))


# ── Run-4 fix: post-fall-like prone spawns + stall weight ───────────────────

def test_reference_recipe_constants():
    """This cfg IS the recipe of the published reference (6op8a8u8@5999, commit d3edc1f).
    Runs 5-6 changed these and were rolled back (see cfg docstring). Change deliberately."""
    assert vs.EXPERT_BC_COEF == 1.0 and vs.FALLEN_SMOOTHNESS_SCALE == 0.1
    assert vs.SERVO_STALL_WEIGHT == -0.05 and vs.GENTLE_RISE_WEIGHT == 0.005
    p = vs.make_microduck_velstand_env_cfg().events["random_prone_init"].params
    # post-fall-like spawns were OFF in the reference recipe; the 2026-10 stand-up coverage turns them
    # back on together with the head-masked teacher (runs 5/6 failed on the teacher label, not the spawns)
    assert p.get("joint_random_prob", 0.0) == (vs.WIDE_JOINT_RANDOM_PROB if vs.ENABLE_WIDE_SPAWNS else 0.0)


def test_randomize_servo_joints_uniform_respects_limits(monkeypatch):
    lo = torch.tensor([-1.0] * 14); hi = torch.tensor([1.0] * 14); hi[7] = 3.0; lo[7] = -3.0
    written = {}

    class _D:
        joint_pos_limits = torch.stack([lo, hi], dim=-1).unsqueeze(0).repeat(4, 1, 1)
        joint_pos = torch.zeros(4, 14)

    class _A:
        data = _D()
        def find_joints(self, pattern): return list(range(14)), [f"j{i}" for i in range(14)]
        def write_joint_position_to_sim(self, pos, joint_ids=None, env_ids=None): assert joint_ids is None and env_ids is not None; written["pos"] = pos.clone()
        def write_joint_velocity_to_sim(self, vel, joint_ids=None, env_ids=None): assert env_ids is not None; written["vel"] = vel.clone()

    class _E:
        device = "cpu"; scene = {"robot": _A()}

    torch.manual_seed(0)
    microduck_mdp.randomize_servo_joints_uniform(_E(), torch.arange(4), range_frac=0.8)
    pos = written["pos"]
    assert pos.shape == (4, 14) and (written["vel"] == 0).all()
    assert (pos[:, :7].abs() <= 0.8 + 1e-6).all() and (pos[:, 7].abs() <= 2.4 + 1e-6).all()
    assert pos.std() > 0.3  # actually randomized, not HOME


# ── Rough-terrain spawns: absolute z must be offset by the env origin ────────

def test_prone_spawn_adds_terrain_origin_z():
    """wy8gcaus lesson: an absolute trunk z on rough terrain spawned robots inside the slope pyramids."""
    class _Scene(dict):
        env_origins = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.30], [2.0, 0.0, 0.12], [3.0, 0.0, 0.05]])
    env = _SpawnEnv(4); env.scene = _Scene(robot=object())
    microduck_mdp.maybe_set_random_prone_orientation(env, torch.arange(4), prone_prob=1.0, prone_z_min=0.07, prone_z_max=0.07)
    z = env.sim.data.qpos[:, 2]
    assert torch.allclose(z, torch.tensor([0.07, 0.37, 0.19, 0.12]), atol=1e-6)
    assert (torch.rad2deg(torch.acos((1 - 2 * (env.sim.data.qpos[:, 4] ** 2 + env.sim.data.qpos[:, 5] ** 2)).clamp(-1, 1))) > 89).all()


def test_body_frame_masks():
    """Standing + tilt-only → stand expert; standing + z → off-anchor, no teacher; walking or zero body → untouched."""
    from mjlab_microduck.tasks.distill import body_frame_masks
    obs = torch.zeros(5, 61)
    obs[0, 55 + 3] = 0.1                       # standing, roll only → teach
    obs[1, 55 + 2] = -0.02; obs[1, 55 + 4] = 0.1  # standing, z + pitch → free
    obs[2, 48] = 0.3; obs[2, 55 + 3] = 0.1     # walking with a (stale) body value → untouched
    obs[3, 55 + 0] = 0.002                     # standing, x/y alive noise only → untouched (stays anchored)
    frames, teach = body_frame_masks(obs, (48, 51), (55, 61), (2, 3, 4), (2,))
    assert frames.tolist() == [True, True, False, False, False]
    assert teach.tolist() == [True, False, False, False, False]


def test_body_control_wired():
    if not vs.ENABLE_BODY_CONTROL:
        return
    cfg = vs.make_microduck_velstand_env_cfg()
    cmds = list(cfg.commands)
    assert cmds.index("twist") < cmds.index("body_pose")  # gate reads the twist term computed this step
    b = cfg.commands["body_pose"]
    assert type(b).__name__ == "StandingGatedPoseCommandCfg" and b.zero_command_prob == vs.BODY_ZERO_CMD_PROB
    assert len(b.ranges) == 6 and b.axis_zero_prob[2] == vs.BODY_Z_ZERO_PROB
    # final stage covers the full range; x/y/yaw stay inside alpha_stand's normalizer comfort zone
    final = cfg.curriculum["body_pose_range"].params["range_stages"][-1]["ranges"]
    assert final[3][1] == pytest.approx(vs.BODY_CMD_MAX_ANGLE) and final[2] == pytest.approx(vs.BODY_CMD_Z_RANGE)
    assert abs(final[0][1]) <= 0.005 and abs(final[5][1]) <= 0.05
    r = cfg.rewards["body_pose_tracking"]
    assert r.weight > 0 and r.params["standing_gate_command_name"] == "twist"
    assert r.params["axis_weights"] == (0.0, 0.0, 1.0, 1.0, 1.0, 0.0) and r.params["nominal_height"] == vs.BODY_NOMINAL_Z
    assert cfg.rewards["upright"].func.__name__ == "upright_body_cmd_relative" and cfg.rewards["upright"].weight == 2.0
    assert cfg.rewards["pose"].func.__name__ == "variable_posture_body_relaxed"
    bc = vs.MicroduckVelStandRlCfg.algorithm.bc_cfg
    assert tuple(bc["body_slice"]) == (55, 61) and tuple(bc["body_no_teacher_axes"]) == (2,)


def test_turn_in_place_mask():
    from mjlab_microduck.tasks.distill import turn_in_place_mask
    obs = torch.zeros(4, 61)
    obs[0, 50] = 0.3                   # turn in place → unanchored
    obs[1, 48] = 0.2; obs[1, 50] = 0.6  # walking + turning → stays anchored
    obs[2, 48] = 0.2                   # walking straight
    assert turn_in_place_mask(obs, (48, 51)).tolist() == [True, False, False, False]


def test_yaw_fix_wired():
    if not vs.ENABLE_YAW_FIX:
        return
    cfg = vs.make_microduck_velstand_env_cfg()
    # stock term untouched (its far-error gradient is what run sape62zb lost), fine term added
    base = make_microduck_velocity_env_cfg()
    assert cfg.rewards["track_angular_velocity"].func is base.rewards["track_angular_velocity"].func
    assert cfg.rewards["track_angular_velocity"].params["std"] == base.rewards["track_angular_velocity"].params["std"]
    f = cfg.rewards["track_yaw_fine"]
    assert f.func.__name__ == "track_yaw_rate_fine" and f.weight > 0 and f.params["std"] < base.rewards["track_angular_velocity"].params["std"]
    tw = cfg.commands["twist"]
    assert tw.turn_in_place_min_frac == vs.TURN_IN_PLACE_MIN_FRAC and tw.rel_turn_in_place_envs == vs.TURN_IN_PLACE_FRACTION_VELSTAND
    assert vs.MicroduckVelStandRlCfg.algorithm.bc_cfg["unanchor_turn_in_place"] is True


# ── 2026-10 stand-up coverage: wide spawns + head-masked teacher ─────────────

def test_wide_spawns_wired():
    p = vs.make_microduck_velstand_env_cfg().events["random_prone_init"].params
    if not vs.ENABLE_WIDE_SPAWNS:
        assert p.get("so3_prob", 0.0) == 0.0 and p.get("sit_prob", 0.0) == 0.0
        return
    assert p["so3_prob"] == vs.WIDE_SO3_PROB and p["sit_prob"] == vs.WIDE_SIT_PROB
    assert p["sit_params"]["sitting_joint_overrides"] == vs.SIT_JOINT_OVERRIDES
    # the curriculum stages must not overwrite the coverage params (shallow merge)
    for st in vs.PRONE_RAMP_STAGES:
        assert not {"joint_random_prob", "so3_prob", "sit_prob"} & set(st["params"])
    # total spawn slice stays bounded so walking keeps the majority of resets
    final = vs.PRONE_RAMP_STAGES[-1]["params"]
    assert final["prone_prob"] + final["crouch_prob"] + p["sit_prob"] <= 0.75


def test_so3_spawns_are_random_orientations_with_random_joints(monkeypatch):
    torch.manual_seed(0)
    n = 512
    env = _SpawnEnv(n)
    jr_ids = []
    monkeypatch.setattr(microduck_mdp, "randomize_servo_joints_uniform", lambda e, ids, **k: jr_ids.append(ids.clone()))
    microduck_mdp.maybe_set_random_prone_orientation(env, torch.arange(n), prone_prob=1.0, prone_z_min=0.07, prone_z_max=0.07,
                                                    so3_prob=1.0, so3_extra_z=0.04, joint_random_extra_z=0.06)
    q = env.sim.data.qpos[:, 3:7]
    assert torch.allclose(q.norm(dim=1), torch.ones(n), atol=1e-5)
    tilt, _, _ = _tilt_and_axes(q)
    # uniform on SO(3): cos(tilt) uniform on [-1, 1] → tilt spread over the whole range, not ±90° only
    assert (tilt < 45).float().mean() > 0.1 and (tilt > 135).float().mean() > 0.1
    assert len(jr_ids) == 1 and jr_ids[0].numel() == n  # every so3 spawn gets random joints
    assert torch.allclose(env.sim.data.qpos[:, 2], torch.full((n,), 0.07 + 0.04 + 0.06), atol=1e-6)


def test_sit_slice_is_exclusive(monkeypatch):
    torch.manual_seed(0)
    n = 2000
    env = _SpawnEnv(n)
    got = {}
    monkeypatch.setattr(microduck_mdp, "set_random_ground_state", lambda e, ids, **k: got.update(ids=ids.clone(), k=k))
    monkeypatch.setattr(microduck_mdp, "set_random_crouch_state", lambda e, ids, **k: got.update(crouch=ids.clone()))
    microduck_mdp.maybe_set_random_prone_orientation(env, torch.arange(n), prone_prob=0.4, crouch_prob=0.2, sit_prob=0.1,
                                                    prone_z_min=0.07, prone_z_max=0.07, sit_params={"sitting_z_min": 0.06})
    assert abs(got["ids"].numel() / n - 0.1) < 0.03 and got["k"]["sitting_prob"] == 1.0 and got["k"]["sitting_z_min"] == 0.06
    assert not set(got["ids"].tolist()) & set(got["crouch"].tolist())
    tilt, _, _ = _tilt_and_axes(env.sim.data.qpos[:, 3:7])
    assert not set(got["ids"].tolist()) & set(torch.nonzero(tilt > 89).flatten().tolist())  # not also prone


def test_expert_input_masks_only_selected_rows():
    from mjlab_microduck.tasks.distill import expert_input
    obs = torch.ones(3, 61)
    out = expert_input(obs, (48, 51), mask_cols=vs.TEACHER_MASK_COLS, mask_rows=torch.tensor([True, False, True]))
    assert (out[:, 48:51] == 0).all()                       # twist always zeroed for the stand expert
    assert (out[0, list(vs.TEACHER_MASK_COLS)] == 0).all() and (out[1, list(vs.TEACHER_MASK_COLS)] == 1).all()
    others = [c for c in range(61) if c not in vs.TEACHER_MASK_COLS and not 48 <= c < 51]
    assert (out[:, others] == 1).all()                      # ang_vel, gravity, head/body cmds kept
    assert (obs == 1).all()                                 # input untouched


def test_teacher_mask_cols_are_the_joint_blocks():
    """ang_vel 0:3, gravity 3:6, joint_pos 6:20, joint_vel 20:34, last action 34:48 — mask = all joint state."""
    assert vs.TEACHER_MASK_COLS == tuple(range(6, 48))
    cfg = vs.make_microduck_velstand_env_cfg()
    assert list(cfg.observations["actor"].terms)[:5] == ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions"]


def test_joint_masked_teacher_wired():
    bc = vs.MicroduckVelStandRlCfg.algorithm.bc_cfg
    if vs.ENABLE_JOINT_MASKED_TEACHER:
        assert tuple(bc["expert_mask_cols"]) == vs.TEACHER_MASK_COLS
        # masking from the 35° BC gate broke the end of the rise (emulated) — keep it well above
        assert bc["expert_mask_tilt_deg"] == vs.TEACHER_MASK_TILT_DEG >= 70.0 > bc["gate_tilt_deg"]
    else:
        assert not bc.get("expert_mask_cols")
    assert bc["coef"] == 1.0 and bc["anchor_coef"] == 1.0  # teacher strengths unchanged


def test_remap_anchor_yaw_table_lookup():
    from mjlab_microduck.tasks.distill import remap_anchor_yaw
    t = vs.ANCHOR_YAW_REMAP
    obs = torch.zeros(6, 61)
    obs[:, 48:51] = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.3], [0.0, 0.0, -1.0], [0.2, 0.0, 0.6], [0.15, 0.0, 0.6], [0.3, 0.0, 0.0]])
    obs[:, 0] = 7.0
    out = remap_anchor_yaw(obs, (48, 51), t)
    ti = lambda c: list(t["targets"]).index(c)
    assert out[0, 50] == 0 and out[5, 50] == 0                                  # standing / straight walking unchanged
    assert abs(out[1, 50].item() - t["pos"][0][ti(0.3)]) < 1e-5                   # knot hit, in place
    assert abs(out[2, 50].item() + t["neg"][0][ti(1.0)]) < 1e-5                   # sign kept, own direction table
    assert abs(out[3, 50].item() - t["pos"][2][ti(0.6)]) < 1e-5
    mid = 0.5 * (t["pos"][1][ti(0.6)] + t["pos"][2][ti(0.6)])
    assert abs(out[4, 50].item() - mid) < 1e-5                                   # linear in |v_xy| between knots
    assert (out[:, :50] == obs[:, :50]).all() and (out[:, 51:] == obs[:, 51:]).all()  # only the yaw command changes
    assert (obs[:, 50] != out[:, 50]).any() and obs[1, 50] == 0.3               # input untouched


def test_anchor_yaw_remap_wired():
    bc = vs.MicroduckVelStandRlCfg.algorithm.bc_cfg
    if not vs.ENABLE_ANCHOR_YAW_REMAP:
        assert not bc.get("anchor_yaw_remap")
        return
    assert bc["anchor_yaw_remap"] is vs.ANCHOR_YAW_REMAP and not bc.get("unanchor_turn_in_place")
    t = vs.ANCHOR_YAW_REMAP
    for key in ("pos", "neg"):
        assert len(t[key]) == len(t["lin"]) and all(len(r) == len(t["targets"]) for r in t[key])
        assert all(list(r) == sorted(r) and r[0] > 0 for r in t[key])            # monotone, never maps a turn to 0
    cfg = vs.make_microduck_velstand_env_cfg()
    assert cfg.rewards["track_yaw_fine"].weight > 0 and cfg.commands["twist"].turn_in_place_min_frac == vs.YAW_REMAP_TURN_IN_PLACE_MIN_FRAC
