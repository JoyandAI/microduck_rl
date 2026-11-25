#!/usr/bin/env python3
"""Generate pick-up detector data in MuJoCo-Warp (or evaluate a detector closed-loop).

The prod policy (velstand) runs under the full training DR stack on rough
terrain with backlash; a virtual hand (pickup/hand.py) picks robots up, carries,
tilts, shakes, sets down or drops them. Pause/resume is driven either by an
ORACLE (ground truth + sampled detection delays, plus spurious pauses on the
floor and some pick-ups the "detector" misses) to collect training data, or by
a trained DETECTOR (--detector model.pt) through the runtime state machine to
measure closed-loop false pauses / latencies.

Usage:
  uv run scripts/pickup_datagen.py --out data/pickup/train_0.npz --num-envs 1024 --seconds 60 --seed 0
  uv run scripts/pickup_datagen.py --detector logs/pickup/model.pt --out data/pickup/eval.npz --seed 100
"""

import argparse
import math
import os
import time

import numpy as np
import onnx
import torch
from onnx import numpy_helper

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab_microduck.pickup.features import FEAT_DIM, PAUSED_IDX, CurrentSensorDR, paused_target_rel
from mjlab_microduck.pickup.hand import VirtualHand, HandCfg, add_hand_to_spec
from mjlab_microduck.robot.microduck_constants import MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG
from mjlab_microduck.tasks import mdp as m
from mjlab_microduck.tasks.backlash import make_backlash_variant
from mjlab_microduck.tasks.microduck_velstand_env_cfg import make_microduck_velstand_env_cfg

PROD_POLICY = "logs/bench_onnx/velstand_fhathosb_3750.onnx"


class OnnxMLP(torch.nn.Module):
    """Batched torch replica of an exported actor (Sub/Div normalizer + Gemm/Elu)."""

    def __init__(self, path, device):
        super().__init__()
        g = onnx.load(path).graph
        init = {i.name: torch.as_tensor(numpy_helper.to_array(i).copy(), device=device) for i in g.initializer}
        self.ops = []
        for n in g.node:
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
            self.ops.append((n.op_type, [init.get(i) for i in n.input], attrs))

    @torch.no_grad()
    def forward(self, x):
        for op, ins, at in self.ops:
            if op == "Sub":
                x = x - ins[1]
            elif op == "Div":
                x = x / ins[1]
            elif op == "Gemm":
                w = ins[1].T if at.get("transB", 0) else ins[1]
                x = x @ w + (ins[2] if len(ins) > 2 and ins[2] is not None else 0)
            elif op == "Elu":
                x = torch.nn.functional.elu(x)
            else:
                raise ValueError(op)
        return x


def build_env(num_envs, device, seed, keep_held_tilted=False):
    cfg = make_microduck_velstand_env_cfg(play=False, rough=True)
    cfg = make_backlash_variant(cfg, MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    # the terrain-level curriculum would move robots between levels on reset by
    # walking distance — irrelevant here, keep terrain levels random
    cfg.curriculum.pop("terrain_levels", None)
    # warm-start velstand keeps fell_over (70°) for its first 100 iters; here
    # held robots are tilted way past that and falls must stay in the data
    cfg.terminations.pop("fell_over", None)
    cfg.curriculum.pop("fell_over_disable", None)
    if keep_held_tilted:
        # fallen_too_long resets anything tilted >40° for 8 s — including a robot somebody is
        # holding upside down. For training data that only shortens long flipped holds; for the
        # battery (one long forced hold) it would end the scenario.
        cfg.terminations.pop("fallen_too_long", None)
    prev_fn = cfg.scene.spec_fn  # rough terrain softens its contacts here

    def spec_fn(spec):
        if prev_fn is not None:
            prev_fn(spec)
        add_hand_to_spec(spec)

    cfg.scene.spec_fn = spec_fn
    return ManagerBasedRlEnv(cfg=cfg, device=device)


class HandSim:
    """Binds VirtualHand to the welded mocap hand of a mjlab env (trunk or head grip)."""

    def __init__(self, env, hand):
        import mujoco
        self.env, self.hand = env, hand
        mj = env.sim.mj_model
        self.eq = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_EQUALITY, "pickup_hand_weld")
        self.eq_head = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_EQUALITY, "pickup_hand_weld_head")
        self.mocap = int(mj.body_mocapid[mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "pickup_hand")])
        assert self.eq >= 0 and self.eq_head >= 0 and self.mocap >= 0
        self.head = int(env.scene["robot"].find_bodies("jaw_soft")[0][0])

    def head_pose(self, robot):
        return robot.data.body_link_pos_w[:, self.head], robot.data.body_link_quat_w[:, self.head]

    def start(self, mask, robot):
        self.hand.start(mask, robot.data.root_link_pos_w, robot.data.root_link_quat_w, *self.head_pose(robot))

    def step(self, dt, robot, feet_force, mass, upright):
        pos, quat, active = self.hand.step(dt, robot.data.root_link_pos_w, robot.data.root_link_quat_w, feet_force,
                                           mass, upright, *self.head_pose(robot))
        d = self.env.sim.data
        d.mocap_pos[:, self.mocap] = pos
        d.mocap_quat[:, self.mocap] = quat
        head = self.hand.grip == 1
        d.eq_active[:, self.eq] = active & ~head
        d.eq_active[:, self.eq_head] = active & head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", default=PROD_POLICY)
    ap.add_argument("--detector", default=None, help="closed-loop: model.pt from pickup_train.py")
    ap.add_argument("--pickup-rate", type=float, default=0.08)
    ap.add_argument("--sm", default="", help="state-machine overrides, e.g. p_resume=0.35,n_resume=4")
    ap.add_argument("--hand", default="", help="HandCfg overrides as python literals, e.g. 'head_grip_prob=1.0;orient_prob=0'")
    ap.add_argument("--force-pickup-at", type=float, default=None, help="battery: every env picked up at this time, held to the end")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    dev = args.device
    env = build_env(args.num_envs, dev, args.seed, keep_held_tilted=args.force_pickup_at is not None)
    N, dt = env.num_envs, env.step_dt
    robot = env.scene["robot"]
    sids = m._servo_joint_ids(env, robot)
    from bam.model import load_model
    kt = load_model(motor_name="xl330", model="m6").kt.value
    policy = OnnxMLP(args.policy, dev)
    hcfg = HandCfg(pickup_rate_hz=args.pickup_rate)
    for kv in filter(None, args.hand.split(";")):
        k, v = kv.split("=", 1)
        assert hasattr(hcfg, k.strip()), k
        setattr(hcfg, k.strip(), eval(v, {"math": math, "pi": math.pi}))
    if args.force_pickup_at is not None:
        hcfg.pickup_rate_hz = 0.0
        hcfg.hold_s = (1e4, 1e4)
    hand = VirtualHand(N, dev, hcfg)
    hsim = HandSim(env, hand)
    cur_dr = CurrentSensorDR(N, dev)
    feet = env.scene.sensors["feet_ground_contact"]

    # pause controller state
    paused = torch.zeros(N, dtype=torch.bool, device=dev)
    t_paused = torch.zeros(N, device=dev)
    tgt_at_pause = torch.zeros(N, 14, device=dev)
    # oracle state
    held_t = torch.zeros(N, device=dev); free_t = torch.zeros(N, device=dev)
    d_up = torch.empty(N, device=dev).uniform_(0.06, 0.5); d_down = torch.empty(N, device=dev).uniform_(0.1, 0.8)
    skip = torch.zeros(N, dtype=torch.bool, device=dev)      # this pick-up is "missed" by the oracle
    min_pause = torch.zeros(N, device=dev)                    # spurious pauses last at least this long
    prev_held = torch.zeros(N, dtype=torch.bool, device=dev)

    det = sm = win = None
    if args.detector:
        from mjlab_microduck.pickup.model import PickupNet, PauseStateMachine, StateMachineCfg
        ck = torch.load(args.detector, map_location=dev)
        det = PickupNet(**ck.get("net_kwargs", {})).to(dev); det.load_state_dict(ck["state_dict"]); det.eval()
        smc = StateMachineCfg(**ck.get("sm_cfg", {}))
        for kv in filter(None, args.sm.split(",")):
            k, v = kv.split("="); setattr(smc, k, type(getattr(smc, k))(v))
        print("state machine:", smc)
        sm = PauseStateMachine(N, dev, smc, dt)
        win = None  # (N, W, F) filled at the first tick

    T = int(args.seconds / dt)
    rec = {k: [] for k in ("feat", "held", "paused", "phase", "since_reset", "fallen", "p", "feet_frac", "cmd")}
    since_reset = torch.zeros(N, dtype=torch.int32, device=dev)

    obs, _ = env.reset()
    action = torch.zeros(N, 14, device=dev)
    mass = env.sim.model.body_mass.sum(-1) if env.sim.model.body_mass.ndim == 2 else env.sim.model.body_mass.sum().expand(N)
    t0 = time.time()
    for step in range(T):
        # ── controller ──
        pol = policy(obs["actor"])
        t_paused = torch.where(paused, t_paused + dt, torch.zeros_like(t_paused))
        hold = paused_target_rel(t_paused, tgt_at_pause)
        action = torch.where(paused.unsqueeze(-1), hold, pol)

        # ── hand ──
        q = robot.data.root_link_quat_w
        up = (1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)) > 0.5
        ff = torch.nan_to_num(feet.data.force).norm(dim=-1).sum(-1)
        if args.force_pickup_at is not None and step == int(args.force_pickup_at / dt):
            hsim.start(torch.ones(N, dtype=torch.bool, device=dev), robot)
        hsim.step(dt, robot, ff, mass, up)

        obs, _, term, trunc, _ = env.step(action)
        done = term | trunc

        # ── features (post-step state = what the runtime reads next tick) ──
        a_obs = obs["actor"]
        cur = robot.data.actuator_force[:, :14] / kt
        feat = torch.cat([a_obs[:, 0:3], a_obs[:, 3:6], a_obs[:, 6:20], a_obs[:, 20:34], action,
                          cur_dr(cur), paused.float().unsqueeze(-1)], dim=-1)
        ff = torch.nan_to_num(feet.data.force).norm(dim=-1).sum(-1)
        feet_frac = ff / (mass * 9.81)
        held = hand.active & (feet_frac < 0.5)
        q = robot.data.root_link_quat_w
        fallen = (1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)) < math.cos(math.radians(60))

        # ── pause / resume decision ──
        if det is None:
            started = held & ~prev_held
            if started.any():
                skip[started] = torch.rand(int(started.sum()), device=dev) < 0.15
                d_up[started] = torch.empty(int(started.sum()), device=dev).uniform_(0.06, 0.5)
            held_t = torch.where(held, held_t + dt, torch.zeros_like(held_t))
            free_t = torch.where(~held, free_t + dt, torch.zeros_like(free_t))
            go_pause = ~paused & held & (held_t >= d_up) & ~skip
            spur = ~paused & ~held & (torch.rand(N, device=dev) < 0.03 * dt)
            min_pause = torch.where(spur, torch.empty(N, device=dev).uniform_(0.5, 4.0), torch.where(go_pause, torch.zeros_like(min_pause), min_pause))
            go_pause = go_pause | spur
            go_resume = paused & ~held & (free_t >= d_down) & (t_paused >= min_pause)
            ended = ~held & prev_held
            if ended.any():
                d_down[ended] = torch.empty(int(ended.sum()), device=dev).uniform_(0.1, 0.8)
            p = held.float()
        else:
            if win is None:
                win = feat.unsqueeze(1).repeat(1, det.window, 1)
            win = torch.cat([win[:, 1:], feat.unsqueeze(1)], dim=1)
            with torch.no_grad():
                p = torch.sigmoid(det(win))
            go_pause, go_resume = sm.update(p)
        tgt_at_pause = torch.where(go_pause.unsqueeze(-1), action, tgt_at_pause)
        paused = (paused | go_pause) & ~go_resume
        if go_resume.any():  # runtime controller.reset(): zero the policy's last-action memory
            env.action_manager._action[go_resume] = 0.0
            if hasattr(env.action_manager, "_prev_action"):
                env.action_manager._prev_action[go_resume] = 0.0
        prev_held = held

        # ── resets ──
        if done.any():
            ids = done.nonzero().squeeze(-1)
            hand.reset(ids); cur_dr.resample(ids)
            paused[ids] = False; t_paused[ids] = 0.0; held_t[ids] = 0.0; free_t[ids] = 0.0; prev_held[ids] = False
            since_reset[ids] = -1
            if sm is not None:
                sm.reset(ids)
                win[ids] = feat[ids].unsqueeze(1)  # obs after reset is the new episode's first tick
        since_reset += 1

        rec["feat"].append(feat.half().cpu()); rec["held"].append(held.cpu()); rec["paused"].append(paused.cpu())
        rec["phase"].append(hand.phase.to(torch.int8).cpu()); rec["since_reset"].append(since_reset.clone().cpu())
        rec["fallen"].append(fallen.cpu()); rec["p"].append(p.half().cpu()); rec["feet_frac"].append(feet_frac.half().cpu())
        rec["cmd"].append(env.command_manager.get_command("twist").norm(dim=-1).half().cpu())
        if step % 500 == 0:
            print(f"step {step}/{T}  held {held.float().mean():.2f}  paused {paused.float().mean():.2f}  "
                  f"fallen {fallen.float().mean():.2f}  {time.time() - t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out = {k: torch.stack(v).numpy() for k, v in rec.items()}
    out["grip"] = hand.grip.cpu().numpy()
    np.savez_compressed(args.out, **out)
    print(f"saved {args.out}: {T} ticks x {N} envs, held {out['held'].mean():.3f} paused {out['paused'].mean():.3f}")
    assert out["feat"].shape[-1] == FEAT_DIM and PAUSED_IDX == FEAT_DIM - 1


if __name__ == "__main__":
    main()
