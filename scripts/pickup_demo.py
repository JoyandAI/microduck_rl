#!/usr/bin/env python3
"""Pick-up detector demo in CPU MuJoCo (BAM actuators, prod velstand policy).

The exported ONNX detector runs exactly as the runtime would: every 50 Hz tick
it sees the last 1 s of features (features.py), the state machine pauses the
policy (ramp to the pause pose) when the duck is picked up and resumes it when
put back down. The hand is the same welded mocap hand as in training.

  # scripted scenario → MP4 with a live p(held) strip
  uv run scripts/pickup_demo.py --detector logs/pickup/nocur/pickup_detector.onnx --video pickup_demo.mp4
  # interactive viewer (keys in the TERMINAL): p = pick up, l = put down, d = drop,
  # arrows = walk command, space = stop, q = quit. Ctrl+right-drag in the viewer also lifts the duck.
  uv run scripts/pickup_demo.py --detector logs/pickup/nocur/pickup_detector.onnx --interactive
  # baseline (no detector): what happens today
  uv run scripts/pickup_demo.py --no-detector --video baseline.mp4
"""

import argparse
import math
import os
import sys
import time

import mujoco
import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer_policy import (  # noqa: E402
    BAM_MAX_CURRENT, BAM_STIFF_SOLIMP_FRICTION, BAM_STIFF_SOLREF_FRICTION, DEFAULT_POSE, TerminalInput, load_bam_model,
)

from mjlab_microduck.pickup.features import FEAT_DIM, paused_target_rel  # noqa: E402
from mjlab_microduck.pickup.hand import CARRY, LOWER, HandCfg, VirtualHand, add_hand_to_spec  # noqa: E402
from mjlab_microduck.pickup.model import WINDOW, PauseStateMachine, StateMachineCfg  # noqa: E402

SCENE = "src/mjlab_microduck/robot/microduck/scene_allcollisions.xml"
POLICY = "logs/bench_onnx/velstand_fhathosb_3750.onnx"
DT, DECIM = 0.005, 4
CTRL_DT = DT * DECIM


def build(scene, kp_fw=200.0, vin=7.4):
    from bam.mujoco import MujocoController
    bam = load_bam_model(kp_fw, vin, BAM_MAX_CURRENT)
    kt, R = bam.kt.value, bam.R.value
    spec = mujoco.MjSpec.from_file(scene)
    names = []
    for act in spec.actuators:
        tgt = act.target if isinstance(act.target, str) else act.target.name
        if tgt.startswith("passive_"):
            continue
        act.set_to_motor(); act.forcelimited = True
        act.forcerange = (-vin * kt / R, vin * kt / R); act.ctrllimited = False
        act.gear = [1.0, 0, 0, 0, 0, 0]
        names.append(act.name)
        for j in spec.joints:
            if j.name == tgt:
                j.damping = np.zeros((3, 1)); j.frictionloss = 0.0
                j.solref_friction = BAM_STIFF_SOLREF_FRICTION; j.solimp_friction = BAM_STIFF_SOLIMP_FRICTION
    add_hand_to_spec(spec, trunk_body="trunk_base")
    model = spec.compile(); model.opt.timestep = DT
    model.vis.global_.offwidth, model.vis.global_.offheight = 1280, 960
    data = mujoco.MjData(model)
    ctrl = MujocoController(bam, names, model, data, vin_drop_gain=0.1, vin_min=6.0)
    return model, data, ctrl, names, kt


class Demo:
    def __init__(self, args):
        self.m, self.d, self.bam, names, self.kt = build(args.scene)
        m = self.m
        jn = [m.actuator(n).trnid[0] for n in names]
        self.qadr = np.array([m.jnt_qposadr[j] for j in jn]); self.vadr = np.array([m.jnt_dofadr[j] for j in jn])
        self.act_ids = np.array([m.actuator(n).id for n in names])
        self.trunk = m.body("trunk_base").id
        self.gyro_adr = m.sensor_adr[m.sensor("imu_ang_vel").id]
        self.eq = m.equality("pickup_hand_weld").id
        self.eq_head = m.equality("pickup_hand_weld_head").id
        self.head = m.body("jaw_soft").id
        self.mocap = m.body_mocapid[m.body("pickup_hand").id]
        self.lfoot = [g for g in range(m.ngeom) if "foot" in (m.geom(g).name or "")]
        self.mass = float(m.body_subtreemass[0])
        self.policy = ort.InferenceSession(args.policy)
        self.det = None if args.no_detector else ort.InferenceSession(args.detector)
        self.sm = PauseStateMachine(1, "cpu", StateMachineCfg(), CTRL_DT)
        self.hand = VirtualHand(1, "cpu", HandCfg(pickup_rate_hz=0.0))
        self.rng = np.random.default_rng(args.seed)
        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0) if self.m.nkey else mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.qadr] = DEFAULT_POSE
        mujoco.mj_forward(self.m, self.d)
        self.last_action = np.zeros(14, np.float32)
        self.cmd = np.zeros(13, np.float32)
        self.win = None
        self.paused = False; self.t_paused = 0.0; self.tgt_at_pause = np.zeros(14, np.float32)
        self.target_rel = np.zeros(14, np.float32)
        self.p = 0.0; self.t = 0.0; self.n_ticks = 0

    # ── sensing ─────────────────────────────────────────────────────────
    def grav(self):
        q = self.d.xquat[self.trunk]; w, xyz = q[0], q[1:]
        v = np.array([0, 0, -1.0]); t = np.cross(xyz, v) * 2
        return (v - w * t + np.cross(xyz, t)).astype(np.float32)

    def gyro(self):
        return self.d.sensordata[self.gyro_adr:self.gyro_adr + 3].astype(np.float32)

    def feet_force(self):
        f = 0.0; c6 = np.zeros(6)
        for i in range(self.d.ncon):
            con = self.d.contact[i]
            g1, g2 = con.geom1, con.geom2
            if (g1 in self.lfoot and self.m.geom_bodyid[g2] == 0) or (g2 in self.lfoot and self.m.geom_bodyid[g1] == 0):
                mujoco.mj_contactForce(self.m, self.d, i, c6); f += abs(c6[0])
        return f

    def features(self):
        r = self.rng
        cur = np.abs(self.d.actuator_force[self.act_ids] / self.kt + r.normal(0, 0.01, 14)) + 0.01
        return np.concatenate([
            self.gyro() + r.uniform(-0.03, 0.03, 3), self.grav() + r.uniform(-0.01, 0.01, 3),
            self.d.qpos[self.qadr] - DEFAULT_POSE + r.uniform(-0.001, 0.001, 14),
            self.d.qvel[self.vadr] + r.uniform(-0.25, 0.25, 14),
            self.target_rel, np.round(cur, 3), [float(self.paused)],
        ]).astype(np.float32)

    # ── one 50 Hz tick ──────────────────────────────────────────────────
    def tick(self):
        obs = np.concatenate([self.gyro(), self.grav(), self.d.qpos[self.qadr] - DEFAULT_POSE,
                              self.d.qvel[self.vadr], self.last_action, self.cmd]).astype(np.float32)
        if self.paused:
            self.t_paused += CTRL_DT
            tgt = paused_target_rel(torch.tensor([self.t_paused]), torch.tensor(self.tgt_at_pause)[None])[0].numpy()
        else:
            a = self.policy.run(None, {"obs": obs[None]})[0][0].astype(np.float32)
            self.last_action = a; tgt = a
        self.target_rel = tgt
        self.bam.q_target[:] = DEFAULT_POSE + tgt
        # hand
        pos = torch.tensor(self.d.xpos[self.trunk])[None]; quat = torch.tensor(self.d.xquat[self.trunk])[None]
        up = torch.tensor([self.grav()[2] < -0.5])
        hp, hq, active = self.hand.step(CTRL_DT, pos.float(), quat.float(), torch.tensor([self.feet_force()]),
                                        torch.tensor([self.mass]), up, *self.head_pose())
        self.d.mocap_pos[self.mocap] = hp[0].numpy(); self.d.mocap_quat[self.mocap] = hq[0].numpy()
        head = bool(self.hand.grip[0] == 1)
        self.d.eq_active[self.eq] = bool(active[0]) and not head
        self.d.eq_active[self.eq_head] = bool(active[0]) and head
        for _ in range(DECIM):
            self.bam.update(); mujoco.mj_step(self.m, self.d)
        self.t += CTRL_DT
        # detector
        f = self.features()
        if self.det is not None:
            self.win = np.repeat(f[None], WINDOW, 0) if self.win is None else np.concatenate([self.win[1:], f[None]])
            self.n_ticks += 1
            self.p = float(self.det.run(None, {"features": self.win[None]})[0][0])
            if self.n_ticks < WINDOW:  # no decisions until the window holds 1 s of real history
                return
            just_paused, just_resumed = self.sm.update(torch.tensor([self.p]))
            if bool(just_paused[0]):
                self.paused = True; self.t_paused = 0.0; self.tgt_at_pause = self.target_rel.copy()
            if bool(just_resumed[0]):
                self.paused = False; self.last_action[:] = 0.0  # runtime controller.reset()

    @property
    def held_gt(self):
        return bool(self.hand.active[0]) and self.feet_force() < 0.5 * self.mass * 9.81

    # ── hand helpers ────────────────────────────────────────────────────
    def pick(self, **kw):
        c = self.hand.cfg
        defaults = dict(hold_s=(1e4, 1e4), lift_z=(0.18, 0.25), lift_s=(0.6, 0.6), tilt_buckets=((30.0, 1.0),),
                        shake_prob=0.0, drop_prob=0.0, yaw_rate=(-0.5, 0.5), drift_amp=(0.03, 0.08), lower_speed=(0.15, 0.15),
                        touch_release_s=(0.3, 0.3), head_grip_prob=0.0, orient_prob=0.0, yaw_turn_prob=0.0)
        for k, v in {**defaults, **kw}.items():
            setattr(c, k, v)
        pos = torch.tensor(self.d.xpos[self.trunk])[None].float(); quat = torch.tensor(self.d.xquat[self.trunk])[None].float()
        self.hand.start(torch.tensor([True]), pos, quat, *self.head_pose())

    def head_pose(self):
        return torch.tensor(self.d.xpos[self.head])[None].float(), torch.tensor(self.d.xquat[self.head])[None].float()

    def put_down(self, drop=False):
        self.hand.drop[:] = drop
        self.hand.release(torch.tensor([True]))


SCENARIO = [  # (t, what, kwargs)
    (0.0, "cmd", dict(vx=0.25)),
    (4.0, "pick", dict(tilt_buckets=((35.0, 1.0),), shake_prob=1.0, shake_amp=(0.015, 0.015))),
    (9.0, "put", {}),
    (11.5, "cmd", dict(vx=0.0)),
    (14.0, "pick", dict(tilt_buckets=((10.0, 1.0),), drift_amp=(0.01, 0.02), yaw_rate=(0.0, 0.0), lift_z=(0.12, 0.12))),
    (18.0, "put", dict(lower_speed=(0.06, 0.06), touch_release_s=(0.6, 0.6))),
    (21.0, "cmd", dict(vx=0.15, wz=0.6)),
    (24.0, "pick", dict(tilt_buckets=((150.0, 1.0),), lift_z=(0.3, 0.3), tilt_freq_hz=(0.25, 0.35), drift_amp=(0.1, 0.15))),
    (30.0, "put", {}),
    (33.0, "cmd", dict(vx=0.25, wz=0.0)),
    (36.0, "end", {}),
]

# The two cases the first model missed on the robot: lifted by the head, and turned 180°.
SCENARIO_V2 = [
    (0.0, "cmd", dict(vx=0.2)),
    (3.0, "pick", dict(head_grip_prob=1.0, tilt_buckets=((20.0, 1.0),), lift_z=(0.15, 0.15))),
    (9.0, "put", {}),
    (12.0, "cmd", dict(vx=0.0)),
    (14.0, "pick", dict(orient_prob=1.0, orient_pitch=(3.1, 3.1), orient_roll=(0.0, 0.0), orient_ramp_s=(1.5, 1.5),
                        tilt_buckets=((5.0, 1.0),), lift_z=(0.25, 0.25), yaw_rate=(0.0, 0.0))),
    (21.0, "put", {}),
    (24.0, "pick", dict(yaw_turn_prob=1.0, yaw_turn=(3.14, 3.14), yaw_turn_s=(1.0, 1.0), tilt_buckets=((10.0, 1.0),),
                        yaw_rate=(0.0, 0.0))),
    (30.0, "put", {}),
    (32.0, "cmd", dict(vx=0.2)),
    (35.0, "end", {}),
]


def run_scenario(demo: Demo, video: str | None, width=720, height=540, scenario=SCENARIO):
    from PIL import Image, ImageDraw, ImageFont
    import imageio
    renderer = mujoco.Renderer(demo.m, height, width) if video else None
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING; cam.trackbodyid = demo.trunk
    cam.distance = 0.85; cam.elevation = -12
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20); small = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = small = ImageFont.load_default()
    frames, hist = [], []
    ev = list(scenario); log = []
    n_ticks = int(scenario[-1][0] / CTRL_DT)
    caption = ""
    for k in range(n_ticks):
        while ev and ev[0][0] <= demo.t + 1e-9:
            _, what, kw = ev.pop(0)
            if what == "cmd":
                demo.cmd[0] = kw.get("vx", 0.0); demo.cmd[2] = kw.get("wz", 0.0)
                caption = f"command vx={demo.cmd[0]:.2f} m/s wz={demo.cmd[2]:.1f} rad/s"
            elif what == "pick":
                demo.pick(**kw)
                caption = "hand picks the duck up" + (" BY THE HEAD" if kw.get("head_grip_prob") else "") + \
                    (" and turns it upside down" if kw.get("orient_prob") else "") + (" and spins it 180°" if kw.get("yaw_turn_prob") else "")
            elif what == "put":
                for kk, v in kw.items():
                    setattr(demo.hand.cfg, kk, v)
                demo.hand.lower_v[:] = demo.hand.cfg.lower_speed[0]; demo.hand.touch_release[:] = demo.hand.cfg.touch_release_s[0]
                demo.put_down(); caption = "hand sets the duck down"
        demo.tick()
        held = demo.held_gt
        hist.append((demo.p, demo.paused, held)); log.append((demo.t, demo.p, demo.paused, held, float(-demo.grav()[2])))
        if renderer is not None and k % 2 == 0:  # 25 fps
            cam.azimuth = 120 + 8 * demo.t
            renderer.update_scene(demo.d, cam); img = Image.fromarray(renderer.render())
            dr = ImageDraw.Draw(img)
            state = "PAUSED (picked up)" if demo.paused else "POLICY RUNNING"
            dr.rectangle([0, 0, width, 64], fill=(0, 0, 0))
            dr.text((10, 6), state, font=font, fill=(255, 150, 40) if demo.paused else (90, 220, 90))
            dr.text((260, 6), f"p(held) = {demo.p:.2f}", font=font, fill=(255, 255, 255))
            dr.text((470, 6), f"t = {demo.t:5.1f} s", font=font, fill=(200, 200, 200))
            dr.text((10, 38), f"truth: {'HELD' if held else 'on floor'}   |   {caption}", font=small, fill=(200, 200, 200))
            # p(held) strip: last 10 s
            y0, h = height - 70, 60; x0, w = 10, width - 20
            dr.rectangle([x0, y0, x0 + w, y0 + h], fill=(20, 20, 20))
            seg = hist[-500:]
            for i, (p, ps, hd) in enumerate(seg):
                x = x0 + int(i * w / 500)
                if hd:
                    dr.line([x, y0 + h - 6, x, y0 + h], fill=(80, 140, 255))
                if ps:
                    dr.line([x, y0, x, y0 + 5], fill=(255, 150, 40))
            pts = [(x0 + int(i * w / 500), y0 + h - 8 - int(p * (h - 16))) for i, (p, _, _) in enumerate(seg)]
            if len(pts) > 1:
                dr.line(pts, fill=(255, 255, 255), width=2)
            dr.text((x0 + 4, y0 - 18), "p(held) last 10 s   (blue = truth held, orange = paused)", font=small, fill=(220, 220, 220))
            frames.append(np.asarray(img))
    if video:
        imageio.mimsave(video, frames, fps=25, quality=7)
        print("wrote", video)
    return np.array(log)


def summarize(log):
    t, p, paused, held, upz = log.T
    print("   t     p  paused truth")
    for i in range(0, len(t), 25):
        print(f"{t[i]:5.1f} {p[i]:5.2f} {int(paused[i]):5d}  {'HELD' if held[i] else '-'}")
    paused = paused.astype(bool); held = held.astype(bool)
    print(f"ticks {len(t)}  held {held.mean():.2f}  paused {paused.mean():.2f}")
    # events
    def edges(x):
        x = x.astype(int); return np.nonzero(np.diff(x) == 1)[0] + 1, np.nonzero(np.diff(x) == -1)[0] + 1
    hu, hd = edges(held); pu, pd = edges(paused)
    for i in hu:
        nxt = pu[pu >= i]
        print(f"  pick-up at {t[i]:5.2f}s → paused {'after %.2fs' % (t[nxt[0]] - t[i]) if len(nxt) else 'NEVER'}")
    for i in hd:
        nxt = pd[pd >= i]
        print(f"  put-down at {t[i]:5.2f}s → resumed {'after %.2fs' % (t[nxt[0]] - t[i]) if len(nxt) else 'NEVER'}")
    fp = [t[i] for i in pu if not held[max(0, i - 25):i + 1].any()]
    print(f"  pauses with no hand: {len(fp)} {['%.1f' % x for x in fp]}")
    print(f"  min uprightness (cos tilt) while not held & policy running: {upz[~held & ~paused].min():.2f}")


def interactive(demo: Demo):
    import mujoco.viewer
    print(__doc__)
    with mujoco.viewer.launch_passive(demo.m, demo.d) as v, TerminalInput() as keys:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING; v.cam.trackbodyid = demo.trunk; v.cam.distance = 0.9
        while v.is_running():
            t0 = time.time()
            for k in keys.get_keys():
                if k == "p": demo.pick(tilt_buckets=((45.0, 1.0),), shake_prob=0.5)
                elif k == "l": demo.put_down()
                elif k == "d": demo.put_down(drop=True)
                elif k == "up": demo.cmd[0] = min(0.4, demo.cmd[0] + 0.1)
                elif k == "down": demo.cmd[0] = max(-0.4, demo.cmd[0] - 0.1)
                elif k == "left": demo.cmd[2] = min(1.0, demo.cmd[2] + 0.3)
                elif k == "right": demo.cmd[2] = max(-1.0, demo.cmd[2] - 0.3)
                elif k == " ": demo.cmd[:3] = 0
                elif k == "q": return
            with v.lock():
                demo.tick()
                v.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                             "PAUSED (picked up)" if demo.paused else "POLICY RUNNING",
                             f"p(held)={demo.p:.2f}  truth={'held' if demo.held_gt else 'floor'}  cmd vx={demo.cmd[0]:.1f} wz={demo.cmd[2]:.1f}"))
            v.sync()
            time.sleep(max(0.0, CTRL_DT - (time.time() - t0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="logs/pickup/nocur/pickup_detector.onnx")
    ap.add_argument("--no-detector", action="store_true")
    ap.add_argument("--policy", default=POLICY)
    ap.add_argument("--scene", default=SCENE)
    ap.add_argument("--video", default=None)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenario", choices=["v1", "v2"], default="v1", help="v2 = head grip, upside down, spun 180°")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    demo = Demo(args)
    if args.interactive:
        interactive(demo)
    else:
        summarize(run_scenario(demo, args.video, scenario=SCENARIO_V2 if args.scenario == "v2" else SCENARIO))


if __name__ == "__main__":
    main()
