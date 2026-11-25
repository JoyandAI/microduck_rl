"""End to end: the real `robotd --sim` with `[pickup] enabled`, against a duck-body world that has
the training's welded hand in it.

    uv run python scripts/pickup_e2e_robotd.py [--video logs/pickup/e2e_robotd.mp4] [--scenario v2]

Needs the microduck checkout at ~/Pollen/microduck with `target/debug/robotd` built (the branch
carrying `[pickup]`). Walks the duck with velstand, picks it up mid-walk (shaken), sets it down,
stands, lifts it gently, sets it down, walks and turns, carries it upside down, sets it down — and
prints robotd's pause/resume edges against the hand's ground truth, plus robotd's own log lines.
Exits after ~40 s of scenario; the daemon's state directory is a /tmp/pk* temp dir."""
import json, math, os, socket, subprocess, sys, tempfile, threading, time
from pathlib import Path

import mujoco, numpy as np, torch

from mjlab_microduck.sim import body_server as bs
from mjlab_microduck.pickup.hand import VirtualHand, HandCfg, add_hand_to_spec

MICRODUCK = Path.home() / "Pollen/microduck"
RL = Path.home() / "Pollen/microduck_rl"
PORT = 7811
VIDEO = sys.argv[sys.argv.index("--video") + 1] if "--video" in sys.argv else None


class HandWorld(bs.World):
    def __init__(self, scene):
        spec = mujoco.MjSpec.from_file(str(scene))
        add_hand_to_spec(spec, trunk_body="trunk_base")
        self.model = spec.compile()
        self.model.opt.timestep = bs.TIMESTEP
        self.model.vis.global_.offwidth, self.model.vis.global_.offheight = 1280, 960
        self.data = mujoco.MjData(self.model)
        self.lock = threading.Lock()
        self.bodies = []


world = HandWorld(bs.DEFAULT_SCENE)
pose, trunk_z = None, bs.HOME_TRUNK_Z  # the home pose, as infer_policy.py places a duck
body = bs.Body(world, 0)
body.place(pose, trunk_z, 0.0)
world.bodies.append(body)
mujoco.mj_forward(world.model, world.data)
server = bs.Server(("127.0.0.1", PORT), bs.Handler); server.body = body
threading.Thread(target=server.serve_forever, daemon=True).start()

m, d = world.model, world.data
eq = m.equality("pickup_hand_weld").id
eq_head = m.equality("pickup_hand_weld_head").id
head = m.body("jaw_soft").id
mocap = m.body_mocapid[m.body("pickup_hand").id]
trunk = m.body("trunk_base").id
feet = {m.geom("left_foot_collision").id, m.geom("right_foot_collision").id}
mass = float(m.body_subtreemass[trunk])
hand = VirtualHand(1, "cpu", HandCfg(pickup_rate_hz=0.0))

def feet_force():
    f, c6 = 0.0, np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        if (c.geom1 in feet and m.geom_bodyid[c.geom2] == 0) or (c.geom2 in feet and m.geom_bodyid[c.geom1] == 0):
            mujoco.mj_contactForce(m, d, i, c6); f += abs(c6[0])
    return f

# ── robotd ──
ort = next((RL / ".venv/lib").glob("python*/site-packages/onnxruntime/capi/libonnxruntime.so*"))
state = Path(tempfile.mkdtemp(prefix="pk"))
params = state / "robotd.toml"
params.write_text(f"""[policy]
enabled = true
walk = "{RL}/logs/bench_onnx/velstand_fhathosb_3750.onnx"
stand = "none"
sitstand = "none"
ground_pick = "none"
kick_left = "none"
kick_right = "none"
roulade = "none"

[audio]
enabled = false

[pickup]
enabled = true
model = "{MICRODUCK}/duck-control/models/pickup_detector.onnx"
""")
sock = state / "duck.sock"
log = open(state / "robotd.log", "w")
env = dict(os.environ, ORT_DYLIB_PATH=str(ort), RUST_LOG="info", DUCK_RUNTIME_DIR=str(state))
robotd = subprocess.Popen([str(MICRODUCK / "target/debug/robotd"), "--sim", f"127.0.0.1:{PORT}",
                           "--params", str(params), "--socket", str(sock)], stdout=log, stderr=subprocess.STDOUT, env=env)

def rpc(method, params=None):
    s = socket.socket(socket.AF_UNIX); s.settimeout(5); s.connect(str(sock)); f = s.makefile("rw")
    f.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}) + "\n"); f.flush()
    return json.loads(f.readline())

# ── scenario ──
cmd = [0.0, 0.0]
EVENTS = [  # seconds after the policy has the robot
    (0.0, "cmd", (0.2, 0.0)), (4.0, "pick", dict(tilt_buckets=((35.0, 1.0),), shake_prob=1.0, shake_amp=(0.015, 0.015))),
    (9.0, "put", {}), (12.0, "cmd", (0.0, 0.0)),
    (15.0, "pick", dict(tilt_buckets=((10.0, 1.0),), drift_amp=(0.01, 0.02), yaw_rate=(0.0, 0.0), lift_z=(0.12, 0.12))),
    (19.0, "put", dict(lower_speed=(0.06, 0.06), touch_release_s=(0.6, 0.6))),
    (22.0, "cmd", (0.15, 0.5)), (25.0, "pick", dict(tilt_buckets=((120.0, 1.0),), lift_z=(0.3, 0.3), tilt_freq_hz=(0.25, 0.35))),
    (30.0, "put", {}), (33.0, "cmd", (0.2, 0.0)), (38.0, "end", None),
]
if "--scenario" in sys.argv and sys.argv[sys.argv.index("--scenario") + 1] == "v2":
    # the two cases the first model missed on the robot: lifted by the head, turned 180°
    EVENTS = [
        (0.0, "cmd", (0.2, 0.0)), (3.0, "pick", dict(head_grip_prob=1.0, tilt_buckets=((20.0, 1.0),), lift_z=(0.15, 0.15), lift_s=(1.5, 1.5))),
        (9.0, "put", {}), (12.0, "cmd", (0.0, 0.0)),
        (14.0, "pick", dict(orient_prob=1.0, orient_pitch=(3.1, 3.1), orient_roll=(0.0, 0.0), orient_ramp_s=(1.5, 1.5),
                            tilt_buckets=((5.0, 1.0),), lift_z=(0.25, 0.25), yaw_rate=(0.0, 0.0))),
        (21.0, "put", {}),
        (24.0, "pick", dict(yaw_turn_prob=1.0, yaw_turn=(3.14, 3.14), yaw_turn_s=(1.0, 1.0), tilt_buckets=((10.0, 1.0),), yaw_rate=(0.0, 0.0))),
        (30.0, "put", {}), (32.0, "cmd", (0.2, 0.0)), (37.0, "end", None),
    ]
DEFAULTS = dict(head_grip_prob=0.0, orient_prob=0.0, yaw_turn_prob=0.0, hold_s=(1e4, 1e4), lift_z=(0.18, 0.25), lift_s=(0.6, 0.6), shake_prob=0.0, drop_prob=0.0,
                yaw_rate=(-0.5, 0.5), drift_amp=(0.03, 0.08), lower_speed=(0.15, 0.15), touch_release_s=(0.3, 0.3),
                tilt_freq_hz=(0.05, 0.6))

def mover():
    while robotd.poll() is None:
        try:
            s = socket.socket(socket.AF_UNIX); s.connect(str(sock)); f = s.makefile("w")
            while robotd.poll() is None:
                f.write(json.dumps({"jsonrpc": "2.0", "method": "robot.move", "params": {"vx": cmd[0], "vy": 0.0, "vyaw": cmd[1]}}) + "\n"); f.flush()
                time.sleep(0.1)
        except OSError:
            time.sleep(0.2)

labels = []  # (wall time, policy label)
def subscriber():
    while robotd.poll() is None:
        try:
            s = socket.socket(socket.AF_UNIX); s.connect(str(sock)); f = s.makefile("rw")
            f.write(json.dumps({"jsonrpc": "2.0", "id": 7, "method": "robot.subscribe", "params": {"hz": 50}}) + "\n"); f.flush()
            for line in f:
                msg = json.loads(line)
                p = msg.get("params") or {}
                if "policy" in p:
                    labels.append((time.time(), p["policy"]))
        except OSError:
            time.sleep(0.2)

# physics in real time, hand updated every control tick
truth = []  # (wall time, held, tilt_deg, trunk_z)
frames = []
renderer = mujoco.Renderer(m, 480, 640) if VIDEO else None
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING; cam.trackbodyid = trunk; cam.distance = 0.85; cam.elevation = -12
t_start = None; ev = list(EVENTS); caption = ""
for _ in range(100):
    if sock.exists(): break
    time.sleep(0.1)
threading.Thread(target=mover, daemon=True).start()
threading.Thread(target=subscriber, daemon=True).start()
print("enable:", rpc("robot.enable", {"on": True}).get("result"))
next_t = time.perf_counter(); k = 0
while True:
    world.step(4)
    k += 1
    now = time.time()
    if t_start is None and labels and labels[-1][1] in ("walk", "stand"):
        t_start = now; print("policy has the robot")
    if t_start is not None:
        while ev and ev[0][0] <= now - t_start:
            _, what, arg = ev.pop(0)
            if what == "end": break
            if what == "cmd": cmd[:] = arg; caption = f"cmd vx={arg[0]} wz={arg[1]}"
            elif what == "pick":
                for kk, v in {**DEFAULTS, **arg}.items(): setattr(hand.cfg, kk, v)
                with world.lock:
                    hand.start(torch.tensor([True]), torch.tensor(d.xpos[trunk])[None].float(), torch.tensor(d.xquat[trunk])[None].float(),
                               torch.tensor(d.xpos[head])[None].float(), torch.tensor(d.xquat[head])[None].float())
                caption = "hand picks it up"
            elif what == "put":
                for kk, v in arg.items(): setattr(hand.cfg, kk, v)
                hand.lower_v[:] = hand.cfg.lower_speed[0]; hand.touch_release[:] = hand.cfg.touch_release_s[0]
                hand.release(torch.tensor([True])); caption = "hand sets it down"
            print(f"{now - t_start:6.2f}s {what} {arg if what == 'cmd' else ''}", flush=True)
        if not ev or ev[0][1] == "end" and ev[0][0] <= now - t_start:
            break
    with world.lock:
        ff = feet_force()
        up = torch.tensor([d.xmat[trunk][8] > 0.5])
        hp, hq, act = hand.step(0.02, torch.tensor(d.xpos[trunk])[None].float(), torch.tensor(d.xquat[trunk])[None].float(),
                                torch.tensor([ff]), torch.tensor([mass]), up,
                                torch.tensor(d.xpos[head])[None].float(), torch.tensor(d.xquat[head])[None].float())
        by_head = bool(hand.grip[0] == 1)
        d.mocap_pos[mocap] = hp[0].numpy(); d.mocap_quat[mocap] = hq[0].numpy()
        d.eq_active[eq] = bool(act[0]) and not by_head; d.eq_active[eq_head] = bool(act[0]) and by_head
        tilt = math.degrees(math.acos(max(-1, min(1, d.xmat[trunk][8]))))
        truth.append((now, bool(act[0]) and ff < 0.5 * mass * 9.81, tilt, float(d.xpos[trunk][2])))
        if renderer is not None and k % 2 == 0 and t_start is not None:
            cam.azimuth = 120 + 8 * (now - t_start)
            renderer.update_scene(d, cam); frames.append((now, renderer.render().copy(), caption))
    next_t += 0.02
    sl = next_t - time.perf_counter()
    if sl > 0: time.sleep(sl)

robotd.terminate(); robotd.wait(5)
# ── report ──
T = np.array([t for t, *_ in truth]); held = np.array([h for _, h, *_ in truth]); tilt = np.array([x for *_, x, _ in truth])
L = labels
lab_at = lambda t: next((l for tt, l in reversed(L) if tt <= t), "?")
paused = np.array([lab_at(t) == "picked_up" for t in T])
def edges(x): return np.nonzero(np.diff(x.astype(int)) == 1)[0] + 1, np.nonzero(np.diff(x.astype(int)) == -1)[0] + 1
hu, hd = edges(held); pu, pd = edges(paused)
for i in hu:
    j = pu[pu >= i - 2]; print(f"  pick-up at {T[i]-t_start:5.2f}s → paused {'after %.2fs' % (T[j[0]] - T[i]) if len(j) else 'NEVER'}")
for i in hd:
    j = pd[pd >= i - 2]; print(f"  put-down at {T[i]-t_start:5.2f}s → resumed {'after %.2fs' % (T[j[0]] - T[i]) if len(j) else 'NEVER'}")
fp = [T[i] - t_start for i in pu if not held[max(0, i - 25):i + 1].any()]
print("  pauses with no hand:", ["%.1f" % x for x in fp])
print("  max tilt while not held & not paused:", tilt[~held & ~paused & (T > t_start + 1)].max().round(1))
print("  label counts:", {l: sum(1 for _, x in L if x == l) for l in set(x for _, x in L)})
os.system(f"grep -E 'pickup|picked up|put down' {state}/robotd.log | sed 's/^/    robotd: /'")
if VIDEO:
    import imageio
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18); small = ImageFont.truetype("DejaVuSans.ttf", 14)
    out = []
    for t, img, cap in frames:
        im = Image.fromarray(img); dr = ImageDraw.Draw(im); lab = lab_at(t)
        dr.rectangle([0, 0, 640, 52], fill=(0, 0, 0))
        dr.text((8, 4), f"robotd: {lab}", font=font, fill=(255, 150, 40) if lab == "picked_up" else (90, 220, 90))
        dr.text((330, 4), f"t = {t - t_start:5.1f} s", font=font, fill=(220, 220, 220))
        i = min(len(T) - 1, np.searchsorted(T, t))
        dr.text((8, 30), f"truth: {'HELD' if held[i] else 'on floor'}  |  {cap}", font=small, fill=(200, 200, 200))
        out.append(np.asarray(im))
    imageio.mimsave(VIDEO, out, fps=25, quality=7); print("wrote", VIDEO)
