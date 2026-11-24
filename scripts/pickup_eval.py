#!/usr/bin/env python3
"""Closed-loop metrics from a `pickup_datagen.py --detector ...` rollout.

  uv run scripts/pickup_eval.py data/pickup/eval_nocur.npz [more.npz ...]

Event-level, per env, on ticks at least 1 s after a reset:
  false pause      pause onset with no hand-held tick in the preceding 0.5 s
                   (reported per hour of policy-running-on-floor time, by context)
  pick-up          held episode ≥ 0.5 s: latency held-onset → pause; missed if never paused while held
  put-down         held → not held, staying on the floor ≥ 1.5 s: latency → resume
                   (stuck if not resumed within 1.5 s); tipped = tilt > 40° before resuming
  false resume     resume onset mid-carry (hand carrying, feet off the floor for the previous 0.5 s);
                   resumes during a set-down — feet already loaded, the compliant hand easing
                   off — are label flicker, counted separately
Put-downs are split into set-downs (feet loaded) and drops (released at height, which
land hard — tipping there is the drop, not the detector).
"""

import sys

import numpy as np

DT = 0.02


def onsets(x):
    return np.nonzero(x[1:] & ~x[:-1])[0] + 1


def analyze(path):
    d = np.load(path)
    held, paused, sr = d["held"], d["paused"], d["since_reset"]
    grav = d["feat"][..., 3:6].astype(np.float32)
    tilt = np.degrees(np.arccos(np.clip(-grav[..., 2], -1, 1)))
    cmd = d["cmd"].astype(np.float32); fallen = d["fallen"]
    phase, ff = d["phase"], d["feet_frac"].astype(np.float32)
    T, N = held.shape
    r = dict(fp=[], fp_ctx={"walking": 0, "standing": 0, "fallen": 0}, floor_time={"walking": 0.0, "standing": 0.0, "fallen": 0.0},
             pick_lat=[], pick_miss=0, put_lat=[], put_stuck=0, put_tipped=0, false_resume=0, setdown_flicker=0, held_time=0.0,
             drops=0, drop_tipped=0)
    for n in range(N):
        h, p, s = held[:, n], paused[:, n], sr[:, n]
        ok = s >= 50
        run_floor = ok & ~h & ~p
        for ctx, m in (("walking", (cmd[:, n] > 0.05) & ~fallen[:, n]), ("standing", (cmd[:, n] <= 0.05) & ~fallen[:, n]), ("fallen", fallen[:, n])):
            r["floor_time"][ctx] += (run_floor & m).sum() * DT
        r["held_time"] += (ok & h).sum() * DT
        for t in onsets(p):
            if not ok[t]:
                continue
            if not h[max(0, t - 25):t + 1].any():
                ctx = "fallen" if fallen[t, n] else ("walking" if cmd[t, n] > 0.05 else "standing")
                r["fp_ctx"][ctx] += 1
        for t in onsets(~p):  # resume onsets
            if ok[t] and h[t]:
                if phase[t, n] == 1 and ff[max(0, t - 25):t + 1, n].max() < 0.25:
                    r["false_resume"] += 1
                else:
                    r["setdown_flicker"] += 1
        # held episodes
        on, off = onsets(h), onsets(~h)
        for t in on:
            if not ok[t] or p[t - 1]:
                continue
            e = off[off > t]; e = e[0] if len(e) else T
            if (e - t) * DT < 0.5:
                continue
            pp = np.nonzero(p[t:e])[0]
            if len(pp):
                r["pick_lat"].append(pp[0] * DT)
            else:
                r["pick_miss"] += 1
        for t in off:
            if not ok[t] or not p[t - 1]:
                continue  # only put-downs of a paused robot
            if t + 75 > T or h[t:t + 75].any() or (s[t:t + 75] < s[t]).any():
                continue
            rr = np.nonzero(~p[t:t + 75])[0]
            if ff[t, n] < 0.25:  # released at height: a drop
                r["drops"] += 1
                r["drop_tipped"] += int((tilt[t:t + (rr[0] + 1 if len(rr) else 75), n] > 40).any())
                continue
            if len(rr):
                r["put_lat"].append(rr[0] * DT)
                if (tilt[t:t + rr[0] + 1, n] > 40).any():
                    r["put_tipped"] += 1
            else:
                r["put_stuck"] += 1
                if (tilt[t:t + 75, n] > 40).any():
                    r["put_tipped"] += 1
    return r


def merge(rs):
    out = rs[0]
    for r in rs[1:]:
        for k, v in r.items():
            if isinstance(v, list):
                out[k] += v
            elif isinstance(v, dict):
                for kk in v:
                    out[k][kk] += v[kk]
            else:
                out[k] += v
    return out


def report(r):
    print("FALSE PAUSES (policy running, nobody touching the duck):")
    for ctx in ("walking", "standing", "fallen"):
        hrs = r["floor_time"][ctx] / 3600
        print(f"  {ctx:9s} {r['fp_ctx'][ctx]:5d} in {hrs:6.2f} h  →  {r['fp_ctx'][ctx] / max(hrs, 1e-9):7.2f} / hour")
    pl = np.array(r["pick_lat"]); n_pick = len(pl) + r["pick_miss"]
    print(f"PICK-UPS: {n_pick}  detected {100 * len(pl) / max(1, n_pick):.1f}%  latency p50/p90/p99 "
          f"{np.percentile(pl, 50):.2f}/{np.percentile(pl, 90):.2f}/{np.percentile(pl, 99):.2f} s")
    ql = np.array(r["put_lat"]); n_put = len(ql) + r["put_stuck"]
    print(f"SET-DOWNS: {n_put}  resumed within 1.5 s {100 * len(ql) / max(1, n_put):.1f}%  latency p50/p90/p99 "
          f"{np.percentile(ql, 50):.2f}/{np.percentile(ql, 90):.2f}/{np.percentile(ql, 99):.2f} s  "
          f"tipped >40° before resuming {100 * r['put_tipped'] / max(1, n_put):.1f}%")
    print(f"DROPS: {r['drops']}  tipped >40° before resuming {100 * r['drop_tipped'] / max(1, r['drops']):.1f}% (the fall itself)")
    print(f"set-down label flicker (resumed with feet already loaded): {r['setdown_flicker']}")
    print(f"FALSE RESUMES mid-carry: {r['false_resume']} in {r['held_time'] / 3600:.2f} h held "
          f"→ {r['false_resume'] / max(r['held_time'] / 3600, 1e-9):.1f} / hour")


if __name__ == "__main__":
    report(merge([analyze(p) for p in sys.argv[1:]]))
