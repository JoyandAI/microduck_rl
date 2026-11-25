#!/usr/bin/env python3
"""Per-phase yaw-impulse decomposition from eval_straightness --csv output.

Pure post-processing: reads the (env,t,x,y,z,yaw,yaw_abs,yaw_rate,vx_b,vy_b,tilt,
wz,phase,+14 joints) time series and integrates the BODY-FRAME yaw rate `wz`
(rad/s, un-differenced, straight from the sim) separately over
  * left-single-support   (phase = +1)
  * right-single-support  (phase = -1)
  * double support / swing(phase =  0)
Units: DEGREES of accumulated yaw per window (i.e. the yaw impulse of that phase,
in deg = integral of wz dt).

Definitions are taken verbatim from scripts/eval_straightness.py so the numbers
are comparable with the JSON it writes.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict

import numpy as np

DT = 0.02  # CTRL_DT in eval_straightness.py


def load(path):
    rows = defaultdict(list)
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            e = int(row["env"])
            rows[e].append(row)
    return rows


def analyse(path):
    rows = load(path)
    out = []
    for e, rs in sorted(rows.items()):
        wz = np.array([float(x["wz"]) for x in rs])           # rad/s, body frame
        ph = np.array([float(x["phase"]) for x in rs])
        z = np.array([float(x["z"]) for x in rs])
        tilt = np.array([float(x["tilt"]) for x in rs])
        yaw = np.array([float(x["yaw"]) for x in rs])
        deg = wz * 180.0 / np.pi
        L = ph > 0
        R = ph < 0
        D = ph == 0
        J_L = float(np.sum(deg[L]) * DT)   # deg of yaw accumulated in L-single
        J_R = float(np.sum(deg[R]) * DT)
        J_D = float(np.sum(deg[D]) * DT)
        # event count = number of entries into a single-support episode
        ev = np.count_nonzero(L[1:] & ~L[:-1]) + np.count_nonzero(R[1:] & ~R[:-1])
        # stride cycle = one L episode + one R episode
        cyc = 0.5 * ev
        T = len(rs) * DT
        out.append(dict(
            env=e, T=T,
            J_L=J_L, J_R=J_R, J_D=J_D, J_net=J_L + J_R + J_D,
            mismatch=abs(J_L) - abs(J_R),
            mwz_L=float(np.mean(deg[L])) if L.any() else float("nan"),
            mwz_R=float(np.mean(deg[R])) if R.any() else float("nan"),
            mwz_D=float(np.mean(deg[D])) if D.any() else float("nan"),
            frac_L=float(np.mean(L)), frac_R=float(np.mean(R)), frac_D=float(np.mean(D)),
            n_events=ev, stride_hz=cyc / T, step_hz=ev / T,
            z_mean=float(np.mean(z)), tilt_mean=float(np.mean(tilt)),
            yaw_net=float(yaw[-1] - yaw[0]),
            speed_along=(float(rs[-1]["x"]) - float(rs[0]["x"])),
        ))
    return out


def rep(name, rows):
    agg = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0] if k != "env"}
    med = {k: float(np.median(v)) for k, v in agg.items()}
    mean = {k: float(np.mean(v)) for k, v in agg.items()}
    print(f"\n===== {name}   (N={len(rows)}) =====")
    print(f"  window T                       : {med['T']:.2f} s (median)")
    print(f"  --- yaw IMPULSE per phase (deg accumulated over T) ---")
    print(f"  J_left-single   mean {mean['J_L']:+8.3f}   median {med['J_L']:+8.3f}")
    print(f"  J_right-single  mean {mean['J_R']:+8.3f}   median {med['J_R']:+8.3f}")
    print(f"  J_double-swing  mean {mean['J_D']:+8.3f}   median {med['J_D']:+8.3f}")
    print(f"  J_net = sum     mean {mean['J_net']:+8.3f}   median {med['J_net']:+8.3f}"
          f"   (net yaw median {med['yaw_net']:+.2f} deg)")
    print(f"  |J_L|-|J_R|     mean {mean['mismatch']:+8.3f}   median {med['mismatch']:+8.3f}"
          f"   -> |net|/|mismatch| = "
          f"{abs(mean['J_net'])/max(abs(mean['mismatch']),1e-9):.3f}")
    print(f"  --- mean wz INSIDE each phase (deg/s) ---")
    print(f"  mean wz | L-single {mean['mwz_L']:+8.3f} | R-single {mean['mwz_R']:+8.3f}"
          f" | double {mean['mwz_D']:+8.3f}")
    print(f"  --- phase time budget ---")
    print(f"  frac L-single {mean['frac_L']:.3f} | R-single {mean['frac_R']:.3f}"
          f" | double/swing {mean['frac_D']:.3f}  (coverage {mean['frac_L']+mean['frac_R']:.3f})")
    print(f"  --- gait timing ---")
    print(f"  step events  mean {mean['n_events']:.1f} over {med['T']:.2f} s"
          f" -> step {mean['step_hz']:.3f} Hz, stride {mean['stride_hz']:.3f} Hz")
    print(f"  --- posture / travel ---")
    print(f"  trunk z mean {1000*mean['z_mean']:.1f} mm | tilt mean {mean['tilt_mean']:.2f} deg"
          f" | net forward travel x {mean['speed_along']:.3f} m")
    return mean, med


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        name, _, path = arg.partition("=")
        rep(name, analyse(path))
