#!/usr/bin/env python3
"""Which joint's phase-locked VELOCITY drives the trunk yaw rate, per phase?

Read-only, uses logs/axis/csv_<policy>_0.2.csv (P64).
"""
import csv, math
import numpy as np

DT = 0.02
R2D = 180.0 / math.pi
POLICIES = ["official", "ck7000", "ck12000"]
JOINTS = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee",
          "left_ankle", "right_hip_yaw", "right_hip_roll", "right_hip_pitch",
          "right_knee", "right_ankle"]


def load(path):
    with open(path) as f:
        r = csv.reader(f)
        hdr = next(r)
        a = np.array([[float(v) for v in row] for row in r])
    return {c: i for i, c in enumerate(hdr)}, a


def main():
    for pol in POLICIES:
        ix, a = load(f"logs/axis/csv_{pol}_0.2.csv")
        env = a[:, ix["env"]].astype(int)
        rows = {}
        # per-env regression of wz on joint velocities, split by phase
        store = {p: {j: [] for j in JOINTS} for p in ("L", "R")}
        dcdt = {p: [[] for _ in JOINTS] for p in ("L", "R")}
        for e in range(int(env.max()) + 1):
            m = env == e
            if m.sum() < 50:
                continue
            wz = a[m, ix["wz"]]
            ph = a[m, ix["phase"]]
            qd = {j: np.gradient(a[m, ix[j]], DT) for j in JOINTS}
            for pname, sel in (("L", ph > 0.5), ("R", ph < -0.5)):
                if sel.sum() < 5:
                    continue
                for j in JOINTS:
                    store[pname][j].append(float(np.corrcoef(qd[j][sel], wz[sel])[0, 1]))
                    dcdt[pname][JOINTS.index(j)].append(
                        float((qd[j][sel] * 1.0).mean()) * R2D)
        print("=" * 108)
        print(f"{pol}: per-env corr( d<joint>/dt , wz ) inside each single-support phase")
        print(f"   (positive corr => that joint's motion in the SAME direction as trunk yaw)")
        print(f"{'joint':<18}{'corr@L-support':>16}{'corr@R-support':>16}"
              f"{'signed dq/dt L':>16}{'signed dq/dt R':>16}")
        for j in JOINTS:
            cl = np.nanmean(store["L"][j]); cr = np.nanmean(store["R"][j])
            dl = np.nanmean(dcdt["L"][JOINTS.index(j)])
            dr = np.nanmean(dcdt["R"][JOINTS.index(j)])
            print(f"{j:<18}{cl:>16.3f}{cr:>16.3f}{dl:>16.2f}{dr:>16.2f}")
        # same, for the hip_yaw straight-line motion in absolute terms
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
