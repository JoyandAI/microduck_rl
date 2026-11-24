#!/usr/bin/env python3
"""Does the per-half-step yaw impulse equal the stance hip_yaw sweep?

Physics: in single support one hip_yaw joint is the only yaw DOF between the
planted foot and the trunk, so the trunk yaw produced over that phase must equal
(minus) the change of the stance hip_yaw angle over the same phase. If that holds
numerically, then

    yaw per gait cycle = -(d hip_yaw_L over L-support + d hip_yaw_R over R-support)

i.e. the net yaw is the algebraic SUM of the two stance hip_yaw sweeps and a
"rectified residual" is literally "the two sweeps do not cancel".

Read-only, logs/axis/csv_<policy>_0.2.csv (P64).
"""
import csv
import numpy as np

DT = 0.02
R2D = 180.0 / np.pi
POLICIES = ["official", "ck7000", "ck12000", "R3hrl1", "R4db03"]


def load(path):
    with open(path) as f:
        r = csv.reader(f)
        hdr = next(r)
        a = np.array([[float(v) for v in row] for row in r])
    return {c: i for i, c in enumerate(hdr)}, a


def phases(ph):
    """Yield (kind, start, end) for contiguous L / R / D runs, dropping the
    first and last partial runs."""
    runs = []
    s = 0
    for i in range(1, len(ph)):
        if ph[i] != ph[i - 1]:
            runs.append((ph[i - 1], s, i))
            s = i
    runs.append((ph[s], s, len(ph)))
    return runs[1:-1]


print(f"{'policy':<10}{'n_L':>5}{'n_R':>5}"
      f"{'int_wz_L':>10}{'d_hyaw_L':>11}{'ratio':>8}"
      f"{'int_wz_R':>10}{'d_hyaw_R':>11}{'ratio':>8}"
      f"{'corr_L':>8}{'corr_R':>8}{'cyc_net':>9}{'sum_sweep':>10}")
for pol in POLICIES:
    try:
        ix, a = load(f"logs/axis/csv_{pol}_0.2.csv")
    except FileNotFoundError:
        continue
    env = a[:, ix["env"]].astype(int)
    IWL, DHL, IWR, DHR, NET, SW = [], [], [], [], [], []
    for e in range(int(env.max()) + 1):
        m = env == e
        if m.sum() < 200:
            continue
        wz = a[m, ix["wz"]]
        ph = a[m, ix["phase"]]
        ql = a[m, ix["left_hip_yaw"]]
        qr = a[m, ix["right_hip_yaw"]]
        # per-cycle pairing
        onsets = np.where((ph > 0.5) & (np.r_[0.0, ph[:-1]] <= 0.5))[0]
        for k in range(len(onsets) - 1):
            s, en = onsets[k], onsets[k + 1]
            p = ph[s:en]
            selL = p > 0.5
            selR = p < -0.5
            if selL.sum() < 3 or selR.sum() < 3:
                continue
            iwL = float(wz[s:en][selL].sum()) * DT * R2D
            iwR = float(wz[s:en][selR].sum()) * DT * R2D
            # stance hip_yaw sweep = last minus first sample of that phase
            qLs = ql[s:en][selL]
            qRs = qr[s:en][selR]
            dL = float(qLs[-1] - qLs[0]) * R2D
            dR = float(qRs[-1] - qRs[0]) * R2D
            IWL.append(iwL); IWR.append(iwR)
            DHL.append(dL); DHR.append(dR)
            NET.append(iwL + iwR)
            SW.append(dL + dR)
    IWL = np.array(IWL); IWR = np.array(IWR); DHL = np.array(DHL); DHR = np.array(DHR)
    NET = np.array(NET); SW = np.array(SW)
    print(f"{pol:<10}{len(IWL):>5}{len(IWR):>5}"
          f"{IWL.mean():>10.2f}{DHL.mean():>11.2f}"
          f"{(-DHL.mean()/IWL.mean() if abs(IWL.mean())>1e-9 else float('nan')):>8.2f}"
          f"{IWR.mean():>10.2f}{DHR.mean():>11.2f}"
          f"{(-DHR.mean()/IWR.mean() if abs(IWR.mean())>1e-9 else float('nan')):>8.2f}"
          f"{np.corrcoef(IWL,-DHL)[0,1]:>8.3f}{np.corrcoef(IWR,-DHR)[0,1]:>8.3f}"
          f"{NET.mean():>9.3f}{SW.mean():>10.2f}")
    # cycle-to-cycle residual and its relation to the sweep sum
    print(f"{'':<10}per-cycle net yaw impulse {NET.mean():+.3f} +- {NET.std():.3f} deg "
          f"| (dL+dR) {SW.mean():+.3f} +- {SW.std():.3f} deg "
          f"| corr(cycle net, sweep sum) = {np.corrcoef(NET,SW)[0,1]:+.3f}")
    # how well does -dL-dR reconstruct the whole-cycle yaw?
    print(f"{'':<10}|sweep set| mean|dL| {np.abs(DHL).mean():.2f} mean|dR| {np.abs(DHR).mean():.2f} "
          f"deg ; mean|int_wz_L| {np.abs(IWL).mean():.2f} mean|int_wz_R| {np.abs(IWR).mean():.2f} deg")
