#!/usr/bin/env python3
"""Final phase-attribution table (P64 CSVs), with a walking-env filter.

Read-only. logs/axis/csv_<policy>_{0.2,0.0}.csv
"""
import csv, math
import numpy as np

DT = 0.02
R2D = 180.0 / math.pi
POLICIES = ["official", "ck7000", "ck12000", "R3hrl1", "R4db03"]


def load(path):
    with open(path) as f:
        r = csv.reader(f)
        hdr = next(r)
        a = np.array([[float(v) for v in row] for row in r])
    return {c: i for i, c in enumerate(hdr)}, a


def run_one(wz, ph, vyb, ql, qr, qrl, qrr, qal, qar):
    o = {}
    for nm, sel in (("L", ph > 0.5), ("R", ph < -0.5), ("D", np.abs(ph) < 0.5)):
        o[f"duty_{nm}"] = float(sel.mean())
        o[f"wz_{nm}"] = float(wz[sel].mean()) * R2D if sel.any() else 0.0
        o[f"con_{nm}"] = o[f"wz_{nm}"] * o[f"duty_{nm}"]
        o[f"vy_{nm}"] = float(vyb[sel].mean()) * 1000 if sel.any() else 0.0
    o["dc"] = float(wz.mean()) * R2D
    o["ac"] = float((wz * ph).mean()) * R2D
    o["net"] = o["con_L"] + o["con_R"] + o["con_D"]
    o["mismatch"] = (-o["con_L"]) - o["con_R"]
    # cycles
    on = np.where((ph > 0.5) & (np.r_[0.0, ph[:-1]] <= 0.5))[0]
    cl, cr, cd, ct, per = [], [], [], [], []
    for k in range(len(on) - 1):
        s, e = on[k], on[k + 1]
        p, w = ph[s:e], wz[s:e]
        cl.append(float(w[p > 0.5].sum()) * DT * R2D)
        cr.append(float(w[p < -0.5].sum()) * DT * R2D)
        cd.append(float(w[np.abs(p) < 0.5].sum()) * DT * R2D)
        ct.append(cl[-1] + cr[-1] + cd[-1])
        per.append((e - s) * DT)
    o["ncyc"] = len(ct)
    o["period"] = float(np.mean(per)) if per else np.nan
    o["freq"] = 1.0 / o["period"] if per else np.nan
    o["cyc_L"] = float(np.mean(cl)) if cl else np.nan
    o["cyc_R"] = float(np.mean(cr)) if cr else np.nan
    o["cyc_D"] = float(np.mean(cd)) if cd else np.nan
    o["cyc_net"] = float(np.mean(ct)) if ct else np.nan
    o["cyc_net_sd"] = float(np.std(ct)) if ct else np.nan
    o["cyc_abs"] = float(np.mean(np.abs(ct))) if ct else np.nan
    # hip_yaw pair mode
    cm = 0.5 * (ql + qr)
    o["cm_all"] = float(cm.mean()) * R2D
    o["cm_L"] = float(cm[ph > 0.5].mean()) * R2D if (ph > 0.5).any() else np.nan
    o["cm_R"] = float(cm[ph < -0.5].mean()) * R2D if (ph < -0.5).any() else np.nan
    o["cm_AC"] = float((cm * ph).mean()) * R2D
    # splay per phase
    o["splay_L"] = float((qrl[ph > 0.5] - qrr[ph > 0.5]).mean()) * R2D if (ph > 0.5).any() else np.nan
    o["splay_R"] = float((qrl[ph < -0.5] - qrr[ph < -0.5]).mean()) * R2D if (ph < -0.5).any() else np.nan
    o["ankleL_L"] = float(qal[ph > 0.5].mean()) * R2D if (ph > 0.5).any() else np.nan
    o["ankleL_R"] = float(qal[ph < -0.5].mean()) * R2D if (ph < -0.5).any() else np.nan
    o["ankleR_L"] = float(qar[ph > 0.5].mean()) * R2D if (ph > 0.5).any() else np.nan
    o["ankleR_R"] = float(qar[ph < -0.5].mean()) * R2D if (ph < -0.5).any() else np.nan
    o["hipyawL_L"] = float(ql[ph > 0.5].mean()) * R2D if (ph > 0.5).any() else np.nan
    o["hipyawL_R"] = float(ql[ph < -0.5].mean()) * R2D if (ph < -0.5).any() else np.nan
    o["hipyawL_p2p"] = float(np.percentile(ql, 95) - np.percentile(ql, 5)) * R2D
    o["hipyawR_p2p"] = float(np.percentile(qr, 95) - np.percentile(qr, 5)) * R2D
    return o


def main():
    for pol in POLICIES:
        ix, a = load(f"logs/axis/csv_{pol}_0.2.csv")
        env = a[:, ix["env"]].astype(int)
        rs, allrs = [], []
        for e in range(int(env.max()) + 1):
            m = env == e
            if m.sum() < 200:
                continue
            d = {c: a[m, ix[c]] for c in ("wz", "phase", "vy_b")}
            o = run_one(d["wz"], d["phase"], d["vy_b"],
                        a[m, ix["left_hip_yaw"]], a[m, ix["right_hip_yaw"]],
                        a[m, ix["left_hip_roll"]], a[m, ix["right_hip_roll"]],
                        a[m, ix["left_ankle"]], a[m, ix["right_ankle"]])
            allrs.append(o)
            if o["ncyc"] >= 20:
                rs.append(o)
        g = lambda k: float(np.nanmean([r[k] for r in rs]))
        s = lambda k: float(np.nanstd([r[k] for r in rs]))
        print("=" * 118)
        print(f"{pol}: {len(rs)}/{len(allrs)} envs are walking (>=20 L-single-support onsets in 20 s)")
        if not rs:
            continue
        print(f"  duty   L {g('duty_L'):.3f}  R {g('duty_R'):.3f}  D {g('duty_D'):.3f}   "
              f"(D = double support OR double swing -- the sensor cannot tell them apart)")
        print(f"  gait   period {g('period'):.3f} s  freq {g('freq'):.2f} Hz  cycles/env {g('ncyc'):.1f}")
        print(f"  wz by phase   L {g('wz_L'):+8.2f} +-{s('wz_L'):5.2f}   "
              f"R {g('wz_R'):+8.2f} +-{s('wz_R'):5.2f}   D {g('wz_D'):+8.2f} +-{s('wz_D'):5.2f}  [deg/s]")
        print(f"  contribution  L {g('con_L'):+8.2f}          R {g('con_R'):+8.2f}          "
              f"D {g('con_D'):+8.2f}")
        print(f"  => DC {g('dc'):+8.2f} +-{s('dc'):5.2f} deg/s   "
              f"|  signed mismatch (-conL-conR) {g('mismatch'):+7.2f} +-{s('mismatch'):5.2f}"
              f"   corr(DC, mismatch) = {np.corrcoef([r['dc'] for r in rs],[r['mismatch'] for r in rs])[0,1]:+.3f}")
        print(f"  => AC {g('ac'):+8.2f} +-{s('ac'):5.2f} deg/s   "
              f"|  DAC envelope  {g('wz_L')-g('wz_R'):+7.2f}")
        print(f"  per-cycle yaw impulse: L-half {g('cyc_L'):+7.3f}  R-half {g('cyc_R'):+7.3f}  "
              f"double {g('cyc_D'):+7.3f}  => NET {g('cyc_net'):+7.3f} +-{s('cyc_net'):.3f} deg/cycle "
              f"(|net| {g('cyc_abs'):.3f})")
        print(f"  lateral velocity by phase [mm/s]: L {g('vy_L'):+7.2f}  R {g('vy_R'):+7.2f}  D {g('vy_D'):+7.2f}"
              f"  => L-R {g('vy_L')-g('vy_R'):+7.2f}")
        print(f"  hip_yaw pair: common mode all {g('cm_all'):+.2f} deg  (L-phase {g('cm_L'):+.2f} / "
              f"R-phase {g('cm_R'):+.2f}, phase-locked {g('cm_AC'):+.2f})")
        print(f"  hip_yaw swing p95-p05: L {g('hipyawL_p2p'):.2f} deg  R {g('hipyawR_p2p'):.2f} deg"
              f"   | left hip_yaw L-phase {g('hipyawL_L'):+.2f} / R-phase {g('hipyawL_R'):+.2f}")
        print(f"  leg splay (hip_roll_L - hip_roll_R): L-phase {g('splay_L'):+.2f}  R-phase {g('splay_R'):+.2f}")
        print(f"  ankle L: L-phase {g('ankleL_L'):+.2f} / R-phase {g('ankleL_R'):+.2f}   "
              f"ankle R: L-phase {g('ankleR_L'):+.2f} / R-phase {g('ankleR_R'):+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
