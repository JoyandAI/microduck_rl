#!/usr/bin/env python3
"""Phase / joint attribution of the net yaw, from the P64 eval_straightness CSVs.

Read-only. Inputs: logs/axis/csv_<policy>_{0.2,0.0}.csv  (64 env x 20 s x 50 Hz).
Outputs a text table to stdout.
"""
import csv, math, sys
import numpy as np

DT = 0.02
POLICIES = ["official", "ck7000", "ck12000", "R3hrl1", "R4db03"]
JOINTS = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee",
          "left_ankle", "right_hip_yaw", "right_hip_roll", "right_hip_pitch",
          "right_knee", "right_ankle", "neck_pitch", "head_pitch", "head_yaw",
          "head_roll"]


def load(path):
    with open(path) as f:
        r = csv.reader(f)
        hdr = next(r)
        a = np.array([[float(v) for v in row] for row in r])
    return hdr, a


def per_env(hdr, a):
    ix = {c: i for i, c in enumerate(hdr)}
    env = a[:, ix["env"]].astype(int)
    out = []
    for e in range(int(env.max()) + 1):
        m = env == e
        if m.sum() < 50:
            continue
        d = {c: a[m, ix[c]] for c in hdr if c != "env"}
        out.append(d)
    return out


def analyse(d, home_cm_sum=None):
    """One env. Returns a dict of phase-attribution numbers."""
    wz = d["wz"]                      # rad/s, body-frame yaw rate (sim)
    ph = d["phase"]
    vyb = d["vy_b"]
    t = d["t"]
    n = len(wz)
    r2d = 180.0 / math.pi
    res = {}
    # ---- three-phase decomposition -------------------------------------
    for nm, sel in (("L", ph > 0.5), ("R", ph < -0.5), ("D", np.abs(ph) < 0.5)):
        res[f"duty_{nm}"] = float(sel.mean())
        res[f"wz_{nm}"] = float(wz[sel].mean()) * r2d if sel.any() else 0.0
        res[f"con_{nm}"] = res[f"wz_{nm}"] * res[f"duty_{nm}"]
        res[f"vy_{nm}"] = float(vyb[sel].mean()) if sel.any() else 0.0
    res["dc"] = float(wz.mean()) * r2d
    res["ac"] = float((wz * ph).mean()) * r2d
    res["net"] = res["con_L"] + res["con_R"] + res["con_D"]
    # ---- gait cycles: left-single-support onsets ------------------------
    onsets = np.where((ph > 0.5) & (np.r_[0.0, ph[:-1]] <= 0.5))[0]
    cyc_tot, cyc_L, cyc_R, cyc_D, cyc_p2p = [], [], [], [], []
    ampL, ampR, period = [], [], []
    for k in range(len(onsets) - 1):
        s, e = onsets[k], onsets[k + 1]
        seg = slice(s, e)
        p = ph[seg]
        w = wz[seg]
        cyc_tot.append(float(w.sum()) * DT * r2d)
        cyc_L.append(float(w[p > 0.5].sum()) * DT * r2d)
        cyc_R.append(float(w[p < -0.5].sum()) * DT * r2d)
        cyc_D.append(float(w[np.abs(p) < 0.5].sum()) * DT * r2d)
        cyc_p2p.append(float(w.max() - w.min()) * r2d)
        ampL.append(abs(float(w[p > 0.5].mean())) * r2d if (p > 0.5).any() else np.nan)
        ampR.append(abs(float(w[p < -0.5].mean())) * r2d if (p < -0.5).any() else np.nan)
        period.append(float(e - s) * DT)
    res["n_cycles"] = len(cyc_tot)
    res["freq"] = 1.0 / np.mean(period) if period else np.nan
    res["cyc_tot"] = float(np.mean(cyc_tot)) if cyc_tot else np.nan   # deg per cycle = residual
    res["cyc_tot_sd"] = float(np.std(cyc_tot)) if cyc_tot else np.nan
    res["cyc_abs_tot"] = float(np.mean(np.abs(cyc_tot))) if cyc_tot else np.nan
    res["cyc_L"] = float(np.mean(cyc_L)) if cyc_L else np.nan
    res["cyc_R"] = float(np.mean(cyc_R)) if cyc_R else np.nan
    res["cyc_D"] = float(np.mean(cyc_D)) if cyc_D else np.nan
    res["cyc_p2p"] = float(np.mean(cyc_p2p)) if cyc_p2p else np.nan
    res["ampL"] = float(np.nanmean(ampL)) if ampL else np.nan
    res["ampR"] = float(np.nanmean(ampR)) if ampR else np.nan
    # mismatch measured the way the doc does: |contrib_L| - |contrib_R|
    res["mismatch"] = abs(res["con_L"]) - abs(res["con_R"])
    res["signed_mismatch"] = (-res["con_L"]) - res["con_R"]   # >0 => nets positive
    # ---- joints: common / differential mode of the hip_yaw pair ---------
    cm = 0.5 * (d["left_hip_yaw"] + d["right_hip_yaw"])
    dm = 0.5 * (d["left_hip_yaw"] - d["right_hip_yaw"])
    res["cm_mean_deg"] = float(cm.mean()) * r2d
    if home_cm_sum is not None:
        res["torsion_deg"] = float(cm.mean()) * r2d - home_cm_sum
    res["cm_ac"] = float((cm * ph).mean()) * r2d        # phase-locked steering
    res["dm_ac"] = float((dm * ph).mean()) * r2d        # phase-locked split
    for nm, sel in (("L", ph > 0.5), ("R", ph < -0.5), ("D", np.abs(ph) < 0.5)):
        res[f"cm_{nm}"] = float(cm[sel].mean()) * r2d if sel.any() else np.nan
        res[f"dm_{nm}"] = float(dm[sel].mean()) * r2d if sel.any() else np.nan
    # per-phase joint means
    for j in JOINTS:
        q = d[j] * r2d
        res[f"q_{j}_L"] = float(q[ph > 0.5].mean()) if (ph > 0.5).any() else np.nan
        res[f"q_{j}_R"] = float(q[ph < -0.5].mean()) if (ph < -0.5).any() else np.nan
        res[f"q_{j}_D"] = float(q[np.abs(ph) < 0.5].mean()) if (np.abs(ph) < 0.5).any() else np.nan
        res[f"q_{j}_mean"] = float(q.mean())
        res[f"q_{j}_p2p"] = float(np.percentile(q, 95) - np.percentile(q, 5))
        res[f"q_{j}_aclock"] = float((q * ph).mean())
    # lateral foot-placement proxy: hip_roll splay per phase
    res["splay_L"] = res["q_left_hip_roll_L"] - res["q_right_hip_roll_L"]
    res["splay_R"] = res["q_left_hip_roll_R"] - res["q_right_hip_roll_R"]
    # speed / drift
    res["speed"] = float(np.hypot(d["x"][-1] - d["x"][0], d["y"][-1] - d["y"][0]) / t[-1])
    res["lat_drift"] = float(d["y"][-1] - d["y"][0])
    return res


def agg(rs, k):
    v = np.array([r[k] for r in rs if r.get(k) is not None and not np.isnan(r[k])],
                 dtype=float)
    return (float(v.mean()), float(v.std()), float(np.median(v))) if len(v) else (np.nan,) * 3


def main():
    # ---- calibrate the HOME common-mode offset from the official row ----
    # eval_straightness defines torsion = 0.5*(hipYawL+hipYawR) - home_common.
    # The home constant is policy-independent, so pin it with the official
    # value it already printed: -0.105 deg (logs/axis/j_official_0.2.json).
    hdr, a = load("logs/axis/csv_official_0.2.csv")
    ix = {c: i for i, c in enumerate(hdr)}
    cm_mean_off = float((0.5 * (a[:, ix["left_hip_yaw"]] +
                                a[:, ix["right_hip_yaw"]])).mean()) * 180 / math.pi
    HOME_CM_SUM = cm_mean_off - (-0.105)
    print(f"[calib] official mean 0.5*(ql+qr) = {cm_mean_off:+.4f} deg ; "
          f"torsion reported = -0.105 -> HOME 0.5*(ql+qr) = {HOME_CM_SUM:+.4f} deg")

    data = {}
    for pol in POLICIES:
        for tag in ("0.2", "0.0"):
            try:
                h, aa = load(f"logs/axis/csv_{pol}_{tag}.csv")
            except FileNotFoundError:
                continue
            envs = per_env(h, aa)
            data[(pol, tag)] = [analyse(e, HOME_CM_SUM) for e in envs]

    r2d = 180 / math.pi
    for tag, label in (("0.2", "vx = 0.2 m/s (WALKING)"), ("0.0", "vx = 0.0 (STANDING)")):
        print("\n" + "=" * 108)
        print(f"PHASE DECOMPOSITION -- {label} -- 64 env x 20 s, seed 0, DR on, frozen cmd")
        print("=" * 108)
        hdrf = (f"{'policy':<11}{'n':>4}{'cyc':>5}{'freq':>6}|"
                f"{'dutyL':>7}{'wzL':>8}{'conL':>8}|{'dutyR':>7}{'wzR':>8}{'conR':>8}|"
                f"{'dutyD':>7}{'wzD':>8}{'conD':>8}|{'DC':>8}{'AC':>8}|{'cyc/j':>8}")
        print(hdrf)
        for pol in POLICIES:
            rs = data.get((pol, tag))
            if not rs:
                print(f"{pol:<11}  (no data / died early, n={len(rs) if rs else 0})")
                continue
            g = lambda k: agg(rs, k)[0]
            print(f"{pol:<11}{len(rs):>4}{g('n_cycles'):>5.0f}{g('freq'):>6.2f}|"
                  f"{g('duty_L'):>7.3f}{g('wz_L'):>8.2f}{g('con_L'):>8.2f}|"
                  f"{g('duty_R'):>7.3f}{g('wz_R'):>8.2f}{g('con_R'):>8.2f}|"
                  f"{g('duty_D'):>7.3f}{g('wz_D'):>8.2f}{g('con_D'):>8.2f}|"
                  f"{g('dc'):>8.2f}{g('ac'):>8.2f}|{g('cyc_tot'):>8.2f}")
            # scatter of DC and of the signed mismatch
            dcv = np.array([r["dc"] for r in rs])
            acv = np.array([r["ac"] for r in rs])
            mmv = np.array([r["signed_mismatch"] for r in rs])
            ctv = np.array([r["cyc_tot"] for r in rs])
            print(f"{'':<11}DC {dcv.mean():+7.2f}+-{dcv.std():5.2f} "
                  f"pos-envs {100*(dcv>0).mean():3.0f}%  |  AC {acv.mean():+7.2f}+-{acv.std():5.2f} "
                  f"pos-envs {100*(acv>0).mean():3.0f}%  |  signed-mismatch "
                  f"{mmv.mean():+7.2f}+-{mmv.std():5.2f} pos {100*(mmv>0).mean():3.0f}%  |  "
                  f"per-cycle residual {ctv.mean():+7.4f}+-{ctv.std():.4f} deg "
                  f"(|.| {np.abs(ctv).mean():.4f})")
            print(f"{'':<11}corr(DC, signed-mismatch) = {np.corrcoef(dcv, mmv)[0,1]:+.3f} ; "
                  f"corr(DC, AC) = {np.corrcoef(dcv, acv)[0,1]:+.3f}")

        # ---- detailed per-phase joint / torsion table ----
        print(f"\n-- {label}: hip_yaw pair mode + torsion + splay, by phase (deg) --")
        print(f"{'policy':<11}{'cm_all':>8}{'tors':>8}{'cm_L':>8}{'cm_R':>8}{'cm_D':>8}"
              f"{'cm_AC':>8}{'dm_AC':>8}|{'Lhy_L':>8}{'Lhy_R':>8}{'Rhy_L':>8}{'Rhy_R':>8}"
              f"|{'splay_L':>9}{'splay_R':>9}")
        for pol in POLICIES:
            rs = data.get((pol, tag))
            if not rs:
                continue
            g = lambda k: agg(rs, k)[0]
            print(f"{pol:<11}{g('cm_mean_deg'):>8.2f}{g('torsion_deg'):>8.2f}"
                  f"{g('cm_L'):>8.2f}{g('cm_R'):>8.2f}{g('cm_D'):>8.2f}"
                  f"{g('cm_ac'):>8.2f}{g('dm_ac'):>8.2f}|"
                  f"{g('q_left_hip_yaw_L'):>8.2f}{g('q_left_hip_yaw_R'):>8.2f}"
                  f"{g('q_right_hip_yaw_L'):>8.2f}{g('q_right_hip_yaw_R'):>8.2f}|"
                  f"{g('splay_L'):>9.2f}{g('splay_R'):>9.2f}")

        print(f"\n-- {label}: other joints, phase-locked amplitude mean(q*phase) (deg),"
              f" and p5-p95 range --")
        print(f"{'policy':<11}" + "".join(f"{j.replace('left_','L.').replace('right_','R.'):>11}"
                                          for j in JOINTS))
        for pol in POLICIES:
            rs = data.get((pol, tag))
            if not rs:
                continue
            g = lambda k: agg(rs, k)[0]
            print(f"{pol:<11}" + "".join(f"{g('q_'+j+'_aclock'):>11.2f}" for j in JOINTS))
        print(f"{'(p95-p05)':<11}" + "".join(f"{j.replace('left_','L.').replace('right_','R.'):>11}"
                                             for j in JOINTS))
        for pol in POLICIES:
            rs = data.get((pol, tag))
            if not rs:
                continue
            g = lambda k: agg(rs, k)[0]
            print(f"{pol:<11}" + "".join(f"{g('q_'+j+'_p2p'):>11.2f}" for j in JOINTS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
