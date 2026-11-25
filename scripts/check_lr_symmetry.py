#!/usr/bin/env python3
"""Is the ROBOT MODEL itself left-right symmetric? (pure CPU, no rollout)

Motivation (user, 2026-09-17): "有可能存在左右动力学不对称的问题吗". If the MJCF's
left and right limbs are not exact mirrors -- different mass, different inertial
offset, different joint axis or range -- then EVERY policy trained on it is forced
into an asymmetric gait, and no reward change can fix that. The reward-side
symptoms we have measured are consistent with either explanation:

  * gait phase decomposition: our L single-support yaw -42.79 deg/s vs R +23.01
    (a 1.9x magnitude mismatch); the official's -1.80 / +1.79 cancels to 0.01.
  * the mirror-symmetry loss (`MICRODUCK_SYMMETRY`) is OFF by default and was
    judged "not a reliable lever" -- but a mirror LOSS cannot repair a model that
    is not mirror-able in the first place.

This audits, in order of how conclusive each is:

 1. CoM lateral offset in the HOME pose. THE decisive number: a model whose
    centre of mass sits off the sagittal plane has a built-in preferred side, and
    it is a property of the model, not of any policy.
 2. Per-body mass / inertial-pos / fullinertia for every mirrored (-left/-right)
    pair, with the y-component sign-flipped as a mirror requires.
 3. Joint axis and range for each mirrored joint pair.
 4. Whether the two legs' kinematic chains are mirror images in the FK sense:
    drive the left leg joints to a pose and the right leg to the mirrored pose,
    run mj_forward, and compare the two foot positions (x mirrored, y negated).
 5. The actuator model (every servo shares one BAM identification, so any
    per-joint override would be an asymmetry).

Usage:
    uv run python scripts/check_lr_symmetry.py
    uv run python scripts/check_lr_symmetry.py --xml src/mjlab_microduck/robot/microduck/robot_allcollisions.xml
"""

from __future__ import annotations

import argparse
import math

import numpy as np


def mirrored_pairs(mujoco, model) -> list[tuple[int, int]]:
    out = []
    for b in range(model.nbody):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if n.endswith("_left") or n.endswith("left"):
            cand = n.replace("_left", "_right").replace("left", "right")
            r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cand)
            if r >= 0:
                out.append((b, r, n, cand))
    # body names in this repo are mostly bare ("ankle_left"); add hand-listed pairs
    for ln, rn in (("ankle_left", "ankle_right"), ("upper_leg_left", "upper_leg_right"),
                   ("hip_l", None), ("leg", None)):
        if rn is None:
            continue
        l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ln)
        r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, rn)
        if l >= 0 and r >= 0 and (l, r, ln, rn) not in out:
            out.append((l, r, ln, rn))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="src/mjlab_microduck/robot/microduck/robot_walk.xml")
    args = ap.parse_args()

    import mujoco
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    print(f"# model: {args.xml}")
    print(f"# bodies={model.nbody} joints={model.njnt} actuators={model.nu}")

    # ── 1. CoM lateral offset in the default (HOME) pose ─────────────────────
    mujoco.mj_forward(model, data)
    com = data.subtree_com[0]
    print("\n=== 1. HOME 姿态下的整机质心（决定性数字）===")
    print(f"  CoM = ({com[0]*1000:+.3f}, {com[1]*1000:+.3f}, {com[2]*1000:+.3f}) mm")
    print(f"  **横向(y)偏移 = {com[1]*1000:+.3f} mm**  ← 非 0 就是模型自带的偏侧")
    mass = float(model.body_mass.sum())
    print(f"  总质量 = {mass*1000:.1f} g")

    # ── 2. mirrored body pairs: mass / inertial pos / inertia ────────────────
    print("\n=== 2. 镜像 body 对的质量与惯量 ===")
    print(f"{'left':<18}{'right':<18}{'mass L/R (g)':>22}{'ipos y L/R (mm)':>26}  mirror?")
    pairs = mirrored_pairs(mujoco, model)
    if not pairs:
        print("  (没有识别到 -left/-right body 名对；本模型的命名见下)")
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(model.nbody)]
        print("  bodies:", ", ".join(n for n in names if n))
    bad = []
    for l, r, ln, rn in pairs:
        ml, mr = model.body_mass[l] * 1000, model.body_mass[r] * 1000
        ipl, ipr = model.body_ipos[l], model.body_ipos[r]
        # a mirror flips the y component's sign and keeps x, z
        dy = ipl[1] + ipr[1]
        ok_mass = abs(ml - mr) < 1e-6
        ok_pos = abs(dy) * 1000 < 0.05
        flag = "OK" if (ok_mass and ok_pos) else "**DIFF**"
        if flag != "OK":
            bad.append((ln, rn, ml - mr, dy * 1000))
        print(f"{ln:<18}{rn:<18}{ml:>10.3f}/{mr:<10.3f}{ipl[1]*1000:>12.3f}/{ipr[1]*1000:<12.3f}  {flag}")

    # ── 3. mirrored joint axis / range ───────────────────────────────────────
    print("\n=== 3. 镜像关节的轴与行程 ===")
    print(f"{'joint':<18}{'axis':>26}{'range (rad)':>26}")
    jsym = {}
    for j in range(model.njnt):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        jsym[n] = j
    seen = set()
    for n, j in sorted(jsym.items()):
        if "left" not in n and "_l" not in n:
            continue
        rn = n.replace("left", "right").replace("_l", "_r")
        if rn not in jsym or (rn, n) in seen:
            continue
        seen.add((n, rn))
        jr = jsym[rn]
        al, ar = model.jnt_axis[j], model.jnt_axis[jr]
        lim = model.jnt_limited[j] and model.jnt_limited[jr]
        rl = model.jnt_range[j] if lim else [float("nan")] * 2
        rr = model.jnt_range[jr] if lim else [float("nan")] * 2
        # mirror requires the axis y-component to flip sign, range to negate+swap
        mir_axis = np.allclose([al[0], -al[1], al[2]], ar, atol=1e-6)
        mir_rng = (not lim) or (abs(rl[0] + rr[1]) < 1e-6 and abs(rl[1] + rr[0]) < 1e-6)
        flag = "OK" if (mir_axis and mir_rng) else "**DIFF**"
        print(f"{n:<18}{str(np.round(al,3)):>26}{str(np.round(rl,4)):>26}  {flag}")
        print(f"{rn:<18}{str(np.round(ar,3)):>26}{str(np.round(rr,4)):>26}")

    # ── 4. FK mirror test: mirrored joint pose -> mirrored foot position ─────
    print("\n=== 4. 正运动学镜像测试（关键）===")
    lo = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ankle_left")
    ro = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ankle_right")
    if lo >= 0 and ro >= 0:
        left_j, right_j = [], []
        for n, j in jsym.items():
            if n.endswith("_left") or n.endswith("left"):
                rn = n.replace("_left", "_right").replace("left", "right")
                if rn in jsym:
                    left_j.append(j); right_j.append(jsym[rn])
        rng = np.random.default_rng(0)
        worst_pos, worst_lab = 0.0, ""
        for trial in range(200):
            q = rng.uniform(-0.4, 0.4, size=len(left_j))
            d = mujoco.MjData(model)
            for k, j in enumerate(left_j):
                d.qpos[model.jnt_qposadr[j]] = q[k]
            for k, j in enumerate(right_j):
                d.qpos[model.jnt_qposadr[j]] = -q[k]      # mirror = negate the angle
            mujoco.mj_forward(model, d)
            pl, pr = d.xpos[lo].copy(), d.xpos[ro].copy()
            # a perfect mirror maps (x,y,z) -> (x,-y,z) and swaps L/R
            err = float(np.linalg.norm(pl - np.array([pr[0], -pr[1], pr[2]])))
            if err > worst_pos:
                worst_pos, worst_lab = err, f"trial {trial} q={np.round(q,3)}"
        print(f"  镜像姿态下 左脚位置 vs 右脚位置的镜像: 最大偏差 = {worst_pos*1000:.4f} mm")
        print("  （若为 0，说明两条腿的正运动学是严格镜像的）")
    else:
        print("  (找不到 ankle_left/ankle_right)")

    # ── 5. summary ──────────────────────────────────────────────────────────
    print("\n=== 5. 结论 ===")
    lat = com[1] * 1000
    print(f"  HOME 姿态横向质心偏移 : {lat:+.3f} mm  {'← 偏侧' if abs(lat) > 0.5 else '(可忽略)'}")
    print(f"  镜像 body 质量/惯量差 : {len(bad)} 处" + (f" {bad}" if bad else " (无)"))
    if lo >= 0 and ro >= 0:
        print(f"  正运动学镜像最大偏差  : {worst_pos*1000:.4f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
