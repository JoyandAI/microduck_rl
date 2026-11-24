#!/usr/bin/env python
"""模型对比总表：配置 + 能否走路 + 站得稳不稳 + 偏航/直行。

按用户 2026-09-21 定的优先级：
  1. **能正常走路** = 基本要求（多少环境真的在走 / 速度 / 是走还是跑 / 摔不摔）
  2. **站得稳** = 基本要求（零命令：reset 次数、倾角、离地间隙）
  3. 走直线 = 其次（20 s 净转角、每米航向误差、<30° 环境占比）

数据来源与口径（每个数字都标了出处，避免张冠李戴）：
  * 直行  : `<out>/<stem>_vx0.2.json`（或 `_walk.json`）= 64 env / 20 s / 冻结 vx=0.2 / DR 开
  * 站立  : 三种来源，按内容自动识别
            (a) `gait_forensics` 探针 `<out>/<stem>_stand.json`（有 trunk/gait ⇒ reset+倾角+离地）
            (b) align 队列的 `*_stand.json` 其实是 **vx=0 的直行评估**（有 per_env ⇒ 只有 reset 数）
            (c) /tmp/stand_probe/*.json（我手动补跑的 6 个老臂，格式同 (a)）
  * 跑步相: `logs/flight/flight_summary.json`（agent D 的 64-env 探针）
  * 训练期: `<train_dir>/<stem>_train.log` 末段指标
  * 中途点: `logs/queue_sweep*/straightness.tsv` + `marching.tsv`
"""
from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANUAL_STAND = Path("/tmp/stand_probe")

# label, out_dir, stem, 配置, flight_key, eval_json 覆盖, train_dir 覆盖
ARMS = [
    ("famB_head_flight s42", "logs/queue_family_b", "famB_head_flight_s42", "官方配方+头块+flight(12k)", "famB_head_flight_s42", None, None),
    ("famB_head_flight s43", "logs/queue_family_b", "famB_head_flight_s43", "官方配方+头块+flight(12k)", "famB_head_flight_s43", None, None),
    ("align_fwd", "logs/queue_align", "align_fwd", "官方配方+头块(9k)", "align_fwd", None, None),
    ("align_full", "logs/queue_align", "align_full", "官方配方+头块(9k)", None, None, None),
    ("align_official", "logs/queue_align", "align_official", "官方配方 verbatim(9k)", None, None, None),
    ("ours_head s42", "logs/queue_overnight", "ours_head", "12偏差+头块(9k)", None, None, "logs/queue_ours"),
    ("ours_head s43", "logs/queue_overnight", "ours_head_s43", "12偏差+头块(9k)", "ours_head_s43", None, None),
    ("ours_head s44", "logs/queue_overnight", "ours_head_s44", "12偏差+头块(9k)", None, None, None),
    ("ours_flight s42 (r1)", "logs/queue_routeA", "ours_flight_s42", "12偏差+头块+flight(9k)", "ours_flight_s42", None, None),
    ("ours_flight s43 (r1)", "logs/queue_routeA", "ours_flight_s43", "12偏差+头块+flight(9k)", "ours_flight_s43", None, None),
    ("ours_flight s44 (r1)", "logs/queue_routeA", "ours_flight_s44", "12偏差+头块+flight(9k)", "ours_flight_s44", None, None),
    ("ours_stance s42", "logs/queue_overnight", "ours_stance_s42", "12偏差+头块+feet_max_width", None, None, None),
    ("ours_ztrack s42", "logs/queue_overnight", "ours_ztrack_s42", "12偏差+头块+yaw_z_track", None, None, None),
    ("ours_flight_stance s42", "logs/queue_overnight", "ours_flight_stance_s42", "12偏差+头块+flight+feet", None, None, None),
    ("ours_yawdc s42", "logs/queue_overnight", "ours_yawdc_s42", "12偏差+头块+yaw_dc", "ours_yawdc_s42", None, None),
    ("turn_ours s42", "logs/queue_overnight", "turn_ours_s42", "12偏差(无头块)+turn_first", None, None, None),
    ("turn_official s42", "logs/queue_overnight", "turn_official_s42", "官方配方+turn_first", "turn_official_s42", None, None),
    ("ck7000 (旧交付)", None, None, "旧配方，无头块", "ck7000", "logs/axis/j_ck7000_0.2.json", None),
]


def _f(v, fmt="{:.1f}", none="—"):
    return fmt.format(v) if isinstance(v, (int, float)) and v == v else none


def _num(d, key, default):
    v = (d or {}).get(key, None)
    return default if v is None else v


def straightness(out, stem, override=None) -> dict:
    paths = [Path(override)] if override else []
    if out:
        paths += [REPO / out / f"{stem}_vx0.2.json", REPO / out / f"{stem}_walk.json"]
    for p in paths:
        if not p.is_file():
            continue
        d = json.loads(p.read_text())
        per = [e for e in d.get("per_env", []) if "net_yaw_deg" in e]
        if not per:
            return {"empty": True, "falls": d.get("falls")}
        w = [e for e in per if e["speed_along_mps"] > 0.02]
        ny = sorted(abs(e["net_yaw_deg"]) for e in per)
        netw = st.median([abs(e["net_yaw_deg"]) for e in w]) if w else float("nan")
        sp = st.median([e["speed_along_mps"] for e in per])
        return {
            "nwalk": len(w), "n": len(per), "net": st.median(ny), "net_walk": netw,
            "lt30": 100 * sum(1 for v in ny if v < 30) / len(ny),
            "dc": st.median([abs(e["yaw_dc_body_deg_s"]) for e in per]),
            "sp": sp, "pe": st.median([e["path_efficiency"] for e in per]),
            "dpm": netw / abs(sp * 20) if sp > 0.005 else float("inf"),
            "falls": d.get("falls", 0), "full": d.get("frac_envs_full_window"),
        }
    return {}


def stand(out, stem) -> dict:
    """零命令站立：自动识别三种来源（见文件头注释）。"""
    # gait_forensics 探针优先（它同时给 reset/倾角/离地）；align 队列那份 `*_stand.json`
    # 只是 vx=0 的直行评估，没有离地间隙，所以放在后面兜底。
    cands = ([MANUAL_STAND / f"{stem}.json"] if stem else []) + \
            ([REPO / out / f"{stem}_stand.json"] if out else [])
    for p in cands:
        if not p.is_file():
            continue
        d = json.loads(p.read_text())
        cmd = d.get("cmd") or {}
        if cmd and abs(cmd.get("vx", 1)) > 1e-6:
            continue
        if "trunk" in d or "gait" in d:          # gait_forensics 探针
            t, g = d.get("trunk", {}) or {}, d.get("gait", {}) or {}
            clr = g.get("foot_clearance_body_z_m")
            return {"resets": d.get("resets"), "tilt": t.get("tilt_max_deg"),
                    "clr": clr * 1000 if isinstance(clr, (int, float)) else float("nan"),
                    "src": "gait"}
        if "per_env" in d:                        # vx=0 的直行评估
            per = [e for e in d["per_env"] if "net_yaw_deg" in e]
            if not per:
                return {"resets": d.get("n_envs_insufficient_data"), "tilt": None,
                        "clr": None, "src": "eval"}
            return {"resets": d.get("n_envs_with_resets", d.get("falls")), "tilt": None,
                    "clr": None, "src": "eval",
                    "sp": st.median([e["speed_along_mps"] for e in per]),
                    "yaw": st.median([abs(e["net_yaw_deg"]) for e in per])}
    return {}


def flight(key) -> dict:
    p = REPO / "logs/flight/flight_summary.json"
    if not key or not p.is_file():
        return {}
    for r in json.loads(p.read_text()):
        if r.get("name") == key:
            return r
    return {}


def train(out, stem, tdir=None) -> dict:
    if not (out or tdir):
        return {}
    p = Path(tdir or out) / f"{stem}_train.log"
    if not p.is_file():
        return {}
    txt = p.read_text(errors="replace")
    m: dict[str, float] = {}
    for line in txt.splitlines():
        mm = re.match(r"^\s*([A-Za-z_][\w/]*):\s*(-?[\d.]+)\s*$", line)
        if mm:
            m[mm.group(1)] = float(mm.group(2))
    it = re.findall(r"Learning iteration (\d+)/(\d+)", txt)
    m["_done"], m["_target"] = (float(it[-1][0]), float(it[-1][1])) if it else (0.0, 0.0)
    return m


def sweep_rows(arm: str) -> dict:
    out = {}
    for sweep in ("logs/queue_sweep", "logs/queue_sweep_famB"):
        tsv = REPO / sweep / "straightness.tsv"
        if not tsv.is_file():
            continue
        for line in tsv.read_text().splitlines()[1:]:
            c = line.split("\t")
            if len(c) < 11 or c[1] != arm:
                continue
            try:
                net = abs(float(c[4].replace("+", "")))
            except ValueError:
                continue
            row = {"label": c[0], "ck": c[2], "net": net, "pe": float(c[5]), "sp": float(c[9])}
            if "best" not in out or net < out["best"]["net"]:
                out["best"] = row
            out["final"] = row
        m = REPO / sweep / "marching.tsv"
        if m.is_file() and "best" in out:
            for line in m.read_text().splitlines()[1:]:
                c = line.split("\t")
                if c and c[0] == out["best"]["label"]:
                    out["best"]["clr"], out["best"]["tilt"] = c[1], c[2]
    return out


def verdict(s, sd, fl) -> str:
    if not s:
        return "无评估"
    if s.get("empty"):
        return "❌ 命令 vx=0.2 时全部环境 0.4 s 内重置"
    tags = []
    walks = s["nwalk"] >= 45 and s["sp"] >= 0.08
    tags.append("✅会走" if walks else f"❌走不动({s['nwalk']}/64,{s['sp']:.3f})")
    if fl:
        tags.append("✅不跑" if fl.get("flight", 1) <= 0.05 else f"❌在跑({fl['flight']:.2f})")
    if sd:
        r = _num(sd, "resets", 99)
        if r <= 1:
            clr = sd.get("clr")
            tags.append(f"✅站得住(离地{clr:.0f}mm)" if isinstance(clr, (int, float)) and clr == clr and clr > 8
                        else "✅站得稳")
        else:
            tags.append(f"❌站不稳({r}次reset)")
    else:
        tags.append("?站立未测")
    return " ".join(tags)


def main() -> int:
    print("=" * 182)
    print("表 1 — 最终 checkpoint 对比（按你的优先级：先会走 → 再站得稳 → 最后看直）")
    print("口径：走动/速度/flight/摔 = 64 env、20 s、冻结 vx=0.2、DR 开；站立 = 零命令 20 s；净转角 = 20 s 累计航向变化的中位数")
    print("=" * 182)
    hdr = (f"{'模型':22s} {'配置':30s} {'it':>6s} | {'走动':>6s} {'速度':>7s} {'flight':>6s} {'摔':>3s} {'窗%':>4s} | "
           f"{'站reset':>7s} {'倾角':>5s} {'离地mm':>7s} | {'净转角':>7s} {'yawDC':>6s} {'°/m':>6s} {'<30°%':>5s} | 判定")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for label, out, stem, cfg, fkey, ev, tdir in ARMS:
        rows.append((label, cfg, train(out, stem, tdir), straightness(out, stem, ev),
                     stand(out, stem), flight(fkey)))

    def key(r):
        _, _, _, s, sd, _fl = r
        walks = 0 if (s and not s.get("empty") and s.get("nwalk", 0) >= 45 and s.get("sp", 0) >= 0.08) else 1
        stand_bad = 1 if (not sd or _num(sd, "resets", 99) > 1) else 0
        return (walks, stand_bad, abs(s["net"]) if s and not s.get("empty") else 9999)

    for label, cfg, tm, s, sd, fl in sorted(rows, key=key):
        good = bool(s) and not s.get("empty")
        def g(v, fmt="{:.1f}"):
            return _f(v, fmt) if good else "—"
        print(f"{label:22s} {cfg:30s} {int(tm.get('_done', 0)):>6d} | "
              f"{(str(s['nwalk']) + '/64') if good else '—':>6s} {g(s['sp'] if good else None,'{:.4f}'):>7s} "
              f"{_f(fl.get('flight'),'{:.3f}'):>6s} {(str(s['falls']) if good else '—'):>3s} "
              f"{g((s['full'] or 0) * 100 if good else None,'{:.0f}'):>4s} | "
              f"{str(sd.get('resets','—')):>7s} {_f(sd.get('tilt'),'{:.1f}'):>5s} {_f(sd.get('clr'),'{:.2f}'):>7s} | "
              f"{g(s['net'] if good else None,'{:.1f}'):>7s} {g(s['dc'] if good else None,'{:.2f}'):>6s} "
              f"{g(s['dpm'] if good else None,'{:.1f}'):>6s} {g(s['lt30'] if good else None,'{:.0f}'):>5s} | "
              f"{verdict(s, sd, fl)}")

    print()
    print("=" * 182)
    print("表 2 — 中途 checkpoint 的最好点（直行度不是训练时长的单调函数；站离地 = 该点的零命令 foot clearance）")
    print("=" * 182)
    hdr = (f"{'run 目录':26s} {'最好ck':>7s} {'|净转角|':>8s} {'pathEff':>8s} {'速度':>7s} {'站离地mm':>8s} | "
           f"{'最终ck':>7s} {'|净转角|最终':>11s} {'退化':>6s}")
    print(hdr); print("-" * len(hdr))
    for arm in ("famB_head_flight_s42", "famB_head_flight_s43", "ours_flight_s42", "ours_flight_s42_r2",
                "ours_flight_s43", "ours_flight_s43_r2", "ours_flight_s44", "ours_flight_s44_r2",
                "align_fwd", "align_full", "ours_head", "ours_head_s43", "ours_head_s44"):
        sw = sweep_rows(arm)
        if not sw.get("best"):
            continue
        b, f = sw["best"], sw.get("final", {})
        deg = (f.get("net", float("nan")) / b["net"]) if b["net"] else float("nan")
        print(f"{arm:26s} {b['ck']:>7s} {b['net']:>8.1f} {b['pe']:>8.3f} {b['sp']:>7.4f} "
              f"{str(b.get('clr','—')):>8s} | {f.get('ck','—'):>7s} {f.get('net',float('nan')):>11.1f} {deg:>5.1f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
