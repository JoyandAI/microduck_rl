#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Microduck 腿部摆锤标定 —— ID↔关节 目视映射工具。

用法(只微动, 不整条轨迹; 每个 +10° 后回位):
    /usr/bin/python3 scripts/map_joint_ids.py --port /dev/ttyACM0 --ids 10,11,12,13,14,20,21,22,23,24

原理: 对每个 ID 依次开扭矩→目标位置 +120 LSB(≈10.5°)→ 读 Δpos → 回位 → 关扭矩。
你在旁边看"哪个连杆在动", 把对应关系记下来, 回填到 record_leg_pendulum.py 的 --id 参数。

安全: 临时限流 0.975A(reg44=150, 结束恢复 300); 温度>50 停; 每次微动后回位。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_hls_registers import read_regs, write_regs, u16_le  # noqa: E402

STEP_LSB = 120          # ≈ 10.5°


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--ids", default="10,11,12,13,14,20,21,22,23,24,30,31,32,33,34")
    ap.add_argument("--step-lsb", type=int, default=STEP_LSB)
    args = ap.parse_args()

    import serial
    ser = serial.Serial(args.port, 1_000_000, timeout=0.2)
    ids = [int(x) for x in args.ids.split(",")]

    def r16(p, a):
        r = read_regs(ser, p, a, 2)
        return None if r is None else u16_le(r, 0)

    print(f"[i] 依次微动 ID: {ids} —— 你看哪个连杆在动, 记下来")
    for pid in ids:
        write_regs(ser, pid, 44, [0x96, 0x00])           # 0.975A 临时限流
        write_regs(ser, pid, 41, [254])
        write_regs(ser, pid, 46, [0xFF, 0x7F])
        p0 = r16(pid, 56)
        if p0 is None:
            print(f"  ID{pid:2d}: 无响应, 跳过")
            continue
        write_regs(ser, pid, 40, [1])
        write_regs(ser, pid, 55, [1])
        time.sleep(0.1)
        goal = (p0 + args.step_lsb) % 4096
        write_regs(ser, pid, 42, [goal & 0xFF, (goal >> 8) & 0xFF])
        time.sleep(0.6)
        p1 = r16(pid, 56)
        mv = ((p1 - p0 + 2048) % 4096) - 2048 if p1 is not None else None
        write_regs(ser, pid, 42, [p0 & 0xFF, (p0 >> 8) & 0xFF])
        time.sleep(0.4)
        write_regs(ser, pid, 40, [0])
        write_regs(ser, pid, 55, [0])
        write_regs(ser, pid, 44, [0x2C, 0x01])            # 恢复 300(1.95A)
        temps = read_regs(ser, pid, 63, 1)
        t = temps[0] if temps else None
        print(f"  ID{pid:2d}: Δ={mv} LSB ({mv*360/4096 if mv else 0:+.1f}°)  温度={t}°C  ← 哪根连杆动?")

    ser.close()
    print("[i] 完成。把『哪个 ID = 哪个关节』的对应关系记下来, 回填到 record_leg_pendulum.py 的 --id/--joint。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
