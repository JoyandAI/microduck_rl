#!/usr/bin/env python3
"""Read-only zero-convention check: stand the duck up, run this, read the table.

No writes, no torque — it only PINGs the 15 servos and reads present position
(reg 56). It tells you whether the servos' zeros sit at the joint MECHANICAL
zero (the convention the runtime and scripts/replay_walk_angles.py assume:
tick 2048 = joint angle 0, sign +1, i.e. standing pose reads the home ticks
below) or somewhere else (typ. all 2048 = zeroed while standing).

Usage:
    uv run --with pyserial python scripts/check_hls_zero.py [--port /dev/ttyACM0]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from replay_walk_angles import (  # noqa: E402  (imports are serial-free)
    HOME_POSE_WIRE,
    RAD_PER_TICK,
    SERIAL_AVAILABLE,
    WIRE_JOINT_IDS,
    ping,
    print_zero_check,
    read_present_ticks,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=None, help="default: first ttyACM/ttyUSB")
    ap.add_argument("--baud", type=int, default=1000000)
    ap.add_argument("--pose", choices=("auto", "zero", "home"), default="auto",
                    help="which pose the duck is in right now: zero = 零位姿势"
                         "(腿竖直, 标定姿势), home = 正常站立姿势。"
                         "'auto' can't distinguish the two — it will ask.")
    args = ap.parse_args()

    if not SERIAL_AVAILABLE:
        raise SystemExit("[!] pyserial missing: uv run --with pyserial python "
                         "scripts/check_hls_zero.py")
    import serial
    from serial.tools import list_ports

    if args.port is None:
        ports = sorted({p.device for p in list_ports.comports()
                        if p.device.startswith(("/dev/ttyACM", "/dev/ttyUSB"))},
                       key=lambda d: d)
        if not ports:
            raise SystemExit("[!] no serial port (USB-TTL) found")
        args.port = ports[0]
        print(f"[i] auto-selected port: {args.port}")

    ser = serial.Serial(args.port, args.baud, timeout=0.3)
    ser.reset_input_buffer()
    try:
        missing = [pid for pid in WIRE_JOINT_IDS if not ping(ser, pid)]
        if missing:
            raise SystemExit(f"[!] servos missing: {missing} "
                             f"(12V power? signal wire? torque state irrelevant)")
        present = read_present_ticks(ser, WIRE_JOINT_IDS)
        if any(t < 0 for t in present):
            raise SystemExit(f"[!] reads failed: {present}")
    finally:
        ser.close()
    print("[i] read-only: no register written, torque untouched. "
          "Duck should be standing in its HOME pose now.")
    print_zero_check(WIRE_JOINT_IDS, present)

    home_tick = [int(round((float(v) + math.pi) / RAD_PER_TICK)) for v in HOME_POSE_WIRE]
    devs = []
    for got, want in zip(present, home_tick):
        d = (got - want) % 4096
        if d > 2048:
            d -= 4096
        devs.append(d)
    worst = max(abs(d) for d in devs)
    at_zero = all(abs(g - 2048) <= 3 for g in present)
    at_home = worst <= 3

    if args.pose == "zero" or (args.pose == "auto" and at_zero and not at_home):
        if at_zero:
            print("\n[✓] 正确: 零位标定已生效, 当前在零位姿势(全部 ≈2048)。"
                  "回到正常站立姿势再跑一次(不带 --pose zero)即可确认约定一致。")
        else:
            print("\n[!] 在零位姿势却未读到 2048 —— 还有偏差, 检查是否真的摆到了"
                  "零位(见 render_zero_pose.py 图)。")
    elif args.pose == "home" or (args.pose == "auto" and at_home and not at_zero):
        if at_home:
            print("\n[✓] 零位与约定一致(站立读数 == home ticks): 可以直接回放。")
        else:
            print(f"\n[!] 站立姿势读数与 home ticks 偏差最大 "
                  f"{worst*RAD_PER_TICK*180/math.pi:.1f}° —— 零位不对, 需重标。")
    else:
        print("\n[?] 无法用'auto'判断(读数既不像'全 2048'也不像 home ticks, "
              "或两者都说得通):")
        print("    - 说清楚: 鸭子现在是什么姿势?再用 --pose zero / --pose home 指定:")
        print("      --pose zero → 应全 2048(零位标定正确);")
        print("      --pose home → 应等于表格 home 列(约定一致)。")
        if at_zero:
            print("    - (注: 全 2048 可能是'标定正确且现在在零位姿势', 也可能是"
                  "'在站立姿势标的错'——两种都长这样, 所以需要你告诉它姿势。)")


if __name__ == "__main__":
    main()
