#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立摆锤台架 —— 标定前准备脚本(你执行)。

一台单独 HL-2909(专用 ID, 如 1) + 摆臂 + 砝码 组成 BAM 摆锤实验台。
本脚本: ①扫描总线找舵机 ②(可选)把 ID 改为目标值 ③打印出厂参数/模式/电压
④确认位置伺服模式 & 量程是否 0-4095(多圈)。

用法:
    /usr/bin/python3 scripts/setup_bench_servo.py --port /dev/ttyACM0
    /usr/bin/python3 scripts/setup_bench_servo.py --port /dev/ttyACM0 --find-id 23 --set-id 1
安全: 只读+写 ID(5 号寄存器)/模式(33=0); 全程不使能扭矩, 不会转动。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_hls_registers import read_regs, write_regs, u16_le  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--find-id", type=int, default=None, help="扫描找到的 ID(打印列表)")
    ap.add_argument("--set-id", type=int, default=None, help="把舵机 ID 改成这个值(新 ID, 1-253)")
    args = ap.parse_args()

    import serial
    ser = serial.Serial(args.port, 1_000_000, timeout=0.15)

    def r8(pid, a):
        x = read_regs(ser, pid, a, 1)
        return None if x is None else x[0]

    def r16(pid, a):
        x = read_regs(ser, pid, a, 2)
        return None if x is None else u16_le(x, 0)

    # 1) 扫描
    print("[i] 扫描总线…")
    found = []
    for cand in range(254):
        if r8(cand, 62) is not None:
            found.append(cand)
    print(f"[i] 发现舵机 ID: {found}")

    if args.find_id is not None:
        pid = args.find_id
        if pid not in found:
            print(f"[!] 未发现 ID={pid}")
            return 1
    elif len(found) == 1:
        pid = found[0]
    else:
        print("[!] 请用 --find-id 指定要配置的 ID")
        return 1

    # 2) 改 ID(如需)
    if args.set_id is not None and args.set_id != pid:
        ok = write_regs(ser, pid, 5, [args.set_id & 0xFF])
        time.sleep(0.15)
        new_found = [c for c in range(254) if r8(c, 62) is not None]
        if args.set_id in new_found:
            print(f"[i] ID 已改: {pid} → {args.set_id}")
            pid = args.set_id
        else:
            print("[!] 改 ID 后未发现新 ID, 请断电重启再试")
            return 1

    # 3) 状态打印
    print(f"\n===== 舵机 ID={pid} =====")
    for a, name in ((0, "固件主版本"), (1, "固件次版本"), (3, "型号号"), (16, "最大扭矩(0.1%)"),
                    (21, "Kp"), (22, "Kd"), (23, "Ki"), (28, "保护电流(6.5mA)"),
                    (30, "角度分辨率"), (31, "位置偏移(0.087°×?)"), (33, "运行模式"),
                    (56, "当前位置(LSB)"), (62, "电压(0.1V)"), (63, "温度(°C)")):
        v = r8(pid, a) if a < 2 or a in (13, 21, 22, 23, 28, 30, 33, 62, 63) else r16(pid, a)
        print(f"  reg{a:>2} {name:<24} = {v}")

    mode = r8(pid, 33)
    if mode != 0:
        print(f"[!] 运行模式={mode}(需要 0=位置伺服) → 写入 33=0…")
        write_regs(ser, pid, 33, [0])
        time.sleep(0.2)
        print(f"    回读 mode={r8(pid, 33)}")
    lo, hi = r16(pid, 9), r16(pid, 11)
    print(f"[i] 位置限位: min={lo} max={hi} (0..4095 = 多圈不受限)")
    print("[i] 完成。下一步: 装摆锤台架 → 称量/实测几何 → record_pendulum_bench.py")
    ser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
