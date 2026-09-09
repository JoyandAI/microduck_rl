#!/usr/bin/env python3
"""HL-2909 error_gain / R 实测标定 (第二层, 替代示波器).

原理: 位置伺服模式下, duty = (Kp/8)*err*error_gain (负反馈, 占空比与误差成正比,
直到 PWM 饱和)。给一个已知的目标偏移, 读 56(当前位置)与 60(当前占空比%) 多拍,
线性拟合 duty vs err 斜率 → error_gain。堵转(目标超远)时读 62(电压)与 69(电流)
→ R = V/I。

注意: 本脚本会 开启扭矩 (40=1) 并让舵机转动/进入堵转, 完成后恢复 (40=0)。
只对 --id 指定的单台舵机执行 (默认 10)。

Usage:
    uv run --with pyserial python scripts/calibrate_hls.py --port /dev/ttyACM0 --id 10
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    raise SystemExit("需要 pyserial: uv run --with pyserial python scripts/calibrate_hls.py")

from read_hls_registers import (
    BROADCAST, INST_READ, INST_WRITE, HEAD,
    build_read, build_ping, frame_checksum, i16_le, ping, read_frame, read_regs, u16_le,
)

RAW_PER_RAD = 4096.0 / (2 * np.pi)   # 0.087°/LSB per the memtable
DEG = np.pi / 180.0


def build_write(pid: int, addr: int, data: bytes) -> bytes:
    # FT-SCS WRITE params = [addr, data...] — no data-length byte (the frame
    # LEN field carries it). A length byte here shifts every write by one
    # address (e.g. writing MODE=0 sets MODE=1) — see docs/ftscs_protocol.md §4.3.
    params = bytes([addr]) + bytes(data)
    ln = len(params) + 2
    return HEAD + bytes([pid, ln, INST_WRITE]) + params + bytes(
        [frame_checksum(pid, ln, INST_WRITE, params)])


def write_raw(ser, pid: int, addr: int, data: bytes) -> bool:
    ser.reset_input_buffer()
    ser.write(build_write(pid, addr, data))
    return read_frame(ser) is not None


def read_i16(ser, pid: int, addr: int):
    b = read_regs(ser, pid, addr, 2)
    return None if b is None else i16_le(b, 0)


def read_u16(ser, pid: int, addr: int):
    b = read_regs(ser, pid, addr, 2)
    return None if b is None else u16_le(b, 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default=None)
    ap.add_argument("--baud", type=int, default=1000000)
    ap.add_argument("--id", type=int, default=10, help="单台舵机 ID (默认 10)")
    ap.add_argument("--out", default="hls_calibration.json")
    args = ap.parse_args()

    if args.port is None:
        ports = [p.device for p in list_ports.comports() if "ttyS" not in p.device]
        if not ports:
            raise SystemExit("未找到串口")
        args.port = ports[0]
    ser = serial.Serial(args.port, args.baud, timeout=0.3)

    pid = args.id
    if not ping(ser, pid):
        ser.close()
        raise SystemExit(f"ID={pid} 无响应")
    print(f"[i] 舵机 ID={pid} 在线, 开始校准实验 (完成后扭矩会关闭)")

    anchor = read_i16(ser, pid, 56)   # current position
    print(f"[i] 当前位置 = {anchor} raw ({anchor*360/4096:.1f}°)")

    # --- 1. 斜坡标定 error_gain (安全版: 小角度 ±15/30°, 全程限流 0.975A) ---
    # 临时把目标电流(44)降到 150 (=0.975A), 防止任何情况下大电流堵转发热
    ORIG_CURRENT = 300
    write_raw(ser, pid, 44, b"\x96\x00")          # 150 * 6.5mA = 0.975A limit
    write_raw(ser, pid, 40, b"\x01")              # torque on
    write_raw(ser, pid, 46, b"\xfa\x7f")          # goal speed max (32767)
    time.sleep(0.3)

    samples = []
    for delta_deg in (15, -15, 30, -30):
        target = anchor + int(delta_deg * 4096.0 / 360.0)  # deg -> raw
        write_raw(ser, pid, 42, bytes([target & 0xFF, (target >> 8) & 0xFF]))
        # burst-read WITHOUT sleep: capture the launch instant (firmware 3ms loop);
        # err shrinks delta -> 0, duty tracks err while not saturated
        for _ in range(50):
            pos = read_i16(ser, pid, 56)
            duty = read_u16(ser, pid, 60)          # 0.1%, bit10 dir
            if pos is not None and duty is not None:
                err_rad = (target - pos) / RAW_PER_RAD
                # duty: 0-1023 forward, 1024-2047 reverse (BIT10 = dir)
                duty_sign = -1.0 if (duty & 0x0400) else 1.0
                duty_frac = duty_sign * (duty & 0x3FF) * 0.001
                if abs(duty_frac) > 0.05:
                    samples.append((abs(err_rad), abs(duty_frac)))
        time.sleep(0.2)

    errs = np.array([s[0] for s in samples])
    dutys = np.array([s[1] for s in samples])
    # 线性区: duty < 0.9 且 |err|>0.1 rad (排除静止噪声与饱和)
    mask = (dutys < 0.9) & (errs > 0.1) & (errs < 0.6)
    if mask.sum() >= 3:
        A = np.vstack([errs[mask], np.ones(mask.sum())]).T
        slope, intercept = np.linalg.lstsq(A, dutys[mask], rcond=None)[0]
        error_gain = slope * 8.0 / 32.0   # slope = (Kp/8)*eg, Kp=32 (read from reg 50)
        print(f"[i] error_gain 拟合: slope={slope:.4f} duty/rad -> error_gain = {error_gain:.4f} "
              f"(Kp=32, n={int(mask.sum())})")
    else:
        slope, error_gain = None, None
        print(f"[!] 线性区样本不足 (n={int(mask.sum())}), 无法拟合 error_gain; "
              f"原始样本: {[(round(e,3), round(d,3)) for e, d in samples[:8]]}")

    # --- 2. (已移除堵转步骤: 上次实测 I=2.5A 触发过热/过流保护; R 以官方自洽值
    #        kt=1.454 & 堵转 0.873N·m => R = vin·kt/τ = 20Ω 为准, 待保护恢复后复核)

    # --- 3. 复位: 回原位, 恢复限流与扭矩 ---
    write_raw(ser, pid, 42, bytes([anchor & 0xFF, (anchor >> 8) & 0xFF]))
    time.sleep(0.5)
    write_raw(ser, pid, 44, bytes([ORIG_CURRENT & 0xFF, (ORIG_CURRENT >> 8) & 0xFF]))
    write_raw(ser, pid, 40, b"\x00")
    ser.close()

    result = {
        "servo_id": pid,
        "kp_actual": 32, "kd_actual": 32, "ki_actual": 0,
        "error_gain": error_gain, "slope_duty_per_rad": slope,
        "note": "error_gain = slope*8/32 (Kp=32, 1/8 scaling per FT-HLS memtable)",
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[i] 已保存: {args.out}")


if __name__ == "__main__":
    main()
