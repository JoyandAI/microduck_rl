#!/usr/bin/env python3
"""HL-2909-C001 (Feetech FT-SCS 总线) 标定数据一键读取脚本。

操作者（不需要会代码）：
  1. USB-TTL 转换器插到电脑；舵机接 12V 电源；转换器 TTL 3 线接舵机总线口
     （GND / 供电 / 信号，顺序以调试板丝印为准，多数调试板自带 5V 供电可不用外接
      —— 但标定电压就是输出电压，建议直接 12V 供电以获得真实数据）。
  2. 编译/安装本仓库后运行:
        uv run --with pyserial python scripts/read_hls_registers.py
     （或者直接: python3 scripts/read_hls_registers.py --port /dev/ttyUSB0）
  3. 脚本自动: 找串口 → PING 找舵机 ID → 读内存表全部相关寄存器 →
     输出人话对照表 + 标定 JSON (hls_registers.json)。把它发回即可。

协议依据: docs/ftscs_protocol.md (FT-SCS 自定义协议, 帧 0xFF 0xFF|ID|LEN|INSTR|PARA|CHECK,
Check = ~(ID+LEN+INSTR+PARA)&0xFF)；寄存器依据: docs/feetech_hls_memtable.md
(磁编码舵机为小端: 低字节在前)。
"""

from __future__ import annotations

import argparse
import binascii
import json
import sys
import time
from pathlib import Path

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    print("需要 pyserial: uv run --with pyserial python scripts/read_hls_registers.py")
    sys.exit(2)

# ---------------- FT-SCS 协议原语 ----------------

HEAD = b"\xFF\xFF"
INST_PING = 0x01
INST_READ = 0x02
INST_WRITE = 0x03
BROADCAST = 0xFE


def frame_checksum(pid: int, length: int, instr: int, params: bytes) -> int:
    return (~(pid + length + instr + sum(params))) & 0xFF


def build_ping(pid: int) -> bytes:
    return HEAD + bytes([pid, 2, INST_PING]) + bytes([frame_checksum(pid, 2, INST_PING, b"")])


def build_read(pid: int, addr: int, nbytes: int) -> bytes:
    params = bytes([addr, nbytes])
    ln = len(params) + 2
    return HEAD + bytes([pid, ln, INST_READ]) + params + bytes([frame_checksum(pid, ln, INST_READ, params)])


def read_frame(ser, timeout: float = 0.3):
    """Read one reply frame: FF FF ID LEN ERR? DATA... CHECK.

    FT-SCS reply: LEN = N_data + 2, and the frame body (ERR+DATA+CHECK) is
    exactly LEN bytes (PING reply FF FF 01 02 00 FC: LEN=2, ERR=00, CHECK=FC).
    Returns (id, err, dataBytes) or None.
    """
    ser.timeout = timeout
    head = ser.read(2)
    if len(head) != 2 or head != HEAD:
        return None
    pid = ser.read(1)
    ln = ser.read(1)
    if len(pid) != 1 or len(ln) != 1:
        return None
    body = ser.read(ln[0])  # ERR(1) + DATA + CHECK(1) == LEN
    if len(body) != ln[0]:
        return None
    err = body[0]
    data = body[1:-1]
    chk = body[-1]
    expect = (~(pid[0] + ln[0] + err + sum(data))) & 0xFF
    if chk != expect:
        return None  # checksum mismatch → treat as stale/noise
    return pid[0], err, data


def ping(ser, pid: int) -> bool:
    ser.reset_input_buffer()
    ser.write(build_ping(pid))
    return read_frame(ser) is not None


def read_regs(ser, pid: int, addr: int, nbytes: int):
    """读 nbytes 个字节, 返回 bytes 或 None。"""
    ser.reset_input_buffer()
    ser.write(build_read(pid, addr, nbytes))
    f = read_frame(ser, timeout=0.3)
    if f is None or f[0] != pid:
        return None
    return bytes(f[2][:nbytes])

# ---------------- 解码 ----------------

RPM_TO_RAD_S = 2 * 3.141592653589793 / 60.0
DEG_TO_RAD = 3.141592653589793 / 180.0


def u8(b, i): return b[i]
def u16_le(b, i): return b[i] | (b[i + 1] << 8)
def i16_le(b, i):
    v = u16_le(b, i)
    return v - 65536 if v & 0x8000 else v


# 读数计划: (地址, 字节数, 名称, 解码函数, 单位, 备注)
READ_PLAN = [
    (0, 1, "固件主版本", u8, "", "只读"),
    (1, 1, "固件次版本", u8, "", "只读"),
    (5, 1, "主ID", u8, "号", "读写"),
    (6, 1, "波特率档位", u8, "档", "0..7 = 1M..38.4k"),
    (9, 2, "最小角度限制", u16_le, "0.087°", "绝对位置模式"),
    (11, 2, "最大角度限制", u16_le, "0.087°", ""),
    (13, 1, "最高温度上限", u8, "°C", ""),
    (15, 1, "最低输入电压", u8, "0.1V", ""),
    (16, 2, "最大扭矩", u16_le, "0.1%", "上电赋给 48"),
    (21, 1, "位置环P(EPROM)", u8, "", "上电赋给 50"),
    (22, 1, "位置环D(EPROM)", u8, "", "上电赋给 51"),
    (23, 1, "位置环I(EPROM)", u8, "", "上电赋给 52"),
    (24, 1, "最小启动力", u8, "0.1%", ""),
    (26, 1, "正向不灵敏区", u8, "0.087°", "死区"),
    (27, 1, "负向不灵敏区", u8, "0.087°", "死区"),
    (28, 2, "保护电流", u16_le, "6.5mA", "上电赋给 44"),
    (30, 1, "角度分辨率", u8, "×", "传感器放大系数"),
    (31, 2, "位置偏移", i16_le, "0.087°", "中位校准"),
    (33, 1, "运行模式", u8, "", "0=位置伺服 1=恒速 2=恒流 3=PWM"),
    (40, 1, "扭矩开关", u8, "", "0=关 1=开 2=阻尼"),
    (41, 1, "加速度", u8, "8.7°/s²/LSB", "0=最大"),
    (42, 2, "目标位置", i16_le, "0.087°", "SRAM, BIT15方向"),
    (44, 2, "目标电流", i16_le, "6.5mA", "SRAM, BIT15方向"),
    (46, 2, "目标速度(运行速度)", i16_le, "0.732RPM", "SRAM, BIT15方向"),
    (48, 2, "转矩限制", u16_le, "0.1%", "SRAM, 默认=16"),
    (50, 1, "Kp(实际生效)", u8, "", "位置环 P × 1/8 缩放"),
    (51, 1, "Kd(实际生效)", u8, "", "位置环 D × 1/4 缩放"),
    (52, 1, "Ki(实际生效)", u8, "", "位置模式无效"),
    (55, 1, "锁标志", u8, "", "1=锁(EPROM掉电不存)"),
    (56, 2, "当前位置", i16_le, "0.087°", "反馈"),
    (58, 2, "当前速度", i16_le, "0.732RPM", "反馈"),
    (60, 2, "当前负载(占空比)", lambda b, i: u16_le(b, i) & 0x0FFF, "0.1%", "反馈, BIT10方向"),
    (62, 1, "当前输入电压", u8, "0.1V", "反馈 ★电源检查"),
    (63, 1, "当前温度", u8, "°C", "反馈"),
    (65, 1, "舵机状态", u8, "", "bit=错误"),
    (66, 1, "移动标志", u8, "", "bit0/bit1"),
    (67, 2, "目标位置(反馈)", i16_le, "0.087°", "反馈"),
    (69, 2, "当前电流", i16_le, "6.5mA", "反馈, BIT15方向"),
    (77, 1, "vFk(*10)", u8, "", "出厂参数(只读)"),
    (78, 1, "vKgI", u8, "", "出厂参数(只读)"),
    (79, 1, "pFk(*10)", u8, "", "出厂参数(只读)"),
    (80, 1, "移动速度阀值", u8, "", "出厂参数(只读)"),
    (81, 1, "DTs(ms)", u8, "ms", "出厂参数(只读) 控制周期"),
    (82, 1, "eFk(*10)", u8, "", "出厂参数(只读) 反电动势系数"),
    (83, 1, "Vk(ms)", u8, "ms", "出厂参数(只读)"),
    (84, 1, "最大速度限制", u8, "0.732RPM/LSB", "出厂参数(只读) ★"),
    (85, 1, "加速度限制", u8, "8.7°/s²/LSB", "出厂参数(只读) ★"),
    (86, 1, "加速度倍数", u8, "×", "出厂参数(只读) ★"),
]

# 只读寄存器在不使能扭矩时也能安全读; 读取本身不改变任何状态。


def decode_all(ser, pid: int, plan: list) -> dict:
    out = {}
    for addr, n, name, dec, unit, note in plan:
        # multiple bytes may cross 0x80 boundary; read contiguous chunk per entry
        raw = read_regs(ser, pid, addr, n)
        if raw is None:
            out[addr] = {"name": name, "raw": None, "error": "read failed"}
            continue
        try:
            val = dec(raw, 0) if n > 1 else raw[0]
        except Exception:
            val = binascii.hexlify(raw).decode()
        out[addr] = {"name": name, "raw": [int(x) for x in raw], "value": val,
                     "unit": unit, "note": note}
    return out


def calibration_summary(regs: dict) -> dict:
    def v(addr):
        r = regs.get(addr)
        return None if r is None or r.get("value") is None else r["value"]

    return {
        "firmware": f"{v(0)}.{v(1)}" if v(0) is not None else None,
        "kp_fw_actual": v(50),           # SRAM Kp (1/8 缩放前的固件值)
        "kd_actual": v(51),              # SRAM Kd (1/4 缩放前)
        "ki_actual": v(52),
        "mode": v(33),
        "deadband_deg": (v(26) or 0) * 0.087,
        "max_torque_percent": (v(16) or 0) / 10.0,
        "protection_current_A": (v(28) or 0) * 6.5 / 1000.0,
        "position_offset_deg": (v(31) or 0) * 0.087,
        "accel_raw_limit": v(85),
        "accel_coef": v(86),
        "accel_max_deg_s2": (v(85) or 0) * (v(86) or 0) * 8.7 if v(85) and v(86) else None,
        "accel_max_rad_s2": (((v(85) or 0) * (v(86) or 0) * 8.7) * DEG_TO_RAD) if v(85) and v(86) else None,
        "max_velocity_raw": v(84),
        "max_velocity_rad_s": (v(84) or 0) * 0.732 * RPM_TO_RAD_S if v(84) is not None else None,
        "dt_ms": v(81),
        "vk_ms": v(83),
        "efk_x10": v(82),
        "vfk_x10": v(77),
        "supply_voltage_V": (v(62) or 0) / 10.0,
        "temperature_C": v(63),
        "status_bits": v(65),
        "current_mA": (v(69) or 0) * 6.5 if v(69) is not None else None,
        "load_duty_percent": (v(60) or 0) / 10.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=None, help="串口设备, 如 /dev/ttyUSB0; 缺省自动探测")
    ap.add_argument("--baud", type=int, default=1000000, help="舵机波特率 (默认出厂 1Mbps)")
    ap.add_argument("--id", type=int, default=1, help="舵机 ID (默认 1; 0 自动扫描 0..253)")
    ap.add_argument("--out", default="hls_registers.json", help="输出 JSON 路径")
    ap.add_argument("--list-ports", action="store_true", help="只列出串口设备")
    args = ap.parse_args()

    if args.list_ports:
        for p in sorted(list_ports.comports(), key=lambda p: p.device):
            print(f"{p.device:<16} {p.description}")
        return 0

    ports = list(sorted(list_ports.comports(), key=lambda p: p.device))
    if args.port is None:
        if not ports:
            print("[!] 没有检测到任何串口设备。请插上 USB-TTL 转换器后重试 (或 --list-ports 查看)。")
            return 1
        args.port = ports[0].device
        print(f"[i] 自动选择串口: {args.port} ({ports[0].description})")
        if len(ports) > 1:
            print("[i] 检测到多个串口, 如不正确用 --port 指定 (--list-ports 查看列表)。")

    pyserial = serial.Serial
    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.3)
    except Exception as e:
        print(f"[!] 无法打开 {args.port}: {e}")
        return 1
    ser.reset_input_buffer()
    print(f"[i] 串口 {args.port} @ {args.baud} baud 已打开。舵机接 12V 电源了吗?")

    # 找到舵机 ID
    pid = args.id
    if pid == 0:
        found = []
        for cand in range(254):
            if ping(ser, cand):
                found.append(cand)
                print(f"[i] 发现舵机 ID = {cand}")
                if len(found) > 8:
                    break
        if not found:
            print("[!] 没有 PING 到任何舵机! 检查: 供电 9-14V / 信号线接对 (调试板SIG↔舵机SIG) / 波特率是否 1M (试 --baud 115200 / 500000 ...)")
            ser.close()
            return 1
        pid = found[0]
        if len(found) > 1:
            print(f"[i] 总线多台舵机, 只读第一台 ID={pid} (要读特定台用 --id)")
    else:
        if not ping(ser, pid):
            print(f"[!] ID={pid} 无响应。用 --id 0 自动扫描, 或用 --baud 调整波特率。")
            ser.close()
            return 1
        print(f"[i] 舵机 ID={pid} 应答正常。")

    # 先确认供电正常
    echk = read_regs(ser, pid, 62, 1)
    if echk:
        print(f"[i] 供电: {echk[0] / 10.0:.1f} V (应在 9-14V 之间)")
    else:
        print("[!] 警告: 读电压失败, 数据可能不可靠。")

    print("[i] 开始读取寄存器表...")
    regs = decode_all(ser, pid, READ_PLAN)
    summary = calibration_summary(regs)

    # 人话输出
    print("\n" + "=" * 72)
    print("HL-2909-C001 标定数据 (人话版)")
    print("=" * 72)
    for addr, info in regs.items():
        v = info.get("value")
        if v is None:
            print(f"  {addr:>3}  {info['name']:<18}  读取失败")
        else:
            u = info.get("unit", "")
            print(f"  {addr:>3}  {info['name']:<18}  {v!s:>12} {u:<12} {info.get('note','')}")
    print("\n[★ 标定摘要 → BAM 参数]")
    for k, v in summary.items():
        print(f"  {k:<24} = {v}")

    out_path = Path(args.out)
    payload = {
        "meta": {
            "port": args.port, "baud": args.baud, "servo_id": pid,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "protocol": "FT-SCS", "model": "HL-2909-C001",
        },
        "registers": regs,
        "calibration_summary": summary,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[i] 已保存: {out_path.resolve()}  ← 把这个文件发回即可")
    ser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
