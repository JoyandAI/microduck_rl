"""record_pendulum_bench.py 的零位守卫 / reg31 自动居中回归测试(无需硬件)。

锁定的不变量:
- zero_belt 按所选轨迹真实振幅计算安全带, 默认四轨迹 = [1211, 2884](±64 LSB 余量);
- center_zero_reg31 成功: 保留新 reg31、结尾 reg46=max、返回 (True, msg, val);
- center_zero_reg31 失败: 恢复**进入时的原值**(不是 0);
- 符号幅值双编码(BIT15 / BIT11)都能被正确回退取用。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "record_pendulum_bench",
    Path(__file__).resolve().parent.parent / "scripts" / "record_pendulum_bench.py",
)
rpb = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rpb)  # noqa


def _decode_reg31(v: int) -> int:
    """符号幅值解码: BIT15 符号(BIT11 约定回退), 其余位为幅值。"""
    if v & 0x8000:
        return -(v & 0x7FFF)
    if v & 0x0800:
        return -(v & 0x07FF)
    return v & 0x7FFF


def make_fake(base_pos: int, reg31_orig: int = 0, *,
              apply_offset: bool = True, accept_bit15: bool = True,
              accept_bit11: bool = True, motion_ok: bool = True):
    """极简舵机仿真: reg56 = (raw ± reg31幅值) % 4096; 扭矩开启时位置=reg42。"""
    st = {"reg31": reg31_orig, "base": base_pos, "apply": apply_offset,
          "writes": [], "accept_bit15": accept_bit15, "accept_bit11": accept_bit11,
          "torque": 0, "goal": 0, "reg46": 0}

    def read_regs(ser, pid, addr, nbytes):
        if addr == 56:
            if st["torque"] and motion_ok:
                pos = st["goal"]
            else:
                off = _decode_reg31(st["reg31"]) if st["apply"] else 0
                pos = (st["base"] + off) % 4096
            return bytes([pos & 0xFF, (pos >> 8) & 0xFF])
        if addr == 31:
            return bytes([st["reg31"] & 0xFF, (st["reg31"] >> 8) & 0xFF])
        if addr in (46, 42, 41):
            return bytes([st.get(addr, 0) & 0xFF, (st.get(addr, 0) >> 8) & 0xFF])
        if addr == 40:
            return bytes([st["torque"]])
        return None

    def write_regs(ser, pid, addr, data):
        data = list(data)
        st["writes"].append((addr, list(data)))
        if addr == 31:
            v = data[0] | (data[1] << 8)
            if (v & 0x8000) and not st["accept_bit15"]:
                return False
            if (v & 0x0800) and not st["accept_bit11"]:
                return False
            st["reg31"] = v
        elif addr == 40:
            st["torque"] = data[0]
        elif addr == 42:
            st["goal"] = data[0] | (data[1] << 8)
        elif addr == 46:
            st["reg46"] = data[0] | (data[1] << 8)
        return True

    return st, read_regs, write_regs


@pytest.fixture()
def bench(monkeypatch):
    """把仿真舵机安装进模块命名空间(替换全局 read_regs/write_regs/time.sleep)。"""
    monkeypatch.setattr(rpb.time, "sleep", lambda *a, **k: None)

    def _install(st, rr, ww):
        monkeypatch.setattr(rpb, "read_regs", rr)
        monkeypatch.setattr(rpb, "write_regs", ww)
        return st

    return _install


def test_zero_belt_default_trajs():
    lo, hi, amp = rpb.zero_belt(["sin_time_square", "sin_sin", "lift_and_drop", "up_and_down"])
    assert amp == 1147
    assert (lo, hi) == (1211, 2884)
    assert lo <= 2048 <= hi  # 自动居中目标 2048 永远在安全带内


def test_zero_belt_adapts_to_selection():
    lo, hi, amp = rpb.zero_belt(["nothing"], margin_lsb=10)
    assert amp == 0 and lo == 10 and hi == 4085


def test_center_success_bit15(bench):
    st, rr, ww = make_fake(base_pos=600, reg31_orig=0x1234)
    bench(st, rr, ww)
    ok, msg, val = rpb.center_zero_reg31(object(), 1)
    assert ok and val == 0x05A8 and "居中成功" in msg
    # 成功: 保留新值(不恢复原值), 且 reg46 结尾为 max
    assert st["reg31"] == 0x05A8
    assert (46, [0xFF, 0x7F]) in st["writes"]


def test_center_fallback_bit11(bench):
    # 主编码(BIT15)被固件拒绝 → 自动回退 BIT11 编码
    st, rr, ww = make_fake(base_pos=3000, reg31_orig=0, accept_bit15=False)
    bench(st, rr, ww)
    ok, msg, val = rpb.center_zero_reg31(object(), 1)
    assert ok and val == 0x0BB8
    assert st["reg31"] == 0x0BB8


def test_center_failure_restores_original_reg31(bench):
    # 两种编码都被拒 → 失败, 且恢复**原值 0x0777**(历史偏移), 而不是写 0
    st, rr, ww = make_fake(base_pos=3000, reg31_orig=0x0777,
                           accept_bit15=False, accept_bit11=False)
    bench(st, rr, ww)
    ok, msg, val = rpb.center_zero_reg31(object(), 1)
    assert not ok and val is None and "未能校准" in msg
    assert st["reg31"] == 0x0777


def test_center_failure_restores_when_motion_fails(bench):
    # 读数/编码都成功, 但运动验证失败(mode4 不认 reg31) → 恢复原值
    st, rr, ww = make_fake(base_pos=600, reg31_orig=0, motion_ok=False)
    bench(st, rr, ww)
    ok, msg, val = rpb.center_zero_reg31(object(), 1)
    assert not ok and val is None and "运动验证失败" in msg
    assert st["reg31"] == 0


def test_center_probe_refused(bench):
    # reg31 完全无效(d=0) → 失败并恢复原值
    st, rr, ww = make_fake(base_pos=600, reg31_orig=0x0011, apply_offset=False)
    bench(st, rr, ww)
    ok, msg, val = rpb.center_zero_reg31(object(), 1)
    assert not ok and val is None and "无预期效果" in msg
    assert st["reg31"] == 0x0011
