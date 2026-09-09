#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成《独立摆锤台架 · 软件方案与测试取值方案》单文件 HTML(精简版)。

    .venv/bin/python scripts/make_pendulum_bench_runbook_html.py
输出: docs/pendulum_bench_runbook.html
"""

from __future__ import annotations

import math
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "pendulum_bench_runbook.html"

# 台架几何(仅"建议值", 机械工程师可重设计; 软件只要求可实测/可称重)
BASE_L, BASE_T = 120, 8
COL_H = 230
AX_H = 8 + COL_H + 4 + 10      # ≈252
ARM_L = 200


def fig_physics() -> str:
    """原理小图(示意): 独立台架 + 摆臂 + 已知量公式。"""
    ax = 170
    return f'''<svg viewBox="0 0 620 470" class="fig" role="img" aria-label="原理示意">
<g transform="translate(20,30) scale(1.45)">
 <line x1="-10" y1="0" x2="300" y2="0" stroke="#333" stroke-width="3"/>
 <rect x="0" y="0" width="120" height="8" fill="#c9c9c9" stroke="#333" stroke-width="1"/>
 <rect x="32" y="8" width="56" height="{COL_H}" fill="#d8d0c2" stroke="#333" stroke-width="1"/>
 <rect x="24" y="{8+COL_H}" width="44" height="31" fill="#dcd6cc" stroke="#333" stroke-width="1"/>
 <line x1="20" y1="{AX_H}" x2="96" y2="{AX_H}" stroke="#c00000" stroke-width="2"/>
 <circle cx="{ax-78}" cy="{AX_H}" r="3" fill="#c00000"/>
 <circle cx="{ax-78}" cy="{AX_H}" r="7" fill="#e8c87a" stroke="#333"/>
 <line x1="{ax-84}" y1="{AX_H}" x2="{ax-84}" y2="{AX_H+200}" stroke="#c00000" stroke-width="3"/>
 <rect x="{ax-84-10}" y="{AX_H+193}" width="20" height="13" fill="#a8d8a8" stroke="#333"/>
 <text x="{ax-84}" y="{AX_H+220}" font-size="8" text-anchor="middle">m_tip(50/100/150g)</text>
 <line x1="{ax-84-16}" y1="{AX_H}" x2="{ax-84-16}" y2="{AX_H+80}" stroke="#777" stroke-dasharray="4 3"/>
 <text x="{ax-78}" y="{AX_H-8}" font-size="8">θ</text>
 <circle cx="{ax-46}" cy="{AX_H}" r="5" fill="#eee" stroke="#333"/>
 <text x="14" y="{8+COL_H+40}" font-size="8">立柱+U夹(轴高≈252, 机械工程师可重设计)</text>
 <text x="14" y="290" font-size="8">摆臂长 L=轴心→砝码质心(名义 0.20m, 实测填入)</text>
</g>
<g font-size="14">
 <text x="420" y="70" text-anchor="middle" font-weight="bold">已知量 = 全称重/全实测(无未知)</text>
 <text x="420" y="100" text-anchor="middle">M = m_tip·L² + m_arm·L²/3 + m_hub·r_hub²</text>
 <text x="420" y="124" text-anchor="middle">B_max = (m_tip + m_arm/2)·g·L + m_hub·g·r_hub</text>
 <text x="420" y="156" text-anchor="middle" fill="#c00000">L_eq = M·g/B_max    m_eq = B_max²/(M·g²)</text>
 <text x="420" y="182" text-anchor="middle" fill="#555">(与 bam Pendulum 动力学精确等价, 软件零改动)</text>
 <text x="420" y="214" text-anchor="middle" fill="#555">转子惯量 → bam.fit 的 armature 一并拟合</text>
 <rect x="300" y="240" width="240" height="70" rx="10" fill="#ffebeb" stroke="#c00000"/>
 <text x="420" y="266" text-anchor="middle" font-size="13" fill="#c00000">⚠ 力矩预算 m_tip·g·L ≤ 0.35 N·m</text>
 <text x="420" y="290" text-anchor="middle" font-size="12.5" fill="#c00000">L=0.2m → m_tip ≤ 170g; 档位取 50/100/150g</text>
</g>
</svg>'''


def fig_trajectories() -> str:
    t = [i * 6 / 399 for i in range(400)]

    def cubic(kf, x):
        for i in range(len(kf) - 1):
            t0, x0, d0 = kf[i]
            t1, x1, d1 = kf[i + 1]
            if t0 <= x <= t1:
                h = t1 - t0
                s = (x - t0) / h
                return (x0 * (2 * s**3 - 3 * s**2 + 1) + x1 * (-2 * s**3 + 3 * s**2)
                        + (d0 * h) * (s**3 - 2 * s**2 + s) + (d1 * h) * (s**3 - s**2))
        return kf[-1][1]

    def P(y):
        return " ".join(f"{80 + x * 800 / 6:.1f},{150 - max(-1.57, min(1.57, y)) * 73:.1f}"
                        for x, y in zip(t, y))

    y_lad = [cubic([[0, 0, 0], [2, -math.pi / 2, 0]], x) for x in t]
    y_free = [-math.pi / 2 + 1.2 * math.exp(-0.55 * (x - 2)) * math.sin(3.1 * (x - 2))
              if x >= 2 else 0 for x in t]
    y_sin = [math.sin(x ** 2) for x in t]
    y_ud = [cubic([[0, 0, 0], [3, math.pi / 2, 0], [6, 0.8 * math.pi / 2, 0]], x) for x in t]
    lad_on = [(x, y) for x, y in zip(t, y_lad) if x <= 2]
    lad_off = [(x, y) for x, y in zip(t, y_free) if x >= 2]
    po = " ".join(f"{80 + a * 800 / 6:.1f},{150 - b * 73:.1f}" for a, b in lad_on)
    pf = " ".join(f"{80 + a * 800 / 6:.1f},{150 - b * 73:.1f}" for a, b in lad_off)

    def grid():
        return '''<g font-size="12" fill="#777">
<line x1="80" y1="35" x2="880" y2="35" stroke="#ddd"/><line x1="80" y1="150" x2="880" y2="150" stroke="#ccc" stroke-dasharray="4 4"/>
<line x1="80" y1="265" x2="880" y2="265" stroke="#ddd"/>
<text x="70" y="40" text-anchor="end">+1.57</text><text x="70" y="154" text-anchor="end">0</text><text x="70" y="269" text-anchor="end">-1.57</text></g>'''

    return f'''<svg viewBox="0 0 960 880" class="fig" role="img" aria-label="轨迹图">
<g>
 <rect x="60" y="18" width="840" height="262" rx="8" fill="#fafafa" stroke="#ccc"/>
 <rect x="80" y="185" width="267" height="80" fill="#ffe0e0" opacity="0.55"/>
 <text x="120" y="205" font-size="12" fill="#c00000">扭矩ON (升到 −90°)</text>
 {grid()}
 <polyline points="{po}" fill="none" stroke="#c00000" stroke-width="2.6"/>
 <polyline points="{pf}" fill="none" stroke="#c00000" stroke-width="1.6" stroke-dasharray="5 4"/>
 <text x="620" y="40" font-size="13" fill="#c00000">2s 后 扭矩OFF → 自由落体(背驱/Stribeck)</text>
 <text x="70" y="275" font-size="14" font-weight="bold">① lift_and_drop</text>
</g>
<g transform="translate(0,300)">
 <rect x="60" y="18" width="840" height="262" rx="8" fill="#fafafa" stroke="#ccc"/>{grid()}
 <polyline points="{P(y_sin)}" fill="none" stroke="#1f77b4" stroke-width="2.2"/>
 <text x="600" y="42" font-size="13" fill="#1f77b4">sin(t²): 一次跑完全速度谱(主推荐)</text>
 <text x="70" y="275" font-size="14" font-weight="bold">② sin_time_square</text>
</g>
<g transform="translate(0,600)">
 <rect x="60" y="18" width="840" height="262" rx="8" fill="#fafafa" stroke="#ccc"/>{grid()}
 <polyline points="{P(y_ud)}" fill="none" stroke="#2ca02c" stroke-width="2.2"/>
 <text x="560" y="44" font-size="13" fill="#2ca02c">0→90°→72°: 静态摩擦/负载相关(低速段)</text>
 <text x="70" y="275" font-size="14" font-weight="bold">③ up_and_down</text>
</g>
<text x="480" y="872" font-size="14" text-anchor="middle" fill="#555">t [s] · pendulum 帧 0 = 杆垂直向下 · 舵机 LSB: goal = q_zero + sign·θ/LSB_RAD</text>
</svg>'''


def fig_pipeline() -> str:
    return '''<svg viewBox="0 0 960 170" class="fig">
<defs><marker id="mf" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
<path d="M0,0 L6,3 L0,6 z" fill="#666"/></marker></defs>
<rect x="10" y="15" width="200" height="62" rx="10" fill="#eef" stroke="#446" stroke-width="2"/>
<text x="110" y="42" font-size="14" text-anchor="middle">record_pendulum_bench.py</text>
<text x="110" y="62" font-size="12" text-anchor="middle">(你, /usr/bin/python3)</text>
<line x1="215" y1="46" x2="290" y2="46" stroke="#666" stroke-width="3" marker-end="url(#mf)"/>
<rect x="295" y="15" width="200" height="62" rx="10" fill="#efe" stroke="#464" stroke-width="2"/>
<text x="395" y="42" font-size="14" text-anchor="middle">log JSON ×45 + manifest</text>
<text x="395" y="62" font-size="12" text-anchor="middle">3轨迹×3砝码×5重复</text>
<line x1="500" y1="46" x2="575" y2="46" stroke="#666" stroke-width="3" marker-end="url(#mf)"/>
<rect x="580" y="15" width="200" height="62" rx="10" fill="#fef" stroke="#664" stroke-width="2"/>
<text x="680" y="42" font-size="14" text-anchor="middle">bam.fit(m1 与 m6)</text>
<text x="680" y="62" font-size="12" text-anchor="middle">CMA-ES, CPU 10-40min/个</text>
<line x1="785" y1="46" x2="860" y2="46" stroke="#666" stroke-width="3" marker-end="url(#mf)"/>
<rect x="865" y="15" width="90" height="62" rx="10" fill="#ffe" stroke="#c90" stroke-width="2"/>
<text x="910" y="46" font-size="13" text-anchor="middle">MAE</text><text x="910" y="64" font-size="12" text-anchor="middle">报告</text>
<rect x="295" y="105" width="560" height="46" rx="10" fill="#faf5ec" stroke="#a80" stroke-width="1.4"/>
<text x="575" y="126" font-size="13" text-anchor="middle">验收门: MAE &lt; 0.157 rad(≈10%×π/2, 相对单摆幅) — 过门选较小档落地</text>
<text x="575" y="144" font-size="12" text-anchor="middle" fill="#777">落地 = 建议 + 你点头 → vendor/bam/params/hls2909/mN.json + constants + 冒烟</text>
</svg>'''


CSS = '''
<style>
:root { --accent:#c00000; }
* { box-sizing:border-box; }
body { font-family:"Noto Sans CJK SC","Microsoft YaHei",sans-serif; margin:0;
       background:#fdfdfc; color:#222; line-height:1.7; }
.wrap { max-width:1060px; margin:0 auto; padding:24px 28px 80px; }
h1 { color:var(--accent); border-bottom:4px solid var(--accent); padding-bottom:10px; }
h2 { margin-top:40px; border-left:6px solid var(--accent); padding-left:12px; }
.fig { width:100%; height:auto; background:#fff; border:1px solid #e4e4e4; border-radius:10px;
       margin:12px 0 6px; box-shadow:0 1px 4px rgba(0,0,0,.06); }
.photo { display:block; width:100%; height:auto; margin:14px 0 22px; border:1px solid #d9d9d9; }
table { border-collapse:collapse; width:100%; margin:12px 0 20px; }
th,td { border:1px solid #e4e4e4; padding:8px 12px; text-align:left; font-size:14px; }
th { background:#f5f0e8; }
code,pre { font-family:Consolas,monospace; background:#f4f2ee; border-radius:6px; }
code { padding:1px 6px; font-size:13.5px; }
pre { padding:12px 16px; overflow-x:auto; font-size:13px; }
.note { background:#fff8e1; border:1px solid #e8c96a; border-radius:8px; padding:10px 14px; margin:14px 0; }
.warn { background:#ffebeb; border:1px solid #c00000; border-radius:8px; padding:12px 16px; margin:16px 0; }
.ok { background:#e9f7ec; border:1px solid #7c3; border-radius:8px; padding:10px 14px; margin:14px 0; }
</style>
'''

BODY = '''
<h1>独立摆锤台架 —— 软件方案 + 测试取值方案</h1>
<p>📄 <a href="pendulum_bench_experiment_design.html"><b>完整实验设计文档</b></a>(步骤/变量控制/采集参数/决策规则) · 本页为速查</p>
<p>用途: 用<b>单独一只 HL-2909 + 摆臂 + 砝码</b>的摆锤台架, 标定全套动力学
(kt / R / armature / error_gain / 摩擦 m1→m6)。<b>机械结构由机械工程师重新设计
(本页只给"软件需要的最小几何约束"); 软件与测试取值按本页执行。</b></p>

<h2>实验实物与全流程</h2>
<p>下面两张图用来看懂台架、摆角和四种轨迹。图内个别示例数字不是本次最终参数；<b>实际操作以本手册后面的“本次唯一参数表”和命令为准。</b></p>
<img class="photo" src="6b341df0-14af-4203-b0a2-35cba4876caf.png" alt="HL-2909 摆锤台架、摆角和全部实验流程">
<img class="photo" src="41b5c9d2-f096-4ff0-9ab0-8f2381968c32.png" alt="HL-2909 摆锤实验装置与四种轨迹详图">

<h2>1. 机械: 只给工程师 5 条硬约束(其余自由发挥)</h2>
<table>
<tr><th>#</th><th>约束</th><th>原因(软件依赖)</th></tr>
<tr><td>1</td><td>输出轴<b>水平</b>, 摆臂在竖直平面内 ±90° 自由摆动(无碰撞)</td><td>重力矩 B(θ) 需要 sin θ 形式</td></tr>
<tr><td>2</td><td>轴心距地面/桌面 ≥ 臂长 + 50mm(如臂 200 → 轴高 ≥ 250)</td><td>摆臂不穿地</td></tr>
<tr><td>3</td><td><b>轴心→端部砝码质心 L 可实测</b>(卡尺 ±1mm); 全部<b>移动件可拆下称重</b></td><td>m_tip/m_arm/m_hub/L 是配方输入</td></tr>
<tr><td>4</td><td>台架固定牢靠(重/压胶垫), 振动 < 1°(摆动时不摇晃)</td><td>数据干净; 固定不良整条作废</td></tr>
<tr><td>5</td><td>舵机总线只接这一只(建议 ID=1; 鸭子总线断开)</td><td>避免误写鸭子上的舵机</td></tr>
</table>
<div class="note">参考尺寸(已有可打印 cad/ 目录, 仅供工程师参考): 底座 120×80×8,
立柱 60×24×230, U 夹 44×31×31(内腔 34×23×20), 摆杆 Ø6×200, 托盘 Ø48(M6)。</div>

<h2>2. BOM(可直接交给采购/机械)</h2>
<table>
<tr><th>类别</th><th>物料与建议规格</th><th>数量</th><th>验收要点</th></tr>
<tr><td>被测件</td><td>HL-2909-C001 + 匹配花键的金属舵盘</td><td>1 套</td><td>优先用原厂配套舵盘，不要猜花键齿数</td></tr>
<tr><td>骨架</td><td>2040 铝型材约 300 mm + 180×120×8 mm 铝底板 + 角码</td><td>各 1</td><td>轴心高 250–280 mm；比打印立柱更稳</td></tr>
<tr><td>台架固定</td><td>C 型桌夹，开口≥40 mm</td><td>2</td><td>底板两侧各一，不能只靠自重</td></tr>
<tr><td>摆臂</td><td>Φ6 mm 碳纤管/铝管，约 220 mm</td><td>1</td><td>最终 L 按轴心到端部总质量质心实测</td></tr>
<tr><td>连接件</td><td>舵盘-杆夹、舵机夹座、端部砝码托</td><td>各 1</td><td><code>cad/pendulum_bench/</code> 作参考；打印用 PETG/尼龙，不建议 PLA</td></tr>
<tr><td>负载</td><td>钢垫片/小砝码，含托盘后端部总质量 50/100/150 g</td><td>3 档</td><td>每档整套称重 0.1 g，可不等于名义值</td></tr>
<tr><td>紧固</td><td>M3/M4 螺栓、M6 螺杆、M6 尼龙锁紧螺母、平垫、螺纹胶</td><td>1 批</td><td>砝码必须机械防松，禁止只用胶粘</td></tr>
<tr><td>供电</td><td>12 V 稳压电源≥3 A、3 A 保险丝、串联急停开关</td><td>各 1</td><td>初始 12.0 V；急停直接切断 12 V</td></tr>
<tr><td>通信</td><td>支持 FT-SCS 半双工的 USB-TTL 转接板和线束</td><td>1</td><td>1 Mbps；电源、TTL、舵机必须共地</td></tr>
<tr><td>测量</td><td>0.1 g 电子秤、0.01 mm 数显卡尺、小水平仪</td><td>各 1</td><td>质量记到 0.1 g，L 记到 1 mm</td></tr>
<tr><td>防护</td><td>3 mm 透明 PC 挡板约 400×400 mm + 护目镜</td><td>各 1</td><td>挡板放在摆动平面和操作者之间</td></tr>
</table>

<h2>3. 测试取值方案(核心表)</h2>
<h3>2.1 几何/质量取值(每次装夹后称、量一次)</h3>
<table>
<tr><th>量</th><th>建议值</th><th>精度</th><th>备注</th></tr>
<tr><td>摆长 L(轴→砝码质心)</td><td><b>0.150 m</b></td><td>±1 mm</td><td>砝码厚度中心算质心</td></tr>
<tr><td>m_arm 摆杆(碳纤 Ø6)</td><td>≈12–25 g(越轻越好)</td><td>0.1 g</td><td>均匀细杆假设</td></tr>
<tr><td>m_hub(舵机臂+夹头+螺丝)</td><td>≈15–30 g</td><td>0.1 g</td><td>r_hub 由臂孔位定(≈8–15 mm)</td></tr>
<tr><td>r_hub(轴→臂质心)</td><td>≈0.01 m</td><td>±2 mm</td><td>可直接取臂孔半径附近</td></tr>
<tr><td><b>m_tip 三档</b></td><td><b>50 / 100 / 150 g</b></td><td>0.1 g</td><td>托盘+砝码+夹紧件<b>一起称</b></td></tr>
</table>

<h3>2.2 激励与采样</h3>
<table>
<tr><th>项</th><th>取值</th><th>理由</th></tr>
<tr><td>轨迹(每条 6 s)</td><td>sin_time_square / sin_sin / lift_and_drop / up_and_down</td><td>背驱+Stribeck / 全速度谱 / 低速+负载</td></tr>
<tr><td>砝码档 × 轨迹 × 重复</td><td>3 × 4 × <b>3</b> = <b>36 条</b></td><td>噪声平均; 多档才有负载相关项</td></tr>
<tr><td>采样率</td><td>目标 ~6 ms/拍(实际以时间戳为准, 期望 6–15 ms)</td><td>固件 DTs=3ms; 总线极限</td></tr>
<tr><td>速度</td><td>位置差分(相邻拍), 不用 reg58</td><td>58 号量化 0.077 rad/s 太粗</td></tr>
<tr><td>固件参数</td><td>Mode=0, Kp=32, Kd=32, Ki=0; reg41=0(最大加速度)</td><td>2026-09-07 对 15 只舵机实测；台架舵机开始前仍需回读</td></tr>
<tr><td>临时限流</td><td><b>0.975 A</b>(reg44=150；结束恢复开始前读到的值)</td><td>防堵转发热；实测出厂保护值为 1.95 A</td></tr>
<tr><td>零位</td><td>扭矩关, 杆自然下垂, 30 读中位(q_zero, 脚本自动)</td><td>pendulum 帧原点</td></tr>
</table>

<h3>2.3 拟合与验收</h3>
<table>
<tr><th>项</th><th>取值</th><th>说明</th></tr>
<tr><td>模型档</td><td>M1-M6 六个候选模型</td><td>同一批数据做 6 次软件拟合，不是额外硬件实验</td></tr>
<tr><td>拟合参数</td><td>kt / R / armature / q_offset / friction_* / error_gain / kd / 限幅(出厂值固定)</td><td>用 m1.json 现值作初值即可</td></tr>
<tr><td>验收门</td><td><b>独立验证 MAE &lt; 0.157 rad</b></td><td>每个“轨迹×质量”的第 3 次不参与拟合，仅用于验证</td></tr>
<tr><td>落地</td><td>mN.json → vendor/bam/params/hls2909/ + constants(需你点头)→ 冒烟 → 长训</td><td>—</td></tr>
</table>

<h2>4. 直观实验流程(逐关放行)</h2>
<div class="ok"><b>A 机械验收 → B 只读检查 → C 50g 单条试跑 → D 36 条正式采集 → E M1-M6 拟合 → F 独立验证后落地。</b><br>
任一关失败就停在当前关，不带病采完整批数据。</div>
<h3>A. 机械验收和称量</h3>
<ol><li>单独称 <code>m_arm</code>：仅摆杆，不含舵盘和端部托。</li>
<li>称 <code>m_hub</code>：舵盘、杆夹和近轴紧固件；测质心半径 <code>r_hub</code>。</li>
<li>三档分别称 <code>m_tip</code>：托盘、砝码、杆端夹和紧固件全部计入。</li>
<li>装好后测 <code>L</code>：输出轴中心到端部总质量质心，不是杆的切割长度。</li>
<li>断电手动摆过 ±90°，确认不碰台架、桌面、线缆和挡板。</li></ol>
<div class="ok"><b>A 关通过：</b>底座用两个 C 夹锁死；轴水平；质量/几何量有记录；最大档完整重力矩≤0.35 Nm。</div>
<h3>B. 接线与只读检查</h3>
<p>断开 12 V 后接线：+12 V 经 3 A 保险丝和急停接舵机；电源 GND、USB-TTL GND、舵机 GND 共地；TTL DATA 接半双工信号线。<b>线色不能凭经验猜，按针脚表逐根核对。</b></p>
<pre>
# ① 准备: 扫描/改ID/确认位置模式(只读+写ID+模式, 不动扭矩)
/usr/bin/python3 scripts/setup_bench_servo.py --port /dev/ttyACM0 --find-id 23 --set-id 1

# ② C关: 50g 试录 1 条(下面数字是格式示例，必须换成实测值)
/usr/bin/python3 scripts/record_pendulum_bench.py --port /dev/ttyACM0 --id 1 \\
  --tip-mass 0.05 --arm-mass 0.018 --arm-length 0.15 --hub-mass 0.020 --hub-radius 0.010 \\
  --trajectory up_and_down --reps 1 --out hls2909_calibration/pilot

# ③ D关: 正式 36 条(约 40-60 min)
/usr/bin/python3 scripts/record_pendulum_bench.py --port /dev/ttyACM0 --id 1 \\
  --tip-mass 0.05 --tip-mass 0.10 --tip-mass 0.15 \\
  --arm-mass 0.018 --arm-length 0.15 --hub-mass 0.020 --hub-radius 0.010 \\
  --trajectory sin_time_square --trajectory sin_sin --trajectory lift_and_drop --trajectory up_and_down \\
  --reps 3 --limit-a 0.975 --torque-budget 0.35 --out hls2909_calibration/bench

# ④ E关: 拟合 + 独立验证(每组第 3 条留出)
.venv/bin/python scripts/fit_leg_pendulum.py --logdir hls2909_calibration/bench \\
  --actuator hls2909 --models m1 m2 m3 m4 m5 m6 --trials 20000 --out hls2909_calibration/fit
</pre>
<h3>到底要做几个实验？</h3>
<table>
<tr><th>序号</th><th>端部总质量</th><th>做什么</th><th>条数</th></tr>
<tr><td>0（试跑）</td><td>50 g</td><td><code>up_and_down</code> 跑 1 次，只检查方向、零位、碰撞和急停</td><td><b>1</b></td></tr>
<tr><td>1</td><td>50 g</td><td>4 种轨迹，每种 3 次</td><td><b>12</b></td></tr>
<tr><td>2</td><td>100 g</td><td>4 种轨迹，每种 3 次</td><td><b>12</b></td></tr>
<tr><td>3</td><td>150 g</td><td>4 种轨迹，每种 3 次</td><td><b>12</b></td></tr>
<tr><td colspan="3"><b>合计：1 条试跑 + 36 条正式数据</b></td><td><b>37</b></td></tr>
</table>
<p>每档质量中的 4 种轨迹固定为：① <code>sin_time_square</code> 速度扫频；② <code>sin_sin</code> 多频复合；③ <code>lift_and_drop</code> 抬起后断扭矩自由落体；④ <code>up_and_down</code> 低速往返。</p>
<div class="note"><b>换砣码不需要重新输命令：</b>脚本每进入一档质量会先关扭矩并暂停；安装、锁紧、检查完成后，在终端输入 <code>START</code> 才会继续。</div>
<h3>每种轨迹的现场步骤</h3>
<table>
<tr><th>轨迹</th><th>脚本动作</th><th>你需要做/观察的事</th></tr>
<tr><td><code>sin_time_square</code><br><span style="color:#666">速度扫频</span></td>
<td>扭矩保持开启。摆杆绕<b>垂直向下零位</b>做正弦扫频：θ = sin(t²)，<b>幅值恒定 ±1 rad（≈±57°）</b>，
频率从 0 平滑升到约 <b>5.7 Hz</b>，6 s 一路从慢到快往复。这是"一次跑完全速度谱"的主推荐激励，用于背驱/Stribeck/全速度段辨识。</td>
<td>站在<b>侧面</b>（摆动平面外）；观察是否撞限位、失步、杆夹滑动——<b>不要用手扶</b>。
杆最大摆到 ~±57°（在水平以内），注意杆/砝码与台架边缘、线束保持间隙。</td></tr>
<tr><td><code>sin_sin</code><br><span style="color:#666">多频复合</span></td>
<td>扭矩保持开启。复合摆动 θ = sin(t)·(π/2) + sin(5t)·0.5·sin(2t)：
<b>主摆 ±90°（到水平）</b>叠加 <b>3~4 倍频的小幅调制</b>（快慢叠加：一个宽带慢摆 + 一个窄带快振）。用于激励谐波/共振/齿轮非线性。</td>
<td>听是否有<b>齿轮卡顿/异响</b>（轻微但高频，靠近听）；检查<b>台架不能跟着晃</b>、夹头不滑。
单次最大摆幅 ~±90°，可能到水平，摆杆扫过平面 <b>清空此方向</b>。</td></tr>
<tr><td><code>lift_and_drop</code><br><span style="color:#666">抬升→自由落体</span></td>
<td>前 <b>2 s</b> 用三次样条把杆从下垂(0°)抬到 <b>-90°（对侧水平）</b>，在 <b>2 s 时刻自动断扭矩</b>，让杆自由落体摆回并自然衰减。用于背驱（断扭矩）与摩擦/Stribeck 辨识。</td>
<td><b>这条最需要清空摆动区</b>——杆会大幅摆回并来回衰减几秒。<b>人站侧面，别在摆动平面</b>。
确认断扭矩后杆能<b>自由摆</b>，<b>不被线束拉住</b>；观察自由衰减是否平滑、有无卡阻。</td></tr>
<tr><td><code>up_and_down</code><br><span style="color:#666">低速往返</span></td>
<td>扭矩保持开启，慢速（共 6 s）：从下垂 <b>0°</b> 缓升到 <b>+90°（水平）</b>，3 s 后缓降到 <b>+72°</b> 并保持到结束。用于<b>低速爬行/静摩擦</b>与负载相关摩擦辨识。</td>
<td>盯<b>低速段</b>（启动 ~0° 附近与 72° 停止时）：是否有<b>爬行/抖动/台阶</b>、是否停得干脆。
观察杆是否<u>匀滑慢动</u>还是<u>一顿一顿</u>——后者的量就是静摩擦。</td></tr>
</table>
<p><b>零位与摆幅速查（θ 约定）：</b>θ = 0 = <b>摆杆垂直向下</b>（静平衡，脚本自动找 q_zero）；<b>+90°（+π/2）与 -90° 分别是从下垂抬到两侧水平</b>。
所有工况都<b>从下垂零位开始、结束后回到零位</b>再等 0.5 s。摆幅速查：sin_time_square ≈±57°；sin_sin ≈±90°（叠高频）；lift_and_drop → -90°；up_and_down → 0°…+90°…+72°。</p>
<p><b>每条结束后：</b>脚本会把目标回到零位并等待 0.5 s。如果看到松动、碰撞、异响、线束拉扯，立即按急停，当条数据作废，修复后重新采集整组 3 次。</p>
<h3>软件内部要点(为什么这样设计)</h3>
<ul>
<li><b>等效化</b>: 已知 (M, B_max) → Pendulum 的 (m_eq, L_eq), <code>arm_mass=0</code>,
与 bam 内部零耦合 —— 不改 vendor 代码;</li>
<li><b>log 格式</b>: 直接兼容 bam's <code>rollout_log</code>: dt/kp/vin/mass/arm_mass/length +
entries[{position, speed, goal_position, torque_enable}], speed 用差分;</li>
<li><b>安全链</b>: 运行前自动拒绝完整重力矩&gt;0.35Nm的组合；临时限流 0.975A、温度&gt;50°C、|I|&gt;1.0A×0.7s、
30s 看门狗 → 立即断扭矩并恢复开始前读到的限流值;</li>
<li><b>数据修正</b>: 编码器角度按 12-bit 跨零展开，速度用前向差分，vin 用每条记录的实测中位数;</li>
<li><b>可复现</b>: 每次换砝码档重称重填 --tip-mass; manifest.json 记录全部输入。</li>
</ul>
<h2>5. 每关验收标准</h2>
<table><tr><th>关卡</th><th>通过标准</th></tr>
<tr><td>B 只读</td><td>只发现 1 只舵机；11.5–12.6 V；温度&lt;45°C；Mode=0；无错误状态</td></tr>
<tr><td>C 试跑</td><td>正角方向正确；0 rad=自然下垂；无滑动；&gt;50 样本；温升&lt;5°C；采样中位周期建议≤20 ms</td></tr>
<tr><td>D 采集</td><td><code>manifest.json</code> 中 records=36；每条&gt;50点；没有安全中止；电压/采样周期无明显漂移</td></tr>
<tr><td>E 拟合</td><td>12 条留出数据验证 MAE&lt;0.157 rad；训练/验证无明显分叉；同等通过时选 m1</td></tr></table>

<h2>6. 安全红线(执行时)</h2>
<div class="warn">
<ul>
<li>m_tip·g·L ≤ 0.35 N·m(0.87 N·m 的一半) —— <b>禁止加大砝码</b>;</li>
<li>摆动平面内手勿入/头勿探; 调 M6 螺纹胶防松(砝码甩出是最大风险);</li>
<li>全程人在场(lift_and_drop 第 2s 后扭矩断电自由落体);</li>
<li>异常断电顺序: 先断 12V 再拔 TTL; 数据目录 zip 备份。</li>
</ul>
</div>

<h2>7. 实验交付清单</h2>
<table>
<tr><th>文件</th><th>作用</th></tr>
<tr><td><code>scripts/setup_bench_servo.py</code></td><td>步骤①: 扫描/改ID/确认位置模式</td></tr>
<tr><td><code>scripts/record_pendulum_bench.py</code></td><td>步骤②③: 自动记录(安全内置)</td></tr>
<tr><td><code>scripts/fit_leg_pendulum.py</code></td><td>步骤④: bam.fit 包装 + MAE 选档</td></tr>
<tr><td><code>cad/pendulum_bench/*.scad</code></td><td>参考打印件(工程师可重设计)</td></tr>
<tr><td><code>scripts/make_pendulum_bench_runbook_html.py</code></td><td>本手册生成器</td></tr>
<tr><td><code>docs/pendulum_bench_runbook.html</code></td><td>本文件</td></tr>
<tr><td><code>hls2909_calibration/bench/</code></td><td>36 条原始 JSON + manifest，不要手工改数据</td></tr>
<tr><td><code>hls2909_calibration/fit/</code></td><td>fit_m1.json / fit_m6.json / mae_report.md</td></tr>
</table>

<h2>8. 参考</h2>
<ul>
<li>BAM 论文: <a href="https://arxiv.org/pdf/2410.08650v1">arXiv 2410.08650</a>(摆锤辨识协议/m1-m6)</li>
<li>BAM: <a href="https://github.com/Rhoban/bam">GitHub</a> · <a href="https://bam.readthedocs.io/en/latest/">文档</a>
(vendor/bam 即其 mjlab_frictionloss 分支)</li>
<li>Feetech 辨识参考: <a href="https://github.com/zeroth-robotics/bam-feetech">zeroth-robotics/bam-feetech</a></li>
</ul>
'''


def main() -> None:
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>独立摆锤台架 · 软件方案与测试取值方案</title>
{CSS}</head><body><div class="wrap">
{BODY}
</div></body></html>"""
    OUT.write_text(html, encoding="utf-8")
    print(f"written {OUT}  ({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
