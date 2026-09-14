#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成《HLS-2909 摆锤实验设计文档》单文件 HTML(详细版)。

依赖: 复用 make_pendulum_bench_runbook_html.py 的图函数(物理/轨迹/流程)。
运行: .venv/bin/python scripts/make_pendulum_experiment_design_html.py
输出: docs/pendulum_bench_experiment_design.html
"""

from __future__ import annotations

from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_pendulum_bench_runbook_html import fig_physics, fig_trajectories, fig_pipeline  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "pendulum_bench_experiment_design.html"


# ── 新图 1: 实验因素-响应变量关系 ────────────────────────────────────────────
def fig_factors() -> str:
    return '''<svg viewBox="0 0 960 300" class="fig" role="img" aria-label="因素-响应图">
<defs><marker id="fA" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
<path d="M0,0 L6,3 L0,6 z" fill="#c00000"/></marker></defs>
<!-- 自变量列 -->
<rect x="30" y="35" width="210" height="230" rx="12" fill="#eef6ff" stroke="#468" stroke-width="2"/>
<text x="135" y="62" font-size="15" text-anchor="middle" font-weight="bold">自变量(3 维正交)</text>
<rect x="50" y="80" width="170" height="46" rx="8" fill="#fff" stroke="#468"/>
<text x="135" y="101" font-size="13" text-anchor="middle">m_tip: 50 / 100 / 160 g</text>
<text x="135" y="118" font-size="11" text-anchor="middle" fill="#555">(端部总质量, 上秤)</text>
<rect x="50" y="136" width="170" height="46" rx="8" fill="#fff" stroke="#468"/>
<text x="135" y="157" font-size="13" text-anchor="middle">轨迹: 3 种 × 6 s</text>
<text x="135" y="174" font-size="11" text-anchor="middle" fill="#555">lift/sin²/updown</text>
<rect x="50" y="192" width="170" height="46" rx="8" fill="#fff" stroke="#468"/>
<text x="135" y="213" font-size="13" text-anchor="middle">重复 n = 5</text>
<text x="135" y="230" font-size="11" text-anchor="middle" fill="#555">3×3×5 = 45 条</text>
<text x="135" y="256" font-size="12" text-anchor="middle" fill="#777">网格: 固定其余因素逐格采集 → 可分离度好</text>
<!-- 箭头 -->
<path d="M255,150 L330,150" stroke="#c00000" stroke-width="3" marker-end="url(#fA)"/>
<text x="292" y="140" font-size="12" text-anchor="middle" fill="#c00000">激励</text>
<!-- 被控对象 -->
<rect x="340" y="78" width="230" height="145" rx="12" fill="#fff3e0" stroke="#c80" stroke-width="2"/>
<text x="455" y="105" font-size="15" text-anchor="middle" font-weight="bold">被控对象/观测</text>
<text x="455" y="132" font-size="13" text-anchor="middle">摆锤台架 + HLS-2909</text>
<text x="455" y="155" font-size="12" text-anchor="middle" fill="#555">已知: M, B(θ), m_eq, L_eq</text>
<text x="455" y="175" font-size="12" text-anchor="middle" fill="#555">未知: 电机+摩擦参数(待辨识)</text>
<text x="455" y="198" font-size="12" text-anchor="middle" fill="#555">控制: Kp=32 Kd=32 限流0.975A</text>
<!-- 箭头 -->
<path d="M585,150 L660,150" stroke="#c00000" stroke-width="3" marker-end="url(#fA)"/>
<text x="622" y="140" font-size="12" text-anchor="middle" fill="#c00000">响应</text>
<!-- 因变量列 -->
<rect x="670" y="35" width="260" height="230" rx="12" fill="#eefcef" stroke="#484" stroke-width="2"/>
<text x="800" y="62" font-size="15" text-anchor="middle" font-weight="bold">因变量(逐拍采集)</text>
<g font-size="12.5">
<text x="688" y="92">θ(t) 位置[rad] ← reg56(0.087°)</text>
<text x="688" y="114">dq(t) 速度[rad/s] ← 位置差分</text>
<text x="688" y="136">duty(t) 占空比[%] ← reg60(0.1%)</text>
<text x="688" y="158">vin(t) 电压[V] ← reg62(0.1V)</text>
<text x="688" y="180">I(t) 电流[A] ← reg69(6.5mA)</text>
<text x="688" y="202">T(t) 温度[°C] + torque_enable</text>
<text x="688" y="230" font-size="11.5" fill="#777">每拍附 t 时间戳(总线实测, 抖动记录)</text>
<text x="688" y="250" font-size="11.5" fill="#777">→ 拟合判据: 回放 MAE(θ)</text>
</g>
</svg>'''


# ── 新图 2: 单条记录时间线(采集数据流) ─────────────────────────────────────
def fig_timeline() -> str:
    return '''<svg viewBox="0 0 960 260" class="fig" role="img" aria-label="单条记录时间线">
<defs><marker id="tA" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
<path d="M0,0 L6,3 L0,6 z" fill="#666"/></marker></defs>
<line x1="40" y1="130" x2="920" y2="130" stroke="#999" stroke-width="2"/>
<g font-size="13">
<circle cx="60" cy="130" r="7" fill="#c00000"/><text x="60" y="105" text-anchor="middle">t=0</text>
<circle cx="330" cy="130" r="7" fill="#c00000"/><text x="330" y="105" text-anchor="middle">轨迹 6s 末</text>
<circle cx="400" cy="130" r="7" fill="#c00000"/><text x="400" y="105" text-anchor="middle">回零</text>
<circle cx="520" cy="130" r="7" fill="#c00000"/><text x="520" y="105" text-anchor="middle">冷却 ≥0.5s</text>
<rect x="620" y="112" width="150" height="36" rx="8" fill="#eef6ff" stroke="#468"/>
<text x="695" y="134" text-anchor="middle">下一条(下一档/轨迹)</text>
</g>
<g font-size="12" fill="#555">
<text x="80" y="160">每 ~6ms 一拍: 写目标(42) + 读 56→θ、60→duty、62→vin、69→I、63→T</text>
<text x="80" y="182">(总线 RTT ≈0.6ms, 一轮读写 ≈4-6ms; 实际时间戳含抖动 → 事后 5ms 重采样)</text>
<text x="80" y="204">抽样帧: 先写后读(控制与测量同拍), 温度/电流每拍顺带 → 安全实时监控</text>
<text x="80" y="226">异常即停: T>50°C / |I|>1.0A×0.7s / 看门狗 30s → 46=0,40=0,55=0 → 该条标记 ERROR</text>
</g>
<line x1="60" y1="140" x2="60" y2="225" stroke="#c00000" stroke-width="1.5" stroke-dasharray="4 3"/>
<line x1="330" y1="140" x2="330" y2="225" stroke="#c00000" stroke-width="1.5" stroke-dasharray="4 3"/>
</svg>'''


CSS = '''
<style>
:root { --accent:#c00000; }
* { box-sizing:border-box; }
body { font-family:"Noto Sans CJK SC","Microsoft YaHei",sans-serif; margin:0;
       background:#fdfdfc; color:#222; line-height:1.7; }
.wrap { max-width:1120px; margin:0 auto; padding:24px 28px 80px; }
h1 { color:var(--accent); border-bottom:4px solid var(--accent); padding-bottom:10px; }
h2 { margin-top:44px; border-left:6px solid var(--accent); padding-left:12px; }
h3 { margin-top:26px; color:#99302a; }
.fig { width:100%; height:auto; background:#fff; border:1px solid #e4e4e4; border-radius:10px;
       margin:12px 0 6px; box-shadow:0 1px 4px rgba(0,0,0,.06); }
table { border-collapse:collapse; width:100%; margin:12px 0 20px; }
th,td { border:1px solid #e4e4e4; padding:8px 12px; text-align:left; font-size:14px; }
th { background:#f5f0e8; }
code,pre { font-family:Consolas,monospace; background:#f4f2ee; border-radius:6px; }
code { padding:1px 6px; font-size:13.5px; }
pre { padding:12px 16px; overflow-x:auto; font-size:13px; }
.note { background:#fff8e1; border:1px solid #e8c96a; border-radius:8px; padding:10px 14px; margin:14px 0; }
.warn { background:#ffebeb; border:1px solid #c00000; border-radius:8px; padding:12px 16px; margin:16px 0; }
.ok { background:#e9f7ec; border:1px solid #7c3; border-radius:8px; padding:10px 14px; margin:14px 0; }
.toc { background:#f7f4ee; border:1px solid #e0d8c8; border-radius:10px; padding:14px 22px; }
</style>
'''

BODY = '''
<h1>HLS-2909 摆锤实验设计文档(详细版)</h1>
<p style="margin-top:-6px">对象: 独立摆锤台架 + 单只 HL-2909-C001(FT-HLS 固件 3.45, 型号号 6922)<br>
快速上手指南: <a href="pendulum_bench_runbook.html">📄 pendulum_bench_runbook.html(速查)</a>
· 本文件是<b>完整实验设计</b>(步骤/代码/采集参数/变量控制)。</p>

<div class="toc"><b>目录</b>
<ol>
<li>术语与符号 · 2. 目标与原理 · 3. 装置与剂量定义(采集参数) · 4. 变量控制(核心)
· 5. 实验步骤(12 步) · 6. 代码清单与命令行 · 7. 数据质量与决策规则 · 8. 安全与中止
· 9. 附录(公式/参考值/JSON 示例)</li>
</ol></div>

<h2>1. 术语与符号</h2>
<table>
<tr><th>符号</th><th>含义</th><th>单位</th><th>来源/测量方式</th></tr>
<tr><td>m_tip</td><td>端部总质量(托盘+砝码片+夹紧件+螺丝)</td><td>kg</td><td>电子秤 0.1g, <b>每档重称</b></td></tr>
<tr><td>m_arm</td><td>摆杆质量(均匀细杆)</td><td>kg</td><td>电子秤(建议 12–25g)</td></tr>
<tr><td>m_hub</td><td>舵机臂+摆杆夹头+螺丝(绕轴同转件)</td><td>kg</td><td>电子秤</td></tr>
<tr><td>L</td><td>轴心→端部砝码<b>质心</b>距离</td><td>m</td><td>卡尺 ±1mm</td></tr>
<tr><td>r_hub</td><td>轴心→m_hub 质心距离</td><td>m</td><td>卡尺 ±2mm(≈臂孔半径)</td></tr>
<tr><td>M</td><td>绕轴总惯量 m_tip·L²+m_arm·L²/3+m_hub·r_hub²</td><td>kg·m²</td><td>由上式(几何已知)</td></tr>
<tr><td>B_max</td><td>最大重力矩 (m_tip+m_arm/2)·g·L+m_hub·g·r_hub</td><td>N·m</td><td>由上式</td></tr>
<tr><td>m_eq, L_eq</td><td>标准 Pendulum 等效质量/长度(arm_mass=0)</td><td>kg, m</td><td>L_eq=M·g/B_max; m_eq=B_max²/(M·g²)</td></tr>
<tr><td>θ</td><td>摆角(pendulum 帧, 0=杆垂直向下)</td><td>rad</td><td>(编码器LSB − q_zero)·LSB_RAD·sign</td></tr>
<tr><td>q_zero</td><td>杆垂直向下时的编码器值</td><td>LSB</td><td>扭矩关, 30 读中位(自动)</td></tr>
<tr><td>sign</td><td>轨迹正角→编码器方向</td><td>±1</td><td>±0.2rad 试步自动检测</td></tr>
<tr><td>dq</td><td>角度差分速度(相邻拍)</td><td>rad/s</td><td>θ 差分 / Δt</td></tr>
<tr><td>MAE</td><td>回放位置平均误差</td><td>rad</td><td>判定门: <b>&lt;0.157</b>(≈10%×π/2)</td></tr>
</table>

<h2>2. 目标与原理</h2>
<p><b>目标</b>: 辨识 HLS-2909 的 kt / R / armature / q_offset / error_gain / kd
与齿轮摩擦(friction_base / viscous / Stribeck / 负载相关项; m1→m6)。<b>为什么用摆锤</b>:
总线恒速法拿不到"负载相关项"(载荷不可控), 且速度范围/分辨率受限; 摆锤提供
<b>已知惯量 + 已知重力矩 + 可控载荷</b>, 是 BAM 协议(arXiv 2410.08650)的标准做法。</p>
@@FIG_PHYSICS@@
<div class="note">等价化意义: 台架几何不满足"点质量+均匀杆"时, (M, B_max) 仍与标准
Pendulum 的动力学精确一致 → <b>bam 内部零改动</b>, 每条 log 只写 m_eq/L_eq。</div>

<h2>3. 采集参数定义</h2>
<h3>3.1 每条记录(log JSON)的元数据字段</h3>
<table>
<tr><th>字段</th><th>取值</th><th>说明</th></tr>
<tr><td>motor</td><td>"hls2909"</td><td>执行器注册名(固定)</td></tr>
<tr><td>kp / vin</td><td>32 / 实测≈12.5</td><td>固件 Kp 出厂值; 电压实测写入</td></tr>
<tr><td>dt</td><td>≈0.006(重采样后 0.005)</td><td>记录间隔(中位) / 重采样网格</td></tr>
<tr><td>mass / arm_mass / length</td><td>m_eq / 0.0 / L_eq</td><td>Pendulum 等效参数(每 m_tip 档)</td></tr>
<tr><td>trajectory / tip_mass_kg / rep</td><td>轨迹名/端部质量/重复号</td><td>分组与可追溯</td></tr>
</table>
<h3>3.2 逐拍(entries[])字段</h3>
<table>
<tr><th>字段</th><th>单位/分辨率</th><th>来源</th><th>物理意义(用于哪一个拟合项)</th></tr>
<tr><td>t</td><td>s(perf_counter)</td><td>本机时钟</td><td>重采样网格/延迟分析</td></tr>
<tr><td>position θ</td><td>rad / 0.087°</td><td>reg56 − q_zero, ×sign</td><td><b>拟合目标(回放误差)</b></td></tr>
<tr><td>speed dq</td><td>rad/s(差分)</td><td>θ 差分</td><td>Stribeck/粘性项的响应速度</td></tr>
<tr><td>goal_position</td><td>rad</td><td>轨迹命令</td><td>回放注入</td></tr>
<tr><td>torque_enable</td><td>bool</td><td>轨迹(t&lt;2s…)</td><td>背驱段(自由落体)</td></tr>
<tr><td>duty</td><td>%, 0.1%</td><td>reg60(11bit 符号幅值)</td><td>限制器是否介入/控制律检查</td></tr>
<tr><td>vin</td><td>V, 0.1V</td><td>reg62</td><td>电压漂移/压降</td></tr>
<tr><td>current</td><td>A, 6.5mA</td><td>reg69(16bit 符号幅值)</td><td>安全监控/限流核验</td></tr>
<tr><td>temp</td><td>°C</td><td>reg63</td><td>安全监控</td></tr>
</table>
<h3>3.3 目录与命名</h3>
<pre>hls2909_calibration/bench/            # ① 原始 log(变间隔)
  ├─ lift_and_drop_tip0.05kg_rep0.json  … 45 条
  ├─ manifest.json                     # 每次运行的完整参数快照
hls2909_calibration/bench_processed/  # ② process 重采样(5ms 均匀网格)
hls2909_calibration/fit/               # ③ fit_m1/m6.json + overlay_*.png + mae_report.md</pre>
@@FIG_TIMELINE@@

<h2>4. 变量控制(核心)</h2>
<h3>4.1 自变量 → 3 维正交网格</h3>
@@FIG_FACTORS@@
<table>
<tr><th>维</th><th>水平</th><th>说明/为什么</th></tr>
<tr><td>m_tip(负载档)</td><td>50 / 100 / 160 g</td><td>多档 → 负载相关项(m3-m6)可辨识; ≤170g 满足力矩预算</td></tr>
<tr><td>轨迹(激励)</td><td>lift_and_drop / sin_time_square / up_and_down</td><td>背驱+Stribeck / 全速度谱 / 低速+静态摩擦</td></tr>
<tr><td>重复</td><td>5</td><td>噪声平均; 最后一重复留作<b>独立验证集</b></td></tr>
</table>
<h3>4.2 控制变量(全程恒定, 逐条记录)</h3>
<table>
<tr><th>变量</th><th>设定</th><th>如何保证</th></tr>
<tr><td>Kp / Kd</td><td>32 / 32(出厂 21/22)</td><td>不写; setup 回读确认</td></tr>
<tr><td>电流限幅</td><td>0.975 A(reg44=150)</td><td>脚本写入; 每条结束恢复 300(1.95A)并回读</td></tr>
<tr><td>轨迹时长</td><td>6 s</td><td>脚本按 trajectory.duration</td></tr>
<tr><td>采样目标</td><td>~6 ms/拍</td><td>循环计时; 实际时间戳如实记录</td></tr>
<tr><td>几何</td><td>L, r_f, 杆/臂不换</td><td>单次实验不拆装; 换档只换砝码</td></tr>
<tr><td>q_zero / sign</td><td>每档(或每次运行)重测</td><td>脚本自动测量并写 manifest</td></tr>
<tr><td>温度上限</td><td>50 °C</td><td>每拍监控, 越限即停</td></tr>
<tr><td>电机状态</td><td>位置伺服 mode=0, 41=254, 46=32767</td><td>setup/record 统一设置</td></tr>
</table>
<h3>4.3 环境/干扰与对策</h3>
<table>
<tr><th>干扰</th><th>表现</th><th>对策/判废规则</th></tr>
<tr><td>温度漂移</td><td>连续记录后期 T 升高</td><td>每 10 条打印 T; &gt;45°C 暂停冷却 2min</td></tr>
<tr><td>总线抖动/丢拍</td><td>dt 忽大忽小</td><td>5ms 重采样; 单条丢拍率&gt;30% 判废</td></tr>
<tr><td>人手触碰/振动</td><td>θ 无命令跳变&gt;5°</td><td>逐条体检: 标记跳过(该条不进拟合)</td></tr>
<tr><td>台架共振</td><td>某砝码档全部异常大 MAE</td><td>垫胶防共振; 若为该档专属 → 检查该档装配</td></tr>
<tr><td>砝码松动</td><td>落体段频率异常</td><td>M6 螺纹胶+防松垫; 每档前扭矩检查</td></tr>
<tr><td>电压漂移</td><td>vin 12.0–12.6</td><td>逐拍记录(模型自动吸收); &lt;11.5V 暂停</td></tr>
</table>
<h3>4.4 顺序与随机化</h3>
<pre>外层 m_tip(50→100→160) → 中层 轨迹 → 内层 rep(0..4)
理由: 正交网格; 换档重称/改参数属"刻意状态改变", 与温度缓漂的时间相关度最低。
(可选加固) 每档完成后插 1 条 nothing 轨迹做"台架状态对照"。</pre>

<h2>5. 实验步骤(12 步, 每步含判定)</h2>
<table>
<tr><th>#</th><th>步骤</th><th>命令/动作</th><th>通过判定</th></tr>
<tr><td>0</td><td>准备</td><td>备齐 BOM(秤/砝码 50·100·160/重物/杆/臂/紧固)+ USB-TTL 供电;<br>sudo chmod a+rw /dev/ttyACM0</td><td>ls /dev/ttyACM0 且 12.4-12.6V</td></tr>
<tr><td>1</td><td>装台架</td><td>按 5 条机械约束(轴水平/±90°/轴高≥L+50/可称量可实测/固定牢)</td><td>手推摆臂: 台架不动、摆动自由</td></tr>
<tr><td>2</td><td>称量/测量</td><td>m_tip(3档各称) / m_arm / m_hub / L / r_hub 记录</td><td>秤 0.1g、卡尺 ±1mm 记录表</td></tr>
<tr><td>3</td><td>配置舵机</td><td>setup_bench_servo.py --find-id X --set-id 1</td><td>mode=0, 限位 0..4095</td></tr>
<tr><td>4</td><td>零位/符号</td><td>record 自动</td><td>q_zero 打印; 符号 Δ≠0</td></tr>
<tr><td>5</td><td>试录 1 条</td><td>--trajectory up_and_down --reps 1</td><td>摆幅&gt;0.5 rad, T&lt;45°C, 无异常</td></tr>
<tr><td>6</td><td>正式 45 条</td><td>record 全网格(见 §6)</td><td>进度 100%, 无安全中止</td></tr>
<tr><td>7</td><td>体检</td><td>脚本/人工: 摆幅/样数/dt/温度/跳变</td><td>≥40 条有效</td></tr>
<tr><td>8</td><td>重采样</td><td>process_bench_logs.py</td><td>输出 bench_processed/ 45 条</td></tr>
<tr><td>9</td><td>拟合</td><td>fit_leg_pendulum.py --models m1 m6</td><td>生成 fit_m1/m6 + overlay + 报告</td></tr>
<tr><td>10</td><td>判档</td><td>独立验证 MAE&lt;0.157 且 train/val 无分叉; 目检 overlay</td><td>选 m1 或 m6</td></tr>
<tr><td>11</td><td>落地</td><td>mN.json → vendor params + constants(需确认)→ 冒烟 → 长训</td><td>—</td></tr>
<tr><td>12</td><td>收尾</td><td>扭矩关/限流恢复回读/数据 zip 备份</td><td>44=300, 40=0, 55=0</td></tr>
</table>

<h2>6. 代码清单与命令行</h2>
<h3>6.1 五个脚本</h3>
<table>
<tr><th>脚本</th><th>作用</th><th>安全内建</th></tr>
<tr><td>scripts/setup_bench_servo.py</td><td>扫描/改ID/确认模式限位</td><td>不动扭矩</td></tr>
<tr><td>scripts/record_pendulum_bench.py</td><td>自动记录(m_tip 多档)</td><td>0.975A/50°C/1.0A×0.7s/30s 看门狗</td></tr>
<tr><td>scripts/process_bench_logs.py</td><td>均匀 dt 重采样</td><td>—</td></tr>
<tr><td>scripts/fit_leg_pendulum.py</td><td>bam.fit + 独立验证 + 复盘图</td><td>—</td></tr>
<tr><td>(可选)scripts/eval_sim_vs_real_step.py</td><td>单舵机阶跃交叉验证(备用)</td><td>—</td></tr>
</table>
<h3>6.2 完整命令行序列(复制即用)</h3>
<pre>
# 0-3 准备与配置
/usr/bin/python3 scripts/setup_bench_servo.py --port /dev/ttyACM0 --find-id 23 --set-id 1

# 5 试录
/usr/bin/python3 scripts/record_pendulum_bench.py --port /dev/ttyACM0 --id 1 \\
  --tip-mass 0.10 --arm-mass 0.018 --arm-length 0.20 --hub-mass 0.020 --hub-radius 0.010 \\
  --trajectory up_and_down --reps 1 --out hls2909_calibration/bench

# 6 正式(3×3×5=45 条, 约 40-60min)
/usr/bin/python3 scripts/record_pendulum_bench.py --port /dev/ttyACM0 --id 1 \\
  --tip-mass 0.05 --tip-mass 0.10 --tip-mass 0.16 \\
  --arm-mass 0.018 --arm-length 0.20 --hub-mass 0.020 --hub-radius 0.010 \\
  --trajectory lift_and_drop --trajectory sin_time_square --trajectory up_and_down \\
  --reps 5 --out hls2909_calibration/bench

# 8 重采样
/usr/bin/python3 scripts/process_bench_logs.py --in hls2909_calibration/bench \\
  --out hls2909_calibration/bench_processed --dt 0.005

# 9-10 拟合+判档
.venv/bin/python scripts/fit_leg_pendulum.py --logdir hls2909_calibration/bench_processed \\
  --actuator hls2909 --models m1 m6 --trials 20000 --out hls2909_calibration/fit
</pre>
<h3>6.3 record 关键参数说明</h3>
<table>
<tr><th>参数</th><th>取值</th><th>如果不填/填错 →</th></tr>
<tr><td>--tip-mass / --arm-mass / --arm-length / --hub-mass / --hub-radius</td><td>实测值</td><td>M/B_max 错 → 等效参数错 → MAE 假性失败; 务必<b>换档重称重填</b></td></tr>
<tr><td>--joint(旧脚本)/ --out</td><td>—</td><td>数据目录分开, 便于多轮对比</td></tr>
<tr><td>--limit-a</td><td>0.975(可 1.2)</td><td>越大越热; 优先不动</td></tr>
<tr><td>--reps</td><td>5(≥2)</td><td>缺独立验证集时会明确报错</td></tr>
</table>
@@FIG_TRAJ@@
@@FIG_PIPE@@

<h2>7. 数据质量与决策规则</h2>
<h3>7.1 逐条质量门槛(步骤 7 体检)</h3>
<table>
<tr><th>指标</th><th>门槛</th><th>不合格动作</th></tr>
<tr><td>摆幅 range(θ)</td><td>&gt;0.5 rad</td><td>检查零位/符号/砝码; 删除</td></tr>
<tr><td>样本数</td><td>≥ 50/条(目标 ~1100)</td><td>丢拍过多; 删除</td></tr>
<tr><td>温度</td><td>&lt;45 °C</td><td>冷却后补录</td></tr>
<tr><td>无命令跳变</td><td>删除 θ 突跳段</td><td>若 &gt;10% → 整条删</td></tr>
<tr><td>dt 中位</td><td>3–15 ms</td><td>异常 → 检查 USB 供电/线缆</td></tr>
</table>
<h3>7.2 拟合决策树</h3>
<pre>独立验证 MAE &lt; 0.157 rad ?
 ├─ 是 且 train≈validation → ✅ 选 MAE 更小档落地(m6 优先)
 ├─ 是 但 train ≪ validation(分叉)   → 过拟合: 加 reps/减参数自由度(固定 max_velocity 等)
 └─ 否
      ├─ 检查 overlay: 落体段/低速段偏差 → Stribeck 需要 m6; 全段移相 → 零位/符号
      ├─ 检查数据篇: 摆幅/样板/装配
      └─ 仍失败 → 增加 trials 或检查"每档重称重填"是否执行
</pre>
<h3>7.3 复盘图判读</h3>
<ul>
<li>黑=真机, 红虚=sim; 看三段: <b>驱动段</b>(斜坡/正弦)、<b>落体段</b>(t&gt;2s 自由下垂)、<b>回位段</b>;</li>
<li>落体段不重合 → 摩擦(背驱/Stribeck)不匹配; 驱动段超前/滞后 → Kp/error_gain/电气参数;</li>
<li>饱和段(duty|&gt;0.4 持续)重合好 → 限流模型正确。</li>
</ul>

<h2>8. 安全与中止准则(红线表)</h2>
<div class="warn">
<table>
<tr><th>红线</th><th>阈值</th><th>动作</th></tr>
<tr><td>温度</td><td>&gt;50 °C</td><td>立即 46=0,40=0,55=0; 冷却再续</td></tr>
<tr><td>电流</td><td>|I|&gt;1.0 A 持续 0.7 s</td><td>同上(限流 0.975A 是硬闸)</td></tr>
<tr><td>看门狗</td><td>单条 &gt;30 s</td><td>同上 + 打 ERROR 标记</td></tr>
<tr><td>力矩预算</td><td>m_tip·g·L ≤ 0.35 N·m</td><td><b>禁止加砝码</b>(L=0.2m → m_tip≤170g)</td></tr>
<tr><td>人员安全</td><td>摆动面 ±(L+40)mm</td><td>手勿入/头勿探(落体段有动能)</td></tr>
<tr><td>断电顺序</td><td>—</td><td>先断 12V, 再拔 TTL</td></tr>
</table>
</div>

<h2>9. 附录</h2>
<h3>9.1 公式推导(等效 Pendulum)</h3>
<pre>台架刚体(绕轴):
  M     = m_tip·L² + m_arm·L²/3 + m_hub·r_hub²
  B_max = (m_tip + m_arm/2)·g·L + m_hub·g·r_hub
标准 Pendulum(点质量 m + 均匀杆 m_a, 长 l):
  M_p = m·l² + m_a·l²/3 ;  B_p(θ) = (m + m_a/2)·g·l·sinθ
取 m_a=0 且令 M_p=M, B_p_max=B_max → 动力学恒等:
  L_eq = M·g/B_max ;  m_eq = B_max²/(M·g²)
注: 任何"绕固定轴的平面摆"动力学只由 (M, B(θ)) 决定 → 结论对任意几何成立。</pre>
<h3>9.2 参考几何(已随 cad/ 提供, 工程师可重设计)</h3>
<table>
<tr><th>件</th><th>尺寸</th></tr>
<tr><td>底座</td><td>120×80×8, 4×M4(孔距 88×52)</td></tr>
<tr><td>立柱</td><td>60×24×230(轴心高 ≈252)</td></tr>
<tr><td>U 夹</td><td>44×31×31, 内腔 34×23×20, 壁 4, M3 顶丝</td></tr>
<tr><td>摆杆</td><td>Ø6×300(取 L≈200), 碳纤/铝</td></tr>
<tr><td>砝码托</td><td>Ø48 托盘 + M6 螺柱(砝码片中心孔 20mm)</td></tr>
<tr><td>(可选)对侧轴承</td><td>608ZZ(8×22×7)</td></tr>
</table>
<h3>9.3 单条 log JSON 示例(节选)</h3>
<pre>{ "motor": "hls2909", "kp": 32, "vin": 12.5, "dt": 0.005,
  "mass": 0.1140, "arm_mass": 0.0, "length": 0.1928,
  "trajectory": "up_and_down", "tip_mass_kg": 0.1, "rep": 0,
  "entries": [
    {"position": 0.0000, "speed": 0.0000, "goal_position": 0.0000, "torque_enable": true},
    {"position": 0.0004, "speed": 0.0800, "goal_position": 0.0012, "torque_enable": true},
    ... ] }</pre>
<h3>9.4 参考链接</h3>
<ul>
<li>BAM 论文: <a href="https://arxiv.org/pdf/2410.08650v1">arXiv 2410.08650</a></li>
<li>BAM: <a href="https://github.com/Rhoban/bam">GitHub</a> · <a href="https://bam.readthedocs.io/en/latest/">文档</a></li>
<li>桌面 BAM 参考(官方辨识管线): /home/joyandai/bam(bam/feetech/record.py、process.py、testbench_mujoco.py)</li>
<li>本仓库: 手册 <a href="pendulum_bench_runbook.html">速查版</a></li>
</ul>
'''


def main() -> None:
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HLS-2909 摆锤实验设计文档(详细版)</title>
{CSS}</head><body><div class="wrap">
{BODY.replace("@@FIG_PHYSICS@@", fig_physics())
  .replace("@@FIG_TIMELINE@@", fig_timeline())
  .replace("@@FIG_FACTORS@@", fig_factors())
  .replace("@@FIG_TRAJ@@", fig_trajectories())
  .replace("@@FIG_PIPE@@", fig_pipeline())}
</div></body></html>"""
    OUT.write_text(html, encoding="utf-8")
    print(f"written {OUT}  ({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
